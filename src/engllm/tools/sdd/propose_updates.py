"""Commit-driven SDD update proposal workflow."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path

from engllm.core.analysis.commit_impact import build_commit_impact
from engllm.core.analysis.evidence_builder import build_update_evidence_pack
from engllm.core.analysis.graph_build import build_graph_store
from engllm.core.analysis.hierarchy import iter_hierarchy_chunks
from engllm.core.analysis.metrics import RunMetricsCollector
from engllm.core.analysis.retrieval import (
    build_retrieval_store,
    default_retrieval_store_path,
)
from engllm.core.hierarchy_docs import (
    build_hierarchy_store,
    excerpt_file_path,
)
from engllm.core.render.json_artifacts import write_json_model
from engllm.core.repo.diff_parser import get_git_diff, parse_diff
from engllm.core.repo.scanner import scan_repository_stream
from engllm.core.workspaces import build_workspace_context, tool_artifact_root
from engllm.domain.errors import ConfigError
from engllm.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    KnowledgeChunk,
    ProjectConfig,
    ProposeUpdatesResult,
    ScanResult,
    SDDTemplate,
    SectionUpdateProposal,
    SymbolSummary,
    UpdateProposalReport,
)
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.sdd.builders import build_update_proposal_prompt
from engllm.tools.sdd.markdown import write_markdown
from engllm.tools.sdd.render import render_update_report_markdown


def _parse_existing_sections(markdown_path: Path) -> dict[str, str]:
    if not markdown_path.exists():
        return {}

    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    sections: dict[str, str] = {}
    current_id: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_id, buffer
        if current_id is not None:
            sections[current_id] = "\n".join(buffer).strip() or "TBD"
        current_id = None
        buffer = []

    for line in lines:
        if line.startswith("## "):
            flush()
            tokens = line[3:].split(maxsplit=1)
            if tokens:
                current_id = tokens[0]
            continue
        if current_id is not None:
            buffer.append(line)

    flush()
    return sections


def _proposal_chunks(
    report: UpdateProposalReport, source_path: Path
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for proposal in report.proposals:
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"update::{proposal.section_id}",
                source_type="sdd_section",
                source_path=source_path,
                section_id=proposal.section_id,
                text=proposal.proposed_text,
            )
        )
    return chunks


def _scan_and_spool_chunks(
    *,
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
    output_root: Path,
) -> tuple[
    list[Path],
    list[CodeUnitSummary],
    list[SymbolSummary],
    list[str],
    Path,
    int,
    Path,
]:
    files: list[Path] = []
    code_summaries: list[CodeUnitSummary] = []
    symbol_summaries: list[SymbolSummary] = []
    dependency_values: set[str] = set()
    chunk_count = 0

    spool_path = output_root / ".scan_code_chunks.jsonl"
    excerpt_root = output_root / ".scan_excerpts"
    output_root.mkdir(parents=True, exist_ok=True)
    excerpt_root.mkdir(parents=True, exist_ok=True)

    with spool_path.open("w", encoding="utf-8") as handle:
        for record in scan_repository_stream(
            project_config=project_config,
            csc=csc,
            repo_root=repo_root,
        ):
            files.append(record.path)
            code_summaries.append(record.code_summary)
            symbol_summaries.extend(record.symbol_summaries)
            dependency_values.update(record.code_summary.imports)

            for chunk in record.code_chunks:
                handle.write(json.dumps(chunk.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
                chunk_count += 1

                excerpt_path = excerpt_file_path(excerpt_root, chunk.source_path)
                excerpt_path.parent.mkdir(parents=True, exist_ok=True)
                with excerpt_path.open("a", encoding="utf-8") as excerpt_handle:
                    excerpt_handle.write(chunk.text)
                    excerpt_handle.write("\n\n")

    return (
        files,
        code_summaries,
        symbol_summaries,
        sorted(dependency_values),
        spool_path,
        chunk_count,
        excerpt_root,
    )


def _iter_spooled_chunks(path: Path) -> Iterator[KnowledgeChunk]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            yield KnowledgeChunk.model_validate(raw)


def propose_updates(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    template: SDDTemplate,
    llm_client: LLMClient,
    existing_sdd_path: Path,
    commit_range: str,
    repo_root: Path,
    model_name: str | None = None,
    temperature: float | None = None,
    hierarchy_docs_enabled: bool = True,
    graph_enabled: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> ProposeUpdatesResult:
    """Run commit-impact analysis and generate section update proposals."""

    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if not existing_sdd_path.exists():
        raise ConfigError(f"Existing SDD file not found: {existing_sdd_path}")
    if not existing_sdd_path.is_file():
        raise ConfigError(f"Existing SDD path is not a file: {existing_sdd_path}")

    workspace = build_workspace_context(
        output_root=project_config.workspace.output_root,
        workspace_id=csc.csc_id,
        kind="csc",
        repo_root=repo_root,
    )
    shared_root = workspace.shared_root
    sdd_root = tool_artifact_root(workspace, "sdd")
    retrieval_index_path = default_retrieval_store_path(shared_root)
    run_metrics_path = sdd_root / "run_metrics.json"
    metrics_collector = RunMetricsCollector(csc_id=csc.csc_id)

    progress(f"[{csc.csc_id}] Scanning repository...")
    metrics_collector.start("scan")
    (
        scan_files,
        code_summaries,
        symbol_summaries,
        dependencies,
        scan_chunks_path,
        code_chunk_count,
        scan_excerpts_root,
    ) = _scan_and_spool_chunks(
        project_config=project_config,
        csc=csc,
        repo_root=repo_root,
        output_root=shared_root,
    )
    metrics_collector.finish(
        files_seen=len(scan_files),
        chunks_written=code_chunk_count,
    )
    progress(f"[{csc.csc_id}] Scan complete: {len(scan_files)} files considered.")

    progress(f"[{csc.csc_id}] Parsing commit range {commit_range}...")
    metrics_collector.start("diff_impact")
    raw_diff = get_git_diff(commit_range=commit_range, repo_root=repo_root)
    file_diffs = parse_diff(raw_diff)
    impact = build_commit_impact(commit_range=commit_range, file_diffs=file_diffs)
    metrics_collector.finish(files_seen=len(file_diffs))

    existing_sections = _parse_existing_sections(existing_sdd_path)

    proposals: list[SectionUpdateProposal] = []
    resolved_model = model_name or project_config.llm.model_name
    resolved_temperature = (
        project_config.llm.temperature if temperature is None else temperature
    )

    metrics_collector.start("propose_sections")
    total_sections = len(template.sections)
    for idx, section_spec in enumerate(template.sections, start=1):
        pack = build_update_evidence_pack(
            section=section_spec,
            csc=csc,
            code_summaries=code_summaries,
            symbol_summaries=symbol_summaries,
            dependencies=dependencies,
            commit_impact=impact,
            existing_sections=existing_sections,
        )
        if pack is None:
            continue

        progress(
            f"[{csc.csc_id}] Proposing section update {idx}/{total_sections}: "
            f"{pack.section.id} {pack.section.title}"
        )
        system_prompt, user_prompt = build_update_proposal_prompt(pack)
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SectionUpdateProposal,
                model_name=resolved_model,
                temperature=resolved_temperature,
            )
        )
        proposal = SectionUpdateProposal.model_validate(
            response.content.model_dump(mode="json")
        )

        if proposal.section_id == "TBD":
            proposal = proposal.model_copy(update={"section_id": pack.section.id})
        if proposal.title == "TBD":
            proposal = proposal.model_copy(update={"title": pack.section.title})
        if proposal.existing_text == "TBD":
            proposal = proposal.model_copy(
                update={"existing_text": pack.existing_section_text or "TBD"}
            )
        if not proposal.evidence_refs:
            proposal = proposal.model_copy(
                update={"evidence_refs": pack.evidence_references}
            )

        proposals.append(proposal)
    metrics_collector.finish()

    report = UpdateProposalReport(
        commit_range=commit_range,
        impacted_sections=impact.impacted_sections,
        proposals=proposals,
    )

    report_markdown_path = sdd_root / "update_report.md"
    report_json_path = sdd_root / "update_proposals.json"
    hierarchy_manifest_path: Path | None = None
    hierarchy_store_path: Path | None = None
    graph_manifest_path: Path | None = None
    graph_store_path: Path | None = None
    hierarchy_chunk_count = 0

    write_markdown(report_markdown_path, render_update_report_markdown(report))
    write_json_model(report_json_path, report)
    progress(f"[{csc.csc_id}] Wrote update report artifacts.")

    if hierarchy_docs_enabled:
        metrics_collector.start("hierarchy")
        progress(
            f"[{csc.csc_id}] Refreshing hierarchy documentation for impacted subtree..."
        )

        changed_paths = {item.path for item in impact.changed_files}
        scan_result = ScanResult(
            files=scan_files,
            code_summaries=code_summaries,
            symbol_summaries=symbol_summaries,
            dependencies=dependencies,
            code_chunks=[],
        )
        hierarchy_store = build_hierarchy_store(
            csc_id=csc.csc_id,
            repo_root=repo_root,
            output_root=shared_root,
            scan_result=scan_result,
            llm_client=llm_client,
            model_name=resolved_model,
            temperature=resolved_temperature,
            changed_paths=changed_paths,
            excerpt_root=scan_excerpts_root,
            progress_callback=progress_callback,
        )
        hierarchy_manifest_path = hierarchy_store.manifest_path
        hierarchy_store_path = hierarchy_store.store_root
        hierarchy_chunk_count = hierarchy_store.chunk_count
        metrics_collector.finish(chunks_written=hierarchy_chunk_count)

    metrics_collector.start("build_retrieval")
    new_chunks = _proposal_chunks(report, report_markdown_path)

    hierarchy_chunk_iter: Iterator[KnowledgeChunk]
    if hierarchy_manifest_path is not None:
        hierarchy_chunk_iter = iter_hierarchy_chunks(hierarchy_manifest_path)
    else:
        hierarchy_chunk_iter = iter(())

    all_chunks = chain(
        new_chunks,
        hierarchy_chunk_iter,
        _iter_spooled_chunks(scan_chunks_path),
    )

    retrieval_manifest = build_retrieval_store(
        store_root=retrieval_index_path,
        chunks=all_chunks,
        shard_size=project_config.generation.index_shard_size,
        write_batch_size=project_config.generation.write_batch_size,
        max_in_memory_records=project_config.generation.max_in_memory_records,
    )
    metrics_collector.finish(
        chunks_written=len(new_chunks) + hierarchy_chunk_count + code_chunk_count
    )

    scan_chunks_path.unlink(missing_ok=True)
    shutil.rmtree(scan_excerpts_root, ignore_errors=True)
    write_json_model(run_metrics_path, metrics_collector.metrics)
    progress(f"[{csc.csc_id}] Wrote retrieval store and run metrics.")

    if graph_enabled:
        metrics_collector.start("build_graph")
        progress(f"[{csc.csc_id}] Building engineering graph...")
        scan_result_for_graph = ScanResult(
            files=scan_files,
            code_summaries=code_summaries,
            symbol_summaries=symbol_summaries,
            dependencies=dependencies,
            code_chunks=[],
        )
        graph_result = build_graph_store(
            csc_id=csc.csc_id,
            repo_root=repo_root,
            output_root=shared_root,
            retrieval_root=retrieval_index_path,
            scan_result=scan_result_for_graph,
            update_report=report,
            commit_impact=impact,
            changed_files={item.path for item in impact.changed_files},
            impacted_section_ids={item.section_id for item in report.proposals},
        )
        graph_manifest_path = graph_result.manifest_path
        graph_store_path = graph_result.store_root
        metrics_collector.finish(
            chunks_written=graph_result.node_count + graph_result.edge_count
        )
        write_json_model(run_metrics_path, metrics_collector.metrics)
        progress(f"[{csc.csc_id}] Wrote engineering graph store.")

    return ProposeUpdatesResult(
        impact=impact,
        report=report,
        retrieval_manifest=retrieval_manifest,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        retrieval_index_path=retrieval_index_path,
        run_metrics_path=run_metrics_path,
        hierarchy_manifest_path=hierarchy_manifest_path,
        hierarchy_store_path=hierarchy_store_path,
        graph_manifest_path=graph_manifest_path,
        graph_store_path=graph_store_path,
    )
