"""Initial SDD generation workflow."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path

from engllm.core.analysis.evidence_builder import build_generation_evidence_pack
from engllm.core.analysis.graph_build import build_graph_store
from engllm.core.analysis.hierarchy import iter_hierarchy_chunks
from engllm.core.analysis.metrics import RunMetricsCollector
from engllm.core.analysis.retrieval import (
    build_document_chunks,
    build_retrieval_store,
    default_retrieval_store_path,
)
from engllm.core.hierarchy_docs import (
    build_hierarchy_store,
    excerpt_file_path,
    load_hierarchy_summary_evidence,
)
from engllm.core.render.json_artifacts import write_json_model
from engllm.core.repo.scanner import scan_repository_stream
from engllm.core.workspaces import build_workspace_context, tool_artifact_root
from engllm.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    GenerateResult,
    KnowledgeChunk,
    ProjectConfig,
    ReviewArtifact,
    ScanResult,
    SDDDocument,
    SDDTemplate,
    SectionDraft,
    SectionReviewArtifact,
    SymbolSummary,
)
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.sdd.builders import build_section_generation_prompt
from engllm.tools.sdd.markdown import render_sdd_markdown, write_markdown


def _enforce_template_section_identity(
    section: SectionDraft, section_id: str, title: str
) -> SectionDraft:
    """Force generated section identity to match the template contract."""

    updates: dict[str, object] = {}
    if section.section_id != section_id:
        updates["section_id"] = section_id
    if section.title != title:
        updates["title"] = title
    if updates:
        return section.model_copy(update=updates)
    return section


def _progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _to_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def _to_absolute(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _section_uses_hierarchy(section_kinds: list[str]) -> bool:
    normalized = {item.lower() for item in section_kinds}
    return any(
        key in normalized
        for key in {
            "hierarchy",
            "hierarchy_summary",
            "file_summary",
            "directory_summary",
        }
    )


def _scan_and_spool_chunks(
    *,
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
    output_root: Path,
) -> tuple[ScanResult, Path, int, Path]:
    """Stream scan results to disk so memory use stays bounded.

    Returns:
    - `ScanResult` with file-level summaries (no in-memory code chunks)
    - path to the temporary JSONL chunk spool
    - number of chunks written
    - path to excerpt files used by hierarchy generation
    """

    files: list[Path] = []
    code_summaries: list[CodeUnitSummary] = []
    symbol_summaries: list[SymbolSummary] = []
    dependency_values: set[str] = set()
    chunk_count = 0

    spool_path = output_root / ".scan_code_chunks.jsonl"
    excerpt_root = output_root / ".scan_excerpts"
    output_root.mkdir(parents=True, exist_ok=True)
    excerpt_root.mkdir(parents=True, exist_ok=True)

    # The scanner yields one file at a time. We immediately persist each chunk
    # instead of keeping all chunks in memory.
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

                # Excerpts are appended per file and reused for hierarchy docs.
                excerpt_path = excerpt_file_path(excerpt_root, chunk.source_path)
                excerpt_path.parent.mkdir(parents=True, exist_ok=True)
                with excerpt_path.open("a", encoding="utf-8") as excerpt_handle:
                    excerpt_handle.write(chunk.text)
                    excerpt_handle.write("\n\n")

    scan_result = ScanResult(
        files=files,
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        dependencies=sorted(dependency_values),
        code_chunks=[],
    )
    return scan_result, spool_path, chunk_count, excerpt_root


def _iter_spooled_chunks(path: Path) -> Iterator[KnowledgeChunk]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            yield KnowledgeChunk.model_validate(raw)


def generate_sdd(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    template: SDDTemplate,
    llm_client: LLMClient,
    repo_root: Path,
    model_name: str | None = None,
    temperature: float | None = None,
    hierarchy_docs_enabled: bool = True,
    graph_enabled: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> GenerateResult:
    """Run the end-to-end initial SDD generation workflow."""

    workspace = build_workspace_context(
        output_root=project_config.workspace.output_root,
        workspace_id=csc.csc_id,
        kind="csc",
        repo_root=repo_root,
    )
    shared_root = workspace.shared_root
    sdd_root = tool_artifact_root(workspace, "sdd")
    metrics_collector = RunMetricsCollector(csc_id=csc.csc_id)

    _progress(progress_callback, f"[{csc.csc_id}] Scanning repository...")
    metrics_collector.start("scan")
    scan_result, scan_chunks_path, code_chunk_count, scan_excerpts_root = (
        _scan_and_spool_chunks(
            project_config=project_config,
            csc=csc,
            repo_root=repo_root,
            output_root=shared_root,
        )
    )
    metrics_collector.finish(
        files_seen=len(scan_result.files),
        chunks_written=code_chunk_count,
    )
    _progress(
        progress_callback,
        f"[{csc.csc_id}] Scan complete: {len(scan_result.files)} files considered.",
    )

    resolved_model = model_name or project_config.llm.model_name
    resolved_temperature = (
        project_config.llm.temperature if temperature is None else temperature
    )

    hierarchy_manifest_path: Path | None = None
    hierarchy_store_path: Path | None = None
    hierarchy_chunk_count = 0
    hierarchy_summary_evidence: list[str] = []

    if hierarchy_docs_enabled:
        metrics_collector.start("hierarchy")
        _progress(
            progress_callback, f"[{csc.csc_id}] Preparing hierarchy documentation..."
        )

        changed_paths: set[Path] | None = None
        candidate_manifest_path = shared_root / "hierarchy" / "manifest.json"
        if candidate_manifest_path.exists():
            try:
                manifest_mtime = candidate_manifest_path.stat().st_mtime
                changed_paths = set()
                for file_path in scan_result.files:
                    relative_path = _to_relative(file_path, repo_root)
                    try:
                        if (
                            _to_absolute(file_path, repo_root).stat().st_mtime
                            > manifest_mtime
                        ):
                            changed_paths.add(relative_path)
                    except OSError:
                        changed_paths.add(relative_path)
                _progress(
                    progress_callback,
                    f"[{csc.csc_id}] Found existing hierarchy store; {len(changed_paths)} changed file(s) detected.",
                )
            except OSError:
                changed_paths = None
                _progress(
                    progress_callback,
                    f"[{csc.csc_id}] Existing hierarchy store unreadable; rebuilding hierarchy.",
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

        if any(
            _section_uses_hierarchy(section.evidence_kinds)
            for section in template.sections
        ):
            hierarchy_summary_evidence = load_hierarchy_summary_evidence(
                hierarchy_store.manifest_path
            )

        metrics_collector.finish(chunks_written=hierarchy_chunk_count)

    metrics_collector.start("generate_sections")
    section_drafts: list[SectionDraft] = []
    total_sections = len(template.sections)
    for idx, section_spec in enumerate(template.sections, start=1):
        section_hierarchy_evidence = (
            hierarchy_summary_evidence
            if _section_uses_hierarchy(section_spec.evidence_kinds)
            else []
        )
        pack = build_generation_evidence_pack(
            section=section_spec,
            csc=csc,
            code_summaries=scan_result.code_summaries,
            symbol_summaries=scan_result.symbol_summaries,
            dependencies=scan_result.dependencies,
            hierarchy_summaries=section_hierarchy_evidence,
        )
        _progress(
            progress_callback,
            f"[{csc.csc_id}] Generating section {idx}/{total_sections}: {pack.section.id} {pack.section.title}",
        )
        system_prompt, user_prompt = build_section_generation_prompt(pack)
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SectionDraft,
                model_name=resolved_model,
                temperature=resolved_temperature,
            )
        )

        section = SectionDraft.model_validate(response.content.model_dump(mode="json"))
        section = _enforce_template_section_identity(
            section, pack.section.id, pack.section.title
        )

        if not section.evidence_refs:
            section = section.model_copy(
                update={"evidence_refs": pack.evidence_references}
            )
        section_drafts.append(section)
    metrics_collector.finish()

    document = SDDDocument(
        csc_id=csc.csc_id,
        title=f"{csc.title} Software Design Description",
        sections=section_drafts,
    )

    review_artifact = ReviewArtifact(
        csc_id=csc.csc_id,
        sections=[
            SectionReviewArtifact(
                section_id=section.section_id,
                evidence_references=section.evidence_refs,
                assumptions=section.assumptions,
                missing_information=section.missing_information,
                confidence=section.confidence,
            )
            for section in section_drafts
        ],
    )

    markdown_path = sdd_root / "sdd.md"
    review_json_path = sdd_root / "review_artifact.json"
    retrieval_index_path = default_retrieval_store_path(shared_root)
    run_metrics_path = sdd_root / "run_metrics.json"

    write_markdown(markdown_path, render_sdd_markdown(document))
    write_json_model(review_json_path, review_artifact)
    _progress(
        progress_callback, f"[{csc.csc_id}] Wrote SDD markdown and review artifact."
    )

    document_chunks = build_document_chunks(
        document=document,
        review_artifact=review_artifact,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
    )

    metrics_collector.start("build_retrieval")

    hierarchy_chunk_iter: Iterator[KnowledgeChunk]
    if hierarchy_manifest_path is not None:
        hierarchy_chunk_iter = iter_hierarchy_chunks(hierarchy_manifest_path)
    else:
        hierarchy_chunk_iter = iter(())

    combined_chunks = chain(
        document_chunks,
        hierarchy_chunk_iter,
        _iter_spooled_chunks(scan_chunks_path),
    )
    retrieval_manifest = build_retrieval_store(
        store_root=retrieval_index_path,
        chunks=combined_chunks,
        shard_size=project_config.generation.index_shard_size,
        write_batch_size=project_config.generation.write_batch_size,
        max_in_memory_records=project_config.generation.max_in_memory_records,
    )
    metrics_collector.finish(
        chunks_written=len(document_chunks) + hierarchy_chunk_count + code_chunk_count
    )
    scan_chunks_path.unlink(missing_ok=True)
    shutil.rmtree(scan_excerpts_root, ignore_errors=True)

    write_json_model(run_metrics_path, metrics_collector.metrics)
    _progress(
        progress_callback, f"[{csc.csc_id}] Wrote retrieval store and run metrics."
    )

    graph_manifest_path: Path | None = None
    graph_store_path: Path | None = None
    if graph_enabled:
        metrics_collector.start("build_graph")
        _progress(progress_callback, f"[{csc.csc_id}] Building engineering graph...")
        graph_result = build_graph_store(
            csc_id=csc.csc_id,
            repo_root=repo_root,
            output_root=shared_root,
            retrieval_root=retrieval_index_path,
            scan_result=scan_result,
            document=document,
            review_artifact=review_artifact,
        )
        graph_manifest_path = graph_result.manifest_path
        graph_store_path = graph_result.store_root
        metrics_collector.finish(
            chunks_written=graph_result.node_count + graph_result.edge_count
        )
        write_json_model(run_metrics_path, metrics_collector.metrics)
        _progress(progress_callback, f"[{csc.csc_id}] Wrote engineering graph store.")

    return GenerateResult(
        document=document,
        review_artifact=review_artifact,
        retrieval_manifest=retrieval_manifest,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
        retrieval_index_path=retrieval_index_path,
        run_metrics_path=run_metrics_path,
        hierarchy_manifest_path=hierarchy_manifest_path,
        hierarchy_store_path=hierarchy_store_path,
        graph_manifest_path=graph_manifest_path,
        graph_store_path=graph_store_path,
    )
