"""Commit-driven SDD update proposal workflow."""

from __future__ import annotations

from collections.abc import Callable
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from sddraft.analysis.commit_impact import build_commit_impact
from sddraft.analysis.evidence_builder import build_update_evidence_packs
from sddraft.analysis.retrieval import (
    LexicalIndexer,
    load_retrieval_index,
    save_retrieval_index,
)
from sddraft.domain.errors import ConfigError
from sddraft.domain.models import (
    CSCDescriptor,
    KnowledgeChunk,
    ProjectConfig,
    ProposeUpdatesResult,
    RetrievalIndex,
    SDDTemplate,
    SectionUpdateProposal,
    UpdateProposalReport,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_update_proposal_prompt
from sddraft.render.json_artifacts import write_json_model
from sddraft.render.markdown import write_markdown
from sddraft.render.reports import render_update_report_markdown
from sddraft.repo.diff_parser import get_git_diff, parse_diff
from sddraft.repo.scanner import scan_repository
from sddraft.workflows.hierarchy_docs import (
    build_hierarchy_artifact,
    load_hierarchy_artifact,
    persist_hierarchy_outputs,
)


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

    progress(f"[{csc.csc_id}] Scanning repository...")
    scan_result = scan_repository(
        project_config=project_config, csc=csc, repo_root=repo_root
    )
    progress(
        f"[{csc.csc_id}] Scan complete: {len(scan_result.files)} files considered."
    )

    progress(f"[{csc.csc_id}] Parsing commit range {commit_range}...")
    raw_diff = get_git_diff(commit_range=commit_range, repo_root=repo_root)
    file_diffs = parse_diff(raw_diff)
    impact = build_commit_impact(commit_range=commit_range, file_diffs=file_diffs)

    existing_sections = _parse_existing_sections(existing_sdd_path)

    evidence_packs = build_update_evidence_packs(
        template=template,
        csc=csc,
        code_summaries=scan_result.code_summaries,
        interface_summaries=scan_result.interface_summaries,
        dependencies=scan_result.dependencies,
        commit_impact=impact,
        existing_sections=existing_sections,
    )

    proposals: list[SectionUpdateProposal] = []
    resolved_model = model_name or project_config.llm.model_name
    resolved_temperature = (
        project_config.llm.temperature if temperature is None else temperature
    )

    total_sections = len(evidence_packs)
    for idx, pack in enumerate(evidence_packs, start=1):
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

    report = UpdateProposalReport(
        commit_range=commit_range,
        impacted_sections=impact.impacted_sections,
        proposals=proposals,
    )

    output_root = project_config.output_dir / csc.csc_id
    report_markdown_path = output_root / "update_report.md"
    report_json_path = output_root / "update_proposals.json"
    retrieval_index_path = output_root / "retrieval_index.json"
    hierarchy_json_path: Path | None = None
    hierarchy_index_path: Path | None = None
    hierarchy_chunks: list[KnowledgeChunk] = []

    write_markdown(report_markdown_path, render_update_report_markdown(report))
    write_json_model(report_json_path, report)
    progress(f"[{csc.csc_id}] Wrote update report artifacts.")

    if hierarchy_docs_enabled:
        progress(
            f"[{csc.csc_id}] Refreshing hierarchy documentation for impacted subtree..."
        )
        existing_artifact = None
        candidate_hierarchy_json_path = (
            output_root / "hierarchy" / "hierarchy_artifact.json"
        )
        if candidate_hierarchy_json_path.exists():
            try:
                existing_artifact = load_hierarchy_artifact(
                    candidate_hierarchy_json_path
                )
            except (OSError, JSONDecodeError, PydanticValidationError, ValueError):
                existing_artifact = None

        changed_paths = {item.path for item in impact.changed_files}
        hierarchy_artifact = build_hierarchy_artifact(
            csc_id=csc.csc_id,
            repo_root=repo_root,
            scan_result=scan_result,
            llm_client=llm_client,
            model_name=resolved_model,
            temperature=resolved_temperature,
            changed_paths=changed_paths,
            existing_artifact=existing_artifact,
            progress_callback=progress_callback,
        )
        (
            hierarchy_json_path,
            hierarchy_index_path,
            _,
            hierarchy_chunks,
        ) = persist_hierarchy_outputs(
            artifact=hierarchy_artifact,
            output_root=output_root,
            progress_callback=progress_callback,
        )

    indexer = LexicalIndexer()
    new_chunks = _proposal_chunks(report, report_markdown_path) + hierarchy_chunks
    new_chunks.extend(scan_result.code_chunks)

    if retrieval_index_path.exists():
        existing_index = load_retrieval_index(retrieval_index_path)
        if not hierarchy_docs_enabled:
            existing_index = RetrievalIndex(
                chunks=[
                    chunk
                    for chunk in existing_index.chunks
                    if chunk.source_type not in {"file_summary", "directory_summary"}
                ]
            )
        retrieval_index = indexer.update(existing=existing_index, new_chunks=new_chunks)
    else:
        retrieval_index = indexer.build(
            document_chunks=_proposal_chunks(report, report_markdown_path)
            + hierarchy_chunks,
            code_chunks=scan_result.code_chunks,
        )

    save_retrieval_index(retrieval_index, retrieval_index_path)
    progress(f"[{csc.csc_id}] Wrote retrieval index.")

    return ProposeUpdatesResult(
        impact=impact,
        report=report,
        retrieval_index=retrieval_index,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        retrieval_index_path=retrieval_index_path,
        hierarchy_json_path=hierarchy_json_path,
        hierarchy_index_path=hierarchy_index_path,
    )
