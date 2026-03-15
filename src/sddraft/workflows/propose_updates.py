"""Commit-driven SDD update proposal workflow."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.commit_impact import build_commit_impact
from sddraft.analysis.evidence_builder import build_update_evidence_packs
from sddraft.analysis.retrieval import (
    LexicalIndexer,
    load_retrieval_index,
    save_retrieval_index,
)
from sddraft.domain.models import (
    CSCDescriptor,
    KnowledgeChunk,
    ProjectConfig,
    ProposeUpdatesResult,
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
) -> ProposeUpdatesResult:
    """Run commit-impact analysis and generate section update proposals."""

    scan_result = scan_repository(
        project_config=project_config, csc=csc, repo_root=repo_root
    )

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
    for pack in evidence_packs:
        system_prompt, user_prompt = build_update_proposal_prompt(pack)
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SectionUpdateProposal,
                model_name=project_config.llm.model_name,
                temperature=project_config.llm.temperature,
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

    write_markdown(report_markdown_path, render_update_report_markdown(report))
    write_json_model(report_json_path, report)

    indexer = LexicalIndexer()
    new_chunks = (
        _proposal_chunks(report, report_markdown_path) + scan_result.code_chunks
    )

    if retrieval_index_path.exists():
        existing_index = load_retrieval_index(retrieval_index_path)
        retrieval_index = indexer.update(existing=existing_index, new_chunks=new_chunks)
    else:
        retrieval_index = indexer.build(
            document_chunks=_proposal_chunks(report, report_markdown_path),
            code_chunks=scan_result.code_chunks,
        )

    save_retrieval_index(retrieval_index, retrieval_index_path)

    return ProposeUpdatesResult(
        impact=impact,
        report=report,
        retrieval_index=retrieval_index,
        report_markdown_path=report_markdown_path,
        report_json_path=report_json_path,
        retrieval_index_path=retrieval_index_path,
    )
