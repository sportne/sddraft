"""Initial SDD generation workflow."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.evidence_builder import build_generation_evidence_packs
from sddraft.analysis.retrieval import (
    LexicalIndexer,
    build_document_chunks,
    save_retrieval_index,
)
from sddraft.domain.models import (
    CSCDescriptor,
    GenerateResult,
    ProjectConfig,
    ReviewArtifact,
    SDDDocument,
    SDDTemplate,
    SectionDraft,
    SectionReviewArtifact,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_section_generation_prompt
from sddraft.render.json_artifacts import write_json_model
from sddraft.render.markdown import render_sdd_markdown, write_markdown
from sddraft.repo.scanner import scan_repository


def _ensure_section_defaults(
    section: SectionDraft, section_id: str, title: str
) -> SectionDraft:
    updates: dict[str, object] = {}
    if not section.section_id or section.section_id == "TBD":
        updates["section_id"] = section_id
    if not section.title or section.title == "TBD":
        updates["title"] = title
    if updates:
        return section.model_copy(update=updates)
    return section


def generate_sdd(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    template: SDDTemplate,
    llm_client: LLMClient,
    repo_root: Path,
) -> GenerateResult:
    """Run the end-to-end initial SDD generation workflow."""

    scan_result = scan_repository(
        project_config=project_config, csc=csc, repo_root=repo_root
    )

    evidence_packs = build_generation_evidence_packs(
        template=template,
        csc=csc,
        code_summaries=scan_result.code_summaries,
        interface_summaries=scan_result.interface_summaries,
        dependencies=scan_result.dependencies,
    )

    section_drafts: list[SectionDraft] = []
    for pack in evidence_packs:
        system_prompt, user_prompt = build_section_generation_prompt(pack)
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SectionDraft,
                model_name=project_config.llm.model_name,
                temperature=project_config.llm.temperature,
            )
        )

        section = SectionDraft.model_validate(response.content.model_dump(mode="json"))
        section = _ensure_section_defaults(section, pack.section.id, pack.section.title)

        if not section.evidence_refs:
            section = section.model_copy(
                update={"evidence_refs": pack.evidence_references}
            )
        section_drafts.append(section)

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

    output_root = project_config.output_dir / csc.csc_id
    markdown_path = output_root / "sdd.md"
    review_json_path = output_root / "review_artifact.json"
    retrieval_index_path = output_root / "retrieval_index.json"

    write_markdown(markdown_path, render_sdd_markdown(document))
    write_json_model(review_json_path, review_artifact)

    document_chunks = build_document_chunks(
        document=document,
        review_artifact=review_artifact,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
    )

    indexer = LexicalIndexer()
    retrieval_index = indexer.build(
        document_chunks=document_chunks, code_chunks=scan_result.code_chunks
    )
    save_retrieval_index(retrieval_index, retrieval_index_path)

    return GenerateResult(
        document=document,
        review_artifact=review_artifact,
        retrieval_index=retrieval_index,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
        retrieval_index_path=retrieval_index_path,
    )
