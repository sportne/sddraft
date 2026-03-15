"""Initial SDD generation workflow."""

from __future__ import annotations

from collections.abc import Callable
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from sddraft.analysis.evidence_builder import build_generation_evidence_packs
from sddraft.analysis.retrieval import (
    LexicalIndexer,
    build_document_chunks,
    save_retrieval_index,
)
from sddraft.domain.models import (
    CSCDescriptor,
    GenerateResult,
    KnowledgeChunk,
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
from sddraft.workflows.hierarchy_docs import (
    build_hierarchy_artifact,
    load_hierarchy_artifact,
    persist_hierarchy_outputs,
)


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


def _progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _to_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def generate_sdd(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    template: SDDTemplate,
    llm_client: LLMClient,
    repo_root: Path,
    model_name: str | None = None,
    temperature: float | None = None,
    hierarchy_docs_enabled: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> GenerateResult:
    """Run the end-to-end initial SDD generation workflow."""

    _progress(progress_callback, f"[{csc.csc_id}] Scanning repository...")
    scan_result = scan_repository(
        project_config=project_config, csc=csc, repo_root=repo_root
    )
    _progress(
        progress_callback,
        f"[{csc.csc_id}] Scan complete: {len(scan_result.files)} files considered.",
    )

    _progress(
        progress_callback, f"[{csc.csc_id}] Building deterministic evidence packs..."
    )
    evidence_packs = build_generation_evidence_packs(
        template=template,
        csc=csc,
        code_summaries=scan_result.code_summaries,
        interface_summaries=scan_result.interface_summaries,
        dependencies=scan_result.dependencies,
    )

    resolved_model = model_name or project_config.llm.model_name
    resolved_temperature = (
        project_config.llm.temperature if temperature is None else temperature
    )

    section_drafts: list[SectionDraft] = []
    total_sections = len(evidence_packs)
    for idx, pack in enumerate(evidence_packs, start=1):
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
    _progress(
        progress_callback, f"[{csc.csc_id}] Wrote SDD markdown and review artifact."
    )

    document_chunks = build_document_chunks(
        document=document,
        review_artifact=review_artifact,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
    )
    hierarchy_json_path: Path | None = None
    hierarchy_index_path: Path | None = None
    hierarchy_chunks: list[KnowledgeChunk] = []

    if hierarchy_docs_enabled:
        _progress(
            progress_callback, f"[{csc.csc_id}] Preparing hierarchy documentation..."
        )
        existing_artifact = None
        changed_paths: set[Path] | None = None
        candidate_hierarchy_json_path = (
            output_root / "hierarchy" / "hierarchy_artifact.json"
        )
        if candidate_hierarchy_json_path.exists():
            try:
                existing_artifact = load_hierarchy_artifact(
                    candidate_hierarchy_json_path
                )
                artifact_mtime = candidate_hierarchy_json_path.stat().st_mtime
                changed_paths = set()
                for file_path in scan_result.files:
                    relative_path = _to_relative(file_path, repo_root)
                    try:
                        if file_path.stat().st_mtime > artifact_mtime:
                            changed_paths.add(relative_path)
                    except OSError:
                        changed_paths.add(relative_path)
                _progress(
                    progress_callback,
                    f"[{csc.csc_id}] Found existing hierarchy artifact; {len(changed_paths)} changed file(s) detected.",
                )
            except (OSError, JSONDecodeError, PydanticValidationError, ValueError):
                existing_artifact = None
                changed_paths = None
                _progress(
                    progress_callback,
                    f"[{csc.csc_id}] Existing hierarchy artifact unreadable; rebuilding hierarchy.",
                )

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
    retrieval_index = indexer.build(
        document_chunks=document_chunks + hierarchy_chunks,
        code_chunks=scan_result.code_chunks,
    )
    save_retrieval_index(retrieval_index, retrieval_index_path)
    _progress(progress_callback, f"[{csc.csc_id}] Wrote retrieval index.")

    return GenerateResult(
        document=document,
        review_artifact=review_artifact,
        retrieval_index=retrieval_index,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
        retrieval_index_path=retrieval_index_path,
        hierarchy_json_path=hierarchy_json_path,
        hierarchy_index_path=hierarchy_index_path,
    )
