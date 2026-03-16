"""Initial SDD generation workflow."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from itertools import chain
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from sddraft.analysis.evidence_builder import build_generation_evidence_pack
from sddraft.analysis.metrics import RunMetricsCollector
from sddraft.analysis.retrieval import (
    build_document_chunks,
    build_retrieval_store,
    default_retrieval_store_path,
)
from sddraft.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    GenerateResult,
    InterfaceSummary,
    KnowledgeChunk,
    ProjectConfig,
    ReviewArtifact,
    ScanResult,
    SDDDocument,
    SDDTemplate,
    SectionDraft,
    SectionReviewArtifact,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_section_generation_prompt
from sddraft.render.json_artifacts import write_json_model
from sddraft.render.markdown import render_sdd_markdown, write_markdown
from sddraft.repo.scanner import scan_repository_stream
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


def _scan_and_spool_chunks(
    *,
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
    output_root: Path,
) -> tuple[ScanResult, Path, int, dict[Path, str]]:
    files: list[Path] = []
    code_summaries: list[CodeUnitSummary] = []
    interface_summaries: list[InterfaceSummary] = []
    dependency_values: set[str] = set()
    chunk_count = 0

    excerpt_parts: dict[Path, list[str]] = defaultdict(list)

    spool_path = output_root / ".scan_code_chunks.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

    with spool_path.open("w", encoding="utf-8") as handle:
        for record in scan_repository_stream(
            project_config=project_config,
            csc=csc,
            repo_root=repo_root,
        ):
            files.append(record.path)
            code_summaries.append(record.code_summary)
            interface_summaries.extend(record.interface_summaries)
            dependency_values.update(record.code_summary.imports)

            for chunk in record.code_chunks:
                handle.write(json.dumps(chunk.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
                chunk_count += 1
                path_key = chunk.source_path
                if len(excerpt_parts[path_key]) < 2:
                    excerpt_parts[path_key].append(chunk.text)

    excerpts = {
        path: "\n\n".join(parts).strip()
        for path, parts in excerpt_parts.items()
        if parts
    }

    scan_result = ScanResult(
        files=files,
        code_summaries=code_summaries,
        interface_summaries=interface_summaries,
        dependencies=sorted(dependency_values),
        code_chunks=[],
    )
    return scan_result, spool_path, chunk_count, excerpts


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
    progress_callback: Callable[[str], None] | None = None,
) -> GenerateResult:
    """Run the end-to-end initial SDD generation workflow."""

    output_root = project_config.output_dir / csc.csc_id
    metrics_collector = RunMetricsCollector(csc_id=csc.csc_id)

    _progress(progress_callback, f"[{csc.csc_id}] Scanning repository...")
    metrics_collector.start("scan")
    scan_result, scan_chunks_path, code_chunk_count, code_excerpts = (
        _scan_and_spool_chunks(
            project_config=project_config,
            csc=csc,
            repo_root=repo_root,
            output_root=output_root,
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

    metrics_collector.start("generate_sections")
    section_drafts: list[SectionDraft] = []
    total_sections = len(template.sections)
    for idx, section_spec in enumerate(template.sections, start=1):
        pack = build_generation_evidence_pack(
            section=section_spec,
            csc=csc,
            code_summaries=scan_result.code_summaries,
            interface_summaries=scan_result.interface_summaries,
            dependencies=scan_result.dependencies,
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
        section = _ensure_section_defaults(section, pack.section.id, pack.section.title)

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

    markdown_path = output_root / "sdd.md"
    review_json_path = output_root / "review_artifact.json"
    retrieval_index_path = default_retrieval_store_path(output_root)
    run_metrics_path = output_root / "run_metrics.json"

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
        metrics_collector.start("hierarchy")
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
            code_excerpts_by_path=code_excerpts,
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
        metrics_collector.finish(chunks_written=len(hierarchy_chunks))

    metrics_collector.start("build_retrieval")

    combined_chunks = chain(
        document_chunks, hierarchy_chunks, _iter_spooled_chunks(scan_chunks_path)
    )
    retrieval_manifest = build_retrieval_store(
        store_root=retrieval_index_path,
        chunks=combined_chunks,
        shard_size=project_config.generation.index_shard_size,
        write_batch_size=project_config.generation.write_batch_size,
        max_in_memory_records=project_config.generation.max_in_memory_records,
    )
    metrics_collector.finish(
        chunks_written=len(document_chunks) + len(hierarchy_chunks) + code_chunk_count
    )
    scan_chunks_path.unlink(missing_ok=True)

    write_json_model(run_metrics_path, metrics_collector.metrics)
    _progress(
        progress_callback, f"[{csc.csc_id}] Wrote retrieval store and run metrics."
    )

    return GenerateResult(
        document=document,
        review_artifact=review_artifact,
        retrieval_manifest=retrieval_manifest,
        markdown_path=markdown_path,
        review_json_path=review_json_path,
        retrieval_index_path=retrieval_index_path,
        run_metrics_path=run_metrics_path,
        hierarchy_json_path=hierarchy_json_path,
        hierarchy_index_path=hierarchy_index_path,
    )
