"""Hierarchy documentation generation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from sddraft.analysis.hierarchy import (
    build_hierarchy_index,
    directory_node_id,
    file_node_id,
    hierarchy_chunks,
    save_hierarchy_index,
)
from sddraft.domain.models import (
    CodeUnitSummary,
    DirectorySummaryDoc,
    EvidenceReference,
    FileSummaryDoc,
    HierarchyDocArtifact,
    HierarchyIndex,
    InterfaceSummary,
    KnowledgeChunk,
    ScanResult,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import (
    build_directory_summary_prompt,
    build_file_summary_prompt,
)
from sddraft.render.hierarchy import write_hierarchy_markdown
from sddraft.render.json_artifacts import write_json_model


class _FileSummaryDraft(BaseModel):
    summary: str
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class _DirectorySummaryDraft(BaseModel):
    summary: str
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5


ProgressCallback = Callable[[str], None]


def _progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _should_emit_progress(current: int, total: int) -> bool:
    return current == 1 or current == total or current % 10 == 0


def _to_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def _file_ref(summary: CodeUnitSummary) -> EvidenceReference:
    return EvidenceReference(kind="code_summary", source=summary.path.as_posix())


def _interface_refs(items: list[InterfaceSummary]) -> list[EvidenceReference]:
    return [
        EvidenceReference(
            kind="interface", source=item.source_path.as_posix(), detail=item.name
        )
        for item in items
    ]


def _ensure_tbd_missing(summary: str, missing: list[str]) -> list[str]:
    if missing:
        return missing
    if summary.strip().upper().startswith("TBD"):
        return ["TBD"]
    return []


def _collect_tree(
    files: list[Path],
) -> tuple[set[Path], dict[Path, list[Path]], dict[Path, list[Path]]]:
    directories: set[Path] = {Path(".")}
    files_by_dir: dict[Path, list[Path]] = defaultdict(list)

    for file_path in sorted(files):
        directory = file_path.parent if file_path.parent != Path("") else Path(".")
        files_by_dir[directory].append(file_path)
        cursor = directory
        while True:
            directories.add(cursor)
            if cursor == Path("."):
                break
            cursor = cursor.parent if cursor.parent != Path("") else Path(".")

    children_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for directory in directories:
        if directory == Path("."):
            continue
        parent = directory.parent if directory.parent != Path("") else Path(".")
        children_by_dir[parent].append(directory)

    for key in files_by_dir:
        files_by_dir[key] = sorted(files_by_dir[key], key=lambda item: item.as_posix())
    for key in children_by_dir:
        children_by_dir[key] = sorted(
            children_by_dir[key], key=lambda item: item.as_posix()
        )

    return directories, files_by_dir, children_by_dir


def _rebuild_sets(
    *,
    files: list[Path],
    directories: set[Path],
    files_by_dir: dict[Path, list[Path]],
    children_by_dir: dict[Path, list[Path]],
    existing: HierarchyDocArtifact | None,
    changed_paths: set[Path] | None,
) -> tuple[set[Path], set[Path]]:
    if existing is None:
        return set(files), set(directories)

    existing_file = {item.path: item for item in existing.file_summaries}
    existing_dir = {item.path: item for item in existing.directory_summaries}

    file_set = set(files)
    changed = {path for path in (changed_paths or set()) if path in file_set}
    file_rebuild = changed | {path for path in file_set if path not in existing_file}

    dir_rebuild: set[Path] = set()
    for file_path in file_rebuild:
        cursor = file_path.parent if file_path.parent != Path("") else Path(".")
        while True:
            dir_rebuild.add(cursor)
            if cursor == Path("."):
                break
            cursor = cursor.parent if cursor.parent != Path("") else Path(".")

    for directory in directories:
        expected_local = files_by_dir.get(directory, [])
        expected_children = children_by_dir.get(directory, [])
        existing_item = existing_dir.get(directory)
        if existing_item is None:
            dir_rebuild.add(directory)
            continue
        if existing_item.local_files != expected_local:
            dir_rebuild.add(directory)
        if existing_item.child_directories != expected_children:
            dir_rebuild.add(directory)

    expanded_dirs = set(dir_rebuild)
    for directory in list(dir_rebuild):
        cursor = directory
        while True:
            expanded_dirs.add(cursor)
            if cursor == Path("."):
                break
            cursor = cursor.parent if cursor.parent != Path("") else Path(".")

    return file_rebuild, expanded_dirs


def build_hierarchy_artifact(
    *,
    csc_id: str,
    repo_root: Path,
    scan_result: ScanResult,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    changed_paths: set[Path] | None = None,
    existing_artifact: HierarchyDocArtifact | None = None,
    code_excerpts_by_path: dict[Path, str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HierarchyDocArtifact:
    """Build hierarchy summaries for files and directories."""

    files = sorted({_to_relative(path, repo_root) for path in scan_result.files})
    directories, files_by_dir, children_by_dir = _collect_tree(files)

    summary_by_path: dict[Path, CodeUnitSummary] = {}
    for summary in scan_result.code_summaries:
        summary_by_path[_to_relative(summary.path, repo_root)] = summary

    interfaces_by_path: dict[Path, list[InterfaceSummary]] = defaultdict(list)
    for item in scan_result.interface_summaries:
        interfaces_by_path[_to_relative(item.source_path, repo_root)].append(item)

    chunks_by_path: dict[Path, list[KnowledgeChunk]] = defaultdict(list)
    if code_excerpts_by_path is None:
        for chunk in scan_result.code_chunks:
            chunks_by_path[chunk.source_path].append(chunk)

    rebuild_files, rebuild_dirs = _rebuild_sets(
        files=files,
        directories=directories,
        files_by_dir=files_by_dir,
        children_by_dir=children_by_dir,
        existing=existing_artifact,
        changed_paths=changed_paths,
    )
    _progress(
        progress_callback,
        (
            "Hierarchy planning: "
            f"{len(files) - len(rebuild_files)} reused file summaries, "
            f"{len(rebuild_files)} regenerated file summaries; "
            f"{len(directories) - len(rebuild_dirs)} reused directory summaries, "
            f"{len(rebuild_dirs)} regenerated directory summaries."
        ),
    )

    existing_file_docs = (
        {item.path: item for item in existing_artifact.file_summaries}
        if existing_artifact
        else {}
    )
    existing_dir_docs = (
        {item.path: item for item in existing_artifact.directory_summaries}
        if existing_artifact
        else {}
    )

    file_docs_by_path: dict[Path, FileSummaryDoc] = {}
    regenerated_files = 0
    total_regenerated_files = len(rebuild_files)
    for file_path in files:
        if file_path not in rebuild_files and file_path in existing_file_docs:
            file_docs_by_path[file_path] = existing_file_docs[file_path]
            continue
        regenerated_files += 1
        if _should_emit_progress(regenerated_files, max(total_regenerated_files, 1)):
            _progress(
                progress_callback,
                f"Hierarchy file summaries: {regenerated_files}/{total_regenerated_files}",
            )

        code_summary = summary_by_path.get(file_path)
        if code_summary is None:
            file_docs_by_path[file_path] = FileSummaryDoc(
                node_id=file_node_id(file_path),
                path=file_path,
                language="unknown",
                summary="TBD: No deterministic code summary available.",
                evidence_refs=[],
                missing_information=["TBD"],
                confidence=0.3,
            )
            continue

        interfaces = interfaces_by_path.get(file_path, [])
        excerpt = ""
        if code_excerpts_by_path is not None:
            excerpt = code_excerpts_by_path.get(file_path, "")
        if not excerpt:
            excerpt_chunks = sorted(
                chunks_by_path.get(file_path, []), key=lambda item: item.line_start or 0
            )[:2]
            excerpt = "\n\n".join(chunk.text for chunk in excerpt_chunks).strip()

        system_prompt, user_prompt = build_file_summary_prompt(
            code_summary=code_summary,
            interfaces=interfaces,
            code_excerpt=excerpt,
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=_FileSummaryDraft,
                model_name=model_name,
                temperature=temperature,
            )
        )
        draft = _FileSummaryDraft.model_validate(
            response.content.model_dump(mode="json")
        )
        summary_text = draft.summary.strip() or "TBD"

        file_docs_by_path[file_path] = FileSummaryDoc(
            node_id=file_node_id(file_path),
            path=file_path,
            language=code_summary.language,
            summary=summary_text,
            functions=code_summary.functions,
            classes=code_summary.classes,
            imports=code_summary.imports,
            evidence_refs=[_file_ref(code_summary), *_interface_refs(interfaces)],
            missing_information=_ensure_tbd_missing(
                summary_text, draft.missing_information
            ),
            confidence=draft.confidence,
        )

    directory_docs_by_path: dict[Path, DirectorySummaryDoc] = {}
    directories_ordered = sorted(
        directories,
        key=lambda path: (
            -len(path.parts) if path != Path(".") else -0,
            path.as_posix(),
        ),
    )

    regenerated_directories = 0
    total_regenerated_directories = len(rebuild_dirs)
    for directory in directories_ordered:
        if directory not in rebuild_dirs and directory in existing_dir_docs:
            directory_docs_by_path[directory] = existing_dir_docs[directory]
            continue
        regenerated_directories += 1
        if _should_emit_progress(
            regenerated_directories, max(total_regenerated_directories, 1)
        ):
            _progress(
                progress_callback,
                "Hierarchy directory summaries: "
                f"{regenerated_directories}/{total_regenerated_directories}",
            )

        local_files = [
            file_docs_by_path[path] for path in files_by_dir.get(directory, [])
        ]
        child_summaries = [
            directory_docs_by_path[path] for path in children_by_dir.get(directory, [])
        ]

        system_prompt, user_prompt = build_directory_summary_prompt(
            directory_path=directory.as_posix(),
            local_files=local_files,
            child_directories=child_summaries,
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=_DirectorySummaryDraft,
                model_name=model_name,
                temperature=temperature,
            )
        )
        dir_draft = _DirectorySummaryDraft.model_validate(
            response.content.model_dump(mode="json")
        )
        summary_text = dir_draft.summary.strip() or "TBD"

        refs = [
            EvidenceReference(kind="file_summary", source=item.path.as_posix())
            for item in local_files
        ] + [
            EvidenceReference(kind="directory_summary", source=item.path.as_posix())
            for item in child_summaries
        ]

        directory_docs_by_path[directory] = DirectorySummaryDoc(
            node_id=directory_node_id(directory),
            path=directory,
            summary=summary_text,
            local_files=[item.path for item in local_files],
            child_directories=[item.path for item in child_summaries],
            evidence_refs=refs,
            missing_information=_ensure_tbd_missing(
                summary_text, dir_draft.missing_information
            ),
            confidence=dir_draft.confidence,
        )

    return HierarchyDocArtifact(
        csc_id=csc_id,
        root=Path("."),
        file_summaries=[
            file_docs_by_path[path] for path in sorted(file_docs_by_path, key=str)
        ],
        directory_summaries=[
            directory_docs_by_path[path]
            for path in sorted(directory_docs_by_path, key=lambda item: item.as_posix())
        ],
    )


def load_hierarchy_artifact(path: Path) -> HierarchyDocArtifact:
    """Load persisted hierarchy artifact."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    return HierarchyDocArtifact.model_validate(raw)


def persist_hierarchy_outputs(
    *,
    artifact: HierarchyDocArtifact,
    output_root: Path,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, HierarchyIndex, list[KnowledgeChunk]]:
    """Persist hierarchy markdown/json/index and return retrieval chunks."""

    hierarchy_root = output_root / "hierarchy"
    hierarchy_json_path = hierarchy_root / "hierarchy_artifact.json"
    hierarchy_index_path = hierarchy_root / "hierarchy_index.json"

    _progress(
        progress_callback, "Hierarchy: writing markdown summaries and index artifacts."
    )
    node_doc_paths = write_hierarchy_markdown(hierarchy_root, artifact)
    index = build_hierarchy_index(artifact, node_doc_paths=node_doc_paths)
    write_json_model(hierarchy_json_path, artifact)
    save_hierarchy_index(index, hierarchy_index_path)

    return (
        hierarchy_json_path,
        hierarchy_index_path,
        index,
        hierarchy_chunks(artifact, index),
    )
