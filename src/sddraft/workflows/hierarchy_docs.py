"""Hierarchy documentation generation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, Field

from sddraft.analysis.hierarchy import (
    directory_node_id,
    file_node_id,
)
from sddraft.analysis.retrieval import tokenize
from sddraft.domain.models import (
    CodeUnitSummary,
    DirectorySummaryDoc,
    DirectorySummaryRecord,
    EvidenceReference,
    FileSummaryDoc,
    FileSummaryRecord,
    HierarchyEdgeRecord,
    HierarchyManifest,
    HierarchyNodeRecord,
    InterfaceSummary,
    ScanResult,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import (
    build_directory_summary_prompt,
    build_file_summary_prompt,
)
from sddraft.render.hierarchy import (
    write_directory_summary_markdown,
    write_file_summary_markdown,
)
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


@dataclass(slots=True)
class HierarchyStoreResult:
    """Persisted hierarchy store metadata returned to workflows."""

    store_root: Path
    manifest_path: Path
    chunk_count: int


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


def excerpt_file_path(excerpt_root: Path, source_path: Path) -> Path:
    """Return deterministic excerpt file path for one source file."""

    candidate = excerpt_root / source_path
    suffix = f"{candidate.suffix}.excerpt" if candidate.suffix else ".excerpt"
    return candidate.with_suffix(suffix)


def load_excerpt_text(*, excerpt_root: Path | None, source_path: Path) -> str:
    """Load one file's excerpt from on-disk excerpt store."""

    if excerpt_root is None:
        return ""
    excerpt_path = excerpt_file_path(excerpt_root, source_path)
    if not excerpt_path.exists():
        return ""
    try:
        return excerpt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


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


def _normalize_dir_path(path: Path) -> Path:
    return Path(".") if path in {Path(""), Path(".")} else path


def _keywords(text: str, limit: int = 8) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


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
    existing_files: dict[Path, FileSummaryRecord] | None,
    existing_dirs: dict[Path, DirectorySummaryRecord] | None,
    changed_paths: set[Path] | None,
) -> tuple[set[Path], set[Path]]:
    if existing_files is None or existing_dirs is None:
        return set(files), set(directories)

    file_set = set(files)
    changed = {path for path in (changed_paths or set()) if path in file_set}
    file_rebuild = changed | {path for path in file_set if path not in existing_files}

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
        existing_item = existing_dirs.get(directory)
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


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _write_jsonl_line(handle: TextIO, model: BaseModel) -> None:
    handle.write(model.model_dump_json())
    handle.write("\n")


def load_hierarchy_store_records(
    manifest_path: Path,
) -> tuple[
    HierarchyManifest,
    dict[Path, FileSummaryRecord],
    dict[Path, DirectorySummaryRecord],
]:
    """Load file and directory records from a hierarchy store manifest."""

    manifest = HierarchyManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    store_root = manifest_path.parent
    file_records: dict[Path, FileSummaryRecord] = {}
    for raw in _iter_jsonl(store_root / manifest.file_summaries_path):
        file_record = FileSummaryRecord.model_validate(raw)
        file_records[file_record.path] = file_record

    directory_records: dict[Path, DirectorySummaryRecord] = {}
    for raw in _iter_jsonl(store_root / manifest.directory_summaries_path):
        directory_record = DirectorySummaryRecord.model_validate(raw)
        directory_records[directory_record.path] = directory_record

    return manifest, file_records, directory_records


def _read_existing_records(
    manifest_path: Path,
) -> tuple[
    dict[Path, FileSummaryRecord] | None,
    dict[Path, DirectorySummaryRecord] | None,
]:
    if not manifest_path.exists():
        return None, None

    try:
        manifest, file_records, directory_records = load_hierarchy_store_records(
            manifest_path
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None

    if manifest.version != "v2-stream-jsonl":
        return None, None

    return file_records, directory_records


def _to_file_summary_doc(record: FileSummaryRecord) -> FileSummaryDoc:
    return FileSummaryDoc.model_validate(record.model_dump(mode="json"))


def _to_directory_summary_doc(record: DirectorySummaryRecord) -> DirectorySummaryDoc:
    return DirectorySummaryDoc.model_validate(record.model_dump(mode="json"))


def build_hierarchy_store(
    *,
    csc_id: str,
    repo_root: Path,
    output_root: Path,
    scan_result: ScanResult,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    changed_paths: set[Path] | None = None,
    excerpt_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HierarchyStoreResult:
    """Build and persist hierarchy summaries as streamed JSONL records."""

    hierarchy_root = output_root / "hierarchy"
    hierarchy_root.mkdir(parents=True, exist_ok=True)

    manifest_path = hierarchy_root / "manifest.json"
    file_summaries_path = hierarchy_root / "file_summaries.jsonl"
    directory_summaries_path = hierarchy_root / "directory_summaries.jsonl"
    nodes_path = hierarchy_root / "nodes.jsonl"
    edges_path = hierarchy_root / "edges.jsonl"

    existing_files, existing_dirs = _read_existing_records(manifest_path)

    files = sorted({_to_relative(path, repo_root) for path in scan_result.files})
    directories, files_by_dir, children_by_dir = _collect_tree(files)

    summary_by_path: dict[Path, CodeUnitSummary] = {}
    for summary in scan_result.code_summaries:
        summary_by_path[_to_relative(summary.path, repo_root)] = summary

    interfaces_by_path: dict[Path, list[InterfaceSummary]] = defaultdict(list)
    for item in scan_result.interface_summaries:
        interfaces_by_path[_to_relative(item.source_path, repo_root)].append(item)

    rebuild_files, rebuild_dirs = _rebuild_sets(
        files=files,
        directories=directories,
        files_by_dir=files_by_dir,
        children_by_dir=children_by_dir,
        existing_files=existing_files,
        existing_dirs=existing_dirs,
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

    files_existing = existing_files or {}
    dirs_existing = existing_dirs or {}

    file_records_by_path: dict[Path, FileSummaryRecord] = {}
    directory_records_by_path: dict[Path, DirectorySummaryRecord] = {}

    regenerated_files = 0
    total_regenerated_files = len(rebuild_files)

    with (
        file_summaries_path.open("w", encoding="utf-8") as file_handle,
        directory_summaries_path.open("w", encoding="utf-8") as directory_handle,
        nodes_path.open("w", encoding="utf-8") as node_handle,
        edges_path.open("w", encoding="utf-8") as edge_handle,
    ):
        for file_path in files:
            if file_path not in rebuild_files and file_path in files_existing:
                file_record = files_existing[file_path]
            else:
                regenerated_files += 1
                if _should_emit_progress(
                    regenerated_files, max(total_regenerated_files, 1)
                ):
                    _progress(
                        progress_callback,
                        "Hierarchy file summaries: "
                        f"{regenerated_files}/{total_regenerated_files}",
                    )

                code_summary = summary_by_path.get(file_path)
                if code_summary is None:
                    file_record = FileSummaryRecord(
                        node_id=file_node_id(file_path),
                        path=file_path,
                        language="unknown",
                        summary="TBD: No deterministic code summary available.",
                        evidence_refs=[],
                        missing_information=["TBD"],
                        confidence=0.3,
                    )
                else:
                    interfaces = interfaces_by_path.get(file_path, [])
                    excerpt = load_excerpt_text(
                        excerpt_root=excerpt_root,
                        source_path=file_path,
                    )
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
                    file_draft = _FileSummaryDraft.model_validate(
                        response.content.model_dump(mode="json")
                    )
                    summary_text = file_draft.summary.strip() or "TBD"
                    file_record = FileSummaryRecord(
                        node_id=file_node_id(file_path),
                        path=file_path,
                        language=code_summary.language,
                        summary=summary_text,
                        functions=code_summary.functions,
                        classes=code_summary.classes,
                        imports=code_summary.imports,
                        evidence_refs=[
                            _file_ref(code_summary),
                            *_interface_refs(interfaces),
                        ],
                        missing_information=_ensure_tbd_missing(
                            summary_text,
                            file_draft.missing_information,
                        ),
                        confidence=file_draft.confidence,
                    )

            file_records_by_path[file_path] = file_record
            _write_jsonl_line(file_handle, file_record)

            file_doc_path = write_file_summary_markdown(
                hierarchy_root=hierarchy_root,
                summary=file_record,
            )
            try:
                doc_path = file_doc_path.relative_to(hierarchy_root)
            except ValueError:
                doc_path = file_doc_path

            parent_path = _normalize_dir_path(file_path.parent)
            parent_id = directory_node_id(parent_path)
            _write_jsonl_line(
                node_handle,
                HierarchyNodeRecord(
                    node_id=file_record.node_id,
                    kind="file",
                    path=file_record.path,
                    parent_id=parent_id,
                    doc_path=doc_path,
                    abstract=file_record.summary[:320],
                    keywords=_keywords(
                        " ".join(
                            [
                                file_record.summary,
                                " ".join(file_record.functions),
                                " ".join(file_record.classes),
                                " ".join(file_record.imports),
                            ]
                        )
                    ),
                ),
            )
            _write_jsonl_line(
                edge_handle,
                HierarchyEdgeRecord(parent_id=parent_id, child_id=file_record.node_id),
            )

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
            if directory not in rebuild_dirs and directory in dirs_existing:
                dir_record = dirs_existing[directory]
            else:
                regenerated_directories += 1
                if _should_emit_progress(
                    regenerated_directories,
                    max(total_regenerated_directories, 1),
                ):
                    _progress(
                        progress_callback,
                        "Hierarchy directory summaries: "
                        f"{regenerated_directories}/{total_regenerated_directories}",
                    )

                local_files = [
                    file_records_by_path[path]
                    for path in files_by_dir.get(directory, [])
                ]
                child_summaries = [
                    directory_records_by_path[path]
                    for path in children_by_dir.get(directory, [])
                ]
                local_file_docs = [_to_file_summary_doc(item) for item in local_files]
                child_directory_docs = [
                    _to_directory_summary_doc(item) for item in child_summaries
                ]
                system_prompt, user_prompt = build_directory_summary_prompt(
                    directory_path=directory.as_posix(),
                    local_files=local_file_docs,
                    child_directories=child_directory_docs,
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
                    EvidenceReference(
                        kind="directory_summary",
                        source=item.path.as_posix(),
                    )
                    for item in child_summaries
                ]

                dir_record = DirectorySummaryRecord(
                    node_id=directory_node_id(directory),
                    path=directory,
                    summary=summary_text,
                    local_files=[item.path for item in local_files],
                    child_directories=[item.path for item in child_summaries],
                    evidence_refs=refs,
                    missing_information=_ensure_tbd_missing(
                        summary_text,
                        dir_draft.missing_information,
                    ),
                    confidence=dir_draft.confidence,
                )

            directory_records_by_path[directory] = dir_record
            _write_jsonl_line(directory_handle, dir_record)

            directory_doc_path = write_directory_summary_markdown(
                hierarchy_root=hierarchy_root,
                summary=dir_record,
            )
            try:
                doc_path = directory_doc_path.relative_to(hierarchy_root)
            except ValueError:
                doc_path = directory_doc_path

            parent_path = _normalize_dir_path(directory.parent)
            directory_parent_id: str | None = (
                None if directory == Path(".") else directory_node_id(parent_path)
            )
            _write_jsonl_line(
                node_handle,
                HierarchyNodeRecord(
                    node_id=dir_record.node_id,
                    kind="directory",
                    path=_normalize_dir_path(dir_record.path),
                    parent_id=directory_parent_id,
                    doc_path=doc_path,
                    abstract=dir_record.summary[:320],
                    keywords=_keywords(
                        " ".join(
                            [
                                dir_record.summary,
                                " ".join(
                                    item.as_posix() for item in dir_record.local_files
                                ),
                                " ".join(
                                    item.as_posix()
                                    for item in dir_record.child_directories
                                ),
                            ]
                        )
                    ),
                ),
            )
            if directory_parent_id is not None:
                _write_jsonl_line(
                    edge_handle,
                    HierarchyEdgeRecord(
                        parent_id=directory_parent_id, child_id=dir_record.node_id
                    ),
                )

    manifest = HierarchyManifest(
        csc_id=csc_id,
        root=Path("."),
        file_summaries_path=Path(file_summaries_path.name),
        directory_summaries_path=Path(directory_summaries_path.name),
        nodes_path=Path(nodes_path.name),
        edges_path=Path(edges_path.name),
    )
    write_json_model(manifest_path, manifest)

    return HierarchyStoreResult(
        store_root=hierarchy_root,
        manifest_path=manifest_path,
        chunk_count=len(file_records_by_path) + len(directory_records_by_path),
    )


def load_hierarchy_summary_evidence(manifest_path: Path) -> list[str]:
    """Load hierarchy evidence strings from streamed store records."""

    manifest, file_records, directory_records = load_hierarchy_store_records(
        manifest_path
    )
    _ = manifest

    evidence: list[str] = []
    for directory in sorted(
        directory_records.values(), key=lambda item: item.path.as_posix()
    ):
        evidence.append(f"directory::{directory.path.as_posix()}::{directory.summary}")
    for file_record in sorted(
        file_records.values(), key=lambda item: item.path.as_posix()
    ):
        evidence.append(f"file::{file_record.path.as_posix()}::{file_record.summary}")
    return evidence
