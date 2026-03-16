"""Deterministic hierarchy graph helpers for hierarchy docs and ask expansion."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

from sddraft.analysis.retrieval import tokenize
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    DirectorySummaryRecord,
    FileSummaryRecord,
    HierarchyDocArtifact,
    HierarchyEdgeRecord,
    HierarchyIndex,
    HierarchyIndexEdge,
    HierarchyIndexNode,
    HierarchyManifest,
    HierarchyNodeRecord,
    KnowledgeChunk,
    RetrievalIndex,
)


def _normalize_dir_path(path: Path) -> Path:
    return Path(".") if path in {Path(""), Path(".")} else path


def file_node_id(path: Path) -> str:
    """Return stable node id for file path."""

    return f"file::{path.as_posix()}"


def directory_node_id(path: Path) -> str:
    """Return stable node id for directory path."""

    normalized = _normalize_dir_path(path)
    return f"dir::{normalized.as_posix()}"


def default_hierarchy_index_path(index_path: Path) -> Path:
    """Return conventional hierarchy manifest path for retrieval paths."""

    if index_path.is_dir() and index_path.name == "retrieval":
        return index_path.parent / "hierarchy" / "manifest.json"
    if index_path.is_dir() and index_path.name == "hierarchy":
        return index_path / "manifest.json"
    if index_path.is_file() and index_path.name == "manifest.json":
        if index_path.parent.name == "retrieval":
            return index_path.parent.parent / "hierarchy" / "manifest.json"
        return index_path
    if (index_path / "retrieval").exists():
        return index_path / "hierarchy" / "manifest.json"
    return index_path.parent / "hierarchy" / "manifest.json"


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


def build_hierarchy_index(
    artifact: HierarchyDocArtifact, node_doc_paths: dict[str, Path]
) -> HierarchyIndex:
    """Build deterministic hierarchy index from hierarchy artifact."""

    nodes: list[HierarchyIndexNode] = []
    edges: list[HierarchyIndexEdge] = []

    for directory_summary in sorted(
        artifact.directory_summaries, key=lambda item: item.path.as_posix()
    ):
        path = _normalize_dir_path(directory_summary.path)
        parent_path = _normalize_dir_path(path.parent)
        parent_id = None if path == Path(".") else directory_node_id(parent_path)

        nodes.append(
            HierarchyIndexNode(
                node_id=directory_summary.node_id,
                kind="directory",
                path=path,
                parent_id=parent_id,
                doc_path=node_doc_paths[directory_summary.node_id],
                abstract=directory_summary.summary[:320],
                keywords=_keywords(
                    " ".join(
                        [
                            directory_summary.summary,
                            " ".join(
                                item.as_posix()
                                for item in directory_summary.local_files
                            ),
                            " ".join(
                                item.as_posix()
                                for item in directory_summary.child_directories
                            ),
                        ]
                    )
                ),
            )
        )
        if parent_id is not None:
            edges.append(
                HierarchyIndexEdge(
                    parent_id=parent_id, child_id=directory_summary.node_id
                )
            )

    for file_summary in sorted(
        artifact.file_summaries, key=lambda item: item.path.as_posix()
    ):
        parent_path = _normalize_dir_path(file_summary.path.parent)
        parent_id = directory_node_id(parent_path)
        nodes.append(
            HierarchyIndexNode(
                node_id=file_summary.node_id,
                kind="file",
                path=file_summary.path,
                parent_id=parent_id,
                doc_path=node_doc_paths[file_summary.node_id],
                abstract=file_summary.summary[:320],
                keywords=_keywords(
                    " ".join(
                        [
                            file_summary.summary,
                            " ".join(file_summary.functions),
                            " ".join(file_summary.classes),
                            " ".join(file_summary.imports),
                        ]
                    )
                ),
            )
        )
        edges.append(
            HierarchyIndexEdge(parent_id=parent_id, child_id=file_summary.node_id)
        )

    sorted_nodes = sorted(nodes, key=lambda item: (item.kind, item.path.as_posix()))
    sorted_edges = sorted(edges, key=lambda item: (item.parent_id, item.child_id))
    return HierarchyIndex(
        csc_id=artifact.csc_id,
        root=_normalize_dir_path(artifact.root),
        nodes=sorted_nodes,
        edges=sorted_edges,
    )


def hierarchy_chunks(
    artifact: HierarchyDocArtifact, index: HierarchyIndex
) -> list[KnowledgeChunk]:
    """Convert hierarchy docs into retrieval chunks."""

    node_by_id = {node.node_id: node for node in index.nodes}
    chunks: list[KnowledgeChunk] = []

    for file_summary in sorted(
        artifact.file_summaries, key=lambda item: item.path.as_posix()
    ):
        node = node_by_id.get(file_summary.node_id)
        if node is None:
            continue
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"hier_file::{file_summary.path.as_posix()}",
                source_type="file_summary",
                source_path=node.doc_path,
                text=file_summary.summary,
                metadata={
                    "node_id": file_summary.node_id,
                    "kind": "file",
                    "path": file_summary.path.as_posix(),
                },
            )
        )

    for directory_summary in sorted(
        artifact.directory_summaries, key=lambda item: item.path.as_posix()
    ):
        node = node_by_id.get(directory_summary.node_id)
        if node is None:
            continue
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"hier_dir::{_normalize_dir_path(directory_summary.path).as_posix()}",
                source_type="directory_summary",
                source_path=node.doc_path,
                text=directory_summary.summary,
                metadata={
                    "node_id": directory_summary.node_id,
                    "kind": "directory",
                    "path": _normalize_dir_path(directory_summary.path).as_posix(),
                },
            )
        )

    return chunks


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "manifest.json"
        if candidate.exists():
            return candidate
        return default_hierarchy_index_path(path)
    return default_hierarchy_index_path(path)


def _load_manifest(manifest_path: Path) -> HierarchyManifest:
    if not manifest_path.exists():
        raise AnalysisError(f"Hierarchy manifest not found: {manifest_path}")
    try:
        return HierarchyManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise AnalysisError(
            f"Invalid hierarchy manifest JSON at {manifest_path}: {exc}"
        ) from exc


def iter_hierarchy_chunks(manifest_path: Path) -> Iterator[KnowledgeChunk]:
    """Yield hierarchy summary chunks from a streamed hierarchy store."""

    resolved_manifest = _resolve_manifest_path(manifest_path)
    manifest = _load_manifest(resolved_manifest)
    store_root = resolved_manifest.parent

    node_by_id: dict[str, HierarchyNodeRecord] = {}
    for raw in _iter_jsonl(store_root / manifest.nodes_path):
        node = HierarchyNodeRecord.model_validate(raw)
        node_by_id[node.node_id] = node

    for raw in _iter_jsonl(store_root / manifest.file_summaries_path):
        file_summary = FileSummaryRecord.model_validate(raw)
        file_node = node_by_id.get(file_summary.node_id)
        if file_node is None:
            continue
        doc_path = (
            file_node.doc_path
            if file_node.doc_path.is_absolute()
            else store_root / file_node.doc_path
        )
        yield KnowledgeChunk(
            chunk_id=f"hier_file::{file_summary.path.as_posix()}",
            source_type="file_summary",
            source_path=doc_path,
            text=file_summary.summary,
            metadata={
                "node_id": file_summary.node_id,
                "kind": "file",
                "path": file_summary.path.as_posix(),
            },
        )

    for raw in _iter_jsonl(store_root / manifest.directory_summaries_path):
        directory_summary = DirectorySummaryRecord.model_validate(raw)
        directory_node = node_by_id.get(directory_summary.node_id)
        if directory_node is None:
            continue
        doc_path = (
            directory_node.doc_path
            if directory_node.doc_path.is_absolute()
            else store_root / directory_node.doc_path
        )
        yield KnowledgeChunk(
            chunk_id=f"hier_dir::{_normalize_dir_path(directory_summary.path).as_posix()}",
            source_type="directory_summary",
            source_path=doc_path,
            text=directory_summary.summary,
            metadata={
                "node_id": directory_summary.node_id,
                "kind": "directory",
                "path": _normalize_dir_path(directory_summary.path).as_posix(),
            },
        )


def save_hierarchy_index(index: HierarchyIndex, path: Path) -> None:
    """Persist hierarchy index JSON (legacy helper)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_hierarchy_index(path: Path) -> HierarchyIndex:
    """Load hierarchy graph from streamed manifest-backed store."""

    manifest_path = _resolve_manifest_path(path)
    manifest = _load_manifest(manifest_path)
    store_root = manifest_path.parent

    nodes: list[HierarchyIndexNode] = []
    for raw in _iter_jsonl(store_root / manifest.nodes_path):
        node_record = HierarchyNodeRecord.model_validate(raw)
        doc_path = (
            node_record.doc_path
            if node_record.doc_path.is_absolute()
            else store_root / node_record.doc_path
        )
        nodes.append(
            HierarchyIndexNode(
                node_id=node_record.node_id,
                kind=node_record.kind,
                path=node_record.path,
                parent_id=node_record.parent_id,
                doc_path=doc_path,
                abstract=node_record.abstract,
                keywords=node_record.keywords,
            )
        )

    edges: list[HierarchyIndexEdge] = []
    for raw in _iter_jsonl(store_root / manifest.edges_path):
        edge_record = HierarchyEdgeRecord.model_validate(raw)
        edges.append(
            HierarchyIndexEdge(
                parent_id=edge_record.parent_id,
                child_id=edge_record.child_id,
                relation=edge_record.relation,
            )
        )

    sorted_nodes = sorted(nodes, key=lambda item: (item.kind, item.path.as_posix()))
    sorted_edges = sorted(edges, key=lambda item: (item.parent_id, item.child_id))
    return HierarchyIndex(
        csc_id=manifest.csc_id,
        root=_normalize_dir_path(manifest.root),
        nodes=sorted_nodes,
        edges=sorted_edges,
    )


def expand_chunks_with_hierarchy(
    *,
    initial_chunks: list[KnowledgeChunk],
    retrieval_index: RetrievalIndex | None,
    hierarchy_index: HierarchyIndex,
    top_k: int,
    load_chunks_by_node_ids: (
        Callable[[set[str], int | None], list[KnowledgeChunk]] | None
    ) = None,
) -> list[KnowledgeChunk]:
    """Expand lexical retrieval with related hierarchy nodes."""

    if not initial_chunks:
        return initial_chunks

    node_by_id = {node.node_id: node for node in hierarchy_index.nodes}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    file_node_by_path: dict[str, str] = {}
    for edge in hierarchy_index.edges:
        children_by_parent[edge.parent_id].append(edge.child_id)
    for parent_id in children_by_parent:
        children_by_parent[parent_id] = sorted(children_by_parent[parent_id])
    for node in hierarchy_index.nodes:
        if node.kind == "file":
            file_node_by_path[node.path.as_posix()] = node.node_id

    chunk_by_node_id: dict[str, KnowledgeChunk] = {}
    if retrieval_index is not None:
        for chunk in retrieval_index.chunks:
            node_id = chunk.metadata.get("node_id")
            if node_id:
                chunk_by_node_id[node_id] = chunk

    seed_node_ids: set[str] = set()
    for chunk in initial_chunks:
        explicit = chunk.metadata.get("node_id")
        if explicit and explicit in node_by_id:
            seed_node_ids.add(explicit)
            continue
        if chunk.source_type == "code":
            candidate = file_node_by_path.get(chunk.source_path.as_posix())
            if candidate:
                seed_node_ids.add(candidate)

    if not seed_node_ids:
        return initial_chunks

    expanded_ids: set[str] = set()
    for node_id in seed_node_ids:
        matched_node = node_by_id.get(node_id)
        if matched_node is None:
            continue
        if matched_node.parent_id:
            expanded_ids.add(matched_node.parent_id)
            for sibling in children_by_parent.get(matched_node.parent_id, []):
                expanded_ids.add(sibling)
        for child in children_by_parent.get(node_id, []):
            expanded_ids.add(child)

    expanded_ids.difference_update(seed_node_ids)
    initial_chunk_ids = {item.chunk_id for item in initial_chunks}
    extras: list[KnowledgeChunk] = []
    if load_chunks_by_node_ids is not None:
        loaded = load_chunks_by_node_ids(expanded_ids, max(top_k, top_k * 2))
        extras = [item for item in loaded if item.chunk_id not in initial_chunk_ids]
        extras = sorted(
            extras,
            key=lambda item: (
                0 if item.source_type == "directory_summary" else 1,
                item.source_path.as_posix(),
                item.chunk_id,
            ),
        )
    else:
        extras = [
            chunk_by_node_id[node_id]
            for node_id in sorted(expanded_ids)
            if node_id in chunk_by_node_id
            and chunk_by_node_id[node_id].chunk_id not in initial_chunk_ids
        ]

    if not extras:
        return initial_chunks

    budget = max(top_k, top_k * 2)
    combined = initial_chunks + extras
    return combined[:budget]
