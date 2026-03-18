"""Tests for hierarchy index construction and graph expansion."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.hierarchy import (
    build_hierarchy_index,
    directory_node_id,
    expand_chunks_with_hierarchy,
    file_node_id,
    hierarchy_chunks,
)
from sddraft.domain.models import (
    DirectorySummaryDoc,
    FileSummaryDoc,
    HierarchyDocArtifact,
    KnowledgeChunk,
    RetrievalIndex,
    SubtreeRollup,
)


def _sample_artifact() -> HierarchyDocArtifact:
    return HierarchyDocArtifact(
        csc_id="X",
        root=Path("."),
        file_summaries=[
            FileSummaryDoc(
                node_id=file_node_id(Path("src/app.py")),
                path=Path("src/app.py"),
                language="python",
                summary="Application entrypoint.",
            ),
            FileSummaryDoc(
                node_id=file_node_id(Path("src/sub/worker.py")),
                path=Path("src/sub/worker.py"),
                language="python",
                summary="Background worker tasks.",
            ),
        ],
        directory_summaries=[
            DirectorySummaryDoc(
                node_id=directory_node_id(Path(".")),
                path=Path("."),
                summary="Root overview.",
                child_directories=[Path("src")],
                subtree_rollup=SubtreeRollup(
                    descendant_file_count=2,
                    descendant_directory_count=2,
                    language_counts={"python": 2},
                    key_topics=["workflow", "analysis"],
                    representative_files=[
                        Path("src/app.py"),
                        Path("src/sub/worker.py"),
                    ],
                ),
            ),
            DirectorySummaryDoc(
                node_id=directory_node_id(Path("src")),
                path=Path("src"),
                summary="Source overview.",
                local_files=[Path("src/app.py")],
                child_directories=[Path("src/sub")],
                subtree_rollup=SubtreeRollup(
                    descendant_file_count=2,
                    descendant_directory_count=1,
                    language_counts={"python": 2},
                    key_topics=["source", "worker"],
                    representative_files=[
                        Path("src/app.py"),
                        Path("src/sub/worker.py"),
                    ],
                ),
            ),
            DirectorySummaryDoc(
                node_id=directory_node_id(Path("src/sub")),
                path=Path("src/sub"),
                summary="Subdirectory overview.",
                local_files=[Path("src/sub/worker.py")],
                subtree_rollup=SubtreeRollup(
                    descendant_file_count=1,
                    descendant_directory_count=0,
                    language_counts={"python": 1},
                    key_topics=["worker"],
                    representative_files=[Path("src/sub/worker.py")],
                ),
            ),
        ],
    )


def test_build_hierarchy_index_and_chunks() -> None:
    artifact = _sample_artifact()
    node_doc_paths = {
        directory_node_id(Path(".")): Path("hierarchy/_directory.md"),
        directory_node_id(Path("src")): Path("hierarchy/src/_directory.md"),
        directory_node_id(Path("src/sub")): Path("hierarchy/src/sub/_directory.md"),
        file_node_id(Path("src/app.py")): Path("hierarchy/src/app.py.md"),
        file_node_id(Path("src/sub/worker.py")): Path("hierarchy/src/sub/worker.py.md"),
    }

    index = build_hierarchy_index(artifact, node_doc_paths=node_doc_paths)
    assert len(index.nodes) == 5
    assert any(
        edge.parent_id == "dir::src" and edge.child_id == "file::src/app.py"
        for edge in index.edges
    )
    assert any(
        edge.parent_id == "dir::src" and edge.child_id == "dir::src/sub"
        for edge in index.edges
    )

    chunks = hierarchy_chunks(artifact, index)
    assert any(item.source_type == "file_summary" for item in chunks)
    assert any(item.source_type == "directory_summary" for item in chunks)
    assert all("node_id" in item.metadata for item in chunks)
    assert any(
        item.source_type == "directory_summary" and "Subtree files:" in item.text
        for item in chunks
    )


def test_expand_chunks_with_hierarchy_neighbors() -> None:
    artifact = _sample_artifact()
    node_doc_paths = {
        directory_node_id(Path(".")): Path("hierarchy/_directory.md"),
        directory_node_id(Path("src")): Path("hierarchy/src/_directory.md"),
        directory_node_id(Path("src/sub")): Path("hierarchy/src/sub/_directory.md"),
        file_node_id(Path("src/app.py")): Path("hierarchy/src/app.py.md"),
        file_node_id(Path("src/sub/worker.py")): Path("hierarchy/src/sub/worker.py.md"),
    }
    index = build_hierarchy_index(artifact, node_doc_paths=node_doc_paths)
    hierarchy_doc_chunks = hierarchy_chunks(artifact, index)

    lexical_code_chunk = KnowledgeChunk(
        chunk_id="code::src/app.py::1-10",
        source_type="code",
        source_path=Path("src/app.py"),
        text="def planner() -> None:\n    pass",
        line_start=1,
        line_end=10,
    )

    retrieval_index = RetrievalIndex(chunks=[lexical_code_chunk, *hierarchy_doc_chunks])
    expanded = expand_chunks_with_hierarchy(
        initial_chunks=[lexical_code_chunk],
        retrieval_index=retrieval_index,
        hierarchy_index=index,
        top_k=1,
    )

    assert len(expanded) >= 2
    assert any(item.source_type == "directory_summary" for item in expanded[1:])
