"""Tests for lexical retrieval index and sharded storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from sddraft.analysis.retrieval import (
    BM25Retriever,
    LexicalIndexer,
    build_retrieval_store,
    load_retrieval_manifest,
    migrate_legacy_index,
    open_query_engine,
    resolve_retrieval_store_path,
    save_retrieval_index,
    to_citations,
    tokenize,
)
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import KnowledgeChunk


def _sample_chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="a",
            source_type="sdd_section",
            source_path=Path("doc.md"),
            text="navigation control handles route planning",
        ),
        KnowledgeChunk(
            chunk_id="b",
            source_type="sdd_section",
            source_path=Path("doc.md"),
            text="logging subsystem stores telemetry",
        ),
        KnowledgeChunk(
            chunk_id="code::src/app.py::1-4",
            source_type="code",
            source_path=Path("src/app.py"),
            text="def compute():\n    return 1",
            line_start=1,
            line_end=4,
        ),
    ]


def test_tokenize_and_bm25_ranking() -> None:
    chunks = _sample_chunks()[:2]

    index = LexicalIndexer().build(document_chunks=chunks, code_chunks=[])
    retriever = BM25Retriever(index)

    tokens = tokenize("Route planning")
    assert tokens == ["route", "planning"]

    matches = retriever.search("route planning", top_k=1)
    assert len(matches) == 1
    assert matches[0].chunk_id == "a"


def test_to_citations_bounds_quotes() -> None:
    chunk = KnowledgeChunk(
        chunk_id="chunk",
        source_type="code",
        source_path=Path("src/a.py"),
        text="x" * 500,
        line_start=1,
        line_end=20,
    )

    citations = to_citations([chunk], max_quote_len=40)
    assert len(citations) == 1
    assert citations[0].quote.endswith("...")


def test_sharded_store_build_and_query_roundtrip(tmp_path: Path) -> None:
    store_root = tmp_path / "retrieval"
    manifest = build_retrieval_store(
        store_root=store_root,
        chunks=_sample_chunks(),
        shard_size=2,
        write_batch_size=1,
        max_in_memory_records=8,
    )

    assert manifest.total_chunks == 3
    assert (store_root / "manifest.json").exists()

    engine = open_query_engine(store_root)
    matches = engine.search("route planning", top_k=1)
    assert [item.chunk_id for item in matches] == ["a"]

    fallback = engine.search("totally-unmatched-query", top_k=1)
    assert fallback
    assert fallback[0].source_path.as_posix() <= "src/app.py"


def test_sharded_store_supports_node_id_lookup(tmp_path: Path) -> None:
    store_root = tmp_path / "retrieval"
    chunks = [
        KnowledgeChunk(
            chunk_id="hier::dir",
            source_type="directory_summary",
            source_path=Path("hierarchy/src/_directory.md"),
            text="src summary",
            metadata={"node_id": "dir::src"},
        ),
        KnowledgeChunk(
            chunk_id="hier::file",
            source_type="file_summary",
            source_path=Path("hierarchy/src/app.py.md"),
            text="app summary",
            metadata={"node_id": "file::src/app.py"},
        ),
    ]
    build_retrieval_store(
        store_root=store_root,
        chunks=chunks,
        shard_size=2,
        write_batch_size=1,
        max_in_memory_records=8,
    )

    engine = open_query_engine(store_root)
    loaded = engine.load_chunks_by_node_ids({"dir::src", "file::src/app.py"}, limit=3)
    assert {item.metadata.get("node_id") for item in loaded} == {
        "dir::src",
        "file::src/app.py",
    }


def test_manifest_loading_and_store_resolution(tmp_path: Path) -> None:
    store_root = tmp_path / "retrieval"
    build_retrieval_store(
        store_root=store_root,
        chunks=_sample_chunks(),
        shard_size=2,
        write_batch_size=2,
        max_in_memory_records=8,
    )

    manifest_by_dir, root_by_dir = load_retrieval_manifest(store_root)
    manifest_by_file, root_by_file = load_retrieval_manifest(
        store_root / "manifest.json"
    )
    assert manifest_by_dir.total_chunks == manifest_by_file.total_chunks
    assert root_by_dir == root_by_file == store_root

    assert resolve_retrieval_store_path(store_root) == store_root
    assert resolve_retrieval_store_path(store_root / "manifest.json") == store_root


def test_migrate_legacy_index_to_sharded_store(tmp_path: Path) -> None:
    legacy_path = tmp_path / "retrieval_index.json"
    legacy = LexicalIndexer().build(document_chunks=_sample_chunks(), code_chunks=[])
    save_retrieval_index(legacy, legacy_path)

    with pytest.raises(AnalysisError, match="migrate-index"):
        resolve_retrieval_store_path(legacy_path)

    migrated_path = migrate_legacy_index(
        index_path=legacy_path,
        shard_size=2,
        write_batch_size=1,
        max_in_memory_records=8,
    )
    assert migrated_path == tmp_path / "retrieval"
    assert (migrated_path / "manifest.json").exists()

    # idempotent on already-migrated path
    second = migrate_legacy_index(
        index_path=migrated_path,
        shard_size=2,
        write_batch_size=1,
        max_in_memory_records=8,
    )
    assert second == migrated_path


def test_open_query_engine_rejects_missing_store(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="Retrieval store not found"):
        open_query_engine(tmp_path / "missing")
