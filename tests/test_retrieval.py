"""Tests for lexical retrieval index."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.retrieval import (
    BM25Retriever,
    LexicalIndexer,
    to_citations,
    tokenize,
)
from sddraft.domain.models import KnowledgeChunk


def test_tokenize_and_bm25_ranking() -> None:
    chunks = [
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
    ]

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
