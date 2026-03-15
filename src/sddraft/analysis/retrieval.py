"""Deterministic lexical retrieval index and BM25-style search."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    Citation,
    KnowledgeChunk,
    RetrievalIndex,
    ReviewArtifact,
    SDDDocument,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical retrieval."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class Indexer(Protocol):
    """Retrieval index interface."""

    def build(
        self, document_chunks: list[KnowledgeChunk], code_chunks: list[KnowledgeChunk]
    ) -> RetrievalIndex:
        """Build a fresh index from document and code chunks."""

    def update(
        self, existing: RetrievalIndex, new_chunks: list[KnowledgeChunk]
    ) -> RetrievalIndex:
        """Update an existing index with replacement by chunk id."""


class Retriever(Protocol):
    """Retrieval search interface."""

    def search(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        """Search for the most relevant chunks."""


class LexicalIndexer:
    """In-memory lexical indexer."""

    def _normalize(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        normalized: list[KnowledgeChunk] = []
        for chunk in chunks:
            normalized.append(chunk.model_copy(update={"tokens": tokenize(chunk.text)}))
        return normalized

    def build(
        self,
        document_chunks: list[KnowledgeChunk],
        code_chunks: list[KnowledgeChunk],
    ) -> RetrievalIndex:
        combined = self._normalize(document_chunks + code_chunks)
        return RetrievalIndex(chunks=combined)

    def update(
        self, existing: RetrievalIndex, new_chunks: list[KnowledgeChunk]
    ) -> RetrievalIndex:
        replacements = {chunk.chunk_id: chunk for chunk in self._normalize(new_chunks)}
        merged: dict[str, KnowledgeChunk] = {
            chunk.chunk_id: chunk for chunk in existing.chunks
        }
        merged.update(replacements)
        sorted_chunks = sorted(merged.values(), key=lambda item: item.chunk_id)
        return RetrievalIndex(chunks=sorted_chunks)


class BM25Retriever:
    """Deterministic BM25-style lexical retriever."""

    def __init__(self, index: RetrievalIndex) -> None:
        self._chunks = index.chunks
        self._k1 = 1.5
        self._b = 0.75

        self._doc_lengths = [len(chunk.tokens) for chunk in self._chunks]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._chunks else 0.0
        )

        self._doc_freq: Counter[str] = Counter()
        for chunk in self._chunks:
            self._doc_freq.update(set(chunk.tokens))

    def _idf(self, term: str) -> float:
        doc_count = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        if doc_count == 0:
            return 0.0
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    def _score(
        self, chunk: KnowledgeChunk, query_terms: list[str], doc_len: int
    ) -> float:
        if not query_terms or not chunk.tokens:
            return 0.0

        tf_counter = Counter(chunk.tokens)
        score = 0.0
        avg_len = self._avg_doc_length or 1.0

        for term in query_terms:
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * (doc_len / avg_len))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        query_terms = tokenize(query)
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, doc_len in zip(self._chunks, self._doc_lengths, strict=False):
            score = self._score(chunk, query_terms, doc_len)
            if score <= 0:
                continue
            scored.append((score, chunk))

        ranked = sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1].source_path.as_posix(),
                item[1].line_start or 0,
                item[1].chunk_id,
            ),
        )
        if ranked:
            return [chunk for _, chunk in ranked[:top_k]]

        fallback = sorted(
            self._chunks,
            key=lambda item: (
                item.source_path.as_posix(),
                item.line_start or 0,
                item.chunk_id,
            ),
        )
        return fallback[:top_k]


def build_document_chunks(
    document: SDDDocument,
    review_artifact: ReviewArtifact,
    markdown_path: Path,
    review_json_path: Path,
) -> list[KnowledgeChunk]:
    """Create retrieval chunks from generated document artifacts."""

    chunks: list[KnowledgeChunk] = []

    for section in document.sections:
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"sdd::{document.csc_id}::{section.section_id}",
                source_type="sdd_section",
                source_path=markdown_path,
                section_id=section.section_id,
                text=section.content,
                metadata={"title": section.title},
            )
        )

    for review_section in review_artifact.sections:
        serialized = json.dumps(review_section.model_dump(mode="json"), sort_keys=True)
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"review::{review_artifact.csc_id}::{review_section.section_id}",
                source_type="review_artifact",
                source_path=review_json_path,
                section_id=review_section.section_id,
                text=serialized,
            )
        )

    return chunks


def to_citations(
    chunks: list[KnowledgeChunk], max_quote_len: int = 200
) -> list[Citation]:
    """Convert chunks into citations with bounded quotes."""

    citations: list[Citation] = []
    for chunk in chunks:
        quote = chunk.text.strip().replace("\n", " ")
        if len(quote) > max_quote_len:
            quote = f"{quote[:max_quote_len].rstrip()}..."
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                quote=quote,
            )
        )
    return citations


def save_retrieval_index(index: RetrievalIndex, path: Path) -> None:
    """Persist retrieval index JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_retrieval_index(path: Path) -> RetrievalIndex:
    """Load retrieval index from JSON file."""

    if not path.exists():
        raise AnalysisError(f"Retrieval index not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid retrieval index JSON at {path}: {exc}") from exc
    return RetrievalIndex.model_validate(raw)
