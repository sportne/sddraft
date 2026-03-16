"""Deterministic lexical retrieval with sharded JSON storage."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    ChunkShardRef,
    Citation,
    DocStatRecord,
    KnowledgeChunk,
    PostingShardRef,
    RetrievalIndex,
    RetrievalManifest,
    ReviewArtifact,
    SDDDocument,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

_LEGACY_INDEX_FILENAME = "retrieval_index.json"
_RETRIEVAL_DIRNAME = "retrieval"
_MANIFEST_FILENAME = "manifest.json"
_DOCSTATS_FILENAME = "docstats.jsonl"
_CHUNK_SHARD_PREFIX = "chunks"
_POSTINGS_PREFIX = "postings"


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


class ChunkSink(Protocol):
    """Streaming sink for retrieval chunks."""

    def add(self, chunk: KnowledgeChunk) -> None:
        """Write one chunk to the sink."""

    def finalize(self) -> RetrievalManifest:
        """Finalize write and return retrieval manifest."""


class ChunkSource(Protocol):
    """Streaming source for retrieval chunks."""

    def iter_chunks(self) -> Iterator[KnowledgeChunk]:
        """Yield stored chunks."""


class Retriever(Protocol):
    """Retrieval search interface."""

    def search(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        """Search for the most relevant chunks."""


class LexicalIndexer:
    """In-memory lexical indexer for tests and legacy migration."""

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
    """Deterministic BM25-style lexical retriever for in-memory indices."""

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


class JsonlChunkSink:
    """Sharded JSON chunk writer with bounded in-memory buffers."""

    def __init__(
        self,
        store_root: Path,
        *,
        shard_size: int,
        write_batch_size: int,
        max_in_memory_records: int,
        posting_bucket_count: int = 64,
    ) -> None:
        self._store_root = store_root
        self._shard_size = max(shard_size, 1)
        self._write_batch_size = max(write_batch_size, 1)
        self._max_in_memory_records = max(max_in_memory_records, self._shard_size)
        self._posting_bucket_count = max(posting_bucket_count, 8)

        self._chunk_buffer: list[KnowledgeChunk] = []
        self._docstats_buffer: list[DocStatRecord] = []
        self._posting_buffer: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._current_shard_id = 0

        self._chunk_shards: list[ChunkShardRef] = []
        self._posting_shards: list[PostingShardRef] = []
        self._total_chunks = 0
        self._total_doc_length = 0

        _prepare_store_root(store_root)

    def _buffer_size(self) -> int:
        postings = sum(len(items) for items in self._posting_buffer.values())
        return len(self._chunk_buffer) + len(self._docstats_buffer) + postings

    def add(self, chunk: KnowledgeChunk) -> None:
        """Write one normalized chunk into the active shard buffers."""

        normalized = chunk.model_copy(update={"tokens": tokenize(chunk.text)})
        self._chunk_buffer.append(normalized)

        doc_length = len(normalized.tokens)
        self._docstats_buffer.append(
            DocStatRecord(chunk_id=normalized.chunk_id, doc_length=doc_length)
        )
        self._total_chunks += 1
        self._total_doc_length += doc_length

        term_freq = Counter(normalized.tokens)
        for term, tf in sorted(term_freq.items()):
            self._posting_buffer[
                _posting_bucket(term, self._posting_bucket_count)
            ].append({"term": term, "chunk_id": normalized.chunk_id, "tf": tf})

        if len(self._chunk_buffer) >= self._shard_size:
            self._flush_current_shard()
            return

        if len(self._docstats_buffer) >= self._write_batch_size:
            self._flush_docstats()

        if self._buffer_size() >= self._max_in_memory_records:
            self._flush_current_shard()

    def _flush_docstats(self) -> None:
        if not self._docstats_buffer:
            return
        path = self._store_root / _DOCSTATS_FILENAME
        with path.open("a", encoding="utf-8") as handle:
            for item in self._docstats_buffer:
                handle.write(item.model_dump_json())
                handle.write("\n")
        self._docstats_buffer.clear()

    def _flush_current_shard(self) -> None:
        if not self._chunk_buffer:
            self._flush_docstats()
            return

        shard_id = self._current_shard_id
        chunk_filename = f"{_CHUNK_SHARD_PREFIX}-{shard_id:05d}.jsonl"
        chunk_path = self._store_root / chunk_filename
        with chunk_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunk_buffer:
                handle.write(json.dumps(chunk.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")

        self._chunk_shards.append(
            ChunkShardRef(
                shard_id=shard_id,
                path=Path(chunk_filename),
                count=len(self._chunk_buffer),
            )
        )

        for bucket in sorted(self._posting_buffer):
            records = sorted(
                self._posting_buffer[bucket],
                key=lambda item: (
                    str(item.get("term", "")),
                    str(item.get("chunk_id", "")),
                ),
            )
            if not records:
                continue
            posting_filename = f"{_POSTINGS_PREFIX}-{shard_id:05d}-{bucket}.jsonl"
            posting_path = self._store_root / posting_filename
            with posting_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
            self._posting_shards.append(
                PostingShardRef(
                    shard_id=shard_id,
                    bucket=bucket,
                    path=Path(posting_filename),
                    count=len(records),
                )
            )

        self._chunk_buffer.clear()
        self._posting_buffer.clear()
        self._flush_docstats()
        self._current_shard_id += 1

    def finalize(self) -> RetrievalManifest:
        """Flush all remaining records and return manifest."""

        self._flush_current_shard()

        average_doc_length = (
            (self._total_doc_length / self._total_chunks) if self._total_chunks else 0.0
        )
        manifest = RetrievalManifest(
            shard_size=self._shard_size,
            total_chunks=self._total_chunks,
            average_doc_length=average_doc_length,
            chunk_shards=self._chunk_shards,
            posting_shards=self._posting_shards,
            docstats_path=Path(_DOCSTATS_FILENAME),
        )
        save_retrieval_manifest(manifest, self._store_root)
        return manifest


class RetrievalQueryEngine:
    """BM25-style retrieval over sharded JSON storage."""

    def __init__(self, store_root: Path, manifest: RetrievalManifest) -> None:
        self._store_root = store_root
        self._manifest = manifest
        self._k1 = 1.5
        self._b = 0.75

        bucket_map: dict[str, list[Path]] = defaultdict(list)
        for shard in manifest.posting_shards:
            bucket_map[shard.bucket].append(self._store_root / shard.path)
        self._posting_paths_by_bucket = {
            bucket: sorted(paths) for bucket, paths in bucket_map.items()
        }

    def iter_chunks(self) -> Iterator[KnowledgeChunk]:
        """Yield all chunks from chunk shards."""

        for shard in sorted(
            self._manifest.chunk_shards, key=lambda item: item.shard_id
        ):
            path = self._store_root / shard.path
            for row in _iter_jsonl(path):
                yield KnowledgeChunk.model_validate(row)

    def _load_postings_for_term(self, term: str) -> dict[str, int]:
        bucket = _posting_bucket(term)
        matches: dict[str, int] = {}
        for path in self._posting_paths_by_bucket.get(bucket, []):
            for row in _iter_jsonl(path):
                if row.get("term") != term:
                    continue
                chunk_id = str(row.get("chunk_id", ""))
                tf_raw = row.get("tf", 0)
                tf = int(tf_raw) if isinstance(tf_raw, int | float | str) else 0
                if chunk_id:
                    matches[chunk_id] = tf
        return matches

    def _load_doc_lengths(self, chunk_ids: set[str]) -> dict[str, int]:
        if not chunk_ids:
            return {}
        results: dict[str, int] = {}
        path = self._store_root / self._manifest.docstats_path
        for row in _iter_jsonl(path):
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id not in chunk_ids:
                continue
            doc_length_raw = row.get("doc_length", 0)
            doc_length = (
                int(doc_length_raw)
                if isinstance(doc_length_raw, int | float | str)
                else 0
            )
            results[chunk_id] = doc_length
            if len(results) >= len(chunk_ids):
                break
        return results

    def _load_chunks(self, chunk_ids: set[str]) -> dict[str, KnowledgeChunk]:
        if not chunk_ids:
            return {}
        found: dict[str, KnowledgeChunk] = {}
        for chunk in self.iter_chunks():
            if chunk.chunk_id not in chunk_ids:
                continue
            found[chunk.chunk_id] = chunk
            if len(found) >= len(chunk_ids):
                break
        return found

    def _idf(self, doc_count: int, df: int) -> float:
        if doc_count == 0:
            return 0.0
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    @staticmethod
    def _tie_break_key(chunk: KnowledgeChunk) -> tuple[str, int, str]:
        return (chunk.source_path.as_posix(), chunk.line_start or 0, chunk.chunk_id)

    def _fallback(self, top_k: int) -> list[KnowledgeChunk]:
        smallest: list[KnowledgeChunk] = []
        for chunk in self.iter_chunks():
            smallest.append(chunk)
            smallest.sort(key=self._tie_break_key)
            if len(smallest) > top_k:
                smallest.pop()
        return smallest

    def search(self, query: str, top_k: int = 6) -> list[KnowledgeChunk]:
        """Search chunks using BM25 scoring over postings shards."""

        query_terms = tokenize(query)
        if not query_terms:
            return self._fallback(top_k)

        postings_by_term: dict[str, dict[str, int]] = {}
        candidate_ids: set[str] = set()
        for term in query_terms:
            postings = self._load_postings_for_term(term)
            if postings:
                postings_by_term[term] = postings
                candidate_ids.update(postings)

        if not candidate_ids:
            return self._fallback(top_k)

        doc_lengths = self._load_doc_lengths(candidate_ids)
        chunks_by_id = self._load_chunks(candidate_ids)

        avg_len = self._manifest.average_doc_length or 1.0
        doc_count = self._manifest.total_chunks
        scored: list[tuple[float, KnowledgeChunk]] = []

        for chunk_id in sorted(candidate_ids):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            doc_len = doc_lengths.get(chunk_id, len(chunk.tokens))
            score = 0.0
            for term in query_terms:
                tf_by_chunk = postings_by_term.get(term)
                if not tf_by_chunk:
                    continue
                tf = tf_by_chunk.get(chunk_id, 0)
                if tf <= 0:
                    continue
                df = len(tf_by_chunk)
                idf = self._idf(doc_count=doc_count, df=df)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * (doc_len / avg_len)
                )
                score += idf * (numerator / denominator)
            if score > 0.0:
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

        return self._fallback(top_k)

    def load_chunks_by_node_ids(
        self, node_ids: set[str], *, limit: int | None = None
    ) -> list[KnowledgeChunk]:
        """Load hierarchy summary chunks by node ids."""

        if not node_ids:
            return []

        matches: list[KnowledgeChunk] = []
        for chunk in self.iter_chunks():
            node_id = chunk.metadata.get("node_id")
            if node_id not in node_ids:
                continue
            if chunk.source_type not in {"file_summary", "directory_summary"}:
                continue
            matches.append(chunk)

        matches.sort(key=self._tie_break_key)
        if limit is not None:
            return matches[:limit]
        return matches


def _prepare_store_root(store_root: Path) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    for pattern in (
        f"{_CHUNK_SHARD_PREFIX}-*.jsonl",
        f"{_POSTINGS_PREFIX}-*.jsonl",
        _DOCSTATS_FILENAME,
        _MANIFEST_FILENAME,
    ):
        for path in store_root.glob(pattern):
            path.unlink(missing_ok=True)


def _posting_bucket(term: str, bucket_count: int = 64) -> str:
    digest = hashlib.md5(term.encode("utf-8")).hexdigest()
    bucket_num = int(digest, 16) % bucket_count
    return f"b{bucket_num:02d}"


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Invalid JSONL row in {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(
                    f"Expected object row in {path}, got: {type(value)}"
                )
            yield value


def default_retrieval_store_path(output_root: Path) -> Path:
    """Return default retrieval store directory for an output root."""

    return output_root / _RETRIEVAL_DIRNAME


def save_retrieval_manifest(manifest: RetrievalManifest, store_root: Path) -> Path:
    """Persist retrieval manifest to the store root."""

    store_root.mkdir(parents=True, exist_ok=True)
    path = store_root / _MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_retrieval_manifest(index_path: Path) -> tuple[RetrievalManifest, Path]:
    """Load retrieval manifest from a manifest file or retrieval root."""

    if index_path.is_dir():
        manifest_path = index_path / _MANIFEST_FILENAME
        store_root = index_path
    else:
        manifest_path = index_path
        store_root = index_path.parent

    if not manifest_path.exists():
        raise AnalysisError(
            f"Retrieval manifest not found at {manifest_path}. "
            "Run `sddraft generate` or migrate a legacy index."
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"Invalid retrieval manifest JSON at {manifest_path}: {exc}"
        ) from exc

    return RetrievalManifest.model_validate(raw), store_root


def build_retrieval_store(
    *,
    store_root: Path,
    chunks: Iterable[KnowledgeChunk],
    shard_size: int,
    write_batch_size: int,
    max_in_memory_records: int,
) -> RetrievalManifest:
    """Build a sharded retrieval store from chunk stream."""

    sink = JsonlChunkSink(
        store_root,
        shard_size=shard_size,
        write_batch_size=write_batch_size,
        max_in_memory_records=max_in_memory_records,
    )
    for chunk in chunks:
        sink.add(chunk)
    return sink.finalize()


def resolve_retrieval_store_path(index_path: Path) -> Path:
    """Resolve user-supplied path to retrieval store root."""

    path = index_path
    if path.is_file() and path.name == _LEGACY_INDEX_FILENAME:
        raise AnalysisError(
            "Legacy retrieval index file detected. "
            f"Run `sddraft migrate-index --index-path {path}`."
        )

    if path.is_file() and path.name == _MANIFEST_FILENAME:
        return path.parent

    if path.is_dir() and (path / _MANIFEST_FILENAME).exists():
        return path

    if path.is_dir() and (path / _RETRIEVAL_DIRNAME / _MANIFEST_FILENAME).exists():
        return path / _RETRIEVAL_DIRNAME

    if path.is_dir() and (path / _LEGACY_INDEX_FILENAME).exists():
        raise AnalysisError(
            "Legacy retrieval index file detected. "
            f"Run `sddraft migrate-index --index-path {path / _LEGACY_INDEX_FILENAME}`."
        )

    raise AnalysisError(
        f"Retrieval store not found at {index_path}. "
        "Expected a retrieval directory containing manifest.json."
    )


def open_query_engine(index_path: Path) -> RetrievalQueryEngine:
    """Open query engine from CLI/user path."""

    store_root = resolve_retrieval_store_path(index_path)
    manifest, resolved_root = load_retrieval_manifest(store_root)
    return RetrievalQueryEngine(resolved_root, manifest)


def _resolve_legacy_index_path(index_path: Path) -> Path | None:
    if index_path.is_file() and index_path.name == _LEGACY_INDEX_FILENAME:
        return index_path
    if index_path.is_dir() and (index_path / _LEGACY_INDEX_FILENAME).exists():
        return index_path / _LEGACY_INDEX_FILENAME
    return None


def migrate_legacy_index(
    *,
    index_path: Path,
    shard_size: int,
    write_batch_size: int,
    max_in_memory_records: int,
) -> Path:
    """Migrate legacy retrieval_index.json to the sharded retrieval store."""

    legacy_path = _resolve_legacy_index_path(index_path)
    if legacy_path is None:
        # idempotent: if already migrated, return resolved retrieval root
        resolved = resolve_retrieval_store_path(index_path)
        return resolved

    legacy_index = load_retrieval_index(legacy_path)
    store_root = legacy_path.parent / _RETRIEVAL_DIRNAME
    build_retrieval_store(
        store_root=store_root,
        chunks=legacy_index.chunks,
        shard_size=shard_size,
        write_batch_size=write_batch_size,
        max_in_memory_records=max_in_memory_records,
    )
    return store_root


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
    """Persist legacy retrieval index JSON (migration support)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_retrieval_index(path: Path) -> RetrievalIndex:
    """Load legacy retrieval index from JSON."""

    if not path.exists():
        raise AnalysisError(f"Retrieval index not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid retrieval index JSON at {path}: {exc}") from exc
    return RetrievalIndex.model_validate(raw)
