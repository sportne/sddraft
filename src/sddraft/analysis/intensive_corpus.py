"""Deterministic corpus building for intensive ask mode."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sddraft.analysis.retrieval import tokenize
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    IntensiveCorpusChunk,
    IntensiveCorpusManifest,
    IntensiveCorpusSegment,
    ProjectConfig,
)
from sddraft.repo.scanner import discover_source_files

_CORPUS_DIRNAME = "ask/intensive/corpus"
_MANIFEST_FILENAME = "manifest.json"
_CHUNKS_FILENAME = "chunks.jsonl"
_CORPUS_RENDER_VERSION = "v1-structured-cross-file"


@dataclass(frozen=True, slots=True)
class _CorpusUnit:
    """Atomic packing unit used while building the corpus."""

    source_path: Path
    line_start: int
    line_end: int
    text: str
    token_count: int
    file_fingerprint: str


def default_intensive_corpus_root(output_root: Path) -> Path:
    """Return the shared corpus directory for intensive ask artifacts."""

    return output_root / _CORPUS_DIRNAME


def default_intensive_runs_root(output_root: Path) -> Path:
    """Return the per-question run directory for intensive ask artifacts."""

    return output_root / "ask" / "intensive" / "runs"


def prepare_intensive_corpus(
    *,
    project_config: ProjectConfig,
    repo_root: Path,
    output_root: Path,
    csc_id: str,
    chunk_tokens: int,
) -> tuple[IntensiveCorpusManifest, bool]:
    """Build or reuse the persisted intensive corpus for one CSC output root."""

    corpus_root = default_intensive_corpus_root(output_root)
    corpus_root.mkdir(parents=True, exist_ok=True)

    units = _collect_corpus_units(
        project_config=project_config,
        repo_root=repo_root,
        chunk_tokens=chunk_tokens,
    )
    corpus_fingerprint = _build_corpus_fingerprint(units, chunk_tokens=chunk_tokens)

    existing = _load_existing_manifest(corpus_root)
    if (
        existing is not None
        and existing.corpus_fingerprint == corpus_fingerprint
        and existing.chunk_tokens == chunk_tokens
        and (corpus_root / existing.chunks_path).exists()
    ):
        return existing, True

    chunks = _pack_units(units, chunk_tokens=chunk_tokens)
    chunks_path = corpus_root / _CHUNKS_FILENAME
    _write_chunk_jsonl(chunks_path, chunks)

    manifest = IntensiveCorpusManifest(
        csc_id=csc_id,
        corpus_fingerprint=corpus_fingerprint,
        chunk_tokens=chunk_tokens,
        file_count=len({unit.source_path for unit in units}),
        chunk_count=len(chunks),
        chunks_path=Path(_CHUNKS_FILENAME),
    )
    _write_manifest(corpus_root / _MANIFEST_FILENAME, manifest)
    return manifest, False


def load_intensive_corpus_manifest(path: Path) -> IntensiveCorpusManifest:
    """Load an intensive corpus manifest from disk."""

    if not path.exists():
        raise AnalysisError(f"Intensive corpus manifest not found at {path}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"Invalid intensive corpus manifest at {path}: {exc}"
        ) from exc
    return IntensiveCorpusManifest.model_validate(raw)


def iter_intensive_corpus_chunks(manifest_path: Path) -> Iterator[IntensiveCorpusChunk]:
    """Yield persisted intensive corpus chunks in deterministic order."""

    manifest = load_intensive_corpus_manifest(manifest_path)
    chunks_path = manifest_path.parent / manifest.chunks_path
    if not chunks_path.exists():
        raise AnalysisError(f"Intensive corpus chunk store not found at {chunks_path}.")

    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnalysisError(
                    f"Invalid intensive corpus chunk row in {chunks_path}: {exc}"
                ) from exc
            yield IntensiveCorpusChunk.model_validate(raw)


def _load_existing_manifest(corpus_root: Path) -> IntensiveCorpusManifest | None:
    manifest_path = corpus_root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return load_intensive_corpus_manifest(manifest_path)


def _write_manifest(path: Path, manifest: IntensiveCorpusManifest) -> None:
    try:
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(
            f"Failed writing intensive corpus manifest to {path}: {exc}"
        ) from exc


def _write_chunk_jsonl(path: Path, chunks: list[IntensiveCorpusChunk]) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json())
                handle.write("\n")
    except OSError as exc:
        raise AnalysisError(
            f"Failed writing intensive corpus chunks to {path}: {exc}"
        ) from exc


def _collect_corpus_units(
    *,
    project_config: ProjectConfig,
    repo_root: Path,
    chunk_tokens: int,
) -> list[_CorpusUnit]:
    files = discover_source_files(
        roots=project_config.sources.roots,
        include=project_config.sources.include,
        exclude=project_config.sources.exclude,
        repo_root=repo_root,
    )

    units: list[_CorpusUnit] = []
    for absolute_path in files:
        relative_path = _to_repo_relative(absolute_path, repo_root)
        try:
            content = absolute_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not content.strip():
            continue
        units.extend(
            _split_file_into_units(
                source_path=relative_path,
                content=content,
                chunk_tokens=chunk_tokens,
            )
        )
    return units


def _build_corpus_fingerprint(units: list[_CorpusUnit], *, chunk_tokens: int) -> str:
    encoded = json.dumps(
        {
            "version": _CORPUS_RENDER_VERSION,
            "chunk_tokens": chunk_tokens,
            "units": [
                {
                    "source_path": unit.source_path.as_posix(),
                    "line_start": unit.line_start,
                    "line_end": unit.line_end,
                    "token_count": unit.token_count,
                    "file_fingerprint": unit.file_fingerprint,
                }
                for unit in units
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_file_into_units(
    *,
    source_path: Path,
    content: str,
    chunk_tokens: int,
) -> list[_CorpusUnit]:
    lines = content.splitlines()
    if not lines:
        return []

    line_token_counts = [len(tokenize(line)) for line in lines]
    file_token_count = sum(line_token_counts)
    file_fingerprint = hashlib.sha256(
        f"{source_path.as_posix()}\n{content}".encode()
    ).hexdigest()

    if file_token_count <= chunk_tokens:
        return [
            _CorpusUnit(
                source_path=source_path,
                line_start=1,
                line_end=len(lines),
                text=content,
                token_count=file_token_count,
                file_fingerprint=file_fingerprint,
            )
        ]

    units: list[_CorpusUnit] = []
    start = 0
    while start < len(lines):
        end = start
        running_tokens = 0
        while end < len(lines):
            candidate_tokens = line_token_counts[end]
            if running_tokens and running_tokens + candidate_tokens > chunk_tokens:
                break
            running_tokens += candidate_tokens
            end += 1
            if running_tokens >= chunk_tokens:
                break

        if end == start:
            # Keep one pathological line intact so line-based provenance stays true.
            running_tokens = line_token_counts[start]
            end = start + 1

        units.append(
            _CorpusUnit(
                source_path=source_path,
                line_start=start + 1,
                line_end=end,
                text="\n".join(lines[start:end]),
                token_count=running_tokens,
                file_fingerprint=file_fingerprint,
            )
        )
        start = end

    return units


def _pack_units(
    units: list[_CorpusUnit], *, chunk_tokens: int
) -> list[IntensiveCorpusChunk]:
    chunks: list[IntensiveCorpusChunk] = []
    current_segments: list[IntensiveCorpusSegment] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_segments, current_tokens
        if not current_segments:
            return
        chunks.append(
            IntensiveCorpusChunk(
                chunk_id=f"corpus::{len(chunks):05d}",
                token_count=current_tokens,
                segments=current_segments,
            )
        )
        current_segments = []
        current_tokens = 0

    for unit in units:
        segment = IntensiveCorpusSegment(
            source_path=unit.source_path,
            line_start=unit.line_start,
            line_end=unit.line_end,
            text=unit.text,
            token_count=unit.token_count,
        )
        if current_segments and current_tokens + unit.token_count > chunk_tokens:
            flush()
        current_segments.append(segment)
        current_tokens += unit.token_count
        if current_tokens >= chunk_tokens:
            flush()

    flush()
    return chunks


def _to_repo_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path
