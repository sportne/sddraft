"""Deterministic repository scanning utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sddraft.domain.errors import AnalysisError, RepositoryError
from sddraft.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    InterfaceSummary,
    KnowledgeChunk,
    ProjectConfig,
    ScanResult,
    SourceLanguage,
)
from sddraft.repo.language_analyzers import detect_language, get_analyzer_for_path


def _matches_patterns(path: Path, include: list[str], exclude: list[str]) -> bool:
    path_match = PurePosixPath(path.as_posix())
    include_match = any(path_match.match(pattern) for pattern in include)
    exclude_match = any(path_match.match(pattern) for pattern in exclude)
    return include_match and not exclude_match


def discover_source_files(
    roots: Iterable[Path],
    include: list[str],
    exclude: list[str],
    repo_root: Path,
) -> list[Path]:
    """Discover source files from roots using include/exclude patterns."""

    discovered: list[Path] = []
    for root in roots:
        normalized_root = root if root.is_absolute() else (repo_root / root)
        normalized_root = normalized_root.resolve()
        if not normalized_root.exists():
            raise RepositoryError(f"Source root does not exist: {normalized_root}")

        for file_path in normalized_root.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                relative_path = file_path.resolve().relative_to(repo_root.resolve())
            except ValueError:
                relative_path = Path(file_path.name)
            if _matches_patterns(Path(relative_path), include, exclude):
                discovered.append(file_path.resolve())

    deduplicated = sorted(set(discovered))
    return deduplicated


def _build_chunk_id(source_path: Path, line_start: int, line_end: int) -> str:
    return f"code::{source_path.as_posix()}::{line_start}-{line_end}"


@dataclass(slots=True)
class ScanRecord:
    """Streaming scan record for one source file."""

    path: Path
    code_summary: CodeUnitSummary
    interface_summaries: list[InterfaceSummary]
    code_chunks: list[KnowledgeChunk]


def build_code_chunks(
    paths: Iterable[Path],
    repo_root: Path,
    chunk_lines: int = 40,
    language_by_path: dict[Path, SourceLanguage] | None = None,
    symbol_count_by_path: dict[Path, int] | None = None,
) -> list[KnowledgeChunk]:
    """Split code files into deterministic line-based retrieval chunks."""

    chunks: list[KnowledgeChunk] = []
    language_by_path = language_by_path or {}
    symbol_count_by_path = symbol_count_by_path or {}

    for path in sorted(paths):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        lines = content.splitlines()
        if not lines:
            continue

        language = language_by_path.get(path, detect_language(path))
        symbol_count = symbol_count_by_path.get(path, 0)

        start = 0
        while start < len(lines):
            end = min(start + chunk_lines, len(lines))
            snippet_lines = lines[start:end]
            snippet = "\n".join(snippet_lines).strip()
            if snippet:
                relative_path = path
                try:
                    relative_path = path.relative_to(repo_root)
                except ValueError:
                    relative_path = path

                chunks.append(
                    KnowledgeChunk(
                        chunk_id=_build_chunk_id(relative_path, start + 1, end),
                        source_type="code",
                        source_path=relative_path,
                        text=snippet,
                        line_start=start + 1,
                        line_end=end,
                        metadata={
                            "language": language,
                            "symbol_count": str(symbol_count),
                        },
                    )
                )
            start = end

    return chunks


def analyze_source_file(path: Path) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
    """Analyze one source file with the registered language analyzer."""

    analyzer = get_analyzer_for_path(path)
    if analyzer.language == "unknown":
        raise AnalysisError(f"Unsupported source language for '{path}'")

    try:
        source_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"Failed to read source file '{path}': {exc}") from exc

    return analyzer.analyze(path=path, source_text=source_text)


def scan_repository(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
) -> ScanResult:
    """Run deterministic repository scanning and summary extraction."""

    files: list[Path] = []
    code_summaries: list[CodeUnitSummary] = []
    interface_summaries: list[InterfaceSummary] = []
    code_chunks: list[KnowledgeChunk] = []
    dependency_values: set[str] = set()

    for record in scan_repository_stream(
        project_config=project_config,
        csc=csc,
        repo_root=repo_root,
    ):
        files.append(record.path)
        code_summaries.append(record.code_summary)
        interface_summaries.extend(record.interface_summaries)
        code_chunks.extend(record.code_chunks)
        dependency_values.update(record.code_summary.imports)

    return ScanResult(
        files=files,
        code_summaries=code_summaries,
        interface_summaries=interface_summaries,
        dependencies=sorted(dependency_values),
        code_chunks=code_chunks,
    )


def scan_repository_stream(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
) -> Iterable[ScanRecord]:
    """Yield per-file scan records for bounded-memory workflows."""

    roots = csc.source_roots or project_config.sources.roots
    files = discover_source_files(
        roots=roots,
        include=project_config.sources.include,
        exclude=project_config.sources.exclude,
        repo_root=repo_root,
    )

    if len(files) > project_config.generation.max_files:
        files = files[: project_config.generation.max_files]

    for path in files:
        try:
            summary, interfaces = analyze_source_file(path)
        except AnalysisError:
            continue

        symbol_count = len(summary.functions) + len(summary.classes)
        chunks = build_code_chunks(
            [path],
            repo_root=repo_root,
            chunk_lines=project_config.generation.code_chunk_lines,
            language_by_path={path: summary.language},
            symbol_count_by_path={path: symbol_count},
        )
        yield ScanRecord(
            path=path,
            code_summary=summary,
            interface_summaries=interfaces,
            code_chunks=chunks,
        )
