"""Deterministic repository scanning utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pathspec import PathSpec

from engllm.core.repo.language_analyzers import detect_language, get_analyzer_for_path
from engllm.domain.errors import AnalysisError, RepositoryError
from engllm.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    KnowledgeChunk,
    ProjectConfig,
    ScanResult,
    SourceLanguage,
    SymbolSummary,
)


def _matches_patterns(path: Path, patterns: list[str]) -> bool:
    path_match = PurePosixPath(path.as_posix())
    return any(path_match.match(pattern) for pattern in patterns)


def _load_gitignore_spec(repo_root: Path) -> PathSpec | None:
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return None
    try:
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return PathSpec.from_lines("gitignore", lines)


def discover_source_files(
    roots: Iterable[Path],
    include: list[str],
    exclude: list[str],
    repo_root: Path,
) -> list[Path]:
    """Discover source files from roots using include/exclude patterns."""

    discovered: list[Path] = []
    gitignore_spec = _load_gitignore_spec(repo_root.resolve())
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

            relative_posix_path = relative_path.as_posix()
            if gitignore_spec is not None and gitignore_spec.match_file(
                relative_posix_path
            ):
                continue

            include_match = _matches_patterns(relative_path, include)
            exclude_match = _matches_patterns(relative_path, exclude)
            if exclude_match:
                continue

            # Keep unknown files in scope for project understanding even when
            # include patterns target specific programming-language suffixes.
            if include_match or detect_language(relative_path) == "unknown":
                discovered.append(file_path.resolve())

    deduplicated = sorted(set(discovered))
    return deduplicated


def _build_chunk_id(source_path: Path, line_start: int, line_end: int) -> str:
    return f"code::{source_path.as_posix()}::{line_start}-{line_end}"


def _to_repo_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


@dataclass(slots=True)
class ScanRecord:
    """Streaming scan record for one source file."""

    path: Path
    code_summary: CodeUnitSummary
    symbol_summaries: list[SymbolSummary]
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
        except (OSError, UnicodeDecodeError):
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


def analyze_source_file(
    path: Path,
    repo_root: Path,
) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
    """Analyze one source file with the registered language analyzer."""

    analyzer = get_analyzer_for_path(path)

    try:
        source_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AnalysisError(f"Failed to read source file '{path}': {exc}") from exc

    return analyzer.analyze(
        path=_to_repo_relative(path, repo_root),
        source_text=source_text,
    )


def scan_paths(
    *,
    roots: Iterable[Path],
    include: list[str],
    exclude: list[str],
    repo_root: Path,
    max_files: int,
    chunk_lines: int,
    include_code_chunks: bool = True,
) -> ScanResult:
    """Run deterministic repository scanning over explicit roots."""

    files: list[Path] = []
    code_summaries: list[CodeUnitSummary] = []
    symbol_summaries: list[SymbolSummary] = []
    code_chunks: list[KnowledgeChunk] = []
    dependency_values: set[str] = set()

    for record in scan_paths_stream(
        roots=roots,
        include=include,
        exclude=exclude,
        repo_root=repo_root,
        max_files=max_files,
        chunk_lines=chunk_lines,
        include_code_chunks=include_code_chunks,
    ):
        files.append(record.path)
        code_summaries.append(record.code_summary)
        symbol_summaries.extend(record.symbol_summaries)
        code_chunks.extend(record.code_chunks)
        dependency_values.update(record.code_summary.imports)

    return ScanResult(
        files=files,
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        dependencies=sorted(dependency_values),
        code_chunks=code_chunks,
    )


def scan_paths_stream(
    *,
    roots: Iterable[Path],
    include: list[str],
    exclude: list[str],
    repo_root: Path,
    max_files: int,
    chunk_lines: int,
    include_code_chunks: bool = True,
) -> Iterable[ScanRecord]:
    """Yield per-file scan records over explicit source roots."""

    files = discover_source_files(
        roots=roots,
        include=include,
        exclude=exclude,
        repo_root=repo_root,
    )

    if len(files) > max_files:
        files = files[:max_files]

    for path in files:
        try:
            summary, symbols = analyze_source_file(path, repo_root=repo_root)
        except AnalysisError:
            continue

        symbol_count = len(symbols)
        chunks = (
            build_code_chunks(
                [path],
                repo_root=repo_root,
                chunk_lines=chunk_lines,
                language_by_path={path: summary.language},
                symbol_count_by_path={path: symbol_count},
            )
            if include_code_chunks
            else []
        )
        yield ScanRecord(
            path=_to_repo_relative(path, repo_root),
            code_summary=summary,
            symbol_summaries=symbols,
            code_chunks=chunks,
        )


def scan_repository(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
) -> ScanResult:
    """Run deterministic repository scanning and summary extraction."""

    roots = csc.source_roots or project_config.sources.roots
    return scan_paths(
        roots=roots,
        include=project_config.sources.include,
        exclude=project_config.sources.exclude,
        repo_root=repo_root,
        max_files=project_config.generation.max_files,
        chunk_lines=project_config.generation.code_chunk_lines,
    )


def scan_repository_stream(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
) -> Iterable[ScanRecord]:
    """Yield per-file scan records for bounded-memory workflows."""

    roots = csc.source_roots or project_config.sources.roots
    yield from scan_paths_stream(
        roots=roots,
        include=project_config.sources.include,
        exclude=project_config.sources.exclude,
        repo_root=repo_root,
        max_files=project_config.generation.max_files,
        chunk_lines=project_config.generation.code_chunk_lines,
    )
