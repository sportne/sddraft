"""Deterministic repository scanning utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from sddraft.domain.errors import AnalysisError, RepositoryError
from sddraft.domain.models import (
    CodeUnitSummary,
    CSCDescriptor,
    InterfaceSummary,
    KnowledgeChunk,
    ProjectConfig,
    ScanResult,
)


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


def _safe_parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def summarize_python_file(path: Path) -> CodeUnitSummary:
    """Extract deterministic summary information from a Python file."""

    module = _safe_parse_python(path)
    if module is None:
        raise AnalysisError(f"Failed to parse Python file: {path}")

    functions: list[str] = []
    classes: list[str] = []
    docstrings: list[str] = []
    imports: list[str] = []

    module_docstring = ast.get_docstring(module)
    if module_docstring:
        docstrings.append(module_docstring)

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            function_doc = ast.get_docstring(node)
            if function_doc:
                docstrings.append(function_doc)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            class_doc = ast.get_docstring(node)
            if class_doc:
                docstrings.append(class_doc)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    function_doc = ast.get_docstring(item)
                    if function_doc:
                        docstrings.append(function_doc)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imported = ", ".join(alias.name for alias in node.names)
            imports.append(f"{module_name}:{imported}")

    return CodeUnitSummary(
        path=path,
        language="python",
        functions=sorted(set(functions)),
        classes=sorted(set(classes)),
        docstrings=sorted(set(docstrings)),
        imports=sorted(set(imports)),
    )


def _extract_interface_summaries(
    path: Path, module: ast.Module
) -> list[InterfaceSummary]:
    interfaces: list[InterfaceSummary] = []

    for node in module.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            members = [
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
            ]
            interfaces.append(
                InterfaceSummary(
                    name=node.name,
                    kind="class",
                    source_path=path,
                    members=sorted(set(members)),
                )
            )

        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            interfaces.append(
                InterfaceSummary(
                    name=node.name,
                    kind="function",
                    source_path=path,
                    members=[],
                )
            )

    if interfaces:
        interfaces.append(
            InterfaceSummary(
                name=path.stem,
                kind="module",
                source_path=path,
                members=sorted(
                    {
                        iface.name
                        for iface in interfaces
                        if iface.kind in {"class", "function"}
                    }
                ),
            )
        )

    return interfaces


def extract_interface_summaries(paths: Iterable[Path]) -> list[InterfaceSummary]:
    """Extract public interfaces from source files where feasible."""

    interfaces: list[InterfaceSummary] = []
    for path in paths:
        if path.suffix != ".py":
            continue
        module = _safe_parse_python(path)
        if module is None:
            continue
        interfaces.extend(_extract_interface_summaries(path, module))
    return interfaces


def _build_chunk_id(source_path: Path, line_start: int, line_end: int) -> str:
    return f"code::{source_path.as_posix()}::{line_start}-{line_end}"


def build_code_chunks(
    paths: Iterable[Path],
    repo_root: Path,
    chunk_lines: int = 40,
) -> list[KnowledgeChunk]:
    """Split code files into deterministic line-based retrieval chunks."""

    chunks: list[KnowledgeChunk] = []
    for path in sorted(paths):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        lines = content.splitlines()
        if not lines:
            continue

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
                    )
                )
            start = end

    return chunks


def scan_repository(
    project_config: ProjectConfig,
    csc: CSCDescriptor,
    repo_root: Path,
) -> ScanResult:
    """Run deterministic repository scanning and summary extraction."""

    roots = csc.source_roots or project_config.sources.roots
    files = discover_source_files(
        roots=roots,
        include=project_config.sources.include,
        exclude=project_config.sources.exclude,
        repo_root=repo_root,
    )

    if len(files) > project_config.generation.max_files:
        files = files[: project_config.generation.max_files]

    code_summaries: list[CodeUnitSummary] = []
    for path in files:
        if path.suffix == ".py":
            try:
                code_summaries.append(summarize_python_file(path))
            except AnalysisError:
                continue

    interface_summaries = extract_interface_summaries(files)
    dependency_values: set[str] = set()
    for summary in code_summaries:
        dependency_values.update(summary.imports)

    code_chunks = build_code_chunks(
        files,
        repo_root=repo_root,
        chunk_lines=project_config.generation.code_chunk_lines,
    )

    return ScanResult(
        files=files,
        code_summaries=code_summaries,
        interface_summaries=interface_summaries,
        dependencies=sorted(dependency_values),
        code_chunks=code_chunks,
    )
