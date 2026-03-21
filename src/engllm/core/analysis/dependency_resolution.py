"""Language-aware dependency normalization and repo-local resolution.

This module converts language-specific import/include/use statements into a
single normalized record shape used by graph building. Resolution is
intentionally conservative: we emit repo-local links only when a deterministic
target file can be identified.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from engllm.domain.models import CodeUnitSummary, SourceLanguage, SymbolSummary

DependencyKind = Literal["import", "include", "using", "require", "use"]
DependencyResolutionStatus = Literal[
    "resolved_exact",
    "resolved_heuristic",
    "unresolved",
]

_JS_TS_EXTENSIONS = (".ts", ".tsx", ".js", ".mjs", ".cjs")


@dataclass(frozen=True, slots=True)
class DependencyRecord:
    """Normalized dependency edge candidate."""

    source_path: Path
    language: SourceLanguage
    raw_dependency: str
    normalized_key: str | None
    dependency_kind: DependencyKind
    resolution_status: DependencyResolutionStatus
    target_path: Path | None = None


@dataclass(slots=True)
class _ResolutionIndexes:
    files_set: set[Path]
    files_by_name: dict[str, set[Path]]
    python_module_to_file: dict[str, Path]
    java_type_to_file: dict[str, Path]
    js_ts_module_to_file: dict[str, set[Path]]
    go_module_to_file: dict[str, set[Path]]
    rust_module_to_file: dict[str, set[Path]]
    rust_crate_roots: list[Path]
    csharp_namespace_to_file: dict[str, Path]


def _normalize_path(path: Path) -> Path:
    return Path(PurePosixPath(path.as_posix()))


def _unique_path(values: set[Path]) -> Path | None:
    if len(values) != 1:
        return None
    return next(iter(values))


def _paths_by_name(files: list[Path]) -> dict[str, set[Path]]:
    index: dict[str, set[Path]] = defaultdict(set)
    for file_path in files:
        index[file_path.name].add(file_path)
    return index


def _build_python_module_index(files: list[Path]) -> dict[str, Path]:
    candidates: dict[str, set[Path]] = defaultdict(set)
    for file_path in files:
        if file_path.suffix != ".py":
            continue
        parts = list(file_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        for idx in range(len(parts)):
            module_name = ".".join(parts[idx:])
            if module_name:
                candidates[module_name].add(file_path)
    module_to_file: dict[str, Path] = {}
    for module_name, file_paths in candidates.items():
        unique = _unique_path(file_paths)
        if unique is not None:
            module_to_file[module_name] = unique
    return module_to_file


def _extract_python_modules(raw_imports: list[str]) -> set[str]:
    modules: set[str] = set()
    for line in raw_imports:
        stripped = line.strip()
        if stripped.startswith("import "):
            tail = stripped[len("import ") :]
            for part in tail.split(","):
                name = part.strip().split(" as ")[0].strip()
                if name:
                    modules.add(name)
        elif stripped.startswith("from "):
            match = re.match(r"from\s+([A-Za-z0-9_\.]+)\s+import\s+", stripped)
            if match:
                modules.add(match.group(1))
    return modules


def _java_package_from_imports(raw_imports: list[str]) -> str | None:
    for line in raw_imports:
        match = re.match(r"\s*package\s+([A-Za-z0-9_.]+)\s*;", line.strip())
        if match:
            return match.group(1)
    return None


def _build_java_type_index(
    *,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
) -> dict[str, Path]:
    package_by_file: dict[Path, str | None] = {
        item.path: _java_package_from_imports(item.imports)
        for item in code_summaries
        if item.language == "java"
    }
    candidates: dict[str, set[Path]] = defaultdict(set)
    for symbol in symbol_summaries:
        if symbol.language != "java":
            continue
        if symbol.owner_qualified_name is not None:
            continue
        if symbol.kind not in {"class", "interface", "enum", "record"}:
            continue
        package = package_by_file.get(symbol.source_path)
        key = f"{package}.{symbol.name}" if package else symbol.name
        candidates[key].add(symbol.source_path)
    resolved: dict[str, Path] = {}
    for key, file_paths in candidates.items():
        unique = _unique_path(file_paths)
        if unique is not None:
            resolved[key] = unique
    return resolved


def _build_suffixless_index(
    files: list[Path], *, suffixes: set[str]
) -> dict[str, set[Path]]:
    index: dict[str, set[Path]] = defaultdict(set)
    for file_path in files:
        if file_path.suffix not in suffixes:
            continue
        key = file_path.with_suffix("").as_posix()
        index[key].add(file_path)
        if file_path.stem == "index":
            parent_key = file_path.parent.as_posix()
            if parent_key and parent_key != ".":
                index[parent_key].add(file_path)
        filename_key = file_path.stem
        index[filename_key].add(file_path)
    return index


def _resolve_by_suffix_match(
    *, normalized_key: str, index: dict[str, set[Path]]
) -> tuple[Path | None, DependencyResolutionStatus]:
    direct = _unique_path(index.get(normalized_key, set()))
    if direct is not None:
        return direct, "resolved_heuristic"

    candidates: set[Path] = set()
    for module_key, module_paths in index.items():
        if module_key.endswith(normalized_key):
            candidates.update(module_paths)
    unique = _unique_path(candidates)
    if unique is None:
        return None, "unresolved"
    return unique, "resolved_heuristic"


def _resolve_relative_path(
    *,
    source_path: Path,
    module_key: str,
    files_set: set[Path],
    extensions: tuple[str, ...],
) -> tuple[Path | None, DependencyResolutionStatus]:
    if module_key.startswith("/"):
        base = _normalize_path(Path(module_key.lstrip("/")))
    else:
        base = _normalize_path(source_path.parent / module_key)

    candidates: list[tuple[Path, DependencyResolutionStatus]] = []
    if base.suffix:
        candidates.append((base, "resolved_exact"))
    else:
        for ext in extensions:
            candidates.append((base.with_suffix(ext), "resolved_heuristic"))
        for ext in extensions:
            candidates.append((base / f"index{ext}", "resolved_heuristic"))

    for candidate, status in candidates:
        normalized = _normalize_path(candidate)
        if normalized in files_set:
            return normalized, status
    return None, "unresolved"


def _extract_js_ts_modules(raw_dependency: str) -> list[tuple[DependencyKind, str]]:
    stripped = raw_dependency.strip()
    values: list[tuple[DependencyKind, str]] = []
    patterns: tuple[tuple[str, DependencyKind], ...] = (
        (r"import\s+.+?\s+from\s+['\"]([^'\"]+)['\"]", "import"),
        (r"import\s+['\"]([^'\"]+)['\"]", "import"),
        (r"export\s+.+?\s+from\s+['\"]([^'\"]+)['\"]", "import"),
        (r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", "require"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, stripped)
        if match:
            values.append((kind, match.group(1)))
    return values


def _extract_go_import_path(raw_dependency: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", raw_dependency.strip())
    if match:
        return match.group(1)
    return None


def _extract_rust_use_path(raw_dependency: str) -> str | None:
    stripped = raw_dependency.strip()
    match = re.match(r"(?:pub\s+)?use\s+(.+);", stripped)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.split(" as ", maxsplit=1)[0].strip()
    value = value.split("{", maxsplit=1)[0].strip().rstrip(":")
    return value or None


def _rust_crate_roots(files: list[Path]) -> list[Path]:
    roots: set[Path] = set()
    for path in files:
        if path.suffix != ".rs":
            continue
        if path.name in {"lib.rs", "main.rs"}:
            roots.add(path.parent)
    if not roots:
        roots.add(Path("."))
    return sorted(roots, key=lambda item: item.as_posix())


def _resolve_rust_module(
    *,
    source_path: Path,
    module_key: str,
    files_set: set[Path],
    crate_roots: list[Path],
) -> tuple[Path | None, DependencyResolutionStatus]:
    candidates: list[Path] = []

    def _append_module_candidates(base_path: Path) -> None:
        candidates.append(_normalize_path(base_path.with_suffix(".rs")))
        candidates.append(_normalize_path(base_path / "mod.rs"))

        # `use crate::pkg::item;` frequently targets a symbol inside `pkg.rs`.
        # When there is nesting, also probe the parent module path.
        if len(base_path.parts) > 1:
            parent_path = Path(*base_path.parts[:-1])
            candidates.append(_normalize_path(parent_path.with_suffix(".rs")))
            candidates.append(_normalize_path(parent_path / "mod.rs"))

    if module_key.startswith("crate::"):
        suffix = module_key[len("crate::") :]
        for root in crate_roots:
            module_path = Path(suffix.replace("::", "/"))
            _append_module_candidates(root / module_path)
    elif module_key.startswith("self::"):
        suffix = module_key[len("self::") :]
        module_path = Path(suffix.replace("::", "/"))
        _append_module_candidates(source_path.parent / module_path)
    elif module_key.startswith("super::"):
        suffix = module_key[len("super::") :]
        module_path = Path(suffix.replace("::", "/"))
        parent = (
            source_path.parent.parent if source_path.parent != Path(".") else Path(".")
        )
        _append_module_candidates(parent / module_path)
    else:
        return None, "unresolved"

    unique = _unique_path(
        {candidate for candidate in candidates if candidate in files_set}
    )
    if unique is None:
        return None, "unresolved"
    return unique, "resolved_exact"


def _extract_cpp_include(raw_dependency: str) -> tuple[str, str] | None:
    match = re.match(r"\s*#\s*include\s*([<\"])([^>\"]+)[>\"]", raw_dependency.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _load_csharp_namespaces(files: list[Path], repo_root: Path) -> dict[str, Path]:
    candidates: dict[str, set[Path]] = defaultdict(set)
    for file_path in files:
        if file_path.suffix != ".cs":
            continue
        absolute = repo_root / file_path
        try:
            text = absolute.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in re.finditer(
            r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.]*)",
            text,
            re.MULTILINE,
        ):
            candidates[match.group(1)].add(file_path)
    resolved: dict[str, Path] = {}
    for namespace, namespace_files in candidates.items():
        unique = _unique_path(namespace_files)
        if unique is not None:
            resolved[namespace] = unique
    return resolved


def _build_indexes(
    *,
    files: list[Path],
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    repo_root: Path,
) -> _ResolutionIndexes:
    return _ResolutionIndexes(
        files_set=set(files),
        files_by_name=_paths_by_name(files),
        python_module_to_file=_build_python_module_index(files),
        java_type_to_file=_build_java_type_index(
            code_summaries=code_summaries,
            symbol_summaries=symbol_summaries,
        ),
        js_ts_module_to_file=_build_suffixless_index(
            files, suffixes={".js", ".mjs", ".cjs", ".ts", ".tsx"}
        ),
        go_module_to_file=_build_suffixless_index(files, suffixes={".go"}),
        rust_module_to_file=_build_suffixless_index(files, suffixes={".rs"}),
        rust_crate_roots=_rust_crate_roots(files),
        csharp_namespace_to_file=_load_csharp_namespaces(files, repo_root),
    )


def _resolve_python(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for module_name in sorted(_extract_python_modules(summary.imports)):
        target = indexes.python_module_to_file.get(module_name)
        if target is None:
            records.append(
                DependencyRecord(
                    source_path=summary.path,
                    language=summary.language,
                    raw_dependency=module_name,
                    normalized_key=module_name,
                    dependency_kind="import",
                    resolution_status="unresolved",
                )
            )
            continue
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=module_name,
                normalized_key=module_name,
                dependency_kind="import",
                resolution_status="resolved_exact",
                target_path=target,
            )
        )
    return records


def _resolve_java(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        match = re.match(
            r"\s*import\s+(?:static\s+)?([A-Za-z0-9_.*]+)\s*;",
            raw.strip(),
        )
        if not match:
            continue
        key = match.group(1)
        if key.endswith(".*"):
            records.append(
                DependencyRecord(
                    source_path=summary.path,
                    language=summary.language,
                    raw_dependency=raw.strip(),
                    normalized_key=key,
                    dependency_kind="import",
                    resolution_status="unresolved",
                )
            )
            continue
        target = indexes.java_type_to_file.get(key)
        if target is None:
            records.append(
                DependencyRecord(
                    source_path=summary.path,
                    language=summary.language,
                    raw_dependency=raw.strip(),
                    normalized_key=key,
                    dependency_kind="import",
                    resolution_status="unresolved",
                )
            )
            continue
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=raw.strip(),
                normalized_key=key,
                dependency_kind="import",
                resolution_status="resolved_exact",
                target_path=target,
            )
        )
    return records


def _resolve_js_ts(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        for kind, module_key in _extract_js_ts_modules(raw):
            target: Path | None
            status: DependencyResolutionStatus
            if module_key.startswith((".", "/")):
                target, status = _resolve_relative_path(
                    source_path=summary.path,
                    module_key=module_key,
                    files_set=indexes.files_set,
                    extensions=_JS_TS_EXTENSIONS,
                )
            else:
                target, status = _resolve_by_suffix_match(
                    normalized_key=module_key,
                    index=indexes.js_ts_module_to_file,
                )
            records.append(
                DependencyRecord(
                    source_path=summary.path,
                    language=summary.language,
                    raw_dependency=raw.strip(),
                    normalized_key=module_key,
                    dependency_kind=kind,
                    resolution_status=status,
                    target_path=target,
                )
            )
    return records


def _resolve_go(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        module_key = _extract_go_import_path(raw)
        if module_key is None:
            continue
        if module_key.startswith((".", "/")):
            target, status = _resolve_relative_path(
                source_path=summary.path,
                module_key=module_key,
                files_set=indexes.files_set,
                extensions=(".go",),
            )
        else:
            target, status = _resolve_by_suffix_match(
                normalized_key=module_key,
                index=indexes.go_module_to_file,
            )
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=raw.strip(),
                normalized_key=module_key,
                dependency_kind="import",
                resolution_status=status,
                target_path=target,
            )
        )
    return records


def _resolve_rust(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        module_key = _extract_rust_use_path(raw)
        if module_key is None:
            continue
        target, status = _resolve_rust_module(
            source_path=summary.path,
            module_key=module_key,
            files_set=indexes.files_set,
            crate_roots=indexes.rust_crate_roots,
        )
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=raw.strip(),
                normalized_key=module_key,
                dependency_kind="use",
                resolution_status=status,
                target_path=target,
            )
        )
    return records


def _resolve_csharp(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        match = re.match(r"\s*(?:global\s+)?using\s+([A-Za-z0-9_.]+)\s*;", raw.strip())
        if not match:
            continue
        key = match.group(1)
        target = indexes.csharp_namespace_to_file.get(key)
        status: DependencyResolutionStatus = (
            "resolved_exact" if target else "unresolved"
        )
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=raw.strip(),
                normalized_key=key,
                dependency_kind="using",
                resolution_status=status,
                target_path=target,
            )
        )
    return records


def _resolve_cpp(
    summary: CodeUnitSummary, indexes: _ResolutionIndexes
) -> list[DependencyRecord]:
    records: list[DependencyRecord] = []
    for raw in sorted(summary.imports):
        include = _extract_cpp_include(raw)
        if include is None:
            continue
        delimiter, include_key = include
        if delimiter == "<":
            records.append(
                DependencyRecord(
                    source_path=summary.path,
                    language=summary.language,
                    raw_dependency=raw.strip(),
                    normalized_key=include_key,
                    dependency_kind="include",
                    resolution_status="unresolved",
                )
            )
            continue

        candidates: set[Path] = set()
        candidates.add(_normalize_path(summary.path.parent / include_key))
        candidates.add(_normalize_path(Path(include_key)))
        candidate_path = _unique_path(
            {candidate for candidate in candidates if candidate in indexes.files_set}
        )
        status: DependencyResolutionStatus = "resolved_exact"
        if candidate_path is None:
            by_name = indexes.files_by_name.get(Path(include_key).name, set())
            candidate_path = _unique_path(by_name)
            status = (
                "resolved_heuristic" if candidate_path is not None else "unresolved"
            )
        records.append(
            DependencyRecord(
                source_path=summary.path,
                language=summary.language,
                raw_dependency=raw.strip(),
                normalized_key=include_key,
                dependency_kind="include",
                resolution_status=status,
                target_path=candidate_path,
            )
        )
    return records


def resolve_dependency_records(
    *,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    files: list[Path],
    repo_root: Path,
) -> list[DependencyRecord]:
    """Resolve language-aware dependency records against repo-local files."""

    indexes = _build_indexes(
        files=files,
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        repo_root=repo_root,
    )
    records: list[DependencyRecord] = []

    for summary in sorted(code_summaries, key=lambda item: item.path.as_posix()):
        if summary.language == "python":
            records.extend(_resolve_python(summary, indexes))
        elif summary.language == "java":
            records.extend(_resolve_java(summary, indexes))
        elif summary.language in {"javascript", "typescript"}:
            records.extend(_resolve_js_ts(summary, indexes))
        elif summary.language == "go":
            records.extend(_resolve_go(summary, indexes))
        elif summary.language == "rust":
            records.extend(_resolve_rust(summary, indexes))
        elif summary.language == "csharp":
            records.extend(_resolve_csharp(summary, indexes))
        elif summary.language == "cpp":
            records.extend(_resolve_cpp(summary, indexes))

    return sorted(
        records,
        key=lambda item: (
            item.source_path.as_posix(),
            item.language,
            item.normalized_key or "",
            item.raw_dependency,
            item.target_path.as_posix() if item.target_path is not None else "",
        ),
    )


def dependency_reason_payload(record: DependencyRecord) -> str:
    """Return stable JSON reason payload for graph `imports` edges."""

    payload = {
        "kind": record.dependency_kind,
        "language": record.language,
        "normalized": record.normalized_key,
        "raw": record.raw_dependency,
        "resolution": record.resolution_status,
    }
    if record.target_path is not None:
        payload["target"] = record.target_path.as_posix()
    return json.dumps(payload, sort_keys=True)
