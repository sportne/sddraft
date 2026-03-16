"""Deterministic symbol inventory extraction for engineering graph building."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from sddraft.analysis.graph_models import GraphSymbolRecord, symbol_node_id
from sddraft.domain.models import InterfaceSummary, ScanResult, SourceLanguage

_WORD_BOUNDARY_TEMPLATE = r"\b{symbol}\b"


def _to_absolute(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _line_pattern(language: SourceLanguage, kind: str, name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    if language == "python":
        if kind in {"class", "interface"}:
            return re.compile(rf"^\s*class\s+{escaped}\b")
        if kind in {"module"}:
            return re.compile(r"^$")
        return re.compile(rf"^\s*def\s+{escaped}\b")

    if kind in {"class", "interface"}:
        return re.compile(
            rf"\b(class|interface|struct|enum|record|trait|impl)\s+{escaped}\b"
        )

    if kind == "module":
        return re.compile(r"^$")

    return re.compile(rf"{_WORD_BOUNDARY_TEMPLATE.format(symbol=escaped)}\s*\(")


def _find_symbol_span(
    *,
    source_lines: list[str],
    language: SourceLanguage,
    kind: str,
    name: str,
) -> tuple[int | None, int | None]:
    if not source_lines:
        return None, None

    pattern = _line_pattern(language, kind, name)
    for idx, line in enumerate(source_lines, start=1):
        if pattern.search(line):
            return idx, idx

    return None, None


def _load_source_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _add_symbol(
    *,
    out: dict[str, GraphSymbolRecord],
    file_path: Path,
    language: SourceLanguage,
    kind: str,
    name: str,
    qualified_name: str | None,
    line_start: int | None,
    line_end: int | None,
) -> None:
    symbol_id = symbol_node_id(file_path, kind, qualified_name or name)
    if symbol_id in out:
        return
    out[symbol_id] = GraphSymbolRecord(
        symbol_id=symbol_id,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language=language,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
    )


def _symbol_kind_from_interface(interface: InterfaceSummary) -> str:
    if interface.kind == "class":
        return "interface"
    if interface.kind == "function":
        return "function"
    return "module"


def build_symbol_inventory(
    *,
    scan_result: ScanResult,
    repo_root: Path,
) -> list[GraphSymbolRecord]:
    """Build a conservative symbol inventory from deterministic scan outputs."""

    source_cache: dict[Path, list[str]] = {}
    symbols: dict[str, GraphSymbolRecord] = {}

    summary_by_path = {summary.path: summary for summary in scan_result.code_summaries}

    for path, summary in sorted(
        summary_by_path.items(), key=lambda item: item[0].as_posix()
    ):
        abs_path = _to_absolute(repo_root, path)
        source_lines = source_cache.setdefault(path, _load_source_lines(abs_path))

        for class_name in sorted(summary.classes):
            line_start, line_end = _find_symbol_span(
                source_lines=source_lines,
                language=summary.language,
                kind="class",
                name=class_name,
            )
            _add_symbol(
                out=symbols,
                file_path=path,
                language=summary.language,
                kind="class",
                name=class_name,
                qualified_name=class_name,
                line_start=line_start,
                line_end=line_end,
            )

        for function_name in sorted(summary.functions):
            line_start, line_end = _find_symbol_span(
                source_lines=source_lines,
                language=summary.language,
                kind="function",
                name=function_name,
            )
            _add_symbol(
                out=symbols,
                file_path=path,
                language=summary.language,
                kind="function",
                name=function_name,
                qualified_name=function_name,
                line_start=line_start,
                line_end=line_end,
            )

    interfaces_by_path: dict[Path, list[InterfaceSummary]] = defaultdict(list)
    for item in scan_result.interface_summaries:
        interfaces_by_path[item.source_path].append(item)

    for path, interfaces in sorted(
        interfaces_by_path.items(), key=lambda item: item[0].as_posix()
    ):
        file_summary = summary_by_path.get(path)
        language: SourceLanguage = (
            file_summary.language if file_summary is not None else "unknown"
        )
        abs_path = _to_absolute(repo_root, path)
        source_lines = source_cache.setdefault(path, _load_source_lines(abs_path))

        for interface in sorted(interfaces, key=lambda item: (item.kind, item.name)):
            symbol_kind = _symbol_kind_from_interface(interface)
            line_start, line_end = _find_symbol_span(
                source_lines=source_lines,
                language=language,
                kind=symbol_kind,
                name=interface.name,
            )
            _add_symbol(
                out=symbols,
                file_path=path,
                language=language,
                kind=symbol_kind,
                name=interface.name,
                qualified_name=interface.name,
                line_start=line_start,
                line_end=line_end,
            )

            for member_name in sorted(interface.members):
                member_start, member_end = _find_symbol_span(
                    source_lines=source_lines,
                    language=language,
                    kind="method",
                    name=member_name,
                )
                qualified = f"{interface.name}.{member_name}"
                _add_symbol(
                    out=symbols,
                    file_path=path,
                    language=language,
                    kind="method",
                    name=member_name,
                    qualified_name=qualified,
                    line_start=member_start,
                    line_end=member_end,
                )

    return sorted(
        symbols.values(),
        key=lambda item: (
            item.file_path.as_posix(),
            item.kind,
            item.qualified_name or item.name,
        ),
    )
