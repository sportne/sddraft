"""Helpers for converting scanner symbols into graph symbol records."""

from __future__ import annotations

from sddraft.analysis.graph_models import GraphSymbolRecord, symbol_node_id
from sddraft.domain.models import ScanResult


def build_symbol_inventory(*, scan_result: ScanResult) -> list[GraphSymbolRecord]:
    """Build graph symbol records directly from analyzer-emitted symbol summaries."""

    symbols: dict[str, GraphSymbolRecord] = {}

    for symbol in sorted(
        scan_result.symbol_summaries,
        key=lambda item: (
            item.source_path.as_posix(),
            item.kind,
            item.qualified_name or item.name,
            item.line_start or 0,
            item.line_end or 0,
        ),
    ):
        qualified = symbol.qualified_name or symbol.name
        symbol_id = symbol_node_id(symbol.source_path, symbol.kind, qualified)
        if symbol_id in symbols:
            continue
        symbols[symbol_id] = GraphSymbolRecord(
            symbol_id=symbol_id,
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            owner_qualified_name=symbol.owner_qualified_name,
            kind=symbol.kind,
            language=symbol.language,
            file_path=symbol.source_path,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
        )

    return sorted(
        symbols.values(),
        key=lambda item: (
            item.file_path.as_posix(),
            item.kind,
            item.qualified_name or item.name,
        ),
    )
