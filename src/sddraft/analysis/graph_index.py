"""Load and query persisted engineering graph artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from sddraft.analysis.graph_models import (
    GraphEdgeRecord,
    GraphManifest,
    GraphNodeRecord,
    GraphStore,
    GraphSymbolRecord,
)
from sddraft.domain.errors import AnalysisError


def default_graph_manifest_path(index_path: Path) -> Path:
    """Return the default graph manifest path from retrieval-like paths."""

    if index_path.is_dir() and index_path.name == "retrieval":
        return index_path.parent / "graph" / "manifest.json"
    if index_path.is_dir() and index_path.name == "graph":
        return index_path / "manifest.json"
    if index_path.is_file() and index_path.name == "manifest.json":
        if index_path.parent.name == "retrieval":
            return index_path.parent.parent / "graph" / "manifest.json"
        if index_path.parent.name == "graph":
            return index_path
    if (index_path / "retrieval").exists():
        return index_path / "graph" / "manifest.json"
    return index_path.parent / "graph" / "manifest.json"


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_dir():
        if path.name == "retrieval":
            return default_graph_manifest_path(path)
        candidate = path / "manifest.json"
        if candidate.exists():
            return candidate
        return default_graph_manifest_path(path)
    return default_graph_manifest_path(path)


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


def load_graph_store(path: Path) -> GraphStore:
    """Load graph store from manifest path or neighboring retrieval path."""

    manifest_path = _resolve_manifest_path(path)
    if not manifest_path.exists():
        raise AnalysisError(f"Graph manifest not found: {manifest_path}")

    try:
        manifest = GraphManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisError(
            f"Invalid graph manifest at {manifest_path}: {exc}"
        ) from exc

    store_root = manifest_path.parent
    nodes_by_id: dict[str, GraphNodeRecord] = {}
    for raw in _iter_jsonl(store_root / manifest.nodes_path):
        node = GraphNodeRecord.model_validate(raw)
        nodes_by_id[node.node_id] = node

    edges_by_id: dict[str, GraphEdgeRecord] = {}
    for raw in _iter_jsonl(store_root / manifest.edges_path):
        edge = GraphEdgeRecord.model_validate(raw)
        edges_by_id[edge.edge_id] = edge

    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)

    adjacency_path = store_root / manifest.adjacency_path
    if adjacency_path.exists():
        try:
            raw_adj = json.loads(adjacency_path.read_text(encoding="utf-8"))
            out_raw = raw_adj.get("outgoing", {}) if isinstance(raw_adj, dict) else {}
            in_raw = raw_adj.get("incoming", {}) if isinstance(raw_adj, dict) else {}
            if isinstance(out_raw, dict):
                for node_id, edge_ids in out_raw.items():
                    if isinstance(node_id, str) and isinstance(edge_ids, list):
                        outgoing[node_id] = [str(item) for item in edge_ids]
            if isinstance(in_raw, dict):
                for node_id, edge_ids in in_raw.items():
                    if isinstance(node_id, str) and isinstance(edge_ids, list):
                        incoming[node_id] = [str(item) for item in edge_ids]
        except (OSError, json.JSONDecodeError):
            outgoing.clear()
            incoming.clear()

    if not outgoing and not incoming:
        for edge in edges_by_id.values():
            outgoing[edge.source_id].append(edge.edge_id)
            incoming[edge.target_id].append(edge.edge_id)
        for key in list(outgoing):
            outgoing[key] = sorted(outgoing[key])
        for key in list(incoming):
            incoming[key] = sorted(incoming[key])

    symbols_by_name: dict[str, list[str]] = defaultdict(list)
    symbol_records: dict[str, GraphSymbolRecord] = {}

    symbol_index_path = store_root / manifest.symbol_index_path
    if symbol_index_path.exists():
        try:
            raw = json.loads(symbol_index_path.read_text(encoding="utf-8"))
            symbols_raw = raw.get("symbols", []) if isinstance(raw, dict) else []
            if isinstance(symbols_raw, list):
                for item in symbols_raw:
                    if not isinstance(item, dict):
                        continue
                    record = GraphSymbolRecord.model_validate(item)
                    symbol_records[record.symbol_id] = record
            by_name_raw = raw.get("by_name", {}) if isinstance(raw, dict) else {}
            if isinstance(by_name_raw, dict):
                for name, symbol_ids in by_name_raw.items():
                    if isinstance(name, str) and isinstance(symbol_ids, list):
                        symbols_by_name[name.lower()] = [
                            str(item) for item in symbol_ids
                        ]
        except (OSError, json.JSONDecodeError, ValueError):
            symbols_by_name.clear()
            symbol_records.clear()

    if not symbol_records:
        for node in nodes_by_id.values():
            if node.node_type != "symbol" or node.path is None or node.language is None:
                continue
            record = GraphSymbolRecord(
                symbol_id=node.node_id,
                name=node.symbol_name or node.label,
                qualified_name=node.qualified_name,
                kind=node.symbol_kind or "symbol",
                language=node.language,
                file_path=node.path,
                line_start=node.line_start,
                line_end=node.line_end,
            )
            symbol_records[record.symbol_id] = record
            symbols_by_name[record.name.lower()].append(record.symbol_id)

    for key in list(symbols_by_name):
        symbols_by_name[key] = sorted(set(symbols_by_name[key]))

    return GraphStore(
        manifest_path=manifest_path,
        manifest=manifest,
        nodes_by_id=nodes_by_id,
        edges_by_id=edges_by_id,
        outgoing={key: sorted(value) for key, value in outgoing.items()},
        incoming={key: sorted(value) for key, value in incoming.items()},
        symbols_by_name={key: value for key, value in sorted(symbols_by_name.items())},
        symbol_records=symbol_records,
    )
