"""Engineering graph construction and persistence."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sddraft.analysis.graph_models import (
    GraphEdgeRecord,
    GraphEdgeType,
    GraphManifest,
    GraphNodeRecord,
    GraphNodeType,
    GraphSymbolRecord,
    chunk_node_id,
    commit_node_id,
    directory_node_id,
    edge_record_id,
    file_node_id,
    normalize_dir_path,
    section_node_id,
)
from sddraft.analysis.retrieval import open_query_engine, tokenize
from sddraft.analysis.symbol_inventory import build_symbol_inventory
from sddraft.domain.models import (
    CommitImpact,
    EvidenceReference,
    ReviewArtifact,
    ScanResult,
    SDDDocument,
    UpdateProposalReport,
)
from sddraft.render.json_artifacts import write_json_model


@dataclass(slots=True)
class GraphBuildResult:
    """Metadata for the persisted engineering graph store."""

    store_root: Path
    manifest_path: Path
    node_count: int
    edge_count: int


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def _to_relative(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path


def _add_node(nodes: dict[str, GraphNodeRecord], node: GraphNodeRecord) -> None:
    if node.node_id not in nodes:
        nodes[node.node_id] = node


def _add_edge(
    edges: dict[str, GraphEdgeRecord],
    *,
    edge_type: GraphEdgeType,
    source_id: str,
    target_id: str,
    reason: str | None = None,
) -> None:
    edge_id = edge_record_id(edge_type, source_id, target_id)
    if edge_id not in edges:
        edges[edge_id] = GraphEdgeRecord(
            edge_id=edge_id,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            reason=reason,
        )


def _symbols_by_file(
    symbol_records: list[GraphSymbolRecord],
) -> dict[Path, list[GraphSymbolRecord]]:
    by_file: dict[Path, list[GraphSymbolRecord]] = defaultdict(list)
    for symbol in symbol_records:
        by_file[symbol.file_path].append(symbol)
    for path in list(by_file):
        by_file[path] = sorted(
            by_file[path],
            key=lambda item: (item.kind, item.qualified_name or item.name),
        )
    return by_file


def _symbols_by_name(
    symbol_records: list[GraphSymbolRecord],
) -> dict[str, list[GraphSymbolRecord]]:
    by_name: dict[str, list[GraphSymbolRecord]] = defaultdict(list)
    for symbol in symbol_records:
        by_name[symbol.name.lower()].append(symbol)
    for key in list(by_name):
        by_name[key] = sorted(
            by_name[key], key=lambda item: (item.file_path.as_posix(), item.symbol_id)
        )
    return by_name


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
        if len(file_paths) == 1:
            module_to_file[module_name] = next(iter(file_paths))
    return module_to_file


def _extract_python_import_modules(raw_imports: list[str]) -> set[str]:
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


def _extract_changed_symbol_names(changed_lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in changed_lines:
        stripped = line.strip()
        for pattern in (
            r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)",
        ):
            match = re.search(pattern, stripped)
            if match:
                names.add(match.group(1))
        call_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
        if call_match:
            names.add(call_match.group(1))
    return names


def _symbols_referenced_by_chunk(
    *,
    chunk_text_tokens: set[str],
    chunk_path: Path,
    line_start: int | None,
    line_end: int | None,
    symbols_for_path: list[GraphSymbolRecord],
) -> list[str]:
    referenced: list[str] = []
    for symbol in symbols_for_path:
        if (
            symbol.line_start is not None
            and line_start is not None
            and line_end is not None
        ):
            if line_start <= symbol.line_start <= line_end:
                referenced.append(symbol.symbol_id)
                continue
        if symbol.name.lower() in chunk_text_tokens:
            referenced.append(symbol.symbol_id)
    return sorted(set(referenced))


def _targets_from_evidence_refs(
    refs: list[EvidenceReference],
    *,
    symbols_by_file: dict[Path, list[GraphSymbolRecord]],
    symbols_by_name: dict[str, list[GraphSymbolRecord]],
) -> tuple[set[str], set[str]]:
    file_targets: set[str] = set()
    symbol_targets: set[str] = set()

    for ref in refs:
        if ref.kind == "code_summary":
            file_targets.add(file_node_id(Path(ref.source)))
            continue

        if ref.kind == "interface":
            source_path = Path(ref.source)
            file_targets.add(file_node_id(source_path))
            if ref.detail:
                local = [
                    item
                    for item in symbols_by_file.get(source_path, [])
                    if item.name == ref.detail
                    or item.qualified_name == ref.detail
                    or (item.qualified_name or "").endswith(f".{ref.detail}")
                ]
                if local:
                    symbol_targets.update(item.symbol_id for item in local)
                else:
                    symbol_targets.update(
                        item.symbol_id
                        for item in symbols_by_name.get(ref.detail.lower(), [])
                    )
            continue

        if ref.kind == "hierarchy_summary":
            # hierarchy evidence format is "file::<path>::<summary>" or "directory::..."
            if ref.source.startswith("file::"):
                _, file_part, *_ = ref.source.split("::", maxsplit=2)
                if file_part:
                    file_targets.add(file_node_id(Path(file_part)))

    return file_targets, symbol_targets


def build_graph_store(
    *,
    csc_id: str,
    repo_root: Path,
    output_root: Path,
    retrieval_root: Path,
    scan_result: ScanResult,
    document: SDDDocument | None = None,
    review_artifact: ReviewArtifact | None = None,
    update_report: UpdateProposalReport | None = None,
    commit_impact: CommitImpact | None = None,
) -> GraphBuildResult:
    """Build engineering graph artifacts for retrieval augmentation."""

    graph_root = output_root / "graph"
    graph_root.mkdir(parents=True, exist_ok=True)

    nodes_path = graph_root / "nodes.jsonl"
    edges_path = graph_root / "edges.jsonl"
    symbol_index_path = graph_root / "symbol_index.json"
    adjacency_path = graph_root / "adjacency.json"
    manifest_path = graph_root / "manifest.json"

    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

    files = sorted({_to_relative(path, repo_root) for path in scan_result.files})
    file_summary_by_path = {item.path: item for item in scan_result.code_summaries}

    directories: set[Path] = {Path(".")}
    for file_path in files:
        cursor = file_path.parent if file_path.parent != Path("") else Path(".")
        while True:
            directories.add(cursor)
            if cursor == Path("."):
                break
            cursor = cursor.parent if cursor.parent != Path("") else Path(".")

    for directory in sorted(
        directories, key=lambda item: (len(item.parts), item.as_posix())
    ):
        _add_node(
            nodes,
            GraphNodeRecord(
                node_id=directory_node_id(directory),
                node_type="directory",
                label=directory.as_posix(),
                path=normalize_dir_path(directory),
            ),
        )
        if directory != Path("."):
            parent = normalize_dir_path(
                directory.parent if directory.parent != Path("") else Path(".")
            )
            _add_edge(
                edges,
                edge_type="contains",
                source_id=directory_node_id(parent),
                target_id=directory_node_id(directory),
            )
            _add_edge(
                edges,
                edge_type="parent_of",
                source_id=directory_node_id(parent),
                target_id=directory_node_id(directory),
            )

    for file_path in files:
        summary = file_summary_by_path.get(file_path)
        _add_node(
            nodes,
            GraphNodeRecord(
                node_id=file_node_id(file_path),
                node_type="file",
                label=file_path.name,
                path=file_path,
                language=summary.language if summary is not None else "unknown",
            ),
        )
        parent = normalize_dir_path(
            file_path.parent if file_path.parent != Path("") else Path(".")
        )
        _add_edge(
            edges,
            edge_type="contains",
            source_id=directory_node_id(parent),
            target_id=file_node_id(file_path),
        )

    symbol_records = build_symbol_inventory(
        scan_result=scan_result, repo_root=repo_root
    )
    symbols_by_file = _symbols_by_file(symbol_records)
    symbols_by_name = _symbols_by_name(symbol_records)

    for symbol in symbol_records:
        _add_node(
            nodes,
            GraphNodeRecord(
                node_id=symbol.symbol_id,
                node_type="symbol",
                label=symbol.qualified_name or symbol.name,
                path=symbol.file_path,
                language=symbol.language,
                symbol_kind=symbol.kind,
                symbol_name=symbol.name,
                qualified_name=symbol.qualified_name,
                line_start=symbol.line_start,
                line_end=symbol.line_end,
            ),
        )
        src_file_id = file_node_id(symbol.file_path)
        _add_edge(
            edges,
            edge_type="contains",
            source_id=src_file_id,
            target_id=symbol.symbol_id,
        )
        _add_edge(
            edges,
            edge_type="defines",
            source_id=src_file_id,
            target_id=symbol.symbol_id,
        )

    symbol_lookup = {record.symbol_id: record for record in symbol_records}
    for symbol in symbol_records:
        if not symbol.qualified_name or "." not in symbol.qualified_name:
            continue
        parent_name = symbol.qualified_name.rsplit(".", maxsplit=1)[0]
        parent_symbol = next(
            (
                item
                for item in symbols_by_file.get(symbol.file_path, [])
                if item.qualified_name == parent_name or item.name == parent_name
            ),
            None,
        )
        if parent_symbol is None:
            continue
        _add_edge(
            edges,
            edge_type="parent_of",
            source_id=parent_symbol.symbol_id,
            target_id=symbol.symbol_id,
        )
        _add_edge(
            edges,
            edge_type="contains",
            source_id=parent_symbol.symbol_id,
            target_id=symbol.symbol_id,
        )

    module_index = _build_python_module_index(files)
    for summary in scan_result.code_summaries:
        if summary.language != "python":
            continue
        source_file_id = file_node_id(summary.path)
        for module_name in sorted(_extract_python_import_modules(summary.imports)):
            target_path = module_index.get(module_name)
            if target_path is None or target_path == summary.path:
                continue
            _add_edge(
                edges,
                edge_type="imports",
                source_id=source_file_id,
                target_id=file_node_id(target_path),
                reason=f"python:{module_name}",
            )

    engine = open_query_engine(retrieval_root)
    section_targets_by_section_node: dict[str, tuple[set[str], set[str]]] = {}

    for chunk in sorted(engine.iter_chunks(), key=lambda item: item.chunk_id):
        chunk_id_value = chunk_node_id(chunk.chunk_id)
        _add_node(
            nodes,
            GraphNodeRecord(
                node_id=chunk_id_value,
                node_type="chunk",
                label=chunk.chunk_id,
                path=chunk.source_path,
                section_id=chunk.section_id,
                chunk_id=chunk.chunk_id,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                metadata={"source_type": chunk.source_type},
            ),
        )

        if chunk.source_type == "code":
            source_file_id = file_node_id(chunk.source_path)
            if source_file_id in nodes:
                _add_edge(
                    edges,
                    edge_type="contains",
                    source_id=source_file_id,
                    target_id=chunk_id_value,
                )

        symbols_for_path = symbols_by_file.get(chunk.source_path, [])
        token_set = set(tokenize(chunk.text))
        for symbol_id in _symbols_referenced_by_chunk(
            chunk_text_tokens=token_set,
            chunk_path=chunk.source_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            symbols_for_path=symbols_for_path,
        ):
            _add_edge(
                edges,
                edge_type="references",
                source_id=chunk_id_value,
                target_id=symbol_id,
            )

    review_by_section_id = {
        item.section_id: item.evidence_references
        for item in (review_artifact.sections if review_artifact else [])
    }

    if document is not None:
        for section in sorted(document.sections, key=lambda item: item.section_id):
            sid = section_node_id(section.section_id)
            _add_node(
                nodes,
                GraphNodeRecord(
                    node_id=sid,
                    node_type="sdd_section",
                    label=f"{section.section_id} {section.title}",
                    section_id=section.section_id,
                ),
            )
            refs = review_by_section_id.get(section.section_id, section.evidence_refs)
            targets = _targets_from_evidence_refs(
                refs,
                symbols_by_file=symbols_by_file,
                symbols_by_name=symbols_by_name,
            )
            section_targets_by_section_node[sid] = targets
            file_targets, symbol_targets = targets
            for file_target in sorted(file_targets):
                if file_target in nodes:
                    _add_edge(
                        edges,
                        edge_type="documents",
                        source_id=sid,
                        target_id=file_target,
                    )
            for symbol_target in sorted(symbol_targets):
                if symbol_target in symbol_lookup:
                    _add_edge(
                        edges,
                        edge_type="documents",
                        source_id=sid,
                        target_id=symbol_target,
                    )

    if update_report is not None:
        for proposal in sorted(
            update_report.proposals, key=lambda item: item.section_id
        ):
            sid = section_node_id(proposal.section_id)
            _add_node(
                nodes,
                GraphNodeRecord(
                    node_id=sid,
                    node_type="sdd_section",
                    label=f"{proposal.section_id} {proposal.title}",
                    section_id=proposal.section_id,
                ),
            )
            targets = _targets_from_evidence_refs(
                proposal.evidence_refs,
                symbols_by_file=symbols_by_file,
                symbols_by_name=symbols_by_name,
            )
            section_targets_by_section_node[sid] = targets
            file_targets, symbol_targets = targets
            for file_target in sorted(file_targets):
                if file_target in nodes:
                    _add_edge(
                        edges,
                        edge_type="documents",
                        source_id=sid,
                        target_id=file_target,
                    )
            for symbol_target in sorted(symbol_targets):
                if symbol_target in symbol_lookup:
                    _add_edge(
                        edges,
                        edge_type="documents",
                        source_id=sid,
                        target_id=symbol_target,
                    )

    changed_file_ids: set[str] = set()
    changed_symbol_ids: set[str] = set()
    if commit_impact is not None:
        commit_id = commit_node_id(commit_impact.commit_range)
        _add_node(
            nodes,
            GraphNodeRecord(
                node_id=commit_id,
                node_type="commit",
                label=commit_impact.commit_range,
            ),
        )
        for changed_file in commit_impact.changed_files:
            fid = file_node_id(changed_file.path)
            if fid in nodes:
                changed_file_ids.add(fid)
                _add_edge(
                    edges,
                    edge_type="changed_in",
                    source_id=fid,
                    target_id=commit_id,
                )

            names = _extract_changed_symbol_names(changed_file.signature_changes)
            for symbol in symbols_by_file.get(changed_file.path, []):
                if symbol.name in names or (symbol.qualified_name or "") in names:
                    changed_symbol_ids.add(symbol.symbol_id)
                    _add_edge(
                        edges,
                        edge_type="changed_in",
                        source_id=symbol.symbol_id,
                        target_id=commit_id,
                    )

        for section_id, (
            file_targets,
            symbol_targets,
        ) in section_targets_by_section_node.items():
            for file_target in sorted(file_targets & changed_file_ids):
                _add_edge(
                    edges,
                    edge_type="impacts_section",
                    source_id=file_target,
                    target_id=section_id,
                )
            for symbol_target in sorted(symbol_targets & changed_symbol_ids):
                _add_edge(
                    edges,
                    edge_type="impacts_section",
                    source_id=symbol_target,
                    target_id=section_id,
                )

    node_rows = [
        item.model_dump(mode="json")
        for item in sorted(nodes.values(), key=lambda node: node.node_id)
    ]
    edge_rows = [
        item.model_dump(mode="json")
        for item in sorted(edges.values(), key=lambda edge: edge.edge_id)
    ]
    _write_jsonl(nodes_path, node_rows)
    _write_jsonl(edges_path, edge_rows)

    node_counts_counter: Counter[str] = Counter(
        node.node_type for node in nodes.values()
    )
    edge_counts_counter: Counter[str] = Counter(
        edge.edge_type for edge in edges.values()
    )

    symbol_index_payload = {
        "symbols": [
            item.model_dump(mode="json")
            for item in sorted(symbol_records, key=lambda entry: entry.symbol_id)
        ],
        "by_name": {
            key: sorted(item.symbol_id for item in values)
            for key, values in sorted(symbols_by_name.items())
        },
        "by_file": {
            key.as_posix(): sorted(item.symbol_id for item in values)
            for key, values in sorted(
                symbols_by_file.items(), key=lambda item: item[0].as_posix()
            )
        },
    }
    _write_json(symbol_index_path, symbol_index_payload)

    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in edges.values():
        outgoing[edge.source_id].append(edge.edge_id)
        incoming[edge.target_id].append(edge.edge_id)
    adjacency_payload = {
        "outgoing": {
            key: sorted(value)
            for key, value in sorted(outgoing.items(), key=lambda item: item[0])
        },
        "incoming": {
            key: sorted(value)
            for key, value in sorted(incoming.items(), key=lambda item: item[0])
        },
    }
    _write_json(adjacency_path, adjacency_payload)

    retrieval_manifest_path = retrieval_root / "manifest.json"
    try:
        source_manifest_rel = retrieval_manifest_path.relative_to(output_root)
    except ValueError:
        source_manifest_rel = retrieval_manifest_path

    node_counts: dict[GraphNodeType, int] = {
        "directory": int(node_counts_counter.get("directory", 0)),
        "file": int(node_counts_counter.get("file", 0)),
        "symbol": int(node_counts_counter.get("symbol", 0)),
        "chunk": int(node_counts_counter.get("chunk", 0)),
        "sdd_section": int(node_counts_counter.get("sdd_section", 0)),
        "commit": int(node_counts_counter.get("commit", 0)),
    }
    edge_counts: dict[GraphEdgeType, int] = {
        "contains": int(edge_counts_counter.get("contains", 0)),
        "defines": int(edge_counts_counter.get("defines", 0)),
        "references": int(edge_counts_counter.get("references", 0)),
        "documents": int(edge_counts_counter.get("documents", 0)),
        "parent_of": int(edge_counts_counter.get("parent_of", 0)),
        "imports": int(edge_counts_counter.get("imports", 0)),
        "changed_in": int(edge_counts_counter.get("changed_in", 0)),
        "impacts_section": int(edge_counts_counter.get("impacts_section", 0)),
    }

    manifest = GraphManifest(
        csc_id=csc_id,
        source_retrieval_manifest=source_manifest_rel,
        nodes_path=Path(nodes_path.name),
        edges_path=Path(edges_path.name),
        symbol_index_path=Path(symbol_index_path.name),
        adjacency_path=Path(adjacency_path.name),
        node_counts=node_counts,
        edge_counts=edge_counts,
    )
    write_json_model(manifest_path, manifest)

    return GraphBuildResult(
        store_root=graph_root,
        manifest_path=manifest_path,
        node_count=len(nodes),
        edge_count=len(edges),
    )
