"""Engineering graph construction and persistence.

The graph builder turns deterministic scan/retrieval artifacts into a single
inspectable graph store (`nodes.jsonl` + `edges.jsonl`). `ask` later uses this
store to expand evidence beyond pure lexical matches.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from engllm.core.analysis.dependency_resolution import (
    DependencyRecord,
    dependency_reason_payload,
    resolve_dependency_records,
)
from engllm.core.analysis.graph_models import (
    GraphBuildDecision,
    GraphBuildPlan,
    GraphEdgeRecord,
    GraphEdgeType,
    GraphInputFingerprint,
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
from engllm.core.analysis.retrieval import (
    load_retrieval_manifest,
    open_query_engine,
    tokenize,
)
from engllm.core.analysis.symbol_inventory import build_symbol_inventory
from engllm.core.render.json_artifacts import write_json_model
from engllm.domain.models import (
    CodeUnitSummary,
    CommitImpact,
    EvidenceReference,
    KnowledgeChunk,
    ReviewArtifact,
    ScanResult,
    SDDDocument,
    UpdateProposalReport,
)

_GRAPH_BUILD_VERSION = "v3-incremental-fragment"
_FRAGMENT_ROW_META = "meta"
_FRAGMENT_ROW_NODE = "node"
_FRAGMENT_ROW_EDGE = "edge"


@dataclass(slots=True)
class GraphBuildResult:
    """Metadata for the persisted engineering graph store."""

    store_root: Path
    manifest_path: Path
    node_count: int
    edge_count: int
    planner_decision: GraphBuildDecision
    rebuilt_fragments: int = 0
    reused_fragments: int = 0


@dataclass(slots=True)
class _SectionInput:
    section_id: str
    title: str
    refs: list[EvidenceReference]


@dataclass(slots=True)
class _GraphFragment:
    fragment_id: str
    fingerprint: str
    nodes: list[GraphNodeRecord]
    edges: list[GraphEdgeRecord]


@dataclass(slots=True)
class _PreparedInputs:
    files: list[Path]
    file_summary_by_path: dict[Path, CodeUnitSummary]
    symbol_records: list[GraphSymbolRecord]
    symbols_by_file: dict[Path, list[GraphSymbolRecord]]
    symbols_by_name: dict[str, list[GraphSymbolRecord]]
    symbol_lookup: dict[str, GraphSymbolRecord]
    dependency_records_by_source: dict[Path, list[DependencyRecord]]
    code_chunks_by_source: dict[Path, list[KnowledgeChunk]]
    misc_chunks: list[KnowledgeChunk]
    sections: list[_SectionInput]
    section_targets: dict[str, tuple[set[str], set[str]]]
    all_file_node_ids: set[str]


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


def _stable_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


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
            and line_start <= symbol.line_start <= line_end
        ):
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

        if ref.kind == "symbol":
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

        if ref.kind == "hierarchy_summary" and ref.source.startswith("file::"):
            _, file_part, *_ = ref.source.split("::", maxsplit=2)
            if file_part:
                file_targets.add(file_node_id(Path(file_part)))

    return file_targets, symbol_targets


def _scan_payload(scan_result: ScanResult) -> dict[str, object]:
    code_summaries = sorted(
        (
            {
                "path": item.path.as_posix(),
                "language": item.language,
                "functions": list(item.functions),
                "classes": list(item.classes),
                "docstrings": list(item.docstrings),
                "imports": list(item.imports),
            }
            for item in scan_result.code_summaries
        ),
        key=lambda item: str(item["path"]),
    )
    symbol_summaries = sorted(
        (
            {
                "source_path": item.source_path.as_posix(),
                "language": item.language,
                "kind": item.kind,
                "name": item.name,
                "qualified_name": item.qualified_name,
                "owner_qualified_name": item.owner_qualified_name,
                "line_start": item.line_start,
                "line_end": item.line_end,
            }
            for item in scan_result.symbol_summaries
        ),
        key=lambda item: (
            str(item["source_path"]),
            str(item["kind"]),
            str(item["qualified_name"] or item["name"]),
            int(item["line_start"] or 0),
        ),
    )
    return {
        "files": sorted(path.as_posix() for path in scan_result.files),
        "code_summaries": code_summaries,
        "symbol_summaries": symbol_summaries,
    }


def _retrieval_payload(retrieval_root: Path) -> dict[str, object]:
    manifest, _ = load_retrieval_manifest(retrieval_root)
    return {
        "version": manifest.version,
        "shard_size": manifest.shard_size,
        "total_chunks": manifest.total_chunks,
        "average_doc_length": manifest.average_doc_length,
        "chunk_shards": [
            {
                "shard_id": item.shard_id,
                "path": item.path.as_posix(),
                "count": item.count,
            }
            for item in sorted(
                manifest.chunk_shards,
                key=lambda shard: (shard.shard_id, shard.path.as_posix()),
            )
        ],
        "posting_shards": [
            {
                "shard_id": item.shard_id,
                "bucket": item.bucket,
                "path": item.path.as_posix(),
                "count": item.count,
            }
            for item in sorted(
                manifest.posting_shards,
                key=lambda shard: (shard.shard_id, shard.bucket, shard.path.as_posix()),
            )
        ],
        "docstats_path": manifest.docstats_path.as_posix(),
    }


def _hierarchy_payload(output_root: Path) -> dict[str, object] | None:
    manifest_path = output_root / "hierarchy" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "path": manifest_path.relative_to(output_root).as_posix(),
            "status": "unreadable",
        }
    if not isinstance(raw, dict):
        return {
            "path": manifest_path.relative_to(output_root).as_posix(),
            "status": "invalid",
        }
    return {
        "path": manifest_path.relative_to(output_root).as_posix(),
        "version": raw.get("version"),
        "file_summaries_path": raw.get("file_summaries_path"),
        "directory_summaries_path": raw.get("directory_summaries_path"),
        "nodes_path": raw.get("nodes_path"),
        "edges_path": raw.get("edges_path"),
    }


def _sections_payload(
    document: SDDDocument | None,
    review_artifact: ReviewArtifact | None,
    update_report: UpdateProposalReport | None,
) -> dict[str, object] | None:
    review_by_section_id = {
        item.section_id: item.evidence_references
        for item in (review_artifact.sections if review_artifact else [])
    }

    if document is not None:
        rows = []
        for section in sorted(document.sections, key=lambda item: item.section_id):
            refs = review_by_section_id.get(section.section_id, section.evidence_refs)
            rows.append(
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "refs": [
                        ref.model_dump(mode="json")
                        for ref in sorted(
                            refs,
                            key=lambda value: (
                                value.kind,
                                value.source,
                                value.detail or "",
                            ),
                        )
                    ],
                }
            )
        return {"document_sections": rows}

    if update_report is not None:
        rows = []
        for proposal in sorted(
            update_report.proposals, key=lambda item: item.section_id
        ):
            rows.append(
                {
                    "section_id": proposal.section_id,
                    "title": proposal.title,
                    "refs": [
                        ref.model_dump(mode="json")
                        for ref in sorted(
                            proposal.evidence_refs,
                            key=lambda value: (
                                value.kind,
                                value.source,
                                value.detail or "",
                            ),
                        )
                    ],
                }
            )
        return {"update_sections": rows}

    return None


def _commit_payload(commit_impact: CommitImpact | None) -> dict[str, object] | None:
    if commit_impact is None:
        return None
    return commit_impact.model_dump(mode="json")


def _compute_input_fingerprint(
    *,
    scan_result: ScanResult,
    retrieval_root: Path,
    output_root: Path,
    document: SDDDocument | None,
    review_artifact: ReviewArtifact | None,
    update_report: UpdateProposalReport | None,
    commit_impact: CommitImpact | None,
) -> GraphInputFingerprint:
    scan_payload = _scan_payload(scan_result)
    retrieval_payload = _retrieval_payload(retrieval_root)
    hierarchy_payload = _hierarchy_payload(output_root)
    section_payload = _sections_payload(document, review_artifact, update_report)
    commit_payload = _commit_payload(commit_impact)

    scan_digest = _hash_payload(scan_payload)
    retrieval_digest = _hash_payload(retrieval_payload)
    hierarchy_digest = _hash_payload(hierarchy_payload) if hierarchy_payload else None
    section_digest = _hash_payload(section_payload) if section_payload else None
    commit_digest = _hash_payload(commit_payload) if commit_payload else None

    digest = _hash_payload(
        {
            "build_version": _GRAPH_BUILD_VERSION,
            "scan_digest": scan_digest,
            "retrieval_digest": retrieval_digest,
            "hierarchy_digest": hierarchy_digest,
            "section_digest": section_digest,
            "commit_digest": commit_digest,
        }
    )

    return GraphInputFingerprint(
        digest=digest,
        scan_digest=scan_digest,
        retrieval_digest=retrieval_digest,
        hierarchy_digest=hierarchy_digest,
        section_digest=section_digest,
        commit_digest=commit_digest,
    )


def _chunk_signature(chunk: KnowledgeChunk) -> dict[str, object]:
    text = chunk.text
    return {
        "chunk_id": chunk.chunk_id,
        "source_type": chunk.source_type,
        "source_path": chunk.source_path.as_posix(),
        "section_id": chunk.section_id,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _collect_sections(
    *,
    document: SDDDocument | None,
    review_artifact: ReviewArtifact | None,
    update_report: UpdateProposalReport | None,
) -> list[_SectionInput]:
    review_by_section_id = {
        item.section_id: item.evidence_references
        for item in (review_artifact.sections if review_artifact else [])
    }
    sections: list[_SectionInput] = []

    if document is not None:
        for section in sorted(document.sections, key=lambda item: item.section_id):
            sections.append(
                _SectionInput(
                    section_id=section.section_id,
                    title=section.title,
                    refs=list(
                        review_by_section_id.get(
                            section.section_id, section.evidence_refs
                        )
                    ),
                )
            )

    if update_report is not None:
        for proposal in sorted(
            update_report.proposals, key=lambda item: item.section_id
        ):
            sections.append(
                _SectionInput(
                    section_id=proposal.section_id,
                    title=proposal.title,
                    refs=list(proposal.evidence_refs),
                )
            )

    unique: dict[str, _SectionInput] = {}
    for section_item in sections:
        unique[section_item.section_id] = section_item
    return [unique[key] for key in sorted(unique)]


def _prepare_inputs(
    *,
    repo_root: Path,
    retrieval_root: Path,
    scan_result: ScanResult,
    document: SDDDocument | None,
    review_artifact: ReviewArtifact | None,
    update_report: UpdateProposalReport | None,
) -> _PreparedInputs:
    files = sorted({_to_relative(path, repo_root) for path in scan_result.files})
    file_summary_by_path = {item.path: item for item in scan_result.code_summaries}

    symbol_records = build_symbol_inventory(scan_result=scan_result)
    symbols_by_file = _symbols_by_file(symbol_records)
    symbols_by_name = _symbols_by_name(symbol_records)
    symbol_lookup = {record.symbol_id: record for record in symbol_records}

    dependency_records = resolve_dependency_records(
        code_summaries=scan_result.code_summaries,
        symbol_summaries=scan_result.symbol_summaries,
        files=files,
        repo_root=repo_root,
    )
    dependency_records_by_source: dict[Path, list[DependencyRecord]] = defaultdict(list)
    for record in dependency_records:
        dependency_records_by_source[record.source_path].append(record)
    for path in list(dependency_records_by_source):
        dependency_records_by_source[path] = sorted(
            dependency_records_by_source[path],
            key=lambda item: (
                item.language,
                item.dependency_kind,
                item.normalized_key or item.raw_dependency,
                item.target_path.as_posix() if item.target_path is not None else "",
            ),
        )

    engine = open_query_engine(retrieval_root)
    code_chunks_by_source: dict[Path, list[KnowledgeChunk]] = defaultdict(list)
    misc_chunks: list[KnowledgeChunk] = []
    for chunk in sorted(engine.iter_chunks(), key=lambda item: item.chunk_id):
        if chunk.source_type == "code":
            code_chunks_by_source[chunk.source_path].append(chunk)
        else:
            misc_chunks.append(chunk)

    for path in list(code_chunks_by_source):
        code_chunks_by_source[path] = sorted(
            code_chunks_by_source[path], key=lambda item: item.chunk_id
        )

    sections = _collect_sections(
        document=document,
        review_artifact=review_artifact,
        update_report=update_report,
    )

    section_targets: dict[str, tuple[set[str], set[str]]] = {}
    for section in sections:
        section_targets[section_node_id(section.section_id)] = (
            _targets_from_evidence_refs(
                section.refs,
                symbols_by_file=symbols_by_file,
                symbols_by_name=symbols_by_name,
            )
        )

    return _PreparedInputs(
        files=files,
        file_summary_by_path=file_summary_by_path,
        symbol_records=symbol_records,
        symbols_by_file=symbols_by_file,
        symbols_by_name=symbols_by_name,
        symbol_lookup=symbol_lookup,
        dependency_records_by_source=dependency_records_by_source,
        code_chunks_by_source=code_chunks_by_source,
        misc_chunks=sorted(misc_chunks, key=lambda item: item.chunk_id),
        sections=sections,
        section_targets=section_targets,
        all_file_node_ids={file_node_id(path) for path in files},
    )


def _structure_fragment_id() -> str:
    return "structure"


def _file_fragment_id(path: Path) -> str:
    return f"file::{path.as_posix()}"


def _section_fragment_id(section_id: str) -> str:
    return f"section::{section_id}"


def _misc_fragment_id() -> str:
    return "misc_chunks"


def _commit_fragment_id(commit_range: str) -> str:
    return f"commit::{commit_range}"


def _structure_fingerprint(prepared: _PreparedInputs) -> str:
    payload: list[dict[str, str]] = []
    for path in prepared.files:
        summary = prepared.file_summary_by_path.get(path)
        payload.append(
            {
                "path": path.as_posix(),
                "language": summary.language if summary is not None else "unknown",
            }
        )
    return _hash_payload(payload)


def _file_fragment_fingerprint(prepared: _PreparedInputs, path: Path) -> str:
    summary = prepared.file_summary_by_path.get(path)
    dependencies = prepared.dependency_records_by_source.get(path, [])
    symbols = prepared.symbols_by_file.get(path, [])
    chunks = prepared.code_chunks_by_source.get(path, [])

    payload = {
        "path": path.as_posix(),
        "summary": summary.model_dump(mode="json") if summary is not None else None,
        "symbols": [item.model_dump(mode="json") for item in symbols],
        "dependencies": [
            {
                "language": item.language,
                "kind": item.dependency_kind,
                "raw": item.raw_dependency,
                "normalized": item.normalized_key,
                "resolution": item.resolution_status,
                "target": item.target_path.as_posix() if item.target_path else None,
            }
            for item in dependencies
        ],
        "chunks": [_chunk_signature(item) for item in chunks],
    }
    return _hash_payload(payload)


def _section_fragment_fingerprint(section: _SectionInput) -> str:
    payload = {
        "section_id": section.section_id,
        "title": section.title,
        "refs": [
            ref.model_dump(mode="json")
            for ref in sorted(
                section.refs,
                key=lambda item: (item.kind, item.source, item.detail or ""),
            )
        ],
    }
    return _hash_payload(payload)


def _misc_fragment_fingerprint(prepared: _PreparedInputs) -> str:
    return _hash_payload([_chunk_signature(chunk) for chunk in prepared.misc_chunks])


def _commit_fragment_fingerprint(
    prepared: _PreparedInputs, commit_impact: CommitImpact
) -> str:
    payload = {
        "commit_impact": commit_impact.model_dump(mode="json"),
        "section_targets": {
            key: {
                "files": sorted(value[0]),
                "symbols": sorted(value[1]),
            }
            for key, value in sorted(
                prepared.section_targets.items(), key=lambda item: item[0]
            )
        },
    }
    return _hash_payload(payload)


def _compute_expected_fragment_fingerprints(
    *,
    prepared: _PreparedInputs,
    commit_impact: CommitImpact | None,
) -> dict[str, str]:
    expected: dict[str, str] = {
        _structure_fragment_id(): _structure_fingerprint(prepared)
    }

    for path in prepared.files:
        expected[_file_fragment_id(path)] = _file_fragment_fingerprint(prepared, path)

    expected[_misc_fragment_id()] = _misc_fragment_fingerprint(prepared)

    for section in prepared.sections:
        expected[_section_fragment_id(section.section_id)] = (
            _section_fragment_fingerprint(section)
        )

    if commit_impact is not None:
        expected[_commit_fragment_id(commit_impact.commit_range)] = (
            _commit_fragment_fingerprint(
                prepared,
                commit_impact,
            )
        )

    return dict(sorted(expected.items(), key=lambda item: item[0]))


def _fragment_file_name(fragment_id: str) -> str:
    prefix = fragment_id.split("::", maxsplit=1)[0]
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", prefix)
    digest = hashlib.sha1(fragment_id.encode("utf-8")).hexdigest()[:16]
    return f"{safe_prefix}-{digest}.jsonl"


def _fragment_path(fragments_root: Path, fragment_id: str) -> Path:
    return fragments_root / _fragment_file_name(fragment_id)


def _read_fragment_meta(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
    except OSError:
        return None
    if not line:
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or raw.get("row_type") != _FRAGMENT_ROW_META:
        return None
    fragment_id = raw.get("fragment_id")
    fingerprint = raw.get("fingerprint")
    if not isinstance(fragment_id, str) or not isinstance(fingerprint, str):
        return None
    return fragment_id, fingerprint


def _fragment_status(
    *,
    fragments_root: Path,
    fragment_id: str,
    expected_fingerprint: str,
) -> str:
    path = _fragment_path(fragments_root, fragment_id)
    if not path.exists():
        return "missing"
    meta = _read_fragment_meta(path)
    if meta is None:
        return "corrupt"
    actual_id, actual_fingerprint = meta
    if actual_id != fragment_id:
        return "corrupt"
    if actual_fingerprint != expected_fingerprint:
        return "mismatch"
    return "match"


def _canonical_files_present(manifest: GraphManifest, graph_root: Path) -> bool:
    required = [
        graph_root / manifest.nodes_path,
        graph_root / manifest.edges_path,
        graph_root / manifest.symbol_index_path,
        graph_root / manifest.adjacency_path,
    ]
    return all(path.exists() for path in required)


def _load_prior_manifest(path: Path) -> GraphManifest | None:
    if not path.exists():
        return None
    try:
        return GraphManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _normalize_scoped_paths(paths: set[Path] | None) -> set[Path]:
    if not paths:
        return set()
    normalized: set[Path] = set()
    for path in paths:
        normalized.add(Path(path.as_posix().lstrip("./")))
    return normalized


def _plan_build(
    *,
    prior_manifest: GraphManifest | None,
    graph_root: Path,
    fragments_root: Path,
    input_fingerprint: GraphInputFingerprint,
    expected_fingerprints: dict[str, str],
    changed_files: set[Path],
    impacted_sections: set[str],
    commit_impact: CommitImpact | None,
) -> GraphBuildPlan:
    if prior_manifest is None:
        return GraphBuildPlan(
            decision="full",
            reason="No prior manifest found.",
            impacted_fragments=list(expected_fingerprints),
        )

    if prior_manifest.build_version != _GRAPH_BUILD_VERSION:
        return GraphBuildPlan(
            decision="full",
            reason="Prior manifest build version is incompatible.",
            impacted_fragments=list(expected_fingerprints),
        )

    statuses = {
        fragment_id: _fragment_status(
            fragments_root=fragments_root,
            fragment_id=fragment_id,
            expected_fingerprint=expected_fingerprint,
        )
        for fragment_id, expected_fingerprint in expected_fingerprints.items()
    }

    if prior_manifest.input_fingerprint is not None and (
        prior_manifest.input_fingerprint.digest == input_fingerprint.digest
    ):
        if _canonical_files_present(prior_manifest, graph_root) and all(
            status == "match" for status in statuses.values()
        ):
            return GraphBuildPlan(
                decision="no_op", reason="Input fingerprint unchanged."
            )
        return GraphBuildPlan(
            decision="full",
            reason="Fingerprint unchanged but prior artifacts are incomplete.",
            impacted_fragments=list(expected_fingerprints),
        )

    if any(status in {"missing", "corrupt"} for status in statuses.values()):
        return GraphBuildPlan(
            decision="full",
            reason="Required fragment is missing or corrupt.",
            impacted_fragments=list(expected_fingerprints),
        )

    scope_available = bool(
        changed_files or impacted_sections or commit_impact is not None
    )
    if not scope_available:
        return GraphBuildPlan(
            decision="full",
            reason="No deterministic change scope available for partial build.",
            impacted_fragments=list(expected_fingerprints),
        )

    impacted: set[str] = {
        fragment_id for fragment_id, status in statuses.items() if status == "mismatch"
    }

    for changed_path in changed_files:
        fragment_id = _file_fragment_id(changed_path)
        if fragment_id in expected_fingerprints:
            impacted.add(fragment_id)

    for section_id in impacted_sections:
        fragment_id = _section_fragment_id(section_id)
        if fragment_id in expected_fingerprints:
            impacted.add(fragment_id)

    if commit_impact is not None:
        commit_fragment = _commit_fragment_id(commit_impact.commit_range)
        if commit_fragment in expected_fingerprints:
            impacted.add(commit_fragment)

    if impacted:
        impacted.add(_structure_fragment_id())

    reusable = sorted(set(expected_fingerprints) - impacted)
    impacted_sorted = sorted(impacted)

    if not impacted_sorted or not reusable:
        return GraphBuildPlan(
            decision="full",
            reason="Partial scope would rebuild all fragments.",
            impacted_fragments=list(expected_fingerprints),
        )

    return GraphBuildPlan(
        decision="partial",
        reason="Using prior fragments for unchanged graph regions.",
        impacted_fragments=impacted_sorted,
        reusable_fragments=reusable,
    )


def _build_structure_fragment(
    *,
    prepared: _PreparedInputs,
    fingerprint: str,
) -> _GraphFragment:
    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

    directories: set[Path] = {Path(".")}
    for file_path in prepared.files:
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

    for file_path in prepared.files:
        summary = prepared.file_summary_by_path.get(file_path)
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

    return _GraphFragment(
        fragment_id=_structure_fragment_id(),
        fingerprint=fingerprint,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=sorted(edges.values(), key=lambda item: item.edge_id),
    )


def _build_file_fragment(
    *,
    prepared: _PreparedInputs,
    path: Path,
    fingerprint: str,
) -> _GraphFragment:
    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

    symbols_for_path = prepared.symbols_by_file.get(path, [])

    for symbol in symbols_for_path:
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

    symbol_by_file_and_qualified: dict[tuple[Path, str], GraphSymbolRecord] = {}
    for record in symbols_for_path:
        key = (record.file_path, record.qualified_name or record.name)
        current = symbol_by_file_and_qualified.get(key)
        if current is None or (current.kind == "module" and record.kind != "module"):
            symbol_by_file_and_qualified[key] = record

    for symbol in symbols_for_path:
        if not symbol.owner_qualified_name:
            continue
        parent_symbol = symbol_by_file_and_qualified.get(
            (symbol.file_path, symbol.owner_qualified_name)
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

    source_file_id = file_node_id(path)
    for dependency_record in prepared.dependency_records_by_source.get(path, []):
        if (
            dependency_record.target_path is None
            or dependency_record.target_path == dependency_record.source_path
        ):
            continue
        target_file_id = file_node_id(dependency_record.target_path)
        if (
            source_file_id not in prepared.all_file_node_ids
            or target_file_id not in prepared.all_file_node_ids
        ):
            continue
        _add_edge(
            edges,
            edge_type="imports",
            source_id=source_file_id,
            target_id=target_file_id,
            reason=dependency_reason_payload(dependency_record),
        )

    for chunk in prepared.code_chunks_by_source.get(path, []):
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
        _add_edge(
            edges,
            edge_type="contains",
            source_id=source_file_id,
            target_id=chunk_id_value,
        )

        token_set = set(tokenize(chunk.text))
        for symbol_id in _symbols_referenced_by_chunk(
            chunk_text_tokens=token_set,
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

    return _GraphFragment(
        fragment_id=_file_fragment_id(path),
        fingerprint=fingerprint,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=sorted(edges.values(), key=lambda item: item.edge_id),
    )


def _build_misc_chunk_fragment(
    *,
    prepared: _PreparedInputs,
    fingerprint: str,
) -> _GraphFragment:
    nodes: dict[str, GraphNodeRecord] = {}
    for chunk in prepared.misc_chunks:
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

    return _GraphFragment(
        fragment_id=_misc_fragment_id(),
        fingerprint=fingerprint,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=[],
    )


def _build_section_fragment(
    *,
    prepared: _PreparedInputs,
    section: _SectionInput,
    fingerprint: str,
) -> _GraphFragment:
    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

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

    file_targets, symbol_targets = prepared.section_targets.get(sid, (set(), set()))

    for file_target in sorted(file_targets):
        if file_target in prepared.all_file_node_ids:
            _add_edge(
                edges,
                edge_type="documents",
                source_id=sid,
                target_id=file_target,
            )

    for symbol_target in sorted(symbol_targets):
        if symbol_target in prepared.symbol_lookup:
            _add_edge(
                edges,
                edge_type="documents",
                source_id=sid,
                target_id=symbol_target,
            )

    return _GraphFragment(
        fragment_id=_section_fragment_id(section.section_id),
        fingerprint=fingerprint,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=sorted(edges.values(), key=lambda item: item.edge_id),
    )


def _build_commit_fragment(
    *,
    prepared: _PreparedInputs,
    commit_impact: CommitImpact,
    fingerprint: str,
) -> _GraphFragment:
    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

    commit_id = commit_node_id(commit_impact.commit_range)
    _add_node(
        nodes,
        GraphNodeRecord(
            node_id=commit_id,
            node_type="commit",
            label=commit_impact.commit_range,
        ),
    )

    changed_file_ids: set[str] = set()
    changed_symbol_ids: set[str] = set()

    for changed_file in sorted(
        commit_impact.changed_files,
        key=lambda item: item.path.as_posix(),
    ):
        fid = file_node_id(changed_file.path)
        if fid in prepared.all_file_node_ids:
            changed_file_ids.add(fid)
            _add_edge(
                edges,
                edge_type="changed_in",
                source_id=fid,
                target_id=commit_id,
            )

        names = _extract_changed_symbol_names(changed_file.signature_changes)
        for symbol in prepared.symbols_by_file.get(changed_file.path, []):
            if symbol.name in names or (symbol.qualified_name or "") in names:
                changed_symbol_ids.add(symbol.symbol_id)
                _add_edge(
                    edges,
                    edge_type="changed_in",
                    source_id=symbol.symbol_id,
                    target_id=commit_id,
                )

    for section_id, (file_targets, symbol_targets) in sorted(
        prepared.section_targets.items(), key=lambda item: item[0]
    ):
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

    return _GraphFragment(
        fragment_id=_commit_fragment_id(commit_impact.commit_range),
        fingerprint=fingerprint,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=sorted(edges.values(), key=lambda item: item.edge_id),
    )


def _build_fragment(
    *,
    fragment_id: str,
    expected_fingerprints: dict[str, str],
    prepared: _PreparedInputs,
    commit_impact: CommitImpact | None,
) -> _GraphFragment:
    fingerprint = expected_fingerprints[fragment_id]

    if fragment_id == _structure_fragment_id():
        return _build_structure_fragment(prepared=prepared, fingerprint=fingerprint)

    if fragment_id == _misc_fragment_id():
        return _build_misc_chunk_fragment(prepared=prepared, fingerprint=fingerprint)

    if fragment_id.startswith("file::"):
        path = Path(fragment_id.split("::", maxsplit=1)[1])
        return _build_file_fragment(
            prepared=prepared,
            path=path,
            fingerprint=fingerprint,
        )

    if fragment_id.startswith("section::"):
        section_id = fragment_id.split("::", maxsplit=1)[1]
        section = next(
            item for item in prepared.sections if item.section_id == section_id
        )
        return _build_section_fragment(
            prepared=prepared,
            section=section,
            fingerprint=fingerprint,
        )

    if fragment_id.startswith("commit::"):
        if commit_impact is None:
            raise ValueError(
                f"Commit fragment requested without commit impact: {fragment_id}"
            )
        return _build_commit_fragment(
            prepared=prepared,
            commit_impact=commit_impact,
            fingerprint=fingerprint,
        )

    raise ValueError(f"Unsupported graph fragment id: {fragment_id}")


def _write_fragment(fragments_root: Path, fragment: _GraphFragment) -> None:
    path = _fragment_path(fragments_root, fragment.fragment_id)
    rows: list[dict[str, object]] = [
        {
            "row_type": _FRAGMENT_ROW_META,
            "fragment_id": fragment.fragment_id,
            "fingerprint": fragment.fingerprint,
            "node_count": len(fragment.nodes),
            "edge_count": len(fragment.edges),
        }
    ]
    rows.extend(
        {
            "row_type": _FRAGMENT_ROW_NODE,
            "payload": node.model_dump(mode="json"),
        }
        for node in fragment.nodes
    )
    rows.extend(
        {
            "row_type": _FRAGMENT_ROW_EDGE,
            "payload": edge.model_dump(mode="json"),
        }
        for edge in fragment.edges
    )
    _write_jsonl(path, rows)


def _read_fragment(fragments_root: Path, fragment_id: str) -> _GraphFragment:
    path = _fragment_path(fragments_root, fragment_id)
    if not path.exists():
        raise ValueError(f"Missing graph fragment: {fragment_id}")

    rows = []
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid graph fragment {fragment_id}: {exc}") from exc

    if not rows:
        raise ValueError(f"Empty graph fragment: {fragment_id}")

    meta = rows[0]
    if not isinstance(meta, dict) or meta.get("row_type") != _FRAGMENT_ROW_META:
        raise ValueError(f"Missing fragment metadata row: {fragment_id}")

    meta_fragment_id = meta.get("fragment_id")
    if meta_fragment_id != fragment_id:
        raise ValueError(f"Fragment ID mismatch in {fragment_id}")

    fingerprint = meta.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError(f"Invalid fragment fingerprint in {fragment_id}")

    nodes: list[GraphNodeRecord] = []
    edges: list[GraphEdgeRecord] = []
    for row in rows[1:]:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed fragment row in {fragment_id}")
        row_type = row.get("row_type")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"Missing payload in {fragment_id}")
        if row_type == _FRAGMENT_ROW_NODE:
            nodes.append(GraphNodeRecord.model_validate(payload))
            continue
        if row_type == _FRAGMENT_ROW_EDGE:
            edges.append(GraphEdgeRecord.model_validate(payload))
            continue
        raise ValueError(f"Unknown fragment row type in {fragment_id}: {row_type}")

    return _GraphFragment(
        fragment_id=fragment_id,
        fingerprint=fingerprint,
        nodes=sorted(nodes, key=lambda item: item.node_id),
        edges=sorted(edges, key=lambda item: item.edge_id),
    )


def _cleanup_stale_fragments(
    fragments_root: Path, expected_fragment_ids: set[str]
) -> None:
    if not fragments_root.exists():
        return
    for path in fragments_root.glob("*.jsonl"):
        meta = _read_fragment_meta(path)
        if meta is None:
            path.unlink(missing_ok=True)
            continue
        fragment_id, _ = meta
        if fragment_id not in expected_fragment_ids:
            path.unlink(missing_ok=True)


def _compose_canonical_graph(
    *,
    graph_root: Path,
    fragment_ids: list[str],
    fragments_root: Path,
    prepared: _PreparedInputs,
) -> tuple[dict[str, GraphNodeRecord], dict[str, GraphEdgeRecord]]:
    nodes: dict[str, GraphNodeRecord] = {}
    edges: dict[str, GraphEdgeRecord] = {}

    for fragment_id in sorted(fragment_ids):
        fragment = _read_fragment(fragments_root, fragment_id)
        for node in fragment.nodes:
            nodes.setdefault(node.node_id, node)
        for edge in fragment.edges:
            edges.setdefault(edge.edge_id, edge)

    node_rows = [
        item.model_dump(mode="json")
        for item in sorted(nodes.values(), key=lambda node: node.node_id)
    ]
    edge_rows = [
        item.model_dump(mode="json")
        for item in sorted(edges.values(), key=lambda edge: edge.edge_id)
    ]
    _write_jsonl(graph_root / "nodes.jsonl", node_rows)
    _write_jsonl(graph_root / "edges.jsonl", edge_rows)

    symbol_index_payload = {
        "symbols": [
            item.model_dump(mode="json")
            for item in sorted(
                prepared.symbol_records, key=lambda entry: entry.symbol_id
            )
        ],
        "by_name": {
            key: sorted(item.symbol_id for item in values)
            for key, values in sorted(prepared.symbols_by_name.items())
        },
        "by_file": {
            key.as_posix(): sorted(item.symbol_id for item in values)
            for key, values in sorted(
                prepared.symbols_by_file.items(), key=lambda item: item[0].as_posix()
            )
        },
    }
    _write_json(graph_root / "symbol_index.json", symbol_index_payload)

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
    _write_json(graph_root / "adjacency.json", adjacency_payload)

    return nodes, edges


def _write_manifest(
    *,
    graph_root: Path,
    csc_id: str,
    output_root: Path,
    retrieval_root: Path,
    nodes: dict[str, GraphNodeRecord],
    edges: dict[str, GraphEdgeRecord],
    input_fingerprint: GraphInputFingerprint,
    planner_decision: GraphBuildDecision,
    previous_manifest_rel: Path | None,
    fragment_stats: dict[str, int],
) -> Path:
    node_counts_counter: Counter[str] = Counter(
        node.node_type for node in nodes.values()
    )
    edge_counts_counter: Counter[str] = Counter(
        edge.edge_type for edge in edges.values()
    )

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

    retrieval_manifest_path = retrieval_root / "manifest.json"
    try:
        source_manifest_rel = retrieval_manifest_path.relative_to(output_root)
    except ValueError:
        source_manifest_rel = retrieval_manifest_path

    manifest = GraphManifest(
        csc_id=csc_id,
        source_retrieval_manifest=source_manifest_rel,
        nodes_path=Path("nodes.jsonl"),
        edges_path=Path("edges.jsonl"),
        symbol_index_path=Path("symbol_index.json"),
        adjacency_path=Path("adjacency.json"),
        node_counts=node_counts,
        edge_counts=edge_counts,
        build_version=_GRAPH_BUILD_VERSION,
        input_fingerprint=input_fingerprint,
        planner_decision=planner_decision,
        previous_manifest=previous_manifest_rel,
        fragment_stats=fragment_stats,
        fragments_path=Path("fragments"),
    )
    manifest_path = graph_root / "manifest.json"
    write_json_model(manifest_path, manifest)
    return manifest_path


def _sum_manifest_counts(manifest: GraphManifest) -> tuple[int, int]:
    return sum(manifest.node_counts.values()), sum(manifest.edge_counts.values())


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
    changed_files: set[Path] | None = None,
    impacted_section_ids: set[str] | None = None,
    prior_manifest_path: Path | None = None,
) -> GraphBuildResult:
    """Build engineering graph artifacts with deterministic incremental planning."""

    graph_root = output_root / "graph"
    graph_root.mkdir(parents=True, exist_ok=True)
    fragments_root = graph_root / "fragments"
    fragments_root.mkdir(parents=True, exist_ok=True)

    manifest_path = prior_manifest_path or (graph_root / "manifest.json")
    prior_manifest = _load_prior_manifest(manifest_path)

    prepared = _prepare_inputs(
        repo_root=repo_root,
        retrieval_root=retrieval_root,
        scan_result=scan_result,
        document=document,
        review_artifact=review_artifact,
        update_report=update_report,
    )
    expected_fingerprints = _compute_expected_fragment_fingerprints(
        prepared=prepared,
        commit_impact=commit_impact,
    )

    input_fingerprint = _compute_input_fingerprint(
        scan_result=scan_result,
        retrieval_root=retrieval_root,
        output_root=output_root,
        document=document,
        review_artifact=review_artifact,
        update_report=update_report,
        commit_impact=commit_impact,
    )

    normalized_changed_files = _normalize_scoped_paths(changed_files)
    normalized_impacted_sections = set(impacted_section_ids or set())

    plan = _plan_build(
        prior_manifest=prior_manifest,
        graph_root=graph_root,
        fragments_root=fragments_root,
        input_fingerprint=input_fingerprint,
        expected_fingerprints=expected_fingerprints,
        changed_files=normalized_changed_files,
        impacted_sections=normalized_impacted_sections,
        commit_impact=commit_impact,
    )

    previous_manifest_rel: Path | None = None
    if plan.decision == "no_op":
        if prior_manifest is None:
            raise ValueError("No-op planner decision requires a prior manifest")
        node_count, edge_count = _sum_manifest_counts(prior_manifest)

        refreshed_manifest = prior_manifest.model_copy(
            update={
                "build_version": _GRAPH_BUILD_VERSION,
                "input_fingerprint": input_fingerprint,
                "planner_decision": "no_op",
                "fragment_stats": {
                    "total_fragments": len(expected_fingerprints),
                    "rebuilt_fragments": 0,
                    "reused_fragments": len(expected_fingerprints),
                },
                "fragments_path": Path("fragments"),
            }
        )
        write_json_model(graph_root / "manifest.json", refreshed_manifest)
        return GraphBuildResult(
            store_root=graph_root,
            manifest_path=graph_root / "manifest.json",
            node_count=node_count,
            edge_count=edge_count,
            planner_decision="no_op",
            rebuilt_fragments=0,
            reused_fragments=len(expected_fingerprints),
        )

    if plan.decision != "full" and prior_manifest is not None:
        if manifest_path.exists():
            previous_manifest_rel = Path("manifest.previous.json")
            shutil.copy2(manifest_path, graph_root / previous_manifest_rel)

    expected_fragment_ids = set(expected_fingerprints)

    decision = plan.decision
    impacted_fragment_ids = set(plan.impacted_fragments)
    reusable_fragment_ids = set(plan.reusable_fragments)

    if decision == "full":
        impacted_fragment_ids = set(expected_fragment_ids)
        reusable_fragment_ids = set()

    for fragment_id in sorted(impacted_fragment_ids):
        fragment = _build_fragment(
            fragment_id=fragment_id,
            expected_fingerprints=expected_fingerprints,
            prepared=prepared,
            commit_impact=commit_impact,
        )
        _write_fragment(fragments_root, fragment)

    if decision == "partial":
        if any(
            _fragment_status(
                fragments_root=fragments_root,
                fragment_id=fragment_id,
                expected_fingerprint=expected_fingerprints[fragment_id],
            )
            != "match"
            for fragment_id in expected_fragment_ids
        ):
            # Safety fallback: if partial reuse cannot be validated, rebuild all.
            decision = "full"
            impacted_fragment_ids = set(expected_fragment_ids)
            reusable_fragment_ids = set()
            for fragment_id in sorted(expected_fragment_ids):
                fragment = _build_fragment(
                    fragment_id=fragment_id,
                    expected_fingerprints=expected_fingerprints,
                    prepared=prepared,
                    commit_impact=commit_impact,
                )
                _write_fragment(fragments_root, fragment)

    _cleanup_stale_fragments(fragments_root, expected_fragment_ids)

    nodes, edges = _compose_canonical_graph(
        graph_root=graph_root,
        fragment_ids=sorted(expected_fragment_ids),
        fragments_root=fragments_root,
        prepared=prepared,
    )

    fragment_stats = {
        "total_fragments": len(expected_fragment_ids),
        "rebuilt_fragments": len(impacted_fragment_ids),
        "reused_fragments": len(reusable_fragment_ids),
    }

    written_manifest_path = _write_manifest(
        graph_root=graph_root,
        csc_id=csc_id,
        output_root=output_root,
        retrieval_root=retrieval_root,
        nodes=nodes,
        edges=edges,
        input_fingerprint=input_fingerprint,
        planner_decision=decision,
        previous_manifest_rel=previous_manifest_rel,
        fragment_stats=fragment_stats,
    )

    return GraphBuildResult(
        store_root=graph_root,
        manifest_path=written_manifest_path,
        node_count=len(nodes),
        edge_count=len(edges),
        planner_decision=decision,
        rebuilt_fragments=len(impacted_fragment_ids),
        reused_fragments=len(reusable_fragment_ids),
    )
