"""Typed models and helpers for engineering graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sddraft.domain.models import SourceLanguage

GraphNodeType = Literal["directory", "file", "symbol", "chunk", "sdd_section", "commit"]
GraphBuildDecision = Literal["no_op", "partial", "full"]
GraphEdgeType = Literal[
    "contains",
    "defines",
    "references",
    "documents",
    "parent_of",
    "imports",
    "changed_in",
    "impacts_section",
]


class GraphModel(BaseModel):
    """Strict base model for graph artifacts."""

    model_config = ConfigDict(extra="forbid")


class GraphNodeRecord(GraphModel):
    """One graph node record stored in JSONL."""

    node_id: str
    node_type: GraphNodeType
    label: str
    path: Path | None = None
    language: SourceLanguage | None = None
    symbol_kind: str | None = None
    symbol_name: str | None = None
    qualified_name: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphEdgeRecord(GraphModel):
    """One graph edge record stored in JSONL."""

    edge_id: str
    edge_type: GraphEdgeType
    source_id: str
    target_id: str
    reason: str | None = None


class GraphSymbolRecord(GraphModel):
    """Compact symbol index record."""

    symbol_id: str
    name: str
    qualified_name: str | None = None
    owner_qualified_name: str | None = None
    kind: str
    language: SourceLanguage
    file_path: Path
    line_start: int | None = None
    line_end: int | None = None


class GraphInputFingerprint(GraphModel):
    """Deterministic build fingerprint for graph input artifacts."""

    digest: str
    scan_digest: str
    retrieval_digest: str
    hierarchy_digest: str | None = None
    section_digest: str | None = None
    commit_digest: str | None = None


class GraphBuildPlan(GraphModel):
    """Planner decision metadata for one graph build attempt."""

    decision: GraphBuildDecision
    reason: str
    impacted_fragments: list[str] = Field(default_factory=list)
    reusable_fragments: list[str] = Field(default_factory=list)


class GraphManifest(GraphModel):
    """Manifest for graph store files."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: Literal["v1-engineering-graph-jsonl"] = "v1-engineering-graph-jsonl"
    csc_id: str
    source_retrieval_manifest: Path
    nodes_path: Path
    edges_path: Path
    symbol_index_path: Path
    adjacency_path: Path
    node_counts: dict[GraphNodeType, int] = Field(default_factory=dict)
    edge_counts: dict[GraphEdgeType, int] = Field(default_factory=dict)
    build_version: str | None = None
    input_fingerprint: GraphInputFingerprint | None = None
    planner_decision: GraphBuildDecision | None = None
    previous_manifest: Path | None = None
    fragment_stats: dict[str, int] = Field(default_factory=dict)
    fragments_path: Path | None = None


@dataclass(slots=True)
class GraphStore:
    """Loaded graph store index for query expansion."""

    manifest_path: Path
    manifest: GraphManifest
    nodes_by_id: dict[str, GraphNodeRecord]
    edges_by_id: dict[str, GraphEdgeRecord]
    outgoing: dict[str, list[str]]
    incoming: dict[str, list[str]]
    symbols_by_name: dict[str, list[str]]
    symbol_records: dict[str, GraphSymbolRecord]


def normalize_dir_path(path: Path) -> Path:
    """Normalize root directory path to '.'."""

    return Path(".") if path in {Path(""), Path(".")} else path


def directory_node_id(path: Path) -> str:
    """Deterministic ID for a directory node."""

    return f"dir::{normalize_dir_path(path).as_posix()}"


def file_node_id(path: Path) -> str:
    """Deterministic ID for a file node."""

    return f"file::{path.as_posix()}"


def symbol_node_id(path: Path, kind: str, qualified_name_or_name: str) -> str:
    """Deterministic ID for a symbol node."""

    return f"sym::{path.as_posix()}::{kind}::{qualified_name_or_name}"


def chunk_node_id(chunk_id: str) -> str:
    """Deterministic ID for a chunk node."""

    return f"chunk::{chunk_id}"


def section_node_id(section_id: str) -> str:
    """Deterministic ID for an SDD section node."""

    return f"sdd_section::{section_id}"


def commit_node_id(commit_range: str) -> str:
    """Deterministic ID for a commit-range node."""

    return f"commit::{commit_range}"


def edge_record_id(edge_type: GraphEdgeType, source_id: str, target_id: str) -> str:
    """Deterministic ID for an edge."""

    return f"edge::{edge_type}::{source_id}::{target_id}"
