"""History-docs tool model namespace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from engllm.core.analysis.history import HistoryBuildSource, HistoryCommitSummary
from engllm.domain.models import (
    CodeUnitSummary,
    CommitImpact,
    DomainModel,
    SymbolSummary,
)


class HistorySubsystemCandidate(DomainModel):
    """Deterministic subsystem grouping derived from one checkpoint snapshot."""

    candidate_id: str
    source_root: Path
    group_path: Path
    file_count: int
    symbol_count: int
    language_counts: dict[str, int] = Field(default_factory=dict)
    representative_files: list[Path] = Field(default_factory=list)


class HistorySnapshotStructuralModel(DomainModel):
    """Tool-scoped structural snapshot model for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    requested_source_roots: list[Path] = Field(default_factory=list)
    analyzed_source_roots: list[Path] = Field(default_factory=list)
    skipped_source_roots: list[Path] = Field(default_factory=list)
    files: list[Path] = Field(default_factory=list)
    code_summaries: list[CodeUnitSummary] = Field(default_factory=list)
    symbol_summaries: list[SymbolSummary] = Field(default_factory=list)
    subsystem_candidates: list[HistorySubsystemCandidate] = Field(default_factory=list)
    build_sources: list[HistoryBuildSource] = Field(default_factory=list)


HistoryDeltaEvidenceKind = Literal[
    "commit", "file", "symbol", "subsystem", "build_source"
]
HistoryDeltaStatus = Literal["introduced", "modified", "retired", "observed"]
HistoryCommitSignalKind = Literal[
    "architectural",
    "interface",
    "dependency",
    "infrastructure",
    "algorithm_candidate",
    "documentation_only",
    "logic_only",
]
HistoryAlgorithmSignalKind = Literal[
    "introduced_module",
    "multi_symbol",
    "variant_family",
]
HistoryInterfaceScopeKind = Literal["symbol", "file"]
HistoryDependencyKind = Literal["build_source", "code_import_signal"]
HistoryAlgorithmScopeKind = Literal["file", "subsystem"]


class HistoryDeltaEvidenceLink(DomainModel):
    """Traceable evidence pointer for interval-delta artifacts."""

    kind: HistoryDeltaEvidenceKind
    reference: str
    detail: str | None = None


class HistoryCommitDelta(DomainModel):
    """One commit-level delta record within a checkpoint interval."""

    commit: HistoryCommitSummary
    parent_commit: str | None = None
    diff_basis: Literal["root", "first_parent"]
    impact: CommitImpact
    signal_kinds: list[HistoryCommitSignalKind] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    affected_subsystem_ids: list[str] = Field(default_factory=list)
    touched_build_sources: list[Path] = Field(default_factory=list)
    evidence_links: list[HistoryDeltaEvidenceLink] = Field(default_factory=list)


class HistorySubsystemChangeCandidate(DomainModel):
    """Structured subsystem-level change candidate derived from interval evidence."""

    candidate_id: str
    status: HistoryDeltaStatus
    source_root: Path
    group_path: Path
    commit_ids: list[str] = Field(default_factory=list)
    file_paths: list[Path] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryDeltaEvidenceLink] = Field(default_factory=list)


class HistoryInterfaceChangeCandidate(DomainModel):
    """Structured interface-level change candidate derived from interval evidence."""

    candidate_id: str
    status: HistoryDeltaStatus
    scope_kind: HistoryInterfaceScopeKind
    source_path: Path
    symbol_name: str | None = None
    qualified_name: str | None = None
    commit_ids: list[str] = Field(default_factory=list)
    signature_changes: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryDeltaEvidenceLink] = Field(default_factory=list)


class HistoryDependencyChangeCandidate(DomainModel):
    """Structured dependency/infrastructure change candidate."""

    candidate_id: str
    status: HistoryDeltaStatus
    dependency_kind: HistoryDependencyKind
    path: Path | None = None
    subsystem_id: str | None = None
    ecosystem: str | None = None
    category: str | None = None
    commit_ids: list[str] = Field(default_factory=list)
    file_paths: list[Path] = Field(default_factory=list)
    dependency_change_lines: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryDeltaEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmCandidate(DomainModel):
    """Conservative algorithm or strategy signal candidate."""

    candidate_id: str
    scope_kind: HistoryAlgorithmScopeKind
    scope_path: Path
    subsystem_id: str | None = None
    commit_ids: list[str] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    variant_names: list[str] = Field(default_factory=list)
    signal_kinds: list[HistoryAlgorithmSignalKind] = Field(default_factory=list)
    evidence_links: list[HistoryDeltaEvidenceLink] = Field(default_factory=list)


class HistoryIntervalDeltaModel(DomainModel):
    """Tool-scoped interval-delta model for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    previous_snapshot_available: bool = False
    commit_deltas: list[HistoryCommitDelta] = Field(default_factory=list)
    subsystem_changes: list[HistorySubsystemChangeCandidate] = Field(
        default_factory=list
    )
    interface_changes: list[HistoryInterfaceChangeCandidate] = Field(
        default_factory=list
    )
    dependency_changes: list[HistoryDependencyChangeCandidate] = Field(
        default_factory=list
    )
    algorithm_candidates: list[HistoryAlgorithmCandidate] = Field(default_factory=list)


class HistoryBuildResult(DomainModel):
    """Result for one history-docs build run."""

    workspace_id: str
    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    previous_checkpoint_source: str
    commit_count: int
    checkpoint_plan_path: Path
    intervals_path: Path
    snapshot_manifest_path: Path | None = None
    snapshot_structural_model_path: Path | None = None
    interval_delta_model_path: Path | None = None
    file_count: int = 0
    symbol_count: int = 0
    subsystem_count: int = 0
    build_source_count: int = 0
    subsystem_change_count: int = 0
    interface_change_count: int = 0
    dependency_change_count: int = 0
    algorithm_candidate_count: int = 0


__all__ = [
    "HistoryAlgorithmCandidate",
    "HistoryBuildResult",
    "HistoryCommitDelta",
    "HistoryDeltaEvidenceLink",
    "HistoryDependencyChangeCandidate",
    "HistoryInterfaceChangeCandidate",
    "HistoryIntervalDeltaModel",
    "HistorySnapshotStructuralModel",
    "HistorySubsystemChangeCandidate",
    "HistorySubsystemCandidate",
]
