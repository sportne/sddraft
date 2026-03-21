"""History-docs tool model namespace."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from engllm.core.analysis.history import HistoryBuildSource
from engllm.domain.models import (
    CodeUnitSummary,
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
    file_count: int = 0
    symbol_count: int = 0
    subsystem_count: int = 0
    build_source_count: int = 0


__all__ = [
    "HistoryBuildResult",
    "HistorySnapshotStructuralModel",
    "HistorySubsystemCandidate",
]
