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


HistoryEvidenceKind = Literal["commit", "file", "symbol", "subsystem", "build_source"]
HistoryDeltaEvidenceKind = HistoryEvidenceKind
HistoryDeltaStatus = Literal["introduced", "modified", "retired", "observed"]
HistoryConceptLifecycleStatus = Literal["active", "retired"]
HistoryConceptChangeStatus = Literal[
    "introduced",
    "modified",
    "unchanged",
    "retired",
    "observed",
]
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
HistoryAlgorithmCapsuleStatus = Literal["introduced", "modified", "observed"]
HistoryAlgorithmSharedAbstractionKind = Literal["function", "class", "symbol"]
HistoryAlgorithmDataStructureKind = Literal["class", "symbol"]
HistoryAlgorithmAssumptionSourceKind = Literal["docstring", "signature_change"]
HistorySectionId = Literal[
    "introduction",
    "architectural_overview",
    "subsystems_modules",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
]
HistorySectionPlanId = Literal[
    "introduction",
    "architectural_overview",
    "subsystems_modules",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
]
HistorySectionKind = Literal["core", "optional"]
HistorySectionPlanStatus = Literal["included", "omitted"]
HistorySectionDepth = Literal["brief", "standard", "deep"]
HistorySectionSignalKind = Literal[
    "active_subsystems",
    "active_modules",
    "active_dependencies",
    "architectural_change",
    "interface_change",
    "dependency_change",
    "infrastructure_change",
    "algorithm_candidate",
    "variant_family",
    "data_state_tokens",
    "robustness_tokens",
    "performance_tokens",
    "security_tokens",
    "rationale_change",
    "limitations_tokens",
]


class HistoryEvidenceLink(DomainModel):
    """Traceable evidence pointer for history-docs artifacts."""

    kind: HistoryEvidenceKind
    reference: str
    detail: str | None = None


HistoryDeltaEvidenceLink = HistoryEvidenceLink


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
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySubsystemChangeCandidate(DomainModel):
    """Structured subsystem-level change candidate derived from interval evidence."""

    candidate_id: str
    status: HistoryDeltaStatus
    source_root: Path
    group_path: Path
    commit_ids: list[str] = Field(default_factory=list)
    file_paths: list[Path] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


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
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


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
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


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
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmSharedAbstraction(DomainModel):
    """Cross-module reusable abstraction referenced by one algorithm capsule."""

    name: str
    kind: HistoryAlgorithmSharedAbstractionKind
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmDataStructure(DomainModel):
    """Named data structure signal referenced by one algorithm capsule."""

    name: str
    kind: HistoryAlgorithmDataStructureKind
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmPhase(DomainModel):
    """Ordered execution phase inferred for one algorithm capsule."""

    phase_key: str
    order: int
    matched_names: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmAssumption(DomainModel):
    """Evidence-backed assumption or constraint captured for one capsule."""

    text: str
    source_kind: HistoryAlgorithmAssumptionSourceKind
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmCapsule(DomainModel):
    """Structured deterministic algorithm capsule for one checkpoint."""

    capsule_id: str
    title: str
    status: HistoryAlgorithmCapsuleStatus
    scope_kind: HistoryAlgorithmScopeKind
    scope_path: Path
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    source_candidate_ids: list[str] = Field(default_factory=list)
    commit_ids: list[str] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    variant_names: list[str] = Field(default_factory=list)
    signal_kinds: list[HistoryAlgorithmSignalKind] = Field(default_factory=list)
    shared_abstractions: list[HistoryAlgorithmSharedAbstraction] = Field(
        default_factory=list
    )
    data_structures: list[HistoryAlgorithmDataStructure] = Field(default_factory=list)
    phases: list[HistoryAlgorithmPhase] = Field(default_factory=list)
    assumptions: list[HistoryAlgorithmAssumption] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmCapsuleIndexEntry(DomainModel):
    """Index entry for one serialized algorithm capsule artifact."""

    capsule_id: str
    title: str
    status: HistoryAlgorithmCapsuleStatus
    scope_kind: HistoryAlgorithmScopeKind
    scope_path: Path
    artifact_path: Path


class HistoryAlgorithmCapsuleIndex(DomainModel):
    """Index of algorithm capsule artifacts for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    capsules: list[HistoryAlgorithmCapsuleIndexEntry] = Field(default_factory=list)


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


class HistorySubsystemConcept(DomainModel):
    """Checkpoint-scoped subsystem concept."""

    concept_id: str
    lifecycle_status: HistoryConceptLifecycleStatus
    change_status: HistoryConceptChangeStatus
    first_seen_checkpoint: str
    last_updated_checkpoint: str
    source_root: Path
    group_path: Path
    module_ids: list[str] = Field(default_factory=list)
    file_count: int
    symbol_count: int
    language_counts: dict[str, int] = Field(default_factory=dict)
    representative_files: list[Path] = Field(default_factory=list)
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryModuleConcept(DomainModel):
    """Checkpoint-scoped module concept."""

    concept_id: str
    lifecycle_status: HistoryConceptLifecycleStatus
    change_status: HistoryConceptChangeStatus
    first_seen_checkpoint: str
    last_updated_checkpoint: str
    path: Path
    subsystem_id: str | None = None
    language: str
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    docstrings: list[str] = Field(default_factory=list)
    symbol_names: list[str] = Field(default_factory=list)
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDependencyConcept(DomainModel):
    """Checkpoint-scoped dependency-source concept."""

    concept_id: str
    lifecycle_status: HistoryConceptLifecycleStatus
    change_status: HistoryConceptChangeStatus
    first_seen_checkpoint: str
    last_updated_checkpoint: str
    path: Path
    ecosystem: str
    category: str
    related_subsystem_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySectionState(DomainModel):
    """Deterministic section stub for one checkpoint."""

    section_id: HistorySectionId
    title: str
    concept_ids: list[str] = Field(default_factory=list)
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryCheckpointModel(DomainModel):
    """Structured documentation-state model for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    previous_checkpoint_model_available: bool = False
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    subsystems: list[HistorySubsystemConcept] = Field(default_factory=list)
    modules: list[HistoryModuleConcept] = Field(default_factory=list)
    dependencies: list[HistoryDependencyConcept] = Field(default_factory=list)
    sections: list[HistorySectionState] = Field(default_factory=list)


class HistorySectionPlan(DomainModel):
    """Scored section plan entry for one checkpoint."""

    section_id: HistorySectionPlanId
    title: str
    kind: HistorySectionKind
    status: HistorySectionPlanStatus
    confidence_score: int
    evidence_score: int
    depth: HistorySectionDepth | None = None
    concept_ids: list[str] = Field(default_factory=list)
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)
    trigger_signals: list[HistorySectionSignalKind] = Field(default_factory=list)
    omission_reason: str | None = None


class HistorySectionOutline(DomainModel):
    """Scored section outline for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    sections: list[HistorySectionPlan] = Field(default_factory=list)


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
    checkpoint_model_path: Path | None = None
    section_outline_path: Path | None = None
    algorithm_capsule_index_path: Path | None = None
    file_count: int = 0
    symbol_count: int = 0
    subsystem_count: int = 0
    build_source_count: int = 0
    subsystem_change_count: int = 0
    interface_change_count: int = 0
    dependency_change_count: int = 0
    algorithm_candidate_count: int = 0
    subsystem_concept_count: int = 0
    module_concept_count: int = 0
    dependency_concept_count: int = 0
    retired_concept_count: int = 0
    included_section_count: int = 0
    omitted_section_count: int = 0
    algorithm_capsule_count: int = 0


__all__ = [
    "HistoryAlgorithmAssumption",
    "HistoryAlgorithmCandidate",
    "HistoryAlgorithmCapsule",
    "HistoryAlgorithmCapsuleIndex",
    "HistoryAlgorithmCapsuleIndexEntry",
    "HistoryAlgorithmCapsuleStatus",
    "HistoryAlgorithmDataStructure",
    "HistoryAlgorithmPhase",
    "HistoryAlgorithmSharedAbstraction",
    "HistoryBuildResult",
    "HistoryCommitDelta",
    "HistoryCheckpointModel",
    "HistoryConceptChangeStatus",
    "HistoryConceptLifecycleStatus",
    "HistoryDependencyConcept",
    "HistoryEvidenceKind",
    "HistoryEvidenceLink",
    "HistoryDeltaEvidenceLink",
    "HistoryDependencyChangeCandidate",
    "HistoryInterfaceChangeCandidate",
    "HistoryIntervalDeltaModel",
    "HistoryModuleConcept",
    "HistorySectionDepth",
    "HistorySectionId",
    "HistorySectionKind",
    "HistorySectionOutline",
    "HistorySectionPlan",
    "HistorySectionPlanId",
    "HistorySectionPlanStatus",
    "HistorySectionSignalKind",
    "HistorySectionState",
    "HistorySnapshotStructuralModel",
    "HistorySubsystemConcept",
    "HistorySubsystemChangeCandidate",
    "HistorySubsystemCandidate",
]
