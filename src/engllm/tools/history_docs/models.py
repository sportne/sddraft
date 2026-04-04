"""History-docs tool model namespace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from engllm.core.analysis.history import HistoryBuildSource, HistoryCommitSummary
from engllm.domain.models import (
    CodeUnitSummary,
    CommitImpact,
    DomainModel,
    ProjectConfig,
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
HistoryDependencyRole = Literal[
    "runtime",
    "development",
    "test",
    "build",
    "plugin",
    "toolchain",
    "optional",
    "peer",
    "unknown",
]
HistoryDependencySectionTarget = Literal[
    "dependencies",
    "build_development_infrastructure",
]
HistoryDependencySourceKind = Literal["primary", "lockfile", "metadata"]
HistoryDependencySummaryStatus = Literal["documented", "tbd", "llm_failed"]
HistoryDependencyDescriptionBasis = Literal[
    "package_general_knowledge",
    "project_evidence",
    "tbd",
]
HistoryDependencyNarrativeRenderStyle = Literal["standard", "grouped_tooling"]
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
    "system_context",
    "subsystems_modules",
    "interfaces",
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
HistoryValidationSeverity = Literal["error", "warning"]
HistoryValidationCheckId = Literal[
    "included_section_missing",
    "omitted_section_rendered",
    "render_manifest_mismatch",
    "missing_source_artifact",
    "unknown_concept_reference",
    "unknown_dependency_reference",
    "unknown_algorithm_capsule_reference",
    "dependency_subsection_shape_invalid",
    "release_note_phrase",
    "weak_core_section",
    "weak_optional_section",
    "dependency_summary_tbd",
    "raw_internal_id_leak",
    "contradictory_tbd_phrase",
    "empty_grouped_tooling_stub",
    "algorithm_capsule_thin",
]
HistorySemanticCheckpointSignalKind = Literal[
    "tag_anchor",
    "interface_shift",
    "dependency_shift",
    "build_shift",
    "broad_change",
    "new_top_level_area",
    "merge_anchor",
]
HistorySemanticCheckpointRecommendation = Literal["primary", "supporting", "skip"]
HistorySemanticCheckpointEvaluationStatus = Literal[
    "scored",
    "heuristic_only",
    "llm_failed",
]
HistorySemanticStructureStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistorySemanticContextStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryIntervalInterpretationStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryCheckpointModelEnrichmentStatus = Literal[
    "scored",
    "heuristic_only",
    "llm_failed",
]
HistoryAlgorithmCapsuleEnrichmentStatus = Literal[
    "scored",
    "heuristic_only",
    "llm_failed",
]
HistoryInterfaceInventoryStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryDependencyLandscapeStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistorySectionPlanningStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryDraftStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryDraftReviewStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryRepairStatus = Literal["scored", "heuristic_only", "llm_failed"]
HistoryIntervalInsightKind = Literal[
    "subsystem_change",
    "interface_change",
    "dependency_change",
    "algorithm_change",
    "build_change",
    "design_rationale",
]
HistoryIntervalSignificance = Literal["low", "medium", "high"]
HistoryConceptEnrichmentKind = Literal[
    "subsystem",
    "module",
    "dependency",
    "capability",
    "design_note",
]
HistoryRationaleClueSourceKind = Literal[
    "commit_message",
    "signature_change",
    "docstring",
    "diff_pattern",
]
HistorySystemContextNodeKind = Literal[
    "system",
    "external_actor",
    "external_system",
    "data_store",
    "runtime_environment",
]
HistorySemanticInterfaceKind = Literal[
    "http_api",
    "cli_surface",
    "library_surface",
    "data_contract",
    "internal_service",
    "protocol",
]
HistoryDocsBenchmarkFocusTag = Literal[
    "small",
    "medium",
    "algorithm-heavy",
    "dependency-heavy",
    "architecture-heavy",
]
HistoryDocsBenchmarkExpectationKind = Literal[
    "section_presence",
    "algorithm_signal",
    "dependency_understanding",
    "architectural_distinction",
    "system_context_signal",
    "interface_distinction",
    "present_state_tone",
]
HistoryDocsRubricDimension = Literal[
    "coverage",
    "coherence",
    "specificity",
    "algorithm_understanding",
    "dependency_understanding",
    "rationale_capture",
    "present_state_tone",
]
HistoryDocsQualityEvaluationStatus = Literal["scored", "llm_failed"]


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
    change_id: str | None = None
    status: HistoryDeltaStatus
    source_root: Path
    group_path: Path
    commit_ids: list[str] = Field(default_factory=list)
    file_paths: list[Path] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_change_id(self) -> HistorySubsystemChangeCandidate:
        """Mirror candidate identifiers into the H12 change-id field."""

        self.change_id = self.change_id or self.candidate_id
        return self


class HistoryInterfaceChangeCandidate(DomainModel):
    """Structured interface-level change candidate derived from interval evidence."""

    candidate_id: str
    change_id: str | None = None
    status: HistoryDeltaStatus
    scope_kind: HistoryInterfaceScopeKind
    source_path: Path
    symbol_name: str | None = None
    qualified_name: str | None = None
    commit_ids: list[str] = Field(default_factory=list)
    signature_changes: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_change_id(self) -> HistoryInterfaceChangeCandidate:
        """Mirror candidate identifiers into the H12 change-id field."""

        self.change_id = self.change_id or self.candidate_id
        return self


class HistoryDependencyChangeCandidate(DomainModel):
    """Structured dependency/infrastructure change candidate."""

    candidate_id: str
    change_id: str | None = None
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

    @model_validator(mode="after")
    def default_change_id(self) -> HistoryDependencyChangeCandidate:
        """Mirror candidate identifiers into the H12 change-id field."""

        self.change_id = self.change_id or self.candidate_id
        return self


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


class HistoryAlgorithmInvariant(DomainModel):
    """Evidence-backed invariant attached to one enriched algorithm capsule."""

    text: str
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmTradeoff(DomainModel):
    """One tradeoff or operational tension captured for an algorithm capsule."""

    title: str
    summary: str
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmVariantRelationship(DomainModel):
    """Relationship note for one known algorithm variant family."""

    variant_name: str
    relationship: str
    summary: str
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmCapsuleEnrichment(DomainModel):
    """Structured H13-01 enrichment for one existing algorithm capsule."""

    capsule_id: str
    purpose: str
    phase_flow_summary: str
    invariants: list[HistoryAlgorithmInvariant] = Field(default_factory=list)
    tradeoffs: list[HistoryAlgorithmTradeoff] = Field(default_factory=list)
    variant_relationships: list[HistoryAlgorithmVariantRelationship] = Field(
        default_factory=list
    )
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_rationale_clue_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryAlgorithmCapsuleEnrichmentIndexEntry(DomainModel):
    """Index entry for one serialized enriched algorithm capsule artifact."""

    capsule_id: str
    artifact_path: Path


class HistoryAlgorithmCapsuleEnrichmentIndex(DomainModel):
    """Index of H13-01 enriched algorithm capsule artifacts."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryAlgorithmCapsuleEnrichmentStatus
    capsules: list[HistoryAlgorithmCapsuleEnrichmentIndexEntry] = Field(
        default_factory=list
    )


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


class HistoryDesignChangeInsight(DomainModel):
    """One interpreted design-change insight derived from interval evidence."""

    insight_id: str
    kind: HistoryIntervalInsightKind
    title: str
    summary: str
    significance: HistoryIntervalSignificance
    related_commit_ids: list[str] = Field(default_factory=list)
    related_change_ids: list[str] = Field(default_factory=list)
    related_subsystem_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryRationaleClue(DomainModel):
    """One conservative rationale clue linked to explicit interval evidence."""

    clue_id: str
    text: str
    confidence: float = 0.0
    related_commit_ids: list[str] = Field(default_factory=list)
    related_change_ids: list[str] = Field(default_factory=list)
    source_kind: HistoryRationaleClueSourceKind
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySignificantChangeWindow(DomainModel):
    """One high-signal commit window interpreted from H3 interval evidence."""

    window_id: str
    start_commit: str
    end_commit: str
    commit_ids: list[str] = Field(default_factory=list)
    title: str
    summary: str
    significance: HistoryIntervalSignificance
    related_insight_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryIntervalInterpretation(DomainModel):
    """Checkpoint-scoped H12 interval interpretation artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryIntervalInterpretationStatus
    insights: list[HistoryDesignChangeInsight] = Field(default_factory=list)
    rationale_clues: list[HistoryRationaleClue] = Field(default_factory=list)
    significant_windows: list[HistorySignificantChangeWindow] = Field(
        default_factory=list
    )


class HistorySubsystemConceptEnrichment(DomainModel):
    """Structured enrichment for one existing subsystem concept."""

    concept_id: str
    display_name: str
    summary: str
    capability_labels: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_rationale_clue_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryModuleConceptEnrichment(DomainModel):
    """Structured enrichment for one existing module concept."""

    concept_id: str
    summary: str
    responsibility_labels: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_rationale_clue_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryCapabilityConceptProposal(DomainModel):
    """Proposed capability concept anchored to existing concepts."""

    capability_id: str
    title: str
    summary: str
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDesignNoteAnchor(DomainModel):
    """Proposed design-note anchor grounded in interval interpretation evidence."""

    note_id: str
    title: str
    summary: str
    related_concept_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_rationale_clue_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryCheckpointModelEnrichment(DomainModel):
    """Checkpoint-scoped H12-02 checkpoint-model enrichment artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryCheckpointModelEnrichmentStatus
    subsystem_enrichments: list[HistorySubsystemConceptEnrichment] = Field(
        default_factory=list
    )
    module_enrichments: list[HistoryModuleConceptEnrichment] = Field(
        default_factory=list
    )
    capability_proposals: list[HistoryCapabilityConceptProposal] = Field(
        default_factory=list
    )
    design_note_anchors: list[HistoryDesignNoteAnchor] = Field(default_factory=list)


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
    display_name: str | None = None
    summary: str | None = None
    capability_labels: list[str] = Field(default_factory=list)
    baseline_subsystem_ids: list[str] = Field(default_factory=list)
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
    summary: str | None = None
    responsibility_labels: list[str] = Field(default_factory=list)
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
    documented_dependency_ids: list[str] = Field(default_factory=list)
    documented_dependency_count: int = 0
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDependencyDeclaration(DomainModel):
    """One parsed direct dependency declaration from a manifest-like source."""

    source_path: Path
    source_kind: HistoryDependencySourceKind
    raw_name: str
    normalized_name: str
    role: HistoryDependencyRole
    version_spec: str | None = None
    group_name: str | None = None
    declaration_text: str | None = None


class HistoryDependencyWarning(DomainModel):
    """Non-fatal parse or summary warning for one checkpoint dependency run."""

    source_path: Path
    code: str
    message: str


class HistoryDependencySummary(DomainModel):
    """Structured LLM response for one dependency summary."""

    general_description: str
    project_usage_description: str
    general_description_basis: HistoryDependencyDescriptionBasis = "tbd"
    project_usage_basis: HistoryDependencyDescriptionBasis = "tbd"
    uncertainty: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class HistoryDependencyEntry(DomainModel):
    """Aggregated direct dependency entry for one checkpoint."""

    dependency_id: str
    display_name: str
    normalized_name: str
    ecosystem: str
    declarations: list[HistoryDependencyDeclaration] = Field(default_factory=list)
    source_manifest_paths: list[Path] = Field(default_factory=list)
    source_dependency_concept_ids: list[str] = Field(default_factory=list)
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    scope_roles: list[HistoryDependencyRole] = Field(default_factory=list)
    section_target: HistoryDependencySectionTarget
    usage_signals: list[str] = Field(default_factory=list)
    general_description: str = "TBD"
    project_usage_description: str = "TBD"
    general_description_basis: HistoryDependencyDescriptionBasis = "tbd"
    project_usage_basis: HistoryDependencyDescriptionBasis = "tbd"
    uncertainty: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    summary_status: HistoryDependencySummaryStatus = "tbd"


class HistoryDependencyInventory(DomainModel):
    """Tool-scoped dependency documentation artifact for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    entries: list[HistoryDependencyEntry] = Field(default_factory=list)
    warnings: list[HistoryDependencyWarning] = Field(default_factory=list)


class HistoryDependencyNarrativeShadowEntry(DomainModel):
    """Shadow dependency narrative with separated general and project evidence."""

    dependency_id: str
    display_name: str
    ecosystem: str
    section_target: HistoryDependencySectionTarget
    scope_roles: list[HistoryDependencyRole] = Field(default_factory=list)
    render_style: HistoryDependencyNarrativeRenderStyle = "standard"
    group_title: str | None = None
    general_description: str = "TBD"
    general_description_basis: HistoryDependencyDescriptionBasis = "tbd"
    project_usage_description: str = "TBD"
    project_usage_basis: HistoryDependencyDescriptionBasis = "tbd"
    uncertainty: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    usage_signals: list[str] = Field(default_factory=list)


class HistoryDependencyNarrativeShadow(DomainModel):
    """Checkpoint-scoped shadow dependency prose artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    entries: list[HistoryDependencyNarrativeShadowEntry] = Field(default_factory=list)


class HistoryInterfaceResponsibility(DomainModel):
    """One interface responsibility or obligation statement."""

    title: str
    summary: str
    related_module_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryCrossModuleContract(DomainModel):
    """One cross-module contract signal tied to an interface concept."""

    contract_id: str
    title: str
    summary: str
    provider_module_ids: list[str] = Field(default_factory=list)
    consumer_module_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryInterfaceConcept(DomainModel):
    """Structured H13-02 interface concept built on top of H11-03 candidates."""

    interface_id: str
    title: str
    kind: HistorySemanticInterfaceKind
    summary: str
    provider_subsystem_ids: list[str] = Field(default_factory=list)
    consumer_context_node_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    responsibilities: list[HistoryInterfaceResponsibility] = Field(default_factory=list)
    cross_module_contracts: list[HistoryCrossModuleContract] = Field(
        default_factory=list
    )
    collaboration_notes: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_rationale_clue_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryInterfaceInventory(DomainModel):
    """Checkpoint-scoped H13-02 interface documentation artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryInterfaceInventoryStatus
    interfaces: list[HistoryInterfaceConcept] = Field(default_factory=list)


class HistoryDependencyProjectRole(DomainModel):
    """Project-level role grouping for one set of related dependencies."""

    role_id: str
    title: str
    summary: str
    dependency_ids: list[str] = Field(default_factory=list)
    related_subsystem_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDependencyCluster(DomainModel):
    """Cluster of dependencies sharing a common role or ecosystem pattern."""

    cluster_id: str
    title: str
    summary: str
    dependency_ids: list[str] = Field(default_factory=list)
    ecosystems: list[str] = Field(default_factory=list)
    scope_roles: list[HistoryDependencyRole] = Field(default_factory=list)
    related_subsystem_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDependencyUsagePattern(DomainModel):
    """Project-specific dependency usage pattern across modules or subsystems."""

    pattern_id: str
    title: str
    summary: str
    dependency_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    related_subsystem_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDependencyLandscape(DomainModel):
    """Checkpoint-scoped H13-03 project-level dependency understanding artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryDependencyLandscapeStatus
    project_roles: list[HistoryDependencyProjectRole] = Field(default_factory=list)
    clusters: list[HistoryDependencyCluster] = Field(default_factory=list)
    usage_patterns: list[HistoryDependencyUsagePattern] = Field(default_factory=list)


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
    planning_rationale: str | None = None
    source_insight_ids: list[str] = Field(default_factory=list)
    source_capability_ids: list[str] = Field(default_factory=list)
    source_design_note_ids: list[str] = Field(default_factory=list)


class HistorySectionOutline(DomainModel):
    """Scored section outline for one checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    sections: list[HistorySectionPlan] = Field(default_factory=list)


class HistoryLLMSectionPlanDecision(DomainModel):
    """Structured LLM decision for one known section id."""

    section_id: HistorySectionPlanId
    status: HistorySectionPlanStatus
    depth: HistorySectionDepth | None = None
    confidence_score: int = Field(ge=0, le=100)
    planning_rationale: str
    source_insight_ids: list[str] = Field(default_factory=list)
    source_capability_ids: list[str] = Field(default_factory=list)
    source_design_note_ids: list[str] = Field(default_factory=list)


class HistoryLLMSectionOutlineJudgment(DomainModel):
    """Validated LLM response for H12-03 section planning."""

    sections: list[HistoryLLMSectionPlanDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_section_ids(self) -> HistoryLLMSectionOutlineJudgment:
        """Reject duplicate section decisions for one outline judgment."""

        section_ids = [section.section_id for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("sections must not repeat section_id values")
        return self


class HistoryLLMSectionOutline(DomainModel):
    """Checkpoint-scoped shadow H12-03 LLM section-planning artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistorySectionPlanningStatus
    sections: list[HistorySectionPlan] = Field(default_factory=list)


class HistoryRenderedSection(DomainModel):
    """Rendered section trace record for one checkpoint markdown artifact."""

    section_id: HistorySectionPlanId
    title: str
    order: int
    kind: HistorySectionKind
    concept_ids: list[str] = Field(default_factory=list)
    algorithm_capsule_ids: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    source_artifact_paths: list[Path] = Field(default_factory=list)
    subheading_count: int = 0


class HistoryRenderManifest(DomainModel):
    """Structured debug manifest for one rendered checkpoint markdown file."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    markdown_path: Path
    sections: list[HistoryRenderedSection] = Field(default_factory=list)


HistoryDraftReviewFindingKind = Literal[
    "coherence",
    "omission",
    "unsupported_claim",
    "redundancy",
    "weak_prose",
]
HistoryDraftReviewSeverity = Literal["low", "medium", "high"]


class HistorySectionDraft(DomainModel):
    """One section-scoped H14 draft body."""

    section_id: HistorySectionPlanId
    markdown_body: str
    supporting_concept_ids: list[str] = Field(default_factory=list)
    supporting_algorithm_capsule_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_capability_ids: list[str] = Field(default_factory=list)
    source_design_note_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySectionDraftArtifact(DomainModel):
    """Checkpoint-scoped H14-01 draft artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryDraftStatus
    sections: list[HistorySectionDraft] = Field(default_factory=list)


class HistoryDraftReviewFinding(DomainModel):
    """One document-review finding for a shadow draft."""

    finding_id: str
    kind: HistoryDraftReviewFindingKind
    severity: HistoryDraftReviewSeverity
    section_id: str | None = None
    summary: str
    revision_goal: str


class HistoryDraftReview(DomainModel):
    """Checkpoint-scoped H14-02 review artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryDraftReviewStatus
    strengths: list[str] = Field(default_factory=list)
    findings: list[HistoryDraftReviewFinding] = Field(default_factory=list)
    recommended_repair_section_ids: list[HistorySectionPlanId] = Field(
        default_factory=list
    )
    unrepairable_notes: list[str] = Field(default_factory=list)


class HistorySectionRepair(DomainModel):
    """One section-scoped H14-03 repair body."""

    section_id: HistorySectionPlanId
    revised_markdown_body: str
    addressed_finding_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySectionRepairArtifact(DomainModel):
    """Checkpoint-scoped H14-03 repair artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistoryRepairStatus
    sections: list[HistorySectionRepair] = Field(default_factory=list)


class HistorySectionDraftJudgment(DomainModel):
    """Structured H14-01 section draft response."""

    markdown_body: str
    supporting_concept_ids: list[str] = Field(default_factory=list)
    supporting_algorithm_capsule_ids: list[str] = Field(default_factory=list)
    source_insight_ids: list[str] = Field(default_factory=list)
    source_capability_ids: list[str] = Field(default_factory=list)
    source_design_note_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryDraftReviewJudgment(DomainModel):
    """Structured H14-02 document review response."""

    strengths: list[str] = Field(default_factory=list)
    findings: list[HistoryDraftReviewFinding] = Field(default_factory=list)
    recommended_repair_section_ids: list[HistorySectionPlanId] = Field(
        default_factory=list
    )


class HistorySectionRepairJudgment(DomainModel):
    """Structured H14-03 section repair response."""

    revised_markdown_body: str
    addressed_finding_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistoryValidationFinding(DomainModel):
    """One deterministic validation finding for a rendered checkpoint."""

    check_id: HistoryValidationCheckId
    severity: HistoryValidationSeverity
    message: str
    section_id: str | None = None
    artifact_path: Path | None = None
    reference: str | None = None
    line_number: int | None = None


class HistoryValidationReport(DomainModel):
    """Deterministic validation report for one rendered checkpoint."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    markdown_path: Path
    render_manifest_path: Path
    error_count: int = 0
    warning_count: int = 0
    findings: list[HistoryValidationFinding] = Field(default_factory=list)


class HistorySemanticCheckpointCandidate(DomainModel):
    """Advisory checkpoint candidate derived from first-parent history signals."""

    commit: HistoryCommitSummary
    window_start_commit: str | None = None
    window_commit_count: int = 0
    tag_names: list[str] = Field(default_factory=list)
    top_level_areas: list[str] = Field(default_factory=list)
    change_kinds: list[str] = Field(default_factory=list)
    signal_kinds: list[HistorySemanticCheckpointSignalKind] = Field(
        default_factory=list
    )
    heuristic_score: int = 0
    recommendation: HistorySemanticCheckpointRecommendation
    semantic_title: str
    rationale: str
    uncertainty: str | None = None
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySemanticCheckpointPlan(DomainModel):
    """Checkpoint-scoped semantic planning artifact for one target commit."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistorySemanticCheckpointEvaluationStatus
    current_target_recommended: bool = False
    candidates: list[HistorySemanticCheckpointCandidate] = Field(default_factory=list)


class HistorySemanticSubsystemCluster(DomainModel):
    """Semantic subsystem cluster emitted for one checkpoint snapshot."""

    semantic_subsystem_id: str
    title: str
    summary: str
    module_ids: list[str] = Field(default_factory=list)
    baseline_subsystem_candidate_ids: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)
    representative_files: list[Path] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySemanticCapabilityCluster(DomainModel):
    """Capability note attached to one or more modules or semantic subsystems."""

    capability_id: str
    title: str
    summary: str
    module_ids: list[str] = Field(default_factory=list)
    semantic_subsystem_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySemanticStructureMap(DomainModel):
    """Checkpoint-scoped semantic subsystem and capability map."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistorySemanticStructureStatus
    semantic_subsystems: list[HistorySemanticSubsystemCluster] = Field(
        default_factory=list
    )
    capabilities: list[HistorySemanticCapabilityCluster] = Field(default_factory=list)


class HistorySemanticStructureJudgmentSubsystem(DomainModel):
    """Structured LLM proposal for one semantic subsystem cluster."""

    semantic_subsystem_id: str
    title: str
    summary: str
    module_ids: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(default_factory=list)


class HistorySemanticStructureJudgmentCapability(DomainModel):
    """Structured LLM proposal for one capability cluster."""

    capability_id: str
    title: str
    summary: str
    module_ids: list[str] = Field(default_factory=list)
    semantic_subsystem_ids: list[str] = Field(default_factory=list)


class HistorySemanticStructureJudgment(DomainModel):
    """Validated LLM subsystem/capability clustering response."""

    semantic_subsystems: list[HistorySemanticStructureJudgmentSubsystem] = Field(
        default_factory=list
    )
    capabilities: list[HistorySemanticStructureJudgmentCapability] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> HistorySemanticStructureJudgment:
        """Reject duplicate semantic subsystem or capability identifiers."""

        subsystem_ids = [
            subsystem.semantic_subsystem_id for subsystem in self.semantic_subsystems
        ]
        capability_ids = [capability.capability_id for capability in self.capabilities]
        if len(subsystem_ids) != len(set(subsystem_ids)):
            raise ValueError(
                "semantic_subsystems must not repeat semantic_subsystem_id values"
            )
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capabilities must not repeat capability_id values")
        return self


class HistorySystemContextNode(DomainModel):
    """One semantic system-context node for a checkpoint."""

    node_id: str
    title: str
    kind: HistorySystemContextNodeKind
    summary: str
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySemanticInterfaceCandidate(DomainModel):
    """One evidence-backed semantic interface candidate."""

    interface_id: str
    title: str
    kind: HistorySemanticInterfaceKind
    summary: str
    provider_subsystem_ids: list[str] = Field(default_factory=list)
    consumer_context_node_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)
    evidence_links: list[HistoryEvidenceLink] = Field(default_factory=list)


class HistorySemanticContextMap(DomainModel):
    """Checkpoint-scoped semantic context and interface artifact."""

    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    evaluation_status: HistorySemanticContextStatus
    context_nodes: list[HistorySystemContextNode] = Field(default_factory=list)
    interfaces: list[HistorySemanticInterfaceCandidate] = Field(default_factory=list)


class HistorySemanticContextJudgmentNode(DomainModel):
    """Structured LLM proposal for one context node."""

    node_id: str
    title: str
    kind: HistorySystemContextNodeKind
    summary: str
    related_subsystem_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)


class HistorySemanticContextJudgmentInterface(DomainModel):
    """Structured LLM proposal for one interface candidate."""

    interface_id: str
    title: str
    kind: HistorySemanticInterfaceKind
    summary: str
    provider_subsystem_ids: list[str] = Field(default_factory=list)
    consumer_context_node_ids: list[str] = Field(default_factory=list)
    related_module_ids: list[str] = Field(default_factory=list)


class HistorySemanticContextJudgment(DomainModel):
    """Validated LLM response for semantic context and interface extraction."""

    context_nodes: list[HistorySemanticContextJudgmentNode] = Field(
        default_factory=list
    )
    interfaces: list[HistorySemanticContextJudgmentInterface] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> HistorySemanticContextJudgment:
        """Reject duplicate context-node or interface identifiers."""

        node_ids = [node.node_id for node in self.context_nodes]
        interface_ids = [interface.interface_id for interface in self.interfaces]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("context_nodes must not repeat node_id values")
        if len(interface_ids) != len(set(interface_ids)):
            raise ValueError("interfaces must not repeat interface_id values")
        return self


class HistoryIntervalInterpretationJudgment(DomainModel):
    """Validated LLM response for H12 interval interpretation."""

    insights: list[HistoryDesignChangeInsight] = Field(default_factory=list)
    rationale_clues: list[HistoryRationaleClue] = Field(default_factory=list)
    significant_windows: list[HistorySignificantChangeWindow] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> HistoryIntervalInterpretationJudgment:
        """Reject duplicate insight, clue, or window identifiers."""

        insight_ids = [insight.insight_id for insight in self.insights]
        clue_ids = [clue.clue_id for clue in self.rationale_clues]
        window_ids = [window.window_id for window in self.significant_windows]
        if len(insight_ids) != len(set(insight_ids)):
            raise ValueError("insights must not repeat insight_id values")
        if len(clue_ids) != len(set(clue_ids)):
            raise ValueError("rationale_clues must not repeat clue_id values")
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("significant_windows must not repeat window_id values")
        return self


class HistoryCheckpointModelEnrichmentJudgment(DomainModel):
    """Validated LLM response for H12-02 checkpoint-model enrichment."""

    subsystem_enrichments: list[HistorySubsystemConceptEnrichment] = Field(
        default_factory=list
    )
    module_enrichments: list[HistoryModuleConceptEnrichment] = Field(
        default_factory=list
    )
    capability_proposals: list[HistoryCapabilityConceptProposal] = Field(
        default_factory=list
    )
    design_note_anchors: list[HistoryDesignNoteAnchor] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> HistoryCheckpointModelEnrichmentJudgment:
        """Reject duplicate enrichment or proposal identifiers."""

        subsystem_ids = [
            enrichment.concept_id for enrichment in self.subsystem_enrichments
        ]
        module_ids = [enrichment.concept_id for enrichment in self.module_enrichments]
        capability_ids = [
            proposal.capability_id for proposal in self.capability_proposals
        ]
        note_ids = [anchor.note_id for anchor in self.design_note_anchors]
        if len(subsystem_ids) != len(set(subsystem_ids)):
            raise ValueError("subsystem_enrichments must not repeat concept_id values")
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("module_enrichments must not repeat concept_id values")
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError(
                "capability_proposals must not repeat capability_id values"
            )
        if len(note_ids) != len(set(note_ids)):
            raise ValueError("design_note_anchors must not repeat note_id values")
        return self


class HistoryAlgorithmCapsuleEnrichmentJudgment(DomainModel):
    """Validated LLM response for H13-01 algorithm capsule enrichment."""

    enrichments: list[HistoryAlgorithmCapsuleEnrichment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_capsule_ids(
        self,
    ) -> HistoryAlgorithmCapsuleEnrichmentJudgment:
        """Reject duplicate enriched algorithm capsule identifiers."""

        capsule_ids = [enrichment.capsule_id for enrichment in self.enrichments]
        if len(capsule_ids) != len(set(capsule_ids)):
            raise ValueError("enrichments must not repeat capsule_id values")
        return self


class HistoryInterfaceInventoryJudgment(DomainModel):
    """Validated LLM response for H13-02 interface documentation."""

    interfaces: list[HistoryInterfaceConcept] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_interface_ids(self) -> HistoryInterfaceInventoryJudgment:
        """Reject duplicate interface identifiers."""

        interface_ids = [interface.interface_id for interface in self.interfaces]
        if len(interface_ids) != len(set(interface_ids)):
            raise ValueError("interfaces must not repeat interface_id values")
        return self


class HistoryDependencyLandscapeJudgment(DomainModel):
    """Validated LLM response for H13-03 dependency landscape generation."""

    project_roles: list[HistoryDependencyProjectRole] = Field(default_factory=list)
    clusters: list[HistoryDependencyCluster] = Field(default_factory=list)
    usage_patterns: list[HistoryDependencyUsagePattern] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> HistoryDependencyLandscapeJudgment:
        """Reject duplicate role, cluster, or usage-pattern identifiers."""

        role_ids = [role.role_id for role in self.project_roles]
        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        pattern_ids = [pattern.pattern_id for pattern in self.usage_patterns]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("project_roles must not repeat role_id values")
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("clusters must not repeat cluster_id values")
        if len(pattern_ids) != len(set(pattern_ids)):
            raise ValueError("usage_patterns must not repeat pattern_id values")
        return self


class HistorySemanticCheckpointJudgment(DomainModel):
    """Structured planner judgment for one deterministic candidate commit."""

    candidate_commit_id: str
    recommendation: HistorySemanticCheckpointRecommendation
    semantic_title: str
    rationale: str
    uncertainty: str | None = None


class HistorySemanticCheckpointJudgmentBatch(DomainModel):
    """Validated semantic-planner response across candidate commits."""

    judgments: list[HistorySemanticCheckpointJudgment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_candidate_ids(
        self,
    ) -> HistorySemanticCheckpointJudgmentBatch:
        """Reject duplicate planner judgments for the same candidate commit."""

        candidate_ids = [judgment.candidate_commit_id for judgment in self.judgments]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("judgments must not repeat candidate_commit_id values")
        return self


def _validate_rubric_scores(
    scores: list[HistoryDocsRubricScore],
) -> list[HistoryDocsRubricScore]:
    expected_dimensions = {
        "coverage",
        "coherence",
        "specificity",
        "algorithm_understanding",
        "dependency_understanding",
        "rationale_capture",
        "present_state_tone",
    }
    dimensions = {score.dimension for score in scores}
    if dimensions != expected_dimensions:
        raise ValueError("rubric_scores must contain exactly the H10 rubric dimensions")
    if len(scores) != len(expected_dimensions):
        raise ValueError("rubric_scores must contain exactly one score per dimension")
    return scores


class HistoryDocsBenchmarkExpectation(DomainModel):
    """Inspectable expected property for one benchmark case."""

    expectation_id: str
    kind: HistoryDocsBenchmarkExpectationKind
    description: str
    required_section_ids: list[HistorySectionPlanId] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class HistoryDocsBenchmarkCase(DomainModel):
    """Resolved benchmark case manifest for one history-docs evaluation case."""

    case_id: str
    title: str
    description: str
    focus_tags: list[HistoryDocsBenchmarkFocusTag] = Field(default_factory=list)
    builder_name: str
    project_config: ProjectConfig
    target_commit: str
    previous_checkpoint_commit: str | None = None
    expectations: list[HistoryDocsBenchmarkExpectation] = Field(default_factory=list)


class HistoryDocsRubricScore(DomainModel):
    """One rubric-dimension score assigned by the H10 evaluator."""

    dimension: HistoryDocsRubricDimension
    score: int = Field(ge=0, le=5)
    rationale: str
    matched_expectation_ids: list[str] = Field(default_factory=list)
    cited_section_ids: list[HistorySectionPlanId] = Field(default_factory=list)


class HistoryDocsLooseRubricScore(DomainModel):
    """Looser rubric-shape model used to normalize real-provider judge output."""

    model_config = ConfigDict(extra="allow")

    dimension: str | None = None
    score: int | None = Field(default=None, ge=0, le=5)
    rationale: str | None = None
    matched_expectation_ids: list[str] = Field(default_factory=list)
    cited_section_ids: list[str] = Field(default_factory=list)


class HistoryDocsQualityJudgmentEnvelope(DomainModel):
    """Permissive H10 judge response model for provider-facing structured output."""

    model_config = ConfigDict(extra="allow")

    rubric_scores: object | None = None
    scores: object | None = None
    coverage: object | None = None
    coherence: object | None = None
    specificity: object | None = None
    algorithm_understanding: object | None = None
    dependency_understanding: object | None = None
    rationale_capture: object | None = None
    present_state_tone: object | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_claim_risks: list[str] = Field(default_factory=list)
    tbd_overuse: bool = False
    evaluator_notes: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class HistoryDocsQualityJudgment(DomainModel):
    """Structured LLM-as-judge response for one rendered checkpoint document."""

    rubric_scores: list[HistoryDocsRubricScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_claim_risks: list[str] = Field(default_factory=list)
    tbd_overuse: bool = False
    evaluator_notes: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rubric_scores(self) -> HistoryDocsQualityJudgment:
        """Require exactly the H10 rubric dimension set."""

        self.rubric_scores = _validate_rubric_scores(self.rubric_scores)
        return self


class HistoryDocsQualityReport(DomainModel):
    """Persisted quality report for one benchmarked rendered checkpoint."""

    case_id: str
    variant_id: str
    checkpoint_id: str
    build_failed: bool = False
    evaluation_status: HistoryDocsQualityEvaluationStatus
    validation_error_count: int = 0
    validation_warning_count: int = 0
    rubric_scores: list[HistoryDocsRubricScore] = Field(default_factory=list)
    overall_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_claim_risks: list[str] = Field(default_factory=list)
    tbd_overuse: bool = False
    evaluator_notes: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    failure_note: str | None = None

    @model_validator(mode="after")
    def validate_rubric_scores(self) -> HistoryDocsQualityReport:
        """Require exactly the H10 rubric dimension set."""

        self.rubric_scores = _validate_rubric_scores(self.rubric_scores)
        return self


class HistoryDocsRubricDelta(DomainModel):
    """Deterministic delta for one rubric dimension between two variants."""

    dimension: HistoryDocsRubricDimension
    baseline_score: int
    candidate_score: int
    delta: int


class HistoryDocsVariantComparison(DomainModel):
    """Deterministic comparison between a baseline and candidate variant."""

    case_id: str
    baseline_variant_id: str
    candidate_variant_id: str
    per_dimension_deltas: list[HistoryDocsRubricDelta] = Field(default_factory=list)
    overall_delta: float = 0.0
    preferred_variant_id: str
    comparison_notes: list[str] = Field(default_factory=list)
    baseline_failed: bool = False
    candidate_failed: bool = False


class HistoryDocsBenchmarkCaseComparisonReport(DomainModel):
    """Per-case comparison artifact for one benchmark suite run."""

    case_id: str
    baseline_variant_id: str
    quality_report_paths: dict[str, Path] = Field(default_factory=dict)
    comparisons: list[HistoryDocsVariantComparison] = Field(default_factory=list)


class HistoryDocsBenchmarkCaseReportRef(DomainModel):
    """Suite-level reference to one benchmark case artifact set."""

    case_id: str
    case_manifest_path: Path
    comparison_report_path: Path
    quality_report_paths: dict[str, Path] = Field(default_factory=dict)


class HistoryDocsBenchmarkSuiteReport(DomainModel):
    """Top-level suite report for one history-docs benchmark run."""

    suite_id: str
    case_ids: list[str] = Field(default_factory=list)
    variant_ids: list[str] = Field(default_factory=list)
    case_reports: list[HistoryDocsBenchmarkCaseReportRef] = Field(default_factory=list)
    average_score_by_variant: dict[str, float] = Field(default_factory=dict)
    failed_evaluation_count: int = 0
    coverage_tags: list[HistoryDocsBenchmarkFocusTag] = Field(default_factory=list)


class HistoryDocsPromotionGateVerdict(DomainModel):
    """Deterministic promotion-gate verdict for one candidate variant."""

    phase_id: str
    candidate_variant_id: str
    baseline_variant_ids: list[str] = Field(default_factory=list)
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)


class HistoryDocsPromotionGateReport(DomainModel):
    """Top-level promotion summary for one real-repo H10 benchmark run."""

    suite_id: str
    provider: str
    model_name: str
    temperature: float
    repo_case_ids: list[str] = Field(default_factory=list)
    average_score_by_variant: dict[str, float] = Field(default_factory=dict)
    win_counts_by_variant: dict[str, int] = Field(default_factory=dict)
    unsupported_claim_risk_totals: dict[str, int] = Field(default_factory=dict)
    failed_build_or_evaluation_count_by_variant: dict[str, int] = Field(
        default_factory=dict
    )
    validation_error_count_by_variant: dict[str, int] = Field(default_factory=dict)
    gate_verdicts: list[HistoryDocsPromotionGateVerdict] = Field(default_factory=list)


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
    semantic_checkpoint_plan_path: Path | None = None
    semantic_structure_map_path: Path | None = None
    semantic_context_map_path: Path | None = None
    snapshot_manifest_path: Path | None = None
    snapshot_structural_model_path: Path | None = None
    interval_delta_model_path: Path | None = None
    interval_interpretation_path: Path | None = None
    checkpoint_model_path: Path | None = None
    checkpoint_model_enrichment_path: Path | None = None
    algorithm_capsule_enrichment_index_path: Path | None = None
    interface_inventory_path: Path | None = None
    dependency_landscape_path: Path | None = None
    dependency_narratives_shadow_path: Path | None = None
    section_outline_path: Path | None = None
    section_outline_llm_path: Path | None = None
    algorithm_capsule_index_path: Path | None = None
    dependencies_artifact_path: Path | None = None
    section_drafts_path: Path | None = None
    checkpoint_draft_markdown_path: Path | None = None
    render_manifest_draft_path: Path | None = None
    validation_report_draft_path: Path | None = None
    draft_review_path: Path | None = None
    section_repairs_path: Path | None = None
    checkpoint_repaired_markdown_path: Path | None = None
    render_manifest_repaired_path: Path | None = None
    validation_report_repaired_path: Path | None = None
    targeted_section_rewrites_path: Path | None = None
    checkpoint_targeted_rewrite_markdown_path: Path | None = None
    render_manifest_targeted_rewrite_path: Path | None = None
    validation_report_targeted_rewrite_path: Path | None = None
    checkpoint_markdown_path: Path | None = None
    render_manifest_path: Path | None = None
    validation_report_path: Path | None = None
    file_count: int = 0
    symbol_count: int = 0
    subsystem_count: int = 0
    build_source_count: int = 0
    semantic_candidate_count: int = 0
    semantic_primary_candidate_count: int = 0
    semantic_planner_status: HistorySemanticCheckpointEvaluationStatus | None = None
    semantic_subsystem_count: int = 0
    semantic_capability_count: int = 0
    semantic_structure_status: HistorySemanticStructureStatus | None = None
    semantic_context_status: HistorySemanticContextStatus | None = None
    interval_interpretation_status: HistoryIntervalInterpretationStatus | None = None
    checkpoint_model_enrichment_status: (
        HistoryCheckpointModelEnrichmentStatus | None
    ) = None
    algorithm_capsule_enrichment_status: (
        HistoryAlgorithmCapsuleEnrichmentStatus | None
    ) = None
    interface_inventory_status: HistoryInterfaceInventoryStatus | None = None
    dependency_landscape_status: HistoryDependencyLandscapeStatus | None = None
    dependency_narrative_count: int = 0
    section_planning_status: HistorySectionPlanningStatus | None = None
    draft_status: HistoryDraftStatus | None = None
    draft_review_status: HistoryDraftReviewStatus | None = None
    repair_status: HistoryRepairStatus | None = None
    context_node_count: int = 0
    interface_candidate_count: int = 0
    subsystem_change_count: int = 0
    interface_change_count: int = 0
    dependency_change_count: int = 0
    algorithm_candidate_count: int = 0
    interval_insight_count: int = 0
    interval_significant_window_count: int = 0
    enriched_subsystem_count: int = 0
    enriched_module_count: int = 0
    capability_proposal_count: int = 0
    design_note_anchor_count: int = 0
    enriched_algorithm_capsule_count: int = 0
    interface_inventory_count: int = 0
    dependency_cluster_count: int = 0
    dependency_usage_pattern_count: int = 0
    subsystem_concept_count: int = 0
    module_concept_count: int = 0
    dependency_concept_count: int = 0
    retired_concept_count: int = 0
    included_section_count: int = 0
    omitted_section_count: int = 0
    algorithm_capsule_count: int = 0
    llm_included_section_count: int = 0
    documented_dependency_count: int = 0
    dependency_warning_count: int = 0
    dependency_summary_failure_count: int = 0
    rendered_section_count: int = 0
    validation_error_count: int = 0
    validation_warning_count: int = 0
    draft_section_count: int = 0
    repaired_section_count: int = 0
    draft_review_finding_count: int = 0
    draft_review_recommended_section_count: int = 0
    draft_validation_error_count: int = 0
    draft_validation_warning_count: int = 0
    repaired_validation_error_count: int = 0
    repaired_validation_warning_count: int = 0
    targeted_rewrite_section_count: int = 0
    targeted_rewrite_validation_error_count: int = 0
    targeted_rewrite_validation_warning_count: int = 0


__all__ = [
    "HistoryAlgorithmAssumption",
    "HistoryAlgorithmCandidate",
    "HistoryAlgorithmCapsule",
    "HistoryAlgorithmCapsuleEnrichment",
    "HistoryAlgorithmCapsuleEnrichmentIndex",
    "HistoryAlgorithmCapsuleEnrichmentIndexEntry",
    "HistoryAlgorithmCapsuleEnrichmentJudgment",
    "HistoryAlgorithmCapsuleEnrichmentStatus",
    "HistoryAlgorithmCapsuleIndex",
    "HistoryAlgorithmCapsuleIndexEntry",
    "HistoryAlgorithmCapsuleStatus",
    "HistoryAlgorithmDataStructure",
    "HistoryAlgorithmInvariant",
    "HistoryAlgorithmPhase",
    "HistoryAlgorithmSharedAbstraction",
    "HistoryAlgorithmTradeoff",
    "HistoryAlgorithmVariantRelationship",
    "HistoryBuildResult",
    "HistoryCommitDelta",
    "HistoryCheckpointModel",
    "HistoryCheckpointModelEnrichment",
    "HistoryCheckpointModelEnrichmentJudgment",
    "HistoryCheckpointModelEnrichmentStatus",
    "HistoryCapabilityConceptProposal",
    "HistoryConceptEnrichmentKind",
    "HistoryConceptChangeStatus",
    "HistoryConceptLifecycleStatus",
    "HistoryDesignChangeInsight",
    "HistoryDesignNoteAnchor",
    "HistoryDocsBenchmarkCase",
    "HistoryDocsBenchmarkCaseComparisonReport",
    "HistoryDocsBenchmarkCaseReportRef",
    "HistoryDocsBenchmarkExpectation",
    "HistoryDocsBenchmarkFocusTag",
    "HistoryDocsPromotionGateReport",
    "HistoryDocsPromotionGateVerdict",
    "HistoryDocsBenchmarkSuiteReport",
    "HistoryDocsLooseRubricScore",
    "HistoryDocsQualityEvaluationStatus",
    "HistoryDocsQualityJudgmentEnvelope",
    "HistoryDocsQualityJudgment",
    "HistoryDocsQualityReport",
    "HistoryDocsRubricDelta",
    "HistoryDocsRubricDimension",
    "HistoryDocsRubricScore",
    "HistoryDocsVariantComparison",
    "HistoryCrossModuleContract",
    "HistoryDependencyCluster",
    "HistoryDependencyDeclaration",
    "HistoryDependencyEntry",
    "HistoryDependencyConcept",
    "HistoryDependencyDescriptionBasis",
    "HistoryDependencyInventory",
    "HistoryDependencyNarrativeRenderStyle",
    "HistoryDependencyNarrativeShadow",
    "HistoryDependencyNarrativeShadowEntry",
    "HistoryDependencyLandscape",
    "HistoryDependencyLandscapeJudgment",
    "HistoryDependencyLandscapeStatus",
    "HistoryDependencyProjectRole",
    "HistoryDependencyRole",
    "HistoryDependencySectionTarget",
    "HistoryDependencySourceKind",
    "HistoryDependencySummary",
    "HistoryDependencySummaryStatus",
    "HistoryDependencyUsagePattern",
    "HistoryDependencyWarning",
    "HistoryDraftReview",
    "HistoryDraftReviewFinding",
    "HistoryDraftReviewFindingKind",
    "HistoryDraftReviewJudgment",
    "HistoryDraftReviewSeverity",
    "HistoryDraftReviewStatus",
    "HistoryDraftStatus",
    "HistoryEvidenceKind",
    "HistoryEvidenceLink",
    "HistoryDeltaEvidenceLink",
    "HistoryDependencyChangeCandidate",
    "HistoryInterfaceChangeCandidate",
    "HistoryInterfaceConcept",
    "HistoryInterfaceInventory",
    "HistoryInterfaceInventoryJudgment",
    "HistoryInterfaceInventoryStatus",
    "HistoryInterfaceResponsibility",
    "HistoryIntervalDeltaModel",
    "HistoryIntervalInsightKind",
    "HistoryIntervalInterpretation",
    "HistoryIntervalInterpretationJudgment",
    "HistoryIntervalInterpretationStatus",
    "HistoryIntervalSignificance",
    "HistoryLLMSectionOutline",
    "HistoryLLMSectionOutlineJudgment",
    "HistoryLLMSectionPlanDecision",
    "HistoryModuleConceptEnrichment",
    "HistoryModuleConcept",
    "HistoryRationaleClue",
    "HistoryRationaleClueSourceKind",
    "HistoryRenderManifest",
    "HistoryRenderedSection",
    "HistoryRepairStatus",
    "HistorySemanticCheckpointCandidate",
    "HistorySemanticCheckpointEvaluationStatus",
    "HistorySemanticCheckpointJudgment",
    "HistorySemanticCheckpointJudgmentBatch",
    "HistorySemanticCheckpointPlan",
    "HistorySemanticCheckpointRecommendation",
    "HistorySemanticCheckpointSignalKind",
    "HistorySemanticContextJudgment",
    "HistorySemanticContextJudgmentInterface",
    "HistorySemanticContextJudgmentNode",
    "HistorySemanticContextMap",
    "HistorySemanticContextStatus",
    "HistorySemanticInterfaceCandidate",
    "HistorySemanticInterfaceKind",
    "HistorySemanticStructureJudgment",
    "HistorySemanticStructureJudgmentCapability",
    "HistorySemanticStructureJudgmentSubsystem",
    "HistorySemanticStructureMap",
    "HistorySemanticStructureStatus",
    "HistorySemanticSubsystemCluster",
    "HistorySemanticCapabilityCluster",
    "HistorySignificantChangeWindow",
    "HistorySystemContextNode",
    "HistorySystemContextNodeKind",
    "HistorySectionDepth",
    "HistorySectionDraft",
    "HistorySectionDraftArtifact",
    "HistorySectionDraftJudgment",
    "HistorySectionId",
    "HistorySectionKind",
    "HistorySectionOutline",
    "HistorySectionPlanningStatus",
    "HistorySectionPlan",
    "HistorySectionPlanId",
    "HistorySectionPlanStatus",
    "HistorySectionRepair",
    "HistorySectionRepairArtifact",
    "HistorySectionRepairJudgment",
    "HistorySectionSignalKind",
    "HistorySectionState",
    "HistorySnapshotStructuralModel",
    "HistorySubsystemConcept",
    "HistorySubsystemConceptEnrichment",
    "HistorySubsystemChangeCandidate",
    "HistorySubsystemCandidate",
    "HistoryValidationCheckId",
    "HistoryValidationFinding",
    "HistoryValidationReport",
    "HistoryValidationSeverity",
]
