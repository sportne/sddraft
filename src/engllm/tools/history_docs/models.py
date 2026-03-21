"""History-docs tool model namespace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

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
    evaluation_status: HistoryDocsQualityEvaluationStatus
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
    snapshot_manifest_path: Path | None = None
    snapshot_structural_model_path: Path | None = None
    interval_delta_model_path: Path | None = None
    checkpoint_model_path: Path | None = None
    section_outline_path: Path | None = None
    algorithm_capsule_index_path: Path | None = None
    dependencies_artifact_path: Path | None = None
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
    documented_dependency_count: int = 0
    dependency_warning_count: int = 0
    dependency_summary_failure_count: int = 0
    rendered_section_count: int = 0
    validation_error_count: int = 0
    validation_warning_count: int = 0


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
    "HistoryDocsBenchmarkCase",
    "HistoryDocsBenchmarkCaseComparisonReport",
    "HistoryDocsBenchmarkCaseReportRef",
    "HistoryDocsBenchmarkExpectation",
    "HistoryDocsBenchmarkFocusTag",
    "HistoryDocsBenchmarkSuiteReport",
    "HistoryDocsQualityEvaluationStatus",
    "HistoryDocsQualityJudgment",
    "HistoryDocsQualityReport",
    "HistoryDocsRubricDelta",
    "HistoryDocsRubricDimension",
    "HistoryDocsRubricScore",
    "HistoryDocsVariantComparison",
    "HistoryDependencyDeclaration",
    "HistoryDependencyEntry",
    "HistoryDependencyConcept",
    "HistoryDependencyInventory",
    "HistoryDependencyRole",
    "HistoryDependencySectionTarget",
    "HistoryDependencySourceKind",
    "HistoryDependencySummary",
    "HistoryDependencySummaryStatus",
    "HistoryDependencyWarning",
    "HistoryEvidenceKind",
    "HistoryEvidenceLink",
    "HistoryDeltaEvidenceLink",
    "HistoryDependencyChangeCandidate",
    "HistoryInterfaceChangeCandidate",
    "HistoryIntervalDeltaModel",
    "HistoryModuleConcept",
    "HistoryRenderManifest",
    "HistoryRenderedSection",
    "HistorySemanticCheckpointCandidate",
    "HistorySemanticCheckpointEvaluationStatus",
    "HistorySemanticCheckpointJudgment",
    "HistorySemanticCheckpointJudgmentBatch",
    "HistorySemanticCheckpointPlan",
    "HistorySemanticCheckpointRecommendation",
    "HistorySemanticCheckpointSignalKind",
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
    "HistoryValidationCheckId",
    "HistoryValidationFinding",
    "HistoryValidationReport",
    "HistoryValidationSeverity",
]
