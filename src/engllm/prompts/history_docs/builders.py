"""Deterministic prompt builders for the history-docs tool."""

from __future__ import annotations

import json

from engllm.prompts.history_docs.templates import (
    ALGORITHM_CAPSULE_ENRICHMENT_SYSTEM_PROMPT,
    CHECKPOINT_MODEL_ENRICHMENT_SYSTEM_PROMPT,
    DEPENDENCY_LANDSCAPE_SYSTEM_PROMPT,
    DEPENDENCY_SUMMARY_SYSTEM_PROMPT,
    DRAFT_REVIEW_SYSTEM_PROMPT,
    HISTORY_DOCS_QUALITY_JUDGE_SYSTEM_PROMPT,
    INTERFACE_INVENTORY_SYSTEM_PROMPT,
    INTERVAL_INTERPRETATION_SYSTEM_PROMPT,
    SECTION_DRAFTING_SYSTEM_PROMPT,
    SECTION_PLANNING_LLM_SYSTEM_PROMPT,
    SECTION_REPAIR_SYSTEM_PROMPT,
    SEMANTIC_CHECKPOINT_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_CONTEXT_SYSTEM_PROMPT,
    SEMANTIC_STRUCTURE_SYSTEM_PROMPT,
)
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsuleEnrichment,
    HistoryAlgorithmCapsuleEnrichmentIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyEntry,
    HistoryDependencyLandscape,
    HistoryDependencyNarrativeShadow,
    HistoryDocsBenchmarkCase,
    HistoryInterfaceInventory,
    HistoryLLMSectionOutline,
    HistoryRenderManifest,
    HistorySectionDraftArtifact,
    HistorySemanticContextMap,
    HistoryValidationReport,
)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_dependency_summary_prompt(
    entry: HistoryDependencyEntry,
) -> tuple[str, str]:
    """Return prompts for one dependency summary generation request."""

    user_prompt = (
        "Document this direct dependency using only the provided evidence.\n"
        f"Dependency Evidence:\n{_json(entry.model_dump(mode='json'))}\n"
    )
    return DEPENDENCY_SUMMARY_SYSTEM_PROMPT, user_prompt


def build_algorithm_capsule_enrichment_prompt(
    *,
    checkpoint_summary: dict[str, object],
    capsules: list[dict[str, object]],
    design_note_anchors: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H13-01 algorithm-capsule enrichment request."""

    user_prompt = (
        "Enrich the listed algorithm capsules using only the supplied structured evidence.\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Algorithm Capsules:\n{_json(capsules)}\n"
        f"Design Note Anchors:\n{_json(design_note_anchors)}\n"
    )
    return ALGORITHM_CAPSULE_ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_checkpoint_model_enrichment_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    subsystems: list[dict[str, object]],
    modules: list[dict[str, object]],
    interval_interpretation: dict[str, object],
    semantic_labels: dict[str, list[dict[str, object]]],
) -> tuple[str, str]:
    """Return prompts for one H12-02 checkpoint-model enrichment request."""

    user_prompt = (
        "Enrich the checkpoint model using only the supplied structured evidence.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Subsystem Concepts:\n{_json(subsystems)}\n"
        f"Module Concepts:\n{_json(modules)}\n"
        f"Interval Interpretation:\n{_json(interval_interpretation)}\n"
        f"Semantic Labels:\n{_json(semantic_labels)}\n"
    )
    return CHECKPOINT_MODEL_ENRICHMENT_SYSTEM_PROMPT, user_prompt


def build_interface_inventory_prompt(
    *,
    checkpoint_summary: dict[str, object],
    interface_context: dict[str, object],
    design_note_anchors: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H13-02 interface-inventory request."""

    user_prompt = (
        "Build a richer interface inventory using only the supplied structured evidence.\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Interface Context:\n{_json(interface_context)}\n"
        f"Design Note Anchors:\n{_json(design_note_anchors)}\n"
    )
    return INTERFACE_INVENTORY_SYSTEM_PROMPT, user_prompt


def build_dependency_landscape_prompt(
    *,
    checkpoint_summary: dict[str, object],
    dependency_context: dict[str, object],
    design_note_anchors: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H13-03 dependency-landscape request."""

    user_prompt = (
        "Build a project-level dependency landscape using only the supplied structured evidence.\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Dependency Context:\n{_json(dependency_context)}\n"
        f"Design Note Anchors:\n{_json(design_note_anchors)}\n"
    )
    return DEPENDENCY_LANDSCAPE_SYSTEM_PROMPT, user_prompt


def build_section_drafting_prompt(
    *,
    checkpoint_context: dict[str, object],
    section_metadata: dict[str, object],
    supporting_evidence: dict[str, object],
) -> tuple[str, str]:
    """Return prompts for one H14-01 section drafting request."""

    user_prompt = (
        "Draft one checkpoint documentation section using only the supplied "
        "structured evidence.\n"
        f"Checkpoint Context:\n{_json(checkpoint_context)}\n"
        f"Section Metadata:\n{_json(section_metadata)}\n"
        f"Supporting Evidence:\n{_json(supporting_evidence)}\n"
    )
    return SECTION_DRAFTING_SYSTEM_PROMPT, user_prompt


def build_draft_review_prompt(
    *,
    checkpoint_context: dict[str, object],
    draft_summary: dict[str, object],
    validation_summary: dict[str, object],
    evidence_summary: dict[str, object],
    markdown: str,
) -> tuple[str, str]:
    """Return prompts for one H14-02 draft review request."""

    user_prompt = (
        "Review this drafted checkpoint document using only the supplied "
        "structured evidence.\n"
        f"Checkpoint Context:\n{_json(checkpoint_context)}\n"
        f"Draft Summary:\n{_json(draft_summary)}\n"
        f"Validation Summary:\n{_json(validation_summary)}\n"
        f"Evidence Summary:\n{_json(evidence_summary)}\n"
        "Draft Markdown:\n"
        f"{markdown}\n"
    )
    return DRAFT_REVIEW_SYSTEM_PROMPT, user_prompt


def build_section_repair_prompt(
    *,
    checkpoint_context: dict[str, object],
    section_metadata: dict[str, object],
    findings: list[dict[str, object]],
    supporting_evidence: dict[str, object],
    original_markdown_body: str,
) -> tuple[str, str]:
    """Return prompts for one H14-03 section repair request."""

    user_prompt = (
        "Repair one drafted checkpoint documentation section using only the "
        "supplied structured evidence.\n"
        f"Checkpoint Context:\n{_json(checkpoint_context)}\n"
        f"Section Metadata:\n{_json(section_metadata)}\n"
        f"Section Findings:\n{_json(findings)}\n"
        f"Supporting Evidence:\n{_json(supporting_evidence)}\n"
        "Original Draft Body:\n"
        f"{original_markdown_body}\n"
    )
    return SECTION_REPAIR_SYSTEM_PROMPT, user_prompt


def build_history_docs_quality_judge_prompt(
    *,
    case: HistoryDocsBenchmarkCase,
    markdown: str,
    render_manifest: HistoryRenderManifest,
    validation_report: HistoryValidationReport,
    checkpoint_model: HistoryCheckpointModel,
    semantic_context_map: HistorySemanticContextMap | None = None,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment | None = None,
    llm_section_outline: HistoryLLMSectionOutline | None = None,
    algorithm_capsule_enrichment_index: (
        HistoryAlgorithmCapsuleEnrichmentIndex | None
    ) = None,
    algorithm_capsule_enrichments: (
        list[HistoryAlgorithmCapsuleEnrichment] | None
    ) = None,
    interface_inventory: HistoryInterfaceInventory | None = None,
    dependency_landscape: HistoryDependencyLandscape | None = None,
    dependency_narratives_shadow: HistoryDependencyNarrativeShadow | None = None,
    section_drafts: HistorySectionDraftArtifact | None = None,
    targeted_section_rewrites: HistorySectionDraftArtifact | None = None,
    draft_validation_report: HistoryValidationReport | None = None,
    repaired_validation_report: HistoryValidationReport | None = None,
    targeted_rewrite_validation_report: HistoryValidationReport | None = None,
    draft_review_summary: dict[str, object] | None = None,
    repaired_section_ids: list[str] | None = None,
) -> tuple[str, str]:
    """Return prompts for one H10 single-document quality evaluation."""

    active_subsystems = [
        subsystem
        for subsystem in checkpoint_model.subsystems
        if subsystem.lifecycle_status == "active"
    ]
    checkpoint_summary = {
        "checkpoint_id": checkpoint_model.checkpoint_id,
        "subsystem_count": len(checkpoint_model.subsystems),
        "module_count": len(checkpoint_model.modules),
        "dependency_concept_count": len(checkpoint_model.dependencies),
        "algorithm_capsule_count": len(checkpoint_model.algorithm_capsule_ids),
        "rendered_section_count": len(render_manifest.sections),
        "validation_error_count": validation_report.error_count,
        "validation_warning_count": validation_report.warning_count,
    }
    structure_summary = [
        {
            "subsystem_id": subsystem.concept_id,
            "display_name": subsystem.display_name or subsystem.group_path.as_posix(),
            "module_count": len(subsystem.module_ids),
            "capability_labels": subsystem.capability_labels,
            "baseline_subsystem_ids": subsystem.baseline_subsystem_ids,
        }
        for subsystem in active_subsystems
    ]
    render_sections = [
        {
            "section_id": section.section_id,
            "title": section.title,
            "kind": section.kind,
            "dependency_ids": section.dependency_ids,
            "algorithm_capsule_ids": section.algorithm_capsule_ids,
        }
        for section in render_manifest.sections
    ]
    context_summary = (
        []
        if semantic_context_map is None
        else [
            {
                "node_id": node.node_id,
                "title": node.title,
                "kind": node.kind,
                "related_subsystem_ids": node.related_subsystem_ids,
                "related_module_count": len(node.related_module_ids),
            }
            for node in semantic_context_map.context_nodes
        ]
    )
    interface_summary = (
        []
        if semantic_context_map is None
        else [
            {
                "interface_id": interface.interface_id,
                "title": interface.title,
                "kind": interface.kind,
                "provider_subsystem_ids": interface.provider_subsystem_ids,
                "related_module_count": len(interface.related_module_ids),
            }
            for interface in semantic_context_map.interfaces
        ]
    )
    enrichment_summary = (
        {"subsystems": [], "modules": [], "design_note_anchor_titles": []}
        if checkpoint_model_enrichment is None
        else {
            "subsystems": [
                {
                    "concept_id": enrichment.concept_id,
                    "display_name": enrichment.display_name,
                    "capability_labels": enrichment.capability_labels,
                }
                for enrichment in checkpoint_model_enrichment.subsystem_enrichments
            ],
            "modules": [
                {
                    "concept_id": enrichment.concept_id,
                    "summary": enrichment.summary,
                    "responsibility_labels": enrichment.responsibility_labels,
                }
                for enrichment in checkpoint_model_enrichment.module_enrichments
            ],
            "design_note_anchor_titles": [
                anchor.title
                for anchor in checkpoint_model_enrichment.design_note_anchors
            ],
        }
    )
    section_planning_summary: dict[str, object] = (
        {
            "evaluation_status": None,
            "included_section_ids": [],
            "omitted_section_ids": [],
            "depth_by_section": {},
            "rationale_snippets": [],
        }
        if llm_section_outline is None
        else {
            "evaluation_status": llm_section_outline.evaluation_status,
            "included_section_ids": [
                section.section_id
                for section in llm_section_outline.sections
                if section.status == "included"
            ],
            "omitted_section_ids": [
                section.section_id
                for section in llm_section_outline.sections
                if section.status == "omitted"
            ],
            "depth_by_section": {
                section.section_id: section.depth
                for section in llm_section_outline.sections
                if section.depth is not None
            },
            "rationale_snippets": [
                {
                    "section_id": section.section_id,
                    "planning_rationale": section.planning_rationale,
                }
                for section in llm_section_outline.sections
                if section.planning_rationale is not None
            ][:6],
        }
    )
    algorithm_enrichment_summary: dict[str, object] = (
        {"evaluation_status": None, "capsules": []}
        if algorithm_capsule_enrichment_index is None
        or algorithm_capsule_enrichments is None
        else {
            "evaluation_status": algorithm_capsule_enrichment_index.evaluation_status,
            "capsules": [
                {
                    "capsule_id": enrichment.capsule_id,
                    "purpose": enrichment.purpose,
                    "phase_flow_summary": enrichment.phase_flow_summary,
                    "invariant_count": len(enrichment.invariants),
                    "tradeoff_count": len(enrichment.tradeoffs),
                    "variant_relationship_count": len(enrichment.variant_relationships),
                }
                for enrichment in algorithm_capsule_enrichments
            ],
        }
    )
    interface_inventory_summary: dict[str, object] = (
        {"evaluation_status": None, "interfaces": []}
        if interface_inventory is None
        else {
            "evaluation_status": interface_inventory.evaluation_status,
            "interfaces": [
                {
                    "interface_id": interface.interface_id,
                    "title": interface.title,
                    "kind": interface.kind,
                    "provider_subsystem_ids": interface.provider_subsystem_ids,
                    "responsibility_titles": [
                        responsibility.title
                        for responsibility in interface.responsibilities
                    ],
                    "contract_titles": [
                        contract.title for contract in interface.cross_module_contracts
                    ],
                }
                for interface in interface_inventory.interfaces
            ],
        }
    )
    dependency_landscape_summary: dict[str, object] = (
        {
            "evaluation_status": None,
            "project_roles": [],
            "clusters": [],
            "usage_patterns": [],
        }
        if dependency_landscape is None
        else {
            "evaluation_status": dependency_landscape.evaluation_status,
            "project_roles": [
                {
                    "role_id": role.role_id,
                    "title": role.title,
                    "dependency_count": len(role.dependency_ids),
                    "related_subsystem_ids": role.related_subsystem_ids,
                }
                for role in dependency_landscape.project_roles
            ],
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "title": cluster.title,
                    "dependency_count": len(cluster.dependency_ids),
                    "ecosystems": cluster.ecosystems,
                    "scope_roles": cluster.scope_roles,
                }
                for cluster in dependency_landscape.clusters
            ],
            "usage_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "title": pattern.title,
                    "dependency_count": len(pattern.dependency_ids),
                    "related_subsystem_ids": pattern.related_subsystem_ids,
                }
                for pattern in dependency_landscape.usage_patterns
            ],
        }
    )
    dependency_narrative_summary: dict[str, object] = (
        {"entry_count": 0, "grouped_tooling_count": 0, "general_basis_counts": {}}
        if dependency_narratives_shadow is None
        else {
            "entry_count": len(dependency_narratives_shadow.entries),
            "grouped_tooling_count": sum(
                entry.render_style == "grouped_tooling"
                for entry in dependency_narratives_shadow.entries
            ),
            "general_basis_counts": {
                basis: sum(
                    entry.general_description_basis == basis
                    for entry in dependency_narratives_shadow.entries
                )
                for basis in ("package_general_knowledge", "project_evidence", "tbd")
            },
        }
    )
    draft_summary: dict[str, object] = (
        {"evaluation_status": None, "section_count": 0, "section_ids": []}
        if section_drafts is None
        else {
            "evaluation_status": section_drafts.evaluation_status,
            "section_count": len(section_drafts.sections),
            "section_ids": [section.section_id for section in section_drafts.sections],
        }
    )
    targeted_rewrite_summary: dict[str, object] = (
        {"evaluation_status": None, "section_count": 0, "section_ids": []}
        if targeted_section_rewrites is None
        else {
            "evaluation_status": targeted_section_rewrites.evaluation_status,
            "section_count": len(targeted_section_rewrites.sections),
            "section_ids": [
                section.section_id for section in targeted_section_rewrites.sections
            ],
        }
    )
    draft_validation_summary: dict[str, int | None] = (
        {"error_count": None, "warning_count": None}
        if draft_validation_report is None
        else {
            "error_count": draft_validation_report.error_count,
            "warning_count": draft_validation_report.warning_count,
        }
    )
    repaired_validation_summary: dict[str, int | None] = (
        {"error_count": None, "warning_count": None}
        if repaired_validation_report is None
        else {
            "error_count": repaired_validation_report.error_count,
            "warning_count": repaired_validation_report.warning_count,
        }
    )
    targeted_rewrite_validation_summary: dict[str, int | None] = (
        {"error_count": None, "warning_count": None}
        if targeted_rewrite_validation_report is None
        else {
            "error_count": targeted_rewrite_validation_report.error_count,
            "warning_count": targeted_rewrite_validation_report.warning_count,
        }
    )
    draft_review_compact = draft_review_summary or {
        "evaluation_status": None,
        "finding_counts_by_kind": {},
        "finding_counts_by_severity": {},
        "recommended_repair_section_ids": [],
    }
    validation_summary = {
        "error_count": validation_report.error_count,
        "warning_count": validation_report.warning_count,
        "findings": [
            {
                "check_id": finding.check_id,
                "severity": finding.severity,
                "section_id": finding.section_id,
                "reference": finding.reference,
            }
            for finding in validation_report.findings
        ],
    }
    user_prompt = (
        "Evaluate this rendered history-docs checkpoint document against the "
        "benchmark expectations.\n"
        f"Benchmark Case:\n{_json(case.model_dump(mode='json'))}\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Structure Summary:\n{_json(structure_summary)}\n"
        f"Context Summary:\n{_json(context_summary)}\n"
        f"Interface Summary:\n{_json(interface_summary)}\n"
        f"Model Enrichment Summary:\n{_json(enrichment_summary)}\n"
        f"Section Planning Summary:\n{_json(section_planning_summary)}\n"
        f"Algorithm Enrichment Summary:\n{_json(algorithm_enrichment_summary)}\n"
        f"Interface Inventory Summary:\n{_json(interface_inventory_summary)}\n"
        f"Dependency Landscape Summary:\n{_json(dependency_landscape_summary)}\n"
        f"Dependency Narrative Summary:\n{_json(dependency_narrative_summary)}\n"
        f"Draft Summary:\n{_json(draft_summary)}\n"
        f"Targeted Rewrite Summary:\n{_json(targeted_rewrite_summary)}\n"
        f"Draft Validation Summary:\n{_json(draft_validation_summary)}\n"
        f"Draft Review Summary:\n{_json(draft_review_compact)}\n"
        f"Repaired Validation Summary:\n{_json(repaired_validation_summary)}\n"
        f"Targeted Rewrite Validation Summary:\n{_json(targeted_rewrite_validation_summary)}\n"
        f"Repaired Section IDs:\n{_json(repaired_section_ids or [])}\n"
        f"Rendered Sections:\n{_json(render_sections)}\n"
        f"Validation Summary:\n{_json(validation_summary)}\n"
        "Checkpoint Markdown:\n"
        f"{markdown}\n"
    )
    return HISTORY_DOCS_QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt


def build_semantic_structure_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    baseline_subsystems: list[dict[str, object]],
    modules: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H11 semantic subsystem/capability clustering request."""

    user_prompt = (
        "Cluster the snapshot modules into semantic subsystems and capability labels "
        "using only the supplied evidence.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Baseline Path-Based Subsystems:\n{_json(baseline_subsystems)}\n"
        f"Active Modules:\n{_json(modules)}\n"
    )
    return SEMANTIC_STRUCTURE_SYSTEM_PROMPT, user_prompt


def build_semantic_context_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    semantic_subsystems: list[dict[str, object]],
    modules: list[dict[str, object]],
    build_sources: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H11-03 semantic context and interface extraction request."""

    user_prompt = (
        "Extract a semantic system context and interface candidates using only the "
        "supplied evidence.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Semantic Subsystems:\n{_json(semantic_subsystems)}\n"
        f"Active Modules:\n{_json(modules)}\n"
        f"Build Sources:\n{_json(build_sources)}\n"
    )
    return SEMANTIC_CONTEXT_SYSTEM_PROMPT, user_prompt


def build_semantic_checkpoint_planner_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    built_checkpoints: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> tuple[str, str]:
    """Return prompts for one H11 semantic checkpoint-planning request."""

    user_prompt = (
        "Recommend meaningful semantic checkpoint anchors using only the supplied "
        "candidate commits.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Existing Built Checkpoints:\n{_json(built_checkpoints)}\n"
        f"Deterministic Candidate Commits:\n{_json(candidates)}\n"
    )
    return SEMANTIC_CHECKPOINT_PLANNER_SYSTEM_PROMPT, user_prompt


def build_interval_interpretation_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    commit_deltas: list[dict[str, object]],
    candidates: dict[str, list[dict[str, object]]],
    modules: list[dict[str, object]],
    semantic_labels: dict[str, list[dict[str, object]]],
) -> tuple[str, str]:
    """Return prompts for one H12 interval interpretation request."""

    user_prompt = (
        "Interpret this checkpoint interval using only the supplied structured "
        "history evidence.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Commit Deltas:\n{_json(commit_deltas)}\n"
        f"Interval Change Candidates:\n{_json(candidates)}\n"
        f"Snapshot Module Labels:\n{_json(modules)}\n"
        f"Semantic Labels:\n{_json(semantic_labels)}\n"
    )
    return INTERVAL_INTERPRETATION_SYSTEM_PROMPT, user_prompt


def build_section_planning_llm_prompt(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    section_scaffold: list[dict[str, object]],
    checkpoint_summary: dict[str, object],
    interval_interpretation: dict[str, object],
    checkpoint_model_enrichment: dict[str, object],
    semantic_context_summary: dict[str, object],
) -> tuple[str, str]:
    """Return prompts for one H12-03 shadow section-planning request."""

    user_prompt = (
        "Plan checkpoint documentation sections using only the supplied "
        "structured evidence.\n"
        f"Checkpoint Context:\n{_json({'checkpoint_id': checkpoint_id, 'target_commit': target_commit, 'previous_checkpoint_commit': previous_checkpoint_commit})}\n"
        f"Deterministic Section Scaffold:\n{_json(section_scaffold)}\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Interval Interpretation:\n{_json(interval_interpretation)}\n"
        f"Checkpoint Model Enrichment:\n{_json(checkpoint_model_enrichment)}\n"
        f"Semantic Context Summary:\n{_json(semantic_context_summary)}\n"
    )
    return SECTION_PLANNING_LLM_SYSTEM_PROMPT, user_prompt
