"""Deterministic prompt builders for the history-docs tool."""

from __future__ import annotations

import json

from engllm.prompts.history_docs.templates import (
    CHECKPOINT_MODEL_ENRICHMENT_SYSTEM_PROMPT,
    DEPENDENCY_SUMMARY_SYSTEM_PROMPT,
    HISTORY_DOCS_QUALITY_JUDGE_SYSTEM_PROMPT,
    INTERVAL_INTERPRETATION_SYSTEM_PROMPT,
    SECTION_PLANNING_LLM_SYSTEM_PROMPT,
    SEMANTIC_CHECKPOINT_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_CONTEXT_SYSTEM_PROMPT,
    SEMANTIC_STRUCTURE_SYSTEM_PROMPT,
)
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyEntry,
    HistoryDocsBenchmarkCase,
    HistoryLLMSectionOutline,
    HistoryRenderManifest,
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
