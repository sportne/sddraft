"""H12-03 shadow LLM section-planning helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_section_planning_llm_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryEvidenceLink,
    HistoryIntervalInterpretation,
    HistoryLLMSectionOutline,
    HistoryLLMSectionOutlineJudgment,
    HistoryLLMSectionPlanDecision,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySectionPlanningStatus,
    HistorySemanticContextMap,
)
from engllm.tools.history_docs.semantic_context import (
    augment_section_outline_with_semantic_context,
)

_STABLE_CORE_SECTION_IDS: tuple[HistorySectionPlanId, ...] = (
    "introduction",
    "architectural_overview",
    "subsystems_modules",
    "dependencies",
    "build_development_infrastructure",
)


def section_outline_llm_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H12-03 shadow section-planning artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "section_outline_llm.json"


def build_section_planning_scaffold(
    *,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    semantic_context_map: HistorySemanticContextMap | None,
    include_semantic_context: bool,
) -> HistorySectionOutline:
    """Return the unified deterministic planning scaffold for H12-03."""

    if not include_semantic_context or semantic_context_map is None:
        return section_outline.model_copy(deep=True)
    return augment_section_outline_with_semantic_context(
        checkpoint_model=checkpoint_model,
        section_outline=section_outline,
        semantic_context_map=semantic_context_map,
    )


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(
    *groups: Iterable[HistoryEvidenceLink],
) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _checkpoint_summary(
    checkpoint_model: HistoryCheckpointModel,
) -> dict[str, object]:
    active_subsystems = [
        subsystem
        for subsystem in checkpoint_model.subsystems
        if subsystem.lifecycle_status == "active"
    ]
    active_modules = [
        module
        for module in checkpoint_model.modules
        if module.lifecycle_status == "active"
    ]
    active_dependencies = [
        dependency
        for dependency in checkpoint_model.dependencies
        if dependency.lifecycle_status == "active"
    ]
    return {
        "checkpoint_id": checkpoint_model.checkpoint_id,
        "target_commit": checkpoint_model.target_commit,
        "previous_checkpoint_commit": checkpoint_model.previous_checkpoint_commit,
        "active_subsystem_count": len(active_subsystems),
        "active_module_count": len(active_modules),
        "active_dependency_count": len(active_dependencies),
        "algorithm_capsule_count": len(checkpoint_model.algorithm_capsule_ids),
        "subsystems": [
            {
                "concept_id": subsystem.concept_id,
                "display_name": subsystem.display_name,
                "summary": subsystem.summary,
                "module_count": len(subsystem.module_ids),
                "capability_labels": subsystem.capability_labels,
            }
            for subsystem in sorted(
                active_subsystems,
                key=lambda item: item.concept_id,
            )[:12]
        ],
        "modules": [
            {
                "concept_id": module.concept_id,
                "path": module.path.as_posix(),
                "summary": module.summary,
                "responsibility_labels": module.responsibility_labels,
            }
            for module in sorted(
                active_modules,
                key=lambda item: item.path.as_posix(),
            )[:24]
        ],
    }


def _scaffold_payload(
    section_outline: HistorySectionOutline,
) -> list[dict[str, object]]:
    return [
        {
            "section_id": section.section_id,
            "title": section.title,
            "kind": section.kind,
            "status": section.status,
            "confidence_score": section.confidence_score,
            "evidence_score": section.evidence_score,
            "depth": section.depth,
            "concept_ids": section.concept_ids,
            "algorithm_capsule_ids": section.algorithm_capsule_ids,
            "trigger_signals": section.trigger_signals,
            "evidence_links": [
                {
                    "kind": link.kind,
                    "reference": link.reference,
                    "detail": link.detail,
                }
                for link in section.evidence_links
            ],
        }
        for section in section_outline.sections
    ]


def _interval_payload(
    interval_interpretation: HistoryIntervalInterpretation,
) -> dict[str, object]:
    return {
        "insights": [
            {
                "insight_id": insight.insight_id,
                "kind": insight.kind,
                "title": insight.title,
                "summary": insight.summary,
                "significance": insight.significance,
                "related_change_ids": insight.related_change_ids,
                "related_subsystem_ids": insight.related_subsystem_ids,
            }
            for insight in sorted(
                interval_interpretation.insights,
                key=lambda item: item.insight_id,
            )[:16]
        ],
        "rationale_clues": [
            {
                "clue_id": clue.clue_id,
                "text": clue.text,
                "confidence": clue.confidence,
                "source_kind": clue.source_kind,
                "related_change_ids": clue.related_change_ids,
            }
            for clue in sorted(
                interval_interpretation.rationale_clues,
                key=lambda item: item.clue_id,
            )[:16]
        ],
        "significant_windows": [
            {
                "window_id": window.window_id,
                "title": window.title,
                "summary": window.summary,
                "significance": window.significance,
                "related_insight_ids": window.related_insight_ids,
            }
            for window in sorted(
                interval_interpretation.significant_windows,
                key=lambda item: item.window_id,
            )[:8]
        ],
    }


def _enrichment_payload(
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
) -> dict[str, object]:
    return {
        "subsystem_enrichments": [
            {
                "concept_id": enrichment.concept_id,
                "display_name": enrichment.display_name,
                "summary": enrichment.summary,
                "capability_labels": enrichment.capability_labels,
                "source_insight_ids": enrichment.source_insight_ids,
                "source_rationale_clue_ids": enrichment.source_rationale_clue_ids,
            }
            for enrichment in checkpoint_model_enrichment.subsystem_enrichments
        ],
        "module_enrichments": [
            {
                "concept_id": enrichment.concept_id,
                "summary": enrichment.summary,
                "responsibility_labels": enrichment.responsibility_labels,
                "source_insight_ids": enrichment.source_insight_ids,
                "source_rationale_clue_ids": enrichment.source_rationale_clue_ids,
            }
            for enrichment in checkpoint_model_enrichment.module_enrichments
        ],
        "capability_proposals": [
            {
                "capability_id": proposal.capability_id,
                "title": proposal.title,
                "summary": proposal.summary,
                "related_subsystem_ids": proposal.related_subsystem_ids,
                "related_module_ids": proposal.related_module_ids,
                "source_insight_ids": proposal.source_insight_ids,
            }
            for proposal in checkpoint_model_enrichment.capability_proposals
        ],
        "design_note_anchors": [
            {
                "note_id": anchor.note_id,
                "title": anchor.title,
                "summary": anchor.summary,
                "related_concept_ids": anchor.related_concept_ids,
                "source_insight_ids": anchor.source_insight_ids,
                "source_rationale_clue_ids": anchor.source_rationale_clue_ids,
            }
            for anchor in checkpoint_model_enrichment.design_note_anchors
        ],
    }


def _semantic_context_payload(
    semantic_context_map: HistorySemanticContextMap | None,
) -> dict[str, object]:
    if semantic_context_map is None:
        return {
            "context_node_count": 0,
            "interface_count": 0,
            "context_nodes": [],
            "interfaces": [],
        }
    return {
        "context_node_count": len(semantic_context_map.context_nodes),
        "interface_count": len(semantic_context_map.interfaces),
        "context_nodes": [
            {
                "node_id": node.node_id,
                "title": node.title,
                "kind": node.kind,
            }
            for node in semantic_context_map.context_nodes
        ],
        "interfaces": [
            {
                "interface_id": interface.interface_id,
                "title": interface.title,
                "kind": interface.kind,
                "provider_subsystem_ids": interface.provider_subsystem_ids,
            }
            for interface in semantic_context_map.interfaces
        ],
    }


def _fallback_outline(
    *,
    scaffold: HistorySectionOutline,
    evaluation_status: HistorySectionPlanningStatus,
    reason: str,
) -> HistoryLLMSectionOutline:
    return HistoryLLMSectionOutline(
        checkpoint_id=scaffold.checkpoint_id,
        target_commit=scaffold.target_commit,
        previous_checkpoint_commit=scaffold.previous_checkpoint_commit,
        evaluation_status=evaluation_status,
        sections=[
            section.model_copy(
                update={
                    "planning_rationale": reason,
                    "source_insight_ids": [],
                    "source_capability_ids": [],
                    "source_design_note_ids": [],
                }
            )
            for section in scaffold.sections
        ],
    )


def _validate_known_reference_ids(
    *,
    decisions: HistoryLLMSectionOutlineJudgment,
    scaffold: HistorySectionOutline,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
) -> dict[HistorySectionPlanId, HistoryLLMSectionPlanDecision]:
    scaffold_ids = [section.section_id for section in scaffold.sections]
    decision_ids = [decision.section_id for decision in decisions.sections]
    if set(decision_ids) != set(scaffold_ids) or len(decision_ids) != len(scaffold_ids):
        raise ValueError(
            "section-planning judgment must cover every known section id exactly once"
        )

    known_insight_ids = {
        insight.insight_id for insight in interval_interpretation.insights
    }
    known_capability_ids = {
        proposal.capability_id
        for proposal in checkpoint_model_enrichment.capability_proposals
    }
    known_design_note_ids = {
        anchor.note_id for anchor in checkpoint_model_enrichment.design_note_anchors
    }
    by_id: dict[HistorySectionPlanId, HistoryLLMSectionPlanDecision] = {
        decision.section_id: decision for decision in decisions.sections
    }
    for section_id in scaffold_ids:
        decision = by_id[section_id]
        if not decision.planning_rationale.strip():
            raise ValueError(
                f"section-planning judgment must provide planning_rationale for {section_id}"
            )
        if decision.status == "included" and decision.depth is None:
            raise ValueError(
                f"included section decisions must include a depth for {section_id}"
            )
        if decision.status == "omitted" and decision.depth is not None:
            raise ValueError(
                f"omitted section decisions must not include a depth for {section_id}"
            )
        if set(decision.source_insight_ids) - known_insight_ids:
            raise ValueError(f"unknown source_insight_ids for {section_id}")
        if set(decision.source_capability_ids) - known_capability_ids:
            raise ValueError(f"unknown source_capability_ids for {section_id}")
        if set(decision.source_design_note_ids) - known_design_note_ids:
            raise ValueError(f"unknown source_design_note_ids for {section_id}")
    return by_id


def _materialize_outline(
    *,
    scaffold: HistorySectionOutline,
    decisions: HistoryLLMSectionOutlineJudgment,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
) -> HistoryLLMSectionOutline:
    decision_by_id = _validate_known_reference_ids(
        decisions=decisions,
        scaffold=scaffold,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
    )
    sections: list[HistorySectionPlan] = []
    for scaffold_section in scaffold.sections:
        decision = decision_by_id[scaffold_section.section_id]
        forced_core = scaffold_section.section_id in _STABLE_CORE_SECTION_IDS
        resolved_status = "included" if forced_core else decision.status
        if resolved_status == "included":
            resolved_depth = decision.depth or scaffold_section.depth
            if resolved_depth is None:
                raise ValueError(
                    f"included section {scaffold_section.section_id} is missing depth"
                )
            omission_reason = None
        else:
            resolved_depth = None
            omission_reason = scaffold_section.omission_reason or "llm_planner_omitted"
        sections.append(
            scaffold_section.model_copy(
                update={
                    "status": resolved_status,
                    "depth": resolved_depth,
                    "confidence_score": decision.confidence_score,
                    "omission_reason": omission_reason,
                    "planning_rationale": decision.planning_rationale.strip(),
                    "source_insight_ids": sorted(set(decision.source_insight_ids)),
                    "source_capability_ids": sorted(
                        set(decision.source_capability_ids)
                    ),
                    "source_design_note_ids": sorted(
                        set(decision.source_design_note_ids)
                    ),
                }
            )
        )
    return HistoryLLMSectionOutline(
        checkpoint_id=scaffold.checkpoint_id,
        target_commit=scaffold.target_commit,
        previous_checkpoint_commit=scaffold.previous_checkpoint_commit,
        evaluation_status="scored",
        sections=sections,
    )


def build_llm_section_outline(
    *,
    checkpoint_model: HistoryCheckpointModel,
    section_scaffold: HistorySectionOutline,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    semantic_context_map: HistorySemanticContextMap | None,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> HistoryLLMSectionOutline:
    """Build the H12-03 shadow LLM section outline from deterministic evidence."""

    if llm_client is None:
        return _fallback_outline(
            scaffold=section_scaffold,
            evaluation_status="heuristic_only",
            reason="Retained baseline section decision after planner fallback.",
        )

    try:
        system_prompt, user_prompt = build_section_planning_llm_prompt(
            checkpoint_id=section_scaffold.checkpoint_id,
            target_commit=section_scaffold.target_commit,
            previous_checkpoint_commit=section_scaffold.previous_checkpoint_commit,
            section_scaffold=_scaffold_payload(section_scaffold),
            checkpoint_summary=_checkpoint_summary(checkpoint_model),
            interval_interpretation=_interval_payload(interval_interpretation),
            checkpoint_model_enrichment=_enrichment_payload(
                checkpoint_model_enrichment
            ),
            semantic_context_summary=_semantic_context_payload(semantic_context_map),
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryLLMSectionOutlineJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryLLMSectionOutlineJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        return _materialize_outline(
            scaffold=section_scaffold,
            decisions=judgment,
            interval_interpretation=interval_interpretation,
            checkpoint_model_enrichment=checkpoint_model_enrichment,
        )
    except Exception:
        return _fallback_outline(
            scaffold=section_scaffold,
            evaluation_status="llm_failed",
            reason="Retained baseline section decision after planner fallback.",
        )


def link_section_planning_outline_to_scaffold(
    *,
    section_scaffold: HistorySectionOutline,
    llm_section_outline: HistoryLLMSectionOutline,
) -> HistoryLLMSectionOutline:
    """Project H12-03 planner decisions onto an updated deterministic scaffold."""

    scaffold_by_id = {
        section.section_id: section for section in section_scaffold.sections
    }
    linked_sections: list[HistorySectionPlan] = []
    for section in llm_section_outline.sections:
        scaffold_section = scaffold_by_id.get(section.section_id)
        if scaffold_section is None:
            continue
        linked_sections.append(
            scaffold_section.model_copy(
                update={
                    "status": section.status,
                    "depth": section.depth,
                    "confidence_score": section.confidence_score,
                    "omission_reason": section.omission_reason,
                    "planning_rationale": section.planning_rationale,
                    "source_insight_ids": section.source_insight_ids,
                    "source_capability_ids": section.source_capability_ids,
                    "source_design_note_ids": section.source_design_note_ids,
                    "algorithm_capsule_ids": section.algorithm_capsule_ids
                    or scaffold_section.algorithm_capsule_ids,
                    "evidence_links": _dedupe_evidence(
                        scaffold_section.evidence_links,
                        section.evidence_links,
                    ),
                }
            )
        )
    return HistoryLLMSectionOutline(
        checkpoint_id=llm_section_outline.checkpoint_id,
        target_commit=llm_section_outline.target_commit,
        previous_checkpoint_commit=llm_section_outline.previous_checkpoint_commit,
        evaluation_status=llm_section_outline.evaluation_status,
        sections=linked_sections,
    )


__all__ = [
    "build_llm_section_outline",
    "build_section_planning_scaffold",
    "link_section_planning_outline_to_scaffold",
    "section_outline_llm_path",
]
