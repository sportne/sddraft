"""H14-03 shadow section repair helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_section_repair_prompt
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleEnrichment,
    HistoryAlgorithmCapsuleEnrichmentIndex,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyInventory,
    HistoryDependencyLandscape,
    HistoryDraftReview,
    HistoryEvidenceLink,
    HistoryInterfaceInventory,
    HistoryIntervalInterpretation,
    HistoryRenderManifest,
    HistoryRepairStatus,
    HistorySectionDraftArtifact,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySectionRepair,
    HistorySectionRepairArtifact,
    HistorySectionRepairJudgment,
    HistorySemanticContextMap,
)
from engllm.tools.history_docs.section_drafting_llm import (
    assemble_shadow_markdown,
    clone_render_manifest,
    extract_section_bodies,
)

_SEVERITY_PRIORITY = {"high": 0, "medium": 1, "low": 2}
_FINDING_KIND_PRIORITY = {
    "unsupported_claim": 0,
    "omission": 1,
    "coherence": 2,
    "redundancy": 3,
    "weak_prose": 4,
}
_MAX_REPAIRED_SECTIONS = 4


def section_repairs_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-03 repair artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "section_repairs_llm.json"


def checkpoint_repaired_markdown_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-03 repaired markdown path."""

    return tool_root / "checkpoints" / checkpoint_id / "checkpoint_repaired_llm.md"


def render_manifest_repaired_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-03 repaired render manifest path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "render_manifest_repaired_llm.json"
    )


def validation_report_repaired_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-03 repaired validation report path."""

    return (
        tool_root
        / "checkpoints"
        / checkpoint_id
        / "validation_report_repaired_llm.json"
    )


def _dedupe_evidence(
    *groups: Iterable[HistoryEvidenceLink],
) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(
        deduped.values(),
        key=lambda link: (link.kind, link.reference, link.detail or ""),
    )


def _select_repair_sections(
    review: HistoryDraftReview,
    render_manifest: HistoryRenderManifest,
) -> list[HistorySectionPlanId]:
    order_by_section: dict[HistorySectionPlanId, int] = {
        section.section_id: section.order for section in render_manifest.sections
    }
    grouped: dict[HistorySectionPlanId, list[tuple[int, int]]] = {}
    for finding in review.findings:
        if finding.section_id is None:
            continue
        section_id = cast(HistorySectionPlanId, finding.section_id)
        grouped.setdefault(section_id, []).append(
            (
                _SEVERITY_PRIORITY[finding.severity],
                _FINDING_KIND_PRIORITY[finding.kind],
            )
        )
    ranked = sorted(
        grouped,
        key=lambda section_id: (
            min(grouped[section_id]),
            order_by_section.get(section_id, 999),
            section_id,
        ),
    )
    preferred = [
        section_id
        for section_id in review.recommended_repair_section_ids
        if section_id in grouped
    ]
    ordered_unique: list[HistorySectionPlanId] = []
    for section_id in [*preferred, *ranked]:
        if section_id not in ordered_unique:
            ordered_unique.append(section_id)
    return ordered_unique[:_MAX_REPAIRED_SECTIONS]


def _build_supporting_evidence(
    *,
    section: HistorySectionPlan,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    semantic_context_map: HistorySemanticContextMap | None,
    interface_inventory: HistoryInterfaceInventory | None,
    dependency_landscape: HistoryDependencyLandscape | None,
) -> dict[str, object]:
    insight_ids = {
        insight.insight_id
        for insight in interval_interpretation.insights
        if set(insight.related_subsystem_ids).intersection(section.concept_ids)
        or section.section_id in {"interfaces", "design_notes_rationale"}
    }
    capability_ids = {
        proposal.capability_id
        for proposal in checkpoint_model_enrichment.capability_proposals
        if set(proposal.related_subsystem_ids).intersection(section.concept_ids)
        or set(proposal.related_module_ids).intersection(section.concept_ids)
    }
    design_note_ids = {
        anchor.note_id
        for anchor in checkpoint_model_enrichment.design_note_anchors
        if set(anchor.related_concept_ids).intersection(section.concept_ids)
    }
    return {
        "insights": [
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "summary": insight.summary,
            }
            for insight in interval_interpretation.insights
            if insight.insight_id in insight_ids
        ],
        "capabilities": [
            {
                "capability_id": proposal.capability_id,
                "title": proposal.title,
                "summary": proposal.summary,
            }
            for proposal in checkpoint_model_enrichment.capability_proposals
            if proposal.capability_id in capability_ids
        ],
        "design_notes": [
            {
                "note_id": anchor.note_id,
                "title": anchor.title,
                "summary": anchor.summary,
            }
            for anchor in checkpoint_model_enrichment.design_note_anchors
            if anchor.note_id in design_note_ids
        ],
        "semantic_context_titles": (
            []
            if semantic_context_map is None or section.section_id != "system_context"
            else [node.title for node in semantic_context_map.context_nodes]
        ),
        "interface_titles": (
            []
            if interface_inventory is None or section.section_id != "interfaces"
            else [interface.title for interface in interface_inventory.interfaces]
        ),
        "dependency_pattern_titles": (
            []
            if dependency_landscape is None
            or section.section_id
            not in {"dependencies", "build_development_infrastructure"}
            else [pattern.title for pattern in dependency_landscape.usage_patterns]
        ),
    }


def build_section_repairs(
    *,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    draft_artifact: HistorySectionDraftArtifact,
    draft_review: HistoryDraftReview,
    draft_markdown: str,
    draft_render_manifest: HistoryRenderManifest,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    semantic_context_map: HistorySemanticContextMap | None,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
    algorithm_capsule_enrichment_index: (
        HistoryAlgorithmCapsuleEnrichmentIndex | None
    ) = None,
    algorithm_capsule_enrichments: (
        list[HistoryAlgorithmCapsuleEnrichment] | None
    ) = None,
    interface_inventory: HistoryInterfaceInventory | None = None,
    dependency_landscape: HistoryDependencyLandscape | None = None,
) -> tuple[HistorySectionRepairArtifact, str, HistoryRenderManifest]:
    """Build H14-03 targeted repairs plus repaired markdown and manifest."""

    del dependency_inventory, capsule_index, capsules
    del algorithm_capsule_enrichment_index, algorithm_capsule_enrichments
    del draft_artifact
    preamble_lines, draft_bodies = extract_section_bodies(
        draft_markdown,
        draft_render_manifest,
    )
    sections_by_id: dict[HistorySectionPlanId, HistorySectionPlan] = {
        section.section_id: section for section in section_outline.sections
    }
    repair_section_ids = _select_repair_sections(draft_review, draft_render_manifest)
    if llm_client is None or not repair_section_ids:
        artifact = HistorySectionRepairArtifact(
            checkpoint_id=checkpoint_model.checkpoint_id,
            target_commit=checkpoint_model.target_commit,
            previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
            evaluation_status=(
                "heuristic_only" if llm_client is None else "heuristic_only"
            ),
            sections=[],
        )
        return (
            artifact,
            draft_markdown,
            clone_render_manifest(
                draft_render_manifest,
                markdown_filename="checkpoint_repaired_llm.md",
            ),
        )

    repairs: list[HistorySectionRepair] = []
    had_failure = False
    known_finding_ids = {finding.finding_id for finding in draft_review.findings}
    findings_by_section: dict[HistorySectionPlanId, list[dict[str, object]]] = {}
    for finding in draft_review.findings:
        if finding.section_id is None:
            continue
        section_id = cast(HistorySectionPlanId, finding.section_id)
        findings_by_section.setdefault(section_id, []).append(
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind,
                "severity": finding.severity,
                "summary": finding.summary,
                "revision_goal": finding.revision_goal,
            }
        )

    for section_id in repair_section_ids:
        section = sections_by_id[section_id]
        original_body = draft_bodies.get(section_id, "")
        try:
            system_prompt, user_prompt = build_section_repair_prompt(
                checkpoint_context={
                    "checkpoint_id": checkpoint_model.checkpoint_id,
                    "target_commit": checkpoint_model.target_commit,
                    "previous_checkpoint_commit": checkpoint_model.previous_checkpoint_commit,
                },
                section_metadata={
                    "section_id": section.section_id,
                    "title": section.title,
                    "kind": section.kind,
                    "depth": section.depth,
                    "concept_ids": section.concept_ids,
                    "algorithm_capsule_ids": section.algorithm_capsule_ids,
                },
                findings=findings_by_section.get(section_id, []),
                supporting_evidence=_build_supporting_evidence(
                    section=section,
                    interval_interpretation=interval_interpretation,
                    checkpoint_model_enrichment=checkpoint_model_enrichment,
                    semantic_context_map=semantic_context_map,
                    interface_inventory=interface_inventory,
                    dependency_landscape=dependency_landscape,
                ),
                original_markdown_body=original_body,
            )
            response = llm_client.generate_structured(
                StructuredGenerationRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=HistorySectionRepairJudgment,
                    model_name=model_name,
                    temperature=temperature,
                )
            )
            judgment = HistorySectionRepairJudgment.model_validate(
                response.content.model_dump(mode="python")
            )
            if not judgment.revised_markdown_body.strip():
                raise ValueError("section repair markdown must not be empty")
            if not set(judgment.addressed_finding_ids) <= known_finding_ids:
                raise ValueError("section repair referenced unknown finding ids")
            if not all(
                (link.kind, link.reference)
                in {
                    (evidence.kind, evidence.reference)
                    for evidence in section.evidence_links
                }
                for link in judgment.evidence_links
            ):
                raise ValueError("section repair referenced unknown evidence links")
            repairs.append(
                HistorySectionRepair(
                    section_id=section_id,
                    revised_markdown_body=judgment.revised_markdown_body.strip(),
                    addressed_finding_ids=list(judgment.addressed_finding_ids),
                    evidence_links=_dedupe_evidence(
                        list(judgment.evidence_links),
                        list(section.evidence_links),
                    ),
                )
            )
        except Exception:
            had_failure = True

    repaired_bodies: dict[HistorySectionPlanId, str] = dict(draft_bodies)
    for repair in repairs:
        repaired_bodies[repair.section_id] = repair.revised_markdown_body
    repaired_markdown = assemble_shadow_markdown(
        preamble_lines=preamble_lines,
        render_manifest=draft_render_manifest,
        section_bodies=repaired_bodies,
    )
    evaluation_status: HistoryRepairStatus = "llm_failed" if had_failure else "scored"
    artifact = HistorySectionRepairArtifact(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        evaluation_status=evaluation_status,
        sections=sorted(
            repairs, key=lambda item: repair_section_ids.index(item.section_id)
        ),
    )
    repaired_manifest = clone_render_manifest(
        draft_render_manifest,
        markdown_filename="checkpoint_repaired_llm.md",
    )
    if not repairs:
        artifact.evaluation_status = "llm_failed" if had_failure else "heuristic_only"
        repaired_markdown = draft_markdown
    return artifact, repaired_markdown, repaired_manifest


__all__ = [
    "build_section_repairs",
    "checkpoint_repaired_markdown_path",
    "render_manifest_repaired_path",
    "section_repairs_path",
    "validation_report_repaired_path",
]
