"""Shared H13 evidence normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryCapabilityConceptProposal,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryDesignChangeInsight,
    HistoryDesignNoteAnchor,
    HistoryEvidenceLink,
    HistoryIntervalInterpretation,
    HistoryModuleConcept,
    HistoryRationaleClue,
    HistorySemanticContextMap,
    HistorySubsystemConcept,
)


@dataclass(frozen=True)
class H13EvidencePack:
    """Resolved H13 evidence and id maps for one checkpoint."""

    checkpoint_model: HistoryCheckpointModel
    interval_interpretation: HistoryIntervalInterpretation
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment
    semantic_context_map: HistorySemanticContextMap | None
    dependency_inventory: HistoryDependencyInventory
    capsule_index: HistoryAlgorithmCapsuleIndex
    capsules: list[HistoryAlgorithmCapsule]
    subsystems_by_id: dict[str, HistorySubsystemConcept]
    modules_by_id: dict[str, HistoryModuleConcept]
    dependency_entries_by_id: dict[str, HistoryDependencyEntry]
    insights_by_id: dict[str, HistoryDesignChangeInsight]
    rationale_clues_by_id: dict[str, HistoryRationaleClue]
    capability_proposals_by_id: dict[str, HistoryCapabilityConceptProposal]
    design_note_anchors_by_id: dict[str, HistoryDesignNoteAnchor]
    context_node_ids: set[str]
    capsule_ids: set[str]


def evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    """Return a deterministic evidence sort key."""

    return (link.kind, link.reference, link.detail or "")


def dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    """Merge and deterministically deduplicate evidence links."""

    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=evidence_sort_key)


def build_h13_evidence_pack(
    *,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    semantic_context_map: HistorySemanticContextMap | None,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
) -> H13EvidencePack:
    """Resolve shared H13 evidence maps from existing checkpoint artifacts."""

    return H13EvidencePack(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        semantic_context_map=semantic_context_map,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules,
        subsystems_by_id={
            subsystem.concept_id: subsystem for subsystem in checkpoint_model.subsystems
        },
        modules_by_id={
            module.concept_id: module for module in checkpoint_model.modules
        },
        dependency_entries_by_id={
            entry.dependency_id: entry for entry in dependency_inventory.entries
        },
        insights_by_id={
            insight.insight_id: insight for insight in interval_interpretation.insights
        },
        rationale_clues_by_id={
            clue.clue_id: clue for clue in interval_interpretation.rationale_clues
        },
        capability_proposals_by_id={
            proposal.capability_id: proposal
            for proposal in checkpoint_model_enrichment.capability_proposals
        },
        design_note_anchors_by_id={
            anchor.note_id: anchor
            for anchor in checkpoint_model_enrichment.design_note_anchors
        },
        context_node_ids=(
            set()
            if semantic_context_map is None
            else {node.node_id for node in semantic_context_map.context_nodes}
        ),
        capsule_ids={capsule.capsule_id for capsule in capsules},
    )


def module_path_label(
    modules_by_id: dict[str, HistoryModuleConcept], module_id: str
) -> str:
    """Return a compact module label for prompts or render output."""

    module = modules_by_id.get(module_id)
    if module is None:
        return module_id
    return module.path.as_posix()


def compact_checkpoint_summary(pack: H13EvidencePack) -> dict[str, object]:
    """Return a compact, prompt-safe checkpoint summary for H13 builders."""

    return {
        "checkpoint_id": pack.checkpoint_model.checkpoint_id,
        "target_commit": pack.checkpoint_model.target_commit,
        "previous_checkpoint_commit": pack.checkpoint_model.previous_checkpoint_commit,
        "subsystem_count": len(pack.checkpoint_model.subsystems),
        "module_count": len(pack.checkpoint_model.modules),
        "dependency_entry_count": len(pack.dependency_inventory.entries),
        "algorithm_capsule_count": len(pack.capsules),
        "insight_count": len(pack.interval_interpretation.insights),
        "rationale_clue_count": len(pack.interval_interpretation.rationale_clues),
    }


def active_modules(pack: H13EvidencePack) -> list[HistoryModuleConcept]:
    """Return active modules in deterministic id order."""

    return sorted(
        [
            module
            for module in pack.modules_by_id.values()
            if module.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    )


def active_subsystems(pack: H13EvidencePack) -> list[HistorySubsystemConcept]:
    """Return active subsystems in deterministic id order."""

    return sorted(
        [
            subsystem
            for subsystem in pack.subsystems_by_id.values()
            if subsystem.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    )


def artifact_path(tool_root: Path, checkpoint_id: str, filename: str) -> Path:
    """Return one checkpoint-scoped H13 artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / filename
