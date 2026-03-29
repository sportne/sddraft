"""H13-01 enriched algorithm capsule builders."""

from __future__ import annotations

from pathlib import Path

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_algorithm_capsule_enrichment_prompt
from engllm.tools.history_docs.h13_evidence import (
    H13EvidencePack,
    build_h13_evidence_pack,
    compact_checkpoint_summary,
)
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleEnrichment,
    HistoryAlgorithmCapsuleEnrichmentIndex,
    HistoryAlgorithmCapsuleEnrichmentIndexEntry,
    HistoryAlgorithmCapsuleEnrichmentJudgment,
    HistoryAlgorithmCapsuleEnrichmentStatus,
    HistoryAlgorithmCapsuleIndex,
    HistoryAlgorithmInvariant,
    HistoryAlgorithmTradeoff,
    HistoryAlgorithmVariantRelationship,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyInventory,
    HistoryIntervalInterpretation,
    HistorySemanticContextMap,
)


def algorithm_capsule_enrichment_dir(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the per-checkpoint H13-01 enriched capsule directory."""

    return tool_root / "checkpoints" / checkpoint_id / "algorithm_capsules_enriched"


def algorithm_capsule_enrichment_index_path(
    tool_root: Path, checkpoint_id: str
) -> Path:
    """Return the per-checkpoint H13-01 enriched capsule index path."""

    return algorithm_capsule_enrichment_dir(tool_root, checkpoint_id) / "index.json"


def algorithm_capsule_enrichment_filename(capsule_id: str) -> str:
    """Return the deterministic artifact filename for one enriched capsule id."""

    return f"{capsule_id.replace(':', '_').replace('/', '_')}.json"


def _capsule_payloads(pack: H13EvidencePack) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for capsule in sorted(pack.capsules, key=lambda item: item.capsule_id):
        related_insights = [
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "summary": insight.summary,
                "significance": insight.significance,
            }
            for insight in pack.interval_interpretation.insights
            if set(insight.related_subsystem_ids) & set(capsule.related_subsystem_ids)
            or set(insight.related_commit_ids) & set(capsule.commit_ids)
            or set(insight.related_change_ids) & set(capsule.source_candidate_ids)
        ][:5]
        related_clues = [
            {
                "clue_id": clue.clue_id,
                "text": clue.text,
                "source_kind": clue.source_kind,
            }
            for clue in pack.interval_interpretation.rationale_clues
            if set(clue.related_commit_ids) & set(capsule.commit_ids)
            or set(clue.related_change_ids) & set(capsule.source_candidate_ids)
        ][:5]
        payloads.append(
            {
                "capsule_id": capsule.capsule_id,
                "title": capsule.title,
                "scope_kind": capsule.scope_kind,
                "scope_path": capsule.scope_path.as_posix(),
                "related_subsystem_ids": capsule.related_subsystem_ids,
                "related_module_ids": capsule.related_module_ids,
                "changed_symbol_names": capsule.changed_symbol_names,
                "variant_names": capsule.variant_names,
                "phase_keys": [phase.phase_key for phase in capsule.phases],
                "shared_abstractions": [
                    item.name for item in capsule.shared_abstractions
                ],
                "data_structures": [item.name for item in capsule.data_structures],
                "assumptions": [item.text for item in capsule.assumptions],
                "related_insights": related_insights,
                "related_rationale_clues": related_clues,
            }
        )
    return payloads


def _fallback_enrichment(
    capsule: HistoryAlgorithmCapsule,
) -> HistoryAlgorithmCapsuleEnrichment:
    phase_flow_summary = (
        "Current flow stages are reflected by the capsule phases: "
        + ", ".join(f"`{phase.phase_key}`" for phase in capsule.phases)
        + "."
        if capsule.phases
        else "Current flow evidence remains limited in this checkpoint capsule."
    )
    invariants = [
        HistoryAlgorithmInvariant(
            text=assumption.text,
            evidence_links=assumption.evidence_links,
        )
        for assumption in capsule.assumptions
    ]
    tradeoffs = [
        HistoryAlgorithmTradeoff(
            title="Fallback Handling",
            summary=assumption.text,
            evidence_links=assumption.evidence_links,
        )
        for assumption in capsule.assumptions
        if any(
            token in assumption.text.lower()
            for token in ("fallback", "strict", "deterministic", "conservative")
        )
    ]
    variant_relationships = [
        HistoryAlgorithmVariantRelationship(
            variant_name=variant_name,
            relationship="variant_family_member",
            summary=f"`{variant_name}` is part of the current variant family for this algorithm scope.",
            evidence_links=capsule.evidence_links,
        )
        for variant_name in capsule.variant_names
    ]
    purpose = f"This capsule captures the current algorithm behavior around `{capsule.scope_path.as_posix()}`."
    return HistoryAlgorithmCapsuleEnrichment(
        capsule_id=capsule.capsule_id,
        purpose=purpose,
        phase_flow_summary=phase_flow_summary,
        invariants=invariants,
        tradeoffs=tradeoffs,
        variant_relationships=variant_relationships,
        related_subsystem_ids=list(capsule.related_subsystem_ids),
        related_module_ids=list(capsule.related_module_ids),
        source_insight_ids=[],
        source_rationale_clue_ids=[],
        evidence_links=list(capsule.evidence_links),
    )


def _validate_enrichments(
    enrichments: list[HistoryAlgorithmCapsuleEnrichment],
    pack: H13EvidencePack,
) -> list[HistoryAlgorithmCapsuleEnrichment]:
    validated: list[HistoryAlgorithmCapsuleEnrichment] = []
    for enrichment in sorted(enrichments, key=lambda item: item.capsule_id):
        if enrichment.capsule_id not in pack.capsule_ids:
            raise ValueError("algorithm enrichment referenced unknown capsule id")
        if not set(enrichment.related_subsystem_ids) <= set(pack.subsystems_by_id):
            raise ValueError("algorithm enrichment referenced unknown subsystem ids")
        if not set(enrichment.related_module_ids) <= set(pack.modules_by_id):
            raise ValueError("algorithm enrichment referenced unknown module ids")
        if not set(enrichment.source_insight_ids) <= set(pack.insights_by_id):
            raise ValueError("algorithm enrichment referenced unknown insight ids")
        if not set(enrichment.source_rationale_clue_ids) <= set(
            pack.rationale_clues_by_id
        ):
            raise ValueError(
                "algorithm enrichment referenced unknown rationale clue ids"
            )
        if not enrichment.purpose.strip() or not enrichment.phase_flow_summary.strip():
            raise ValueError(
                "algorithm enrichment must include purpose and phase_flow_summary"
            )
        validated.append(enrichment)
    return validated


def build_algorithm_capsule_enrichments(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    semantic_context_map: HistorySemanticContextMap | None,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> tuple[
    HistoryAlgorithmCapsuleEnrichmentIndex, list[HistoryAlgorithmCapsuleEnrichment]
]:
    """Build H13-01 enriched algorithm capsule artifacts."""

    pack = build_h13_evidence_pack(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        semantic_context_map=semantic_context_map,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules,
    )
    fallback = [
        _fallback_enrichment(capsule)
        for capsule in sorted(capsules, key=lambda item: item.capsule_id)
    ]
    status: HistoryAlgorithmCapsuleEnrichmentStatus = "heuristic_only"
    enrichments = fallback
    if llm_client is not None and capsules:
        try:
            system_prompt, user_prompt = build_algorithm_capsule_enrichment_prompt(
                checkpoint_summary=compact_checkpoint_summary(pack),
                capsules=_capsule_payloads(pack),
                design_note_anchors=[
                    {
                        "note_id": anchor.note_id,
                        "title": anchor.title,
                        "summary": anchor.summary,
                    }
                    for anchor in checkpoint_model_enrichment.design_note_anchors
                ],
            )
            response = llm_client.generate_structured(
                StructuredGenerationRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=HistoryAlgorithmCapsuleEnrichmentJudgment,
                    model_name=model_name,
                    temperature=temperature,
                )
            )
            judgment = HistoryAlgorithmCapsuleEnrichmentJudgment.model_validate(
                response.content.model_dump(mode="python")
            )
            enrichments = _validate_enrichments(judgment.enrichments, pack)
            if len(enrichments) != len(capsules):
                raise ValueError(
                    "algorithm enrichment response did not cover every known capsule"
                )
            status = "scored"
        except Exception:
            enrichments = fallback
            status = "llm_failed"

    index = HistoryAlgorithmCapsuleEnrichmentIndex(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        capsules=[
            HistoryAlgorithmCapsuleEnrichmentIndexEntry(
                capsule_id=enrichment.capsule_id,
                artifact_path=Path(
                    algorithm_capsule_enrichment_filename(enrichment.capsule_id)
                ),
            )
            for enrichment in enrichments
        ],
    )
    return index, enrichments
