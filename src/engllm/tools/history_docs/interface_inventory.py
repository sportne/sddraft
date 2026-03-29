"""H13-02 interface inventory builders."""

from __future__ import annotations

from pathlib import Path

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_interface_inventory_prompt
from engllm.tools.history_docs.h13_evidence import (
    H13EvidencePack,
    build_h13_evidence_pack,
    compact_checkpoint_summary,
)
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryCrossModuleContract,
    HistoryDependencyInventory,
    HistoryInterfaceConcept,
    HistoryInterfaceInventory,
    HistoryInterfaceInventoryJudgment,
    HistoryInterfaceResponsibility,
    HistoryIntervalInterpretation,
    HistorySemanticContextMap,
)


def interface_inventory_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H13-02 interface inventory path."""

    return tool_root / "checkpoints" / checkpoint_id / "interface_inventory.json"


def _interface_payload(pack: H13EvidencePack) -> dict[str, object]:
    semantic_context_map = pack.semantic_context_map
    return {
        "context_nodes": (
            []
            if semantic_context_map is None
            else [
                {
                    "node_id": node.node_id,
                    "title": node.title,
                    "kind": node.kind,
                }
                for node in semantic_context_map.context_nodes
            ]
        ),
        "interfaces": (
            []
            if semantic_context_map is None
            else [
                {
                    "interface_id": interface.interface_id,
                    "title": interface.title,
                    "kind": interface.kind,
                    "summary": interface.summary,
                    "provider_subsystem_ids": interface.provider_subsystem_ids,
                    "consumer_context_node_ids": interface.consumer_context_node_ids,
                    "related_module_ids": interface.related_module_ids,
                }
                for interface in semantic_context_map.interfaces
            ]
        ),
        "subsystems": [
            {
                "concept_id": subsystem.concept_id,
                "display_name": subsystem.display_name
                or subsystem.group_path.as_posix(),
                "capability_labels": subsystem.capability_labels,
            }
            for subsystem in sorted(
                pack.subsystems_by_id.values(), key=lambda item: item.concept_id
            )
            if subsystem.lifecycle_status == "active"
        ],
        "modules": [
            {
                "concept_id": module.concept_id,
                "path": module.path.as_posix(),
                "functions": module.functions[:8],
                "classes": module.classes[:8],
                "imports": module.imports[:8],
                "summary": module.summary,
                "responsibility_labels": module.responsibility_labels,
            }
            for module in sorted(
                pack.modules_by_id.values(), key=lambda item: item.concept_id
            )
            if module.lifecycle_status == "active"
        ],
        "interval_insights": [
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "summary": insight.summary,
                "related_subsystem_ids": insight.related_subsystem_ids,
            }
            for insight in pack.interval_interpretation.insights
            if insight.kind in {"interface_change", "design_rationale"}
        ],
    }


def _fallback_inventory(pack: H13EvidencePack) -> HistoryInterfaceInventory:
    semantic_context_map = pack.semantic_context_map
    interfaces: list[HistoryInterfaceConcept] = []
    if semantic_context_map is not None:
        for interface in sorted(
            semantic_context_map.interfaces, key=lambda item: item.interface_id
        ):
            responsibilities = [
                HistoryInterfaceResponsibility(
                    title="Boundary Responsibility",
                    summary=interface.summary,
                    related_module_ids=list(interface.related_module_ids),
                    evidence_links=list(interface.evidence_links),
                )
            ]
            contracts: list[HistoryCrossModuleContract] = []
            if len(interface.related_module_ids) > 1:
                contracts.append(
                    HistoryCrossModuleContract(
                        contract_id=f"contract::{interface.interface_id}",
                        title="Cross-Module Contract",
                        summary="This interface coordinates behavior across multiple linked modules.",
                        provider_module_ids=interface.related_module_ids[:1],
                        consumer_module_ids=interface.related_module_ids[1:],
                        evidence_links=list(interface.evidence_links),
                    )
                )
            interfaces.append(
                HistoryInterfaceConcept(
                    interface_id=interface.interface_id,
                    title=interface.title,
                    kind=interface.kind,
                    summary=interface.summary,
                    provider_subsystem_ids=list(interface.provider_subsystem_ids),
                    consumer_context_node_ids=list(interface.consumer_context_node_ids),
                    related_module_ids=list(interface.related_module_ids),
                    responsibilities=responsibilities,
                    cross_module_contracts=contracts,
                    collaboration_notes=(
                        [
                            "Coordinates a boundary or contract already surfaced by the semantic context map."
                        ]
                        if interface.related_module_ids
                        else []
                    ),
                    source_insight_ids=[],
                    source_rationale_clue_ids=[],
                    evidence_links=list(interface.evidence_links),
                )
            )
    return HistoryInterfaceInventory(
        checkpoint_id=pack.checkpoint_model.checkpoint_id,
        target_commit=pack.checkpoint_model.target_commit,
        previous_checkpoint_commit=pack.checkpoint_model.previous_checkpoint_commit,
        evaluation_status="heuristic_only",
        interfaces=interfaces,
    )


def _validate_inventory(
    pack: H13EvidencePack, inventory: HistoryInterfaceInventory
) -> HistoryInterfaceInventory:
    validated_interfaces: list[HistoryInterfaceConcept] = []
    for interface in sorted(inventory.interfaces, key=lambda item: item.interface_id):
        if not interface.title.strip() or not interface.summary.strip():
            raise ValueError(
                "interface inventory entries must include title and summary"
            )
        if not set(interface.provider_subsystem_ids) <= set(pack.subsystems_by_id):
            raise ValueError(
                "interface inventory referenced unknown provider subsystem ids"
            )
        if not set(interface.related_module_ids) <= set(pack.modules_by_id):
            raise ValueError(
                "interface inventory referenced unknown related module ids"
            )
        if not set(interface.consumer_context_node_ids) <= pack.context_node_ids:
            raise ValueError("interface inventory referenced unknown context node ids")
        if not set(interface.source_insight_ids) <= set(pack.insights_by_id):
            raise ValueError("interface inventory referenced unknown insight ids")
        if not set(interface.source_rationale_clue_ids) <= set(
            pack.rationale_clues_by_id
        ):
            raise ValueError(
                "interface inventory referenced unknown rationale clue ids"
            )
        for responsibility in interface.responsibilities:
            if not responsibility.title.strip() or not responsibility.summary.strip():
                raise ValueError(
                    "interface responsibilities must include title and summary"
                )
            if not set(responsibility.related_module_ids) <= set(pack.modules_by_id):
                raise ValueError(
                    "interface responsibility referenced unknown module ids"
                )
        for contract in interface.cross_module_contracts:
            if not contract.title.strip() or not contract.summary.strip():
                raise ValueError(
                    "cross-module contracts must include title and summary"
                )
            if not set(contract.provider_module_ids) <= set(pack.modules_by_id):
                raise ValueError(
                    "interface contract referenced unknown provider module ids"
                )
            if not set(contract.consumer_module_ids) <= set(pack.modules_by_id):
                raise ValueError(
                    "interface contract referenced unknown consumer module ids"
                )
        validated_interfaces.append(interface)
    return HistoryInterfaceInventory(
        checkpoint_id=inventory.checkpoint_id,
        target_commit=inventory.target_commit,
        previous_checkpoint_commit=inventory.previous_checkpoint_commit,
        evaluation_status=inventory.evaluation_status,
        interfaces=validated_interfaces,
    )


def build_interface_inventory(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    dependency_inventory: HistoryDependencyInventory,
    semantic_context_map: HistorySemanticContextMap | None,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> HistoryInterfaceInventory:
    """Build H13-02 interface inventory."""

    pack = build_h13_evidence_pack(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        semantic_context_map=semantic_context_map,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules,
    )
    fallback = _fallback_inventory(pack)
    if llm_client is None or semantic_context_map is None:
        return fallback
    try:
        system_prompt, user_prompt = build_interface_inventory_prompt(
            checkpoint_summary=compact_checkpoint_summary(pack),
            interface_context=_interface_payload(pack),
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
                response_model=HistoryInterfaceInventoryJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryInterfaceInventoryJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        return _validate_inventory(
            pack,
            HistoryInterfaceInventory(
                checkpoint_id=checkpoint_id,
                target_commit=target_commit,
                previous_checkpoint_commit=previous_checkpoint_commit,
                evaluation_status="scored",
                interfaces=judgment.interfaces,
            ),
        )
    except Exception:
        fallback.evaluation_status = "llm_failed"
        return fallback
