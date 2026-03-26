"""H12-02 checkpoint-model enrichment helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from engllm.domain.models import CodeUnitSummary
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_checkpoint_model_enrichment_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryCheckpointModelEnrichmentJudgment,
    HistoryCheckpointModelEnrichmentStatus,
    HistoryDesignNoteAnchor,
    HistoryEvidenceLink,
    HistoryIntervalInterpretation,
    HistoryModuleConcept,
    HistoryModuleConceptEnrichment,
    HistoryRationaleClue,
    HistorySemanticContextMap,
    HistorySemanticStructureMap,
    HistorySnapshotStructuralModel,
    HistorySubsystemConcept,
    HistorySubsystemConceptEnrichment,
)

_MAX_PROMPT_SUBSYSTEMS = 12
_MAX_PROMPT_MODULES = 24
_MAX_PROMPT_INSIGHTS = 16
_MAX_PROMPT_CLUES = 16
_MAX_FALLBACK_DESIGN_NOTES = 8


def checkpoint_model_enrichment_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H12-02 checkpoint-model enrichment artifact path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "checkpoint_model_enrichment.json"
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


def _summary_docstrings(summary: CodeUnitSummary) -> list[str]:
    return [value.strip()[:160] for value in summary.docstrings if value.strip()][:2]


def _known_evidence_links(
    *,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    snapshot: HistorySnapshotStructuralModel,
    semantic_structure_map: HistorySemanticStructureMap | None,
    semantic_context_map: HistorySemanticContextMap | None,
) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    for concept_group in (
        checkpoint_model.subsystems,
        checkpoint_model.modules,
        checkpoint_model.dependencies,
    ):
        for concept in concept_group:
            for link in concept.evidence_links:
                links.add((link.kind, link.reference))
    for insight in interval_interpretation.insights:
        for link in insight.evidence_links:
            links.add((link.kind, link.reference))
    for clue in interval_interpretation.rationale_clues:
        for link in clue.evidence_links:
            links.add((link.kind, link.reference))
    for window in interval_interpretation.significant_windows:
        for link in window.evidence_links:
            links.add((link.kind, link.reference))
    for summary in snapshot.code_summaries:
        links.add(("file", summary.path.as_posix()))
    if semantic_structure_map is not None:
        for subsystem in semantic_structure_map.semantic_subsystems:
            links.add(("subsystem", subsystem.semantic_subsystem_id))
    if semantic_context_map is not None:
        for node in semantic_context_map.context_nodes:
            for subsystem_id in node.related_subsystem_ids:
                links.add(("subsystem", subsystem_id))
            for module_id in node.related_module_ids:
                links.add(("file", module_id.removeprefix("module::")))
    return links


def _subsystem_payloads(
    checkpoint_model: HistoryCheckpointModel,
) -> list[dict[str, object]]:
    return [
        {
            "concept_id": subsystem.concept_id,
            "display_name": subsystem.display_name,
            "summary": subsystem.summary,
            "source_root": subsystem.source_root.as_posix(),
            "group_path": subsystem.group_path.as_posix(),
            "module_ids": subsystem.module_ids,
            "file_count": subsystem.file_count,
            "symbol_count": subsystem.symbol_count,
            "language_counts": subsystem.language_counts,
            "capability_labels": subsystem.capability_labels,
            "baseline_subsystem_ids": subsystem.baseline_subsystem_ids,
            "representative_files": [
                path.as_posix() for path in subsystem.representative_files[:8]
            ],
        }
        for subsystem in sorted(
            [
                concept
                for concept in checkpoint_model.subsystems
                if concept.lifecycle_status == "active"
            ],
            key=lambda item: item.concept_id,
        )[:_MAX_PROMPT_SUBSYSTEMS]
    ]


def _module_payloads(
    checkpoint_model: HistoryCheckpointModel,
    snapshot: HistorySnapshotStructuralModel,
) -> list[dict[str, object]]:
    code_summaries_by_path = {
        summary.path: summary
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
    }
    payloads: list[dict[str, object]] = []
    for module in sorted(
        [
            concept
            for concept in checkpoint_model.modules
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.path.as_posix(),
    )[:_MAX_PROMPT_MODULES]:
        code_summary = code_summaries_by_path.get(module.path)
        payloads.append(
            {
                "concept_id": module.concept_id,
                "path": module.path.as_posix(),
                "subsystem_id": module.subsystem_id,
                "language": module.language,
                "functions": module.functions[:8],
                "classes": module.classes[:8],
                "imports": module.imports[:8],
                "docstring_excerpts": (
                    [] if code_summary is None else _summary_docstrings(code_summary)
                ),
                "symbol_names": module.symbol_names[:12],
                "summary": module.summary,
                "responsibility_labels": module.responsibility_labels,
            }
        )
    return payloads


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
                "related_commit_ids": insight.related_commit_ids,
                "related_change_ids": insight.related_change_ids,
                "related_subsystem_ids": insight.related_subsystem_ids,
            }
            for insight in sorted(
                interval_interpretation.insights,
                key=lambda item: item.insight_id,
            )[:_MAX_PROMPT_INSIGHTS]
        ],
        "rationale_clues": [
            {
                "clue_id": clue.clue_id,
                "text": clue.text,
                "confidence": clue.confidence,
                "source_kind": clue.source_kind,
                "related_commit_ids": clue.related_commit_ids,
                "related_change_ids": clue.related_change_ids,
            }
            for clue in sorted(
                interval_interpretation.rationale_clues,
                key=lambda item: item.clue_id,
            )[:_MAX_PROMPT_CLUES]
        ],
        "significant_windows": [
            {
                "window_id": window.window_id,
                "title": window.title,
                "summary": window.summary,
                "significance": window.significance,
                "commit_ids": window.commit_ids,
                "related_insight_ids": window.related_insight_ids,
            }
            for window in sorted(
                interval_interpretation.significant_windows,
                key=lambda item: item.window_id,
            )[:8]
        ],
    }


def _semantic_payload(
    *,
    semantic_structure_map: HistorySemanticStructureMap | None,
    semantic_context_map: HistorySemanticContextMap | None,
) -> dict[str, list[dict[str, object]]]:
    return {
        "semantic_subsystems": (
            []
            if semantic_structure_map is None
            else [
                {
                    "semantic_subsystem_id": subsystem.semantic_subsystem_id,
                    "title": subsystem.title,
                    "summary": subsystem.summary,
                    "module_ids": subsystem.module_ids,
                    "capability_ids": subsystem.capability_ids,
                }
                for subsystem in semantic_structure_map.semantic_subsystems
            ]
        ),
        "capabilities": (
            []
            if semantic_structure_map is None
            else [
                {
                    "capability_id": capability.capability_id,
                    "title": capability.title,
                    "summary": capability.summary,
                    "module_ids": capability.module_ids,
                    "semantic_subsystem_ids": capability.semantic_subsystem_ids,
                }
                for capability in semantic_structure_map.capabilities
            ]
        ),
        "context_nodes": (
            []
            if semantic_context_map is None
            else [
                {
                    "node_id": node.node_id,
                    "title": node.title,
                    "kind": node.kind,
                    "related_subsystem_ids": node.related_subsystem_ids,
                    "related_module_ids": node.related_module_ids,
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
                    "provider_subsystem_ids": interface.provider_subsystem_ids,
                    "related_module_ids": interface.related_module_ids,
                }
                for interface in semantic_context_map.interfaces
            ]
        ),
    }


def _module_summary(module: HistoryModuleConcept) -> str:
    symbols: list[str] = []
    if module.functions:
        symbols.append(f"functions {', '.join(module.functions[:3])}")
    if module.classes:
        symbols.append(f"classes {', '.join(module.classes[:3])}")
    if module.imports:
        symbols.append(f"imports {', '.join(module.imports[:3])}")
    if symbols:
        return (
            f"This {module.language} module at `{module.path.as_posix()}` currently centers on "
            + "; ".join(symbols)
            + "."
        )
    return f"This {module.language} module at `{module.path.as_posix()}` is active in the current checkpoint."


def _fallback_subsystem_enrichments(
    *,
    checkpoint_model: HistoryCheckpointModel,
    semantic_structure_map: HistorySemanticStructureMap | None,
) -> list[HistorySubsystemConceptEnrichment]:
    semantic_by_id = {}
    capability_titles_by_id: dict[str, str] = {}
    if semantic_structure_map is not None:
        semantic_by_id = {
            subsystem.semantic_subsystem_id: subsystem
            for subsystem in semantic_structure_map.semantic_subsystems
        }
        capability_titles_by_id = {
            capability.capability_id: capability.title
            for capability in semantic_structure_map.capabilities
        }
    enrichments: list[HistorySubsystemConceptEnrichment] = []
    for subsystem in sorted(
        [
            concept
            for concept in checkpoint_model.subsystems
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    ):
        semantic_cluster = semantic_by_id.get(subsystem.concept_id)
        display_name = (
            semantic_cluster.title
            if semantic_cluster is not None and semantic_cluster.title.strip()
            else subsystem.display_name or subsystem.group_path.as_posix()
        )
        summary = (
            semantic_cluster.summary.strip()
            if semantic_cluster is not None and semantic_cluster.summary.strip()
            else subsystem.summary
            or (
                f"This subsystem groups active modules under `{subsystem.group_path.as_posix()}` and currently spans "
                f"{subsystem.file_count} files and {subsystem.symbol_count} symbols."
            )
        )
        capability_labels = sorted(
            {
                *subsystem.capability_labels,
                *(
                    []
                    if semantic_cluster is None
                    else [
                        capability_titles_by_id.get(item, item)
                        for item in semantic_cluster.capability_ids
                    ]
                ),
            }
        )
        enrichments.append(
            HistorySubsystemConceptEnrichment(
                concept_id=subsystem.concept_id,
                display_name=display_name,
                summary=summary,
                capability_labels=capability_labels,
                evidence_links=_dedupe_evidence(subsystem.evidence_links),
            )
        )
    return enrichments


def _fallback_module_enrichments(
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistoryModuleConceptEnrichment]:
    enrichments: list[HistoryModuleConceptEnrichment] = []
    for module in sorted(
        [
            concept
            for concept in checkpoint_model.modules
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.path.as_posix(),
    ):
        enrichments.append(
            HistoryModuleConceptEnrichment(
                concept_id=module.concept_id,
                summary=_module_summary(module),
                responsibility_labels=[],
                evidence_links=_dedupe_evidence(module.evidence_links),
            )
        )
    return enrichments


def _fallback_design_note_anchors(
    *,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistoryDesignNoteAnchor]:
    known_concept_ids = {
        concept.concept_id for concept in checkpoint_model.subsystems
    } | {concept.concept_id for concept in checkpoint_model.modules}
    anchors: list[HistoryDesignNoteAnchor] = []
    rationale_by_change_id: dict[str, list[HistoryRationaleClue]] = defaultdict(list)
    for clue in interval_interpretation.rationale_clues:
        for change_id in clue.related_change_ids:
            rationale_by_change_id[change_id].append(clue)
    significance_order = {"high": 3, "medium": 2, "low": 1}
    for insight in sorted(
        interval_interpretation.insights,
        key=lambda item: (
            significance_order.get(item.significance, 0),
            item.insight_id,
        ),
        reverse=True,
    )[:_MAX_FALLBACK_DESIGN_NOTES]:
        related_concepts = sorted(
            subsystem_id
            for subsystem_id in insight.related_subsystem_ids
            if subsystem_id in known_concept_ids
        )
        related_clues = [
            clue
            for change_id in insight.related_change_ids
            for clue in rationale_by_change_id.get(change_id, [])
        ]
        if not related_concepts and not related_clues:
            continue
        anchors.append(
            HistoryDesignNoteAnchor(
                note_id=f"design-note::{insight.insight_id}",
                title=insight.title or "TBD",
                summary=insight.summary or "TBD",
                related_concept_ids=related_concepts,
                source_insight_ids=[insight.insight_id],
                source_rationale_clue_ids=sorted(
                    {clue.clue_id for clue in related_clues}
                ),
                evidence_links=_dedupe_evidence(
                    insight.evidence_links,
                    *[clue.evidence_links for clue in related_clues],
                ),
            )
        )
    return sorted(anchors, key=lambda item: item.note_id)


def _fallback_checkpoint_model_enrichment(
    *,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    semantic_structure_map: HistorySemanticStructureMap | None,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    status: HistoryCheckpointModelEnrichmentStatus,
) -> HistoryCheckpointModelEnrichment:
    return HistoryCheckpointModelEnrichment(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        subsystem_enrichments=_fallback_subsystem_enrichments(
            checkpoint_model=checkpoint_model,
            semantic_structure_map=semantic_structure_map,
        ),
        module_enrichments=_fallback_module_enrichments(checkpoint_model),
        capability_proposals=[],
        design_note_anchors=_fallback_design_note_anchors(
            interval_interpretation=interval_interpretation,
            checkpoint_model=checkpoint_model,
        ),
    )


def _validate_enrichment_judgment(
    judgment: HistoryCheckpointModelEnrichmentJudgment,
    *,
    known_subsystem_ids: set[str],
    known_module_ids: set[str],
    known_insight_ids: set[str],
    known_rationale_clue_ids: set[str],
    known_evidence_links: set[tuple[str, str]],
) -> None:
    known_concept_ids = known_subsystem_ids | known_module_ids
    for subsystem_enrichment in judgment.subsystem_enrichments:
        if subsystem_enrichment.concept_id not in known_subsystem_ids:
            raise ValueError("subsystem enrichment referenced unknown concept id")
        if (
            not subsystem_enrichment.display_name.strip()
            or not subsystem_enrichment.summary.strip()
        ):
            raise ValueError("subsystem enrichment must have non-empty text")
        if not set(subsystem_enrichment.source_insight_ids) <= known_insight_ids:
            raise ValueError("subsystem enrichment referenced unknown insight id")
        if (
            not set(subsystem_enrichment.source_rationale_clue_ids)
            <= known_rationale_clue_ids
        ):
            raise ValueError(
                "subsystem enrichment referenced unknown rationale clue id"
            )
        if (
            not {
                (link.kind, link.reference)
                for link in subsystem_enrichment.evidence_links
            }
            <= known_evidence_links
        ):
            raise ValueError("subsystem enrichment referenced unknown evidence")
    for module_enrichment in judgment.module_enrichments:
        if module_enrichment.concept_id not in known_module_ids:
            raise ValueError("module enrichment referenced unknown concept id")
        if not module_enrichment.summary.strip():
            raise ValueError("module enrichment must have a non-empty summary")
        if not set(module_enrichment.source_insight_ids) <= known_insight_ids:
            raise ValueError("module enrichment referenced unknown insight id")
        if (
            not set(module_enrichment.source_rationale_clue_ids)
            <= known_rationale_clue_ids
        ):
            raise ValueError("module enrichment referenced unknown rationale clue id")
        if (
            not {
                (link.kind, link.reference) for link in module_enrichment.evidence_links
            }
            <= known_evidence_links
        ):
            raise ValueError("module enrichment referenced unknown evidence")
    for proposal in judgment.capability_proposals:
        if not proposal.title.strip() or not proposal.summary.strip():
            raise ValueError("capability proposal must have non-empty text")
        if not set(proposal.related_subsystem_ids) <= known_subsystem_ids:
            raise ValueError("capability proposal referenced unknown subsystem ids")
        if not set(proposal.related_module_ids) <= known_module_ids:
            raise ValueError("capability proposal referenced unknown module ids")
        if not set(proposal.source_insight_ids) <= known_insight_ids:
            raise ValueError("capability proposal referenced unknown insight id")
        if (
            not {(link.kind, link.reference) for link in proposal.evidence_links}
            <= known_evidence_links
        ):
            raise ValueError("capability proposal referenced unknown evidence")
    for anchor in judgment.design_note_anchors:
        if not anchor.title.strip() or not anchor.summary.strip():
            raise ValueError("design note anchor must have non-empty text")
        if not set(anchor.related_concept_ids) <= known_concept_ids:
            raise ValueError("design note anchor referenced unknown concept ids")
        if not set(anchor.source_insight_ids) <= known_insight_ids:
            raise ValueError("design note anchor referenced unknown insight id")
        if not set(anchor.source_rationale_clue_ids) <= known_rationale_clue_ids:
            raise ValueError("design note anchor referenced unknown rationale clue id")
        if (
            not {(link.kind, link.reference) for link in anchor.evidence_links}
            <= known_evidence_links
        ):
            raise ValueError("design note anchor referenced unknown evidence")


def _materialize_enrichment(
    *,
    judgment: HistoryCheckpointModelEnrichmentJudgment,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    status: HistoryCheckpointModelEnrichmentStatus,
) -> HistoryCheckpointModelEnrichment:
    return HistoryCheckpointModelEnrichment(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        subsystem_enrichments=sorted(
            judgment.subsystem_enrichments,
            key=lambda item: item.concept_id,
        ),
        module_enrichments=sorted(
            judgment.module_enrichments,
            key=lambda item: item.concept_id,
        ),
        capability_proposals=sorted(
            judgment.capability_proposals,
            key=lambda item: item.capability_id,
        ),
        design_note_anchors=sorted(
            judgment.design_note_anchors,
            key=lambda item: item.note_id,
        ),
    )


def build_checkpoint_model_enrichment(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    checkpoint_model: HistoryCheckpointModel,
    snapshot: HistorySnapshotStructuralModel,
    interval_interpretation: HistoryIntervalInterpretation,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
    semantic_structure_map: HistorySemanticStructureMap | None = None,
    semantic_context_map: HistorySemanticContextMap | None = None,
) -> HistoryCheckpointModelEnrichment:
    """Build the H12-02 checkpoint-model enrichment artifact."""

    if llm_client is None:
        return _fallback_checkpoint_model_enrichment(
            checkpoint_model=checkpoint_model,
            interval_interpretation=interval_interpretation,
            semantic_structure_map=semantic_structure_map,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="heuristic_only",
        )

    system_prompt, user_prompt = build_checkpoint_model_enrichment_prompt(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        subsystems=_subsystem_payloads(checkpoint_model),
        modules=_module_payloads(checkpoint_model, snapshot),
        interval_interpretation=_interval_payload(interval_interpretation),
        semantic_labels=_semantic_payload(
            semantic_structure_map=semantic_structure_map,
            semantic_context_map=semantic_context_map,
        ),
    )
    known_subsystem_ids = {
        concept.concept_id
        for concept in checkpoint_model.subsystems
        if concept.lifecycle_status == "active"
    }
    known_module_ids = {
        concept.concept_id
        for concept in checkpoint_model.modules
        if concept.lifecycle_status == "active"
    }
    known_insight_ids = {
        insight.insight_id for insight in interval_interpretation.insights
    }
    known_rationale_clue_ids = {
        clue.clue_id for clue in interval_interpretation.rationale_clues
    }
    known_evidence_links = _known_evidence_links(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        snapshot=snapshot,
        semantic_structure_map=semantic_structure_map,
        semantic_context_map=semantic_context_map,
    )
    try:
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryCheckpointModelEnrichmentJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryCheckpointModelEnrichmentJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        _validate_enrichment_judgment(
            judgment,
            known_subsystem_ids=known_subsystem_ids,
            known_module_ids=known_module_ids,
            known_insight_ids=known_insight_ids,
            known_rationale_clue_ids=known_rationale_clue_ids,
            known_evidence_links=known_evidence_links,
        )
        return _materialize_enrichment(
            judgment=judgment,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="scored",
        )
    except Exception:
        return _fallback_checkpoint_model_enrichment(
            checkpoint_model=checkpoint_model,
            interval_interpretation=interval_interpretation,
            semantic_structure_map=semantic_structure_map,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="llm_failed",
        )


def apply_checkpoint_model_enrichment(
    checkpoint_model: HistoryCheckpointModel,
    enrichment: HistoryCheckpointModelEnrichment,
) -> HistoryCheckpointModel:
    """Apply a validated H12-02 enrichment artifact to an existing model."""

    subsystem_enrichments = {
        enrichment_item.concept_id: enrichment_item
        for enrichment_item in enrichment.subsystem_enrichments
    }
    module_enrichments = {
        enrichment_item.concept_id: enrichment_item
        for enrichment_item in enrichment.module_enrichments
    }

    subsystems: list[HistorySubsystemConcept] = []
    for subsystem in checkpoint_model.subsystems:
        enrichment_item = subsystem_enrichments.get(subsystem.concept_id)
        if enrichment_item is None:
            subsystems.append(subsystem)
            continue
        subsystems.append(
            subsystem.model_copy(
                update={
                    "display_name": enrichment_item.display_name,
                    "summary": enrichment_item.summary,
                    "capability_labels": sorted(
                        set(
                            [
                                *subsystem.capability_labels,
                                *enrichment_item.capability_labels,
                            ]
                        )
                    ),
                    "evidence_links": _dedupe_evidence(
                        subsystem.evidence_links,
                        enrichment_item.evidence_links,
                    ),
                }
            )
        )

    modules: list[HistoryModuleConcept] = []
    for module in checkpoint_model.modules:
        module_enrichment = module_enrichments.get(module.concept_id)
        if module_enrichment is None:
            modules.append(module)
            continue
        modules.append(
            module.model_copy(
                update={
                    "summary": module_enrichment.summary,
                    "responsibility_labels": sorted(
                        set(
                            [
                                *module.responsibility_labels,
                                *module_enrichment.responsibility_labels,
                            ]
                        )
                    ),
                    "evidence_links": _dedupe_evidence(
                        module.evidence_links,
                        module_enrichment.evidence_links,
                    ),
                }
            )
        )

    return checkpoint_model.model_copy(
        update={
            "subsystems": subsystems,
            "modules": modules,
        }
    )
