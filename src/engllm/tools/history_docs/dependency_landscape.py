"""H13-03 dependency landscape builders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_dependency_landscape_prompt
from engllm.tools.history_docs.h13_evidence import (
    H13EvidencePack,
    build_h13_evidence_pack,
    compact_checkpoint_summary,
    dedupe_evidence,
)
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyCluster,
    HistoryDependencyInventory,
    HistoryDependencyLandscape,
    HistoryDependencyLandscapeJudgment,
    HistoryDependencyProjectRole,
    HistoryDependencyUsagePattern,
    HistoryIntervalInterpretation,
    HistorySemanticContextMap,
)


def dependency_landscape_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H13-03 dependency landscape artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "dependency_landscape.json"


def _dependency_payload(pack: H13EvidencePack) -> dict[str, object]:
    return {
        "entries": [
            {
                "dependency_id": entry.dependency_id,
                "display_name": entry.display_name,
                "ecosystem": entry.ecosystem,
                "scope_roles": entry.scope_roles,
                "section_target": entry.section_target,
                "related_subsystem_ids": entry.related_subsystem_ids,
                "related_module_ids": entry.related_module_ids,
                "usage_signals": entry.usage_signals[:10],
                "general_description": entry.general_description,
                "project_usage_description": entry.project_usage_description,
            }
            for entry in sorted(
                pack.dependency_inventory.entries,
                key=lambda item: item.dependency_id,
            )
        ],
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
        "interval_insights": [
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "summary": insight.summary,
                "kind": insight.kind,
                "related_subsystem_ids": insight.related_subsystem_ids,
            }
            for insight in pack.interval_interpretation.insights
            if insight.kind in {"dependency_change", "build_change", "design_rationale"}
        ],
    }


def _fallback_landscape(pack: H13EvidencePack) -> HistoryDependencyLandscape:
    entries = sorted(
        pack.dependency_inventory.entries, key=lambda item: item.dependency_id
    )
    project_roles: list[HistoryDependencyProjectRole] = []
    for role in sorted({role for entry in entries for role in entry.scope_roles}):
        matching = [entry for entry in entries if role in entry.scope_roles]
        if not matching:
            continue
        project_roles.append(
            HistoryDependencyProjectRole(
                role_id=f"role::{role}",
                title=role.replace("_", " ").title(),
                summary=(
                    "This checkpoint includes "
                    f"{len(matching)} dependency entries acting in the current {role.replace('_', ' ')} role."
                ),
                dependency_ids=[entry.dependency_id for entry in matching],
                related_subsystem_ids=sorted(
                    {
                        subsystem_id
                        for entry in matching
                        for subsystem_id in entry.related_subsystem_ids
                    }
                ),
                evidence_links=[],
            )
        )
    if not project_roles and entries:
        project_roles.append(
            HistoryDependencyProjectRole(
                role_id="role::mixed",
                title="Mixed Dependency Roles",
                summary="This checkpoint groups direct dependencies without strong role separation evidence.",
                dependency_ids=[entry.dependency_id for entry in entries],
                related_subsystem_ids=sorted(
                    {
                        subsystem_id
                        for entry in entries
                        for subsystem_id in entry.related_subsystem_ids
                    }
                ),
                evidence_links=[],
            )
        )

    clusters: list[HistoryDependencyCluster] = []
    by_ecosystem: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_ecosystem[entry.ecosystem].append(entry.dependency_id)
    for ecosystem in sorted(by_ecosystem):
        dependency_ids = sorted(by_ecosystem[ecosystem])
        matching_entries = [
            pack.dependency_entries_by_id[dependency_id]
            for dependency_id in dependency_ids
        ]
        clusters.append(
            HistoryDependencyCluster(
                cluster_id=f"cluster::{ecosystem}",
                title=f"{ecosystem.title()} Dependency Cluster",
                summary=(
                    f"This checkpoint groups {len(dependency_ids)} direct {ecosystem} dependency entries with similar tooling or runtime context."
                ),
                dependency_ids=dependency_ids,
                ecosystems=[ecosystem],
                scope_roles=sorted(
                    {role for entry in matching_entries for role in entry.scope_roles}
                ),
                related_subsystem_ids=sorted(
                    {
                        subsystem_id
                        for entry in matching_entries
                        for subsystem_id in entry.related_subsystem_ids
                    }
                ),
                evidence_links=dedupe_evidence(
                    *[
                        [
                            link
                            for concept_id in entry.source_dependency_concept_ids
                            for link in (
                                []
                                if concept_id
                                not in {
                                    concept.concept_id
                                    for concept in pack.checkpoint_model.dependencies
                                }
                                else next(
                                    concept.evidence_links
                                    for concept in pack.checkpoint_model.dependencies
                                    if concept.concept_id == concept_id
                                )
                            )
                        ]
                        for entry in matching_entries
                    ]
                ),
            )
        )

    usage_patterns: list[HistoryDependencyUsagePattern] = []
    by_subsystem: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        for subsystem_id in entry.related_subsystem_ids:
            by_subsystem[subsystem_id].append(entry.dependency_id)
    for subsystem_id in sorted(by_subsystem):
        dependency_ids = sorted(set(by_subsystem[subsystem_id]))
        subsystem = pack.subsystems_by_id.get(subsystem_id)
        usage_patterns.append(
            HistoryDependencyUsagePattern(
                pattern_id=f"pattern::{subsystem_id}",
                title=(
                    f"{(subsystem.display_name or subsystem.group_path.as_posix()) if subsystem is not None else subsystem_id} Dependency Usage"
                ),
                summary="This subsystem currently concentrates the listed direct dependency relationships.",
                dependency_ids=dependency_ids,
                related_module_ids=sorted(
                    {
                        module_id
                        for dependency_id in dependency_ids
                        for module_id in pack.dependency_entries_by_id[
                            dependency_id
                        ].related_module_ids
                    }
                ),
                related_subsystem_ids=[subsystem_id],
                source_insight_ids=[
                    insight.insight_id
                    for insight in pack.interval_interpretation.insights
                    if subsystem_id in insight.related_subsystem_ids
                    and insight.kind in {"dependency_change", "build_change"}
                ][:4],
                evidence_links=dedupe_evidence(
                    *[
                        [
                            link
                            for concept_id in pack.dependency_entries_by_id[
                                dependency_id
                            ].source_dependency_concept_ids
                            for link in (
                                []
                                if concept_id
                                not in {
                                    concept.concept_id
                                    for concept in pack.checkpoint_model.dependencies
                                }
                                else next(
                                    concept.evidence_links
                                    for concept in pack.checkpoint_model.dependencies
                                    if concept.concept_id == concept_id
                                )
                            )
                        ]
                        for dependency_id in dependency_ids
                    ]
                ),
            )
        )

    return HistoryDependencyLandscape(
        checkpoint_id=pack.checkpoint_model.checkpoint_id,
        target_commit=pack.checkpoint_model.target_commit,
        previous_checkpoint_commit=pack.checkpoint_model.previous_checkpoint_commit,
        evaluation_status="heuristic_only",
        project_roles=project_roles,
        clusters=clusters,
        usage_patterns=usage_patterns,
    )


def _validate_landscape(
    pack: H13EvidencePack,
    landscape: HistoryDependencyLandscape,
) -> HistoryDependencyLandscape:
    dependency_ids = set(pack.dependency_entries_by_id)
    subsystem_ids = set(pack.subsystems_by_id)
    module_ids = set(pack.modules_by_id)
    insight_ids = set(pack.insights_by_id)
    for role in landscape.project_roles:
        if not role.title.strip() or not role.summary.strip():
            raise ValueError("dependency project roles must include title and summary")
        if not set(role.dependency_ids) <= dependency_ids:
            raise ValueError(
                "dependency project role referenced unknown dependency ids"
            )
        if not set(role.related_subsystem_ids) <= subsystem_ids:
            raise ValueError("dependency project role referenced unknown subsystem ids")
    for cluster in landscape.clusters:
        if not cluster.title.strip() or not cluster.summary.strip():
            raise ValueError("dependency clusters must include title and summary")
        if not set(cluster.dependency_ids) <= dependency_ids:
            raise ValueError("dependency cluster referenced unknown dependency ids")
        if not set(cluster.related_subsystem_ids) <= subsystem_ids:
            raise ValueError("dependency cluster referenced unknown subsystem ids")
    for pattern in landscape.usage_patterns:
        if not pattern.title.strip() or not pattern.summary.strip():
            raise ValueError("dependency usage patterns must include title and summary")
        if not set(pattern.dependency_ids) <= dependency_ids:
            raise ValueError(
                "dependency usage pattern referenced unknown dependency ids"
            )
        if not set(pattern.related_subsystem_ids) <= subsystem_ids:
            raise ValueError(
                "dependency usage pattern referenced unknown subsystem ids"
            )
        if not set(pattern.related_module_ids) <= module_ids:
            raise ValueError("dependency usage pattern referenced unknown module ids")
        if not set(pattern.source_insight_ids) <= insight_ids:
            raise ValueError("dependency usage pattern referenced unknown insight ids")
    return HistoryDependencyLandscape(
        checkpoint_id=landscape.checkpoint_id,
        target_commit=landscape.target_commit,
        previous_checkpoint_commit=landscape.previous_checkpoint_commit,
        evaluation_status=landscape.evaluation_status,
        project_roles=sorted(landscape.project_roles, key=lambda item: item.role_id),
        clusters=sorted(landscape.clusters, key=lambda item: item.cluster_id),
        usage_patterns=sorted(
            landscape.usage_patterns, key=lambda item: item.pattern_id
        ),
    )


def build_dependency_landscape(
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
) -> HistoryDependencyLandscape:
    """Build H13-03 dependency-landscape artifact."""

    pack = build_h13_evidence_pack(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        semantic_context_map=semantic_context_map,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules,
    )
    fallback = _fallback_landscape(pack)
    if llm_client is None or not dependency_inventory.entries:
        return fallback
    try:
        system_prompt, user_prompt = build_dependency_landscape_prompt(
            checkpoint_summary=compact_checkpoint_summary(pack),
            dependency_context=_dependency_payload(pack),
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
                response_model=HistoryDependencyLandscapeJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryDependencyLandscapeJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        return _validate_landscape(
            pack,
            HistoryDependencyLandscape(
                checkpoint_id=checkpoint_id,
                target_commit=target_commit,
                previous_checkpoint_commit=previous_checkpoint_commit,
                evaluation_status="scored",
                project_roles=judgment.project_roles,
                clusters=judgment.clusters,
                usage_patterns=judgment.usage_patterns,
            ),
        )
    except Exception:
        fallback.evaluation_status = "llm_failed"
        return fallback
