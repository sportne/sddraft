"""H14-01 shadow section drafting helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_section_drafting_prompt
from engllm.tools.history_docs.h13_evidence import dedupe_evidence
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleEnrichment,
    HistoryAlgorithmCapsuleEnrichmentIndex,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyInventory,
    HistoryDependencyLandscape,
    HistoryDraftStatus,
    HistoryEvidenceLink,
    HistoryInterfaceInventory,
    HistoryIntervalInterpretation,
    HistoryRenderedSection,
    HistoryRenderManifest,
    HistorySectionDraft,
    HistorySectionDraftArtifact,
    HistorySectionDraftJudgment,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySemanticContextMap,
)

_MAX_CONCEPT_SUMMARIES = 10
_MAX_ALGORITHM_SUMMARIES = 8
_MAX_DEPENDENCY_SUMMARIES = 12
_MAX_INSIGHTS = 10
_MAX_CLUES = 10
_MAX_CAPABILITIES = 10
_MAX_DESIGN_NOTES = 10
_TARGETED_REWRITE_SECTION_IDS: tuple[HistorySectionPlanId, ...] = (
    "architectural_overview",
    "system_context",
    "subsystems_modules",
    "interfaces",
    "dependencies",
    "build_development_infrastructure",
)
_TARGETED_REWRITE_LLM_SECTION_IDS: tuple[HistorySectionPlanId, ...] = (
    "architectural_overview",
    "system_context",
    "subsystems_modules",
    "interfaces",
)
_ARCHITECTURE_SECTION_IDS: tuple[HistorySectionPlanId, ...] = (
    "architectural_overview",
    "subsystems_modules",
)


def section_drafts_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-01 section-draft artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "section_drafts_llm.json"


def checkpoint_draft_markdown_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-01 draft markdown path."""

    return tool_root / "checkpoints" / checkpoint_id / "checkpoint_draft_llm.md"


def render_manifest_draft_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-01 draft render manifest path."""

    return tool_root / "checkpoints" / checkpoint_id / "render_manifest_draft_llm.json"


def validation_report_draft_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-01 draft validation report path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "validation_report_draft_llm.json"
    )


def targeted_section_rewrites_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the targeted shadow rewrite artifact path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "targeted_section_rewrites_llm.json"
    )


def checkpoint_targeted_rewrite_markdown_path(
    tool_root: Path, checkpoint_id: str
) -> Path:
    """Return the targeted shadow rewrite markdown path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "checkpoint_targeted_rewrite_llm.md"
    )


def render_manifest_targeted_rewrite_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the targeted shadow rewrite render-manifest path."""

    return (
        tool_root
        / "checkpoints"
        / checkpoint_id
        / "render_manifest_targeted_rewrite_llm.json"
    )


def validation_report_targeted_rewrite_path(
    tool_root: Path, checkpoint_id: str
) -> Path:
    """Return the targeted shadow rewrite validation report path."""

    return (
        tool_root
        / "checkpoints"
        / checkpoint_id
        / "validation_report_targeted_rewrite_llm.json"
    )


def _known_evidence_keys(
    *groups: Iterable[HistoryEvidenceLink],
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for group in groups:
        for link in group:
            keys.add((link.kind, link.reference))
    return keys


def _split_markdown_blocks(
    markdown: str,
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    lines = markdown.splitlines()
    first_heading = next(
        (index for index, line in enumerate(lines) if line.startswith("## ")),
        len(lines),
    )
    preamble = lines[:first_heading]
    blocks: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []
    for line in lines[first_heading:]:
        if line.startswith("## "):
            if current_title is not None:
                blocks.append((current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)
    if current_title is not None:
        blocks.append((current_title, current_body))
    return preamble, blocks


def extract_section_bodies(
    markdown: str,
    render_manifest: HistoryRenderManifest,
) -> tuple[list[str], dict[HistorySectionPlanId, str]]:
    """Return the document preamble plus one body per rendered section id."""

    preamble, blocks = _split_markdown_blocks(markdown)
    by_title = {title: body for title, body in blocks}
    bodies: dict[HistorySectionPlanId, str] = {}
    for section in render_manifest.sections:
        body_lines = by_title.get(section.title, [])
        body = "\n".join(body_lines).strip("\n")
        bodies[section.section_id] = body
    return preamble, bodies


def assemble_shadow_markdown(
    *,
    preamble_lines: list[str],
    render_manifest: HistoryRenderManifest,
    section_bodies: dict[HistorySectionPlanId, str],
) -> str:
    """Assemble deterministic shadow markdown from known section bodies."""

    lines = list(preamble_lines)
    if lines and lines[-1] != "":
        lines.append("")
    for section in render_manifest.sections:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"## {section.title}")
        lines.append("")
        body = section_bodies.get(section.section_id, "").strip("\n")
        if body:
            lines.extend(body.splitlines())
            lines.append("")
        else:
            lines.append("TBD")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def clone_render_manifest(
    render_manifest: HistoryRenderManifest,
    *,
    markdown_filename: str,
) -> HistoryRenderManifest:
    """Return one manifest clone that points at another markdown artifact."""

    return render_manifest.model_copy(
        update={"markdown_path": Path(markdown_filename)},
        deep=True,
    )


def _concept_summary(
    checkpoint_model: HistoryCheckpointModel,
    concept_id: str,
    *,
    hide_internal_ids: bool = False,
) -> dict[str, object] | None:
    for subsystem in checkpoint_model.subsystems:
        if subsystem.concept_id == concept_id:
            summary: dict[str, object] = {
                "kind": "subsystem",
                "display_name": subsystem.display_name
                or subsystem.group_path.as_posix(),
                "summary": subsystem.summary,
                "capability_labels": subsystem.capability_labels,
            }
            if not hide_internal_ids:
                summary["concept_id"] = subsystem.concept_id
                summary["module_ids"] = subsystem.module_ids
            return summary
    for module in checkpoint_model.modules:
        if module.concept_id == concept_id:
            summary = {
                "kind": "module",
                "path": module.path.as_posix(),
                "summary": module.summary,
                "responsibility_labels": module.responsibility_labels,
                "functions": module.functions[:6],
                "classes": module.classes[:6],
            }
            if not hide_internal_ids:
                summary["concept_id"] = module.concept_id
            return summary
    for dependency in checkpoint_model.dependencies:
        if dependency.concept_id == concept_id:
            summary = {
                "kind": "dependency_concept",
                "path": dependency.path.as_posix(),
                "ecosystem": dependency.ecosystem,
                "category": dependency.category,
            }
            if not hide_internal_ids:
                summary["concept_id"] = dependency.concept_id
                summary["documented_dependency_ids"] = (
                    dependency.documented_dependency_ids
                )
            return summary
    return None


def _capsule_summary(
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    enrichments_by_id: dict[str, HistoryAlgorithmCapsuleEnrichment],
    capsule_id: str,
    *,
    hide_internal_ids: bool = False,
) -> dict[str, object] | None:
    index_entry = next(
        (entry for entry in capsule_index.capsules if entry.capsule_id == capsule_id),
        None,
    )
    capsule = next((item for item in capsules if item.capsule_id == capsule_id), None)
    if index_entry is None or capsule is None:
        return None
    enrichment = enrichments_by_id.get(capsule_id)
    summary: dict[str, object] = {
        "title": index_entry.title,
        "phase_count": len(capsule.phases),
        "purpose": None if enrichment is None else enrichment.purpose,
        "phase_flow_summary": (
            None if enrichment is None else enrichment.phase_flow_summary
        ),
    }
    if not hide_internal_ids:
        summary["capsule_id"] = capsule_id
        summary["related_module_ids"] = capsule.related_module_ids
    return summary


def _dependency_summary(
    dependency_inventory: HistoryDependencyInventory,
    dependency_id: str,
    *,
    hide_internal_ids: bool = False,
) -> dict[str, object] | None:
    entry = next(
        (
            item
            for item in dependency_inventory.entries
            if item.dependency_id == dependency_id
        ),
        None,
    )
    if entry is None:
        return None
    summary: dict[str, object] = {
        "display_name": entry.display_name,
        "ecosystem": entry.ecosystem,
        "section_target": entry.section_target,
        "project_usage_description": entry.project_usage_description,
    }
    if not hide_internal_ids:
        summary["dependency_id"] = entry.dependency_id
        summary["related_subsystem_ids"] = entry.related_subsystem_ids
        summary["related_module_ids"] = entry.related_module_ids
    return summary


def _role_hint_for_subsystem(
    *,
    display_name: str,
    summary: str | None,
    representative_module_path: str | None,
    representative_symbols: list[str],
) -> str | None:
    haystack = " ".join(
        value.lower()
        for value in [
            display_name,
            summary or "",
            representative_module_path or "",
            *representative_symbols,
        ]
        if value
    )
    if any(token in haystack for token in ("api", "router", "route", "request")):
        return "request-entry boundary"
    if any(token in haystack for token in ("engine", "planner", "plan", "orchestr")):
        return "planning and coordination layer"
    if any(
        token in haystack
        for token in ("storage", "repository", "state", "persist", "store")
    ):
        return "state and persistence boundary"
    return None


def _boundary_hint_for_subsystem(role_hint: str | None) -> str | None:
    if role_hint == "request-entry boundary":
        return "entry boundary"
    if role_hint == "planning and coordination layer":
        return "coordination core"
    if role_hint == "state and persistence boundary":
        return "persistence boundary"
    return None


def _architecture_focus(
    *,
    section: HistorySectionPlan,
    checkpoint_model: HistoryCheckpointModel,
    hide_internal_ids: bool = False,
) -> dict[str, object]:
    if section.section_id not in _ARCHITECTURE_SECTION_IDS:
        return {}

    module_by_id = {module.concept_id: module for module in checkpoint_model.modules}
    subsystem_profiles: list[dict[str, object]] = []
    active_subsystems = [
        subsystem
        for subsystem in checkpoint_model.subsystems
        if subsystem.lifecycle_status == "active"
    ]
    for subsystem in active_subsystems:
        display_name = subsystem.display_name or subsystem.group_path.as_posix()
        representative_module = next(
            (
                module_by_id[module_id]
                for module_id in subsystem.module_ids
                if module_id in module_by_id
            ),
            None,
        )
        representative_module_path = (
            None
            if representative_module is None
            else representative_module.path.as_posix()
        )
        representative_symbols = (
            []
            if representative_module is None
            else [
                *representative_module.functions[:3],
                *representative_module.classes[:3],
            ]
        )
        role_hint = _role_hint_for_subsystem(
            display_name=display_name,
            summary=subsystem.summary,
            representative_module_path=representative_module_path,
            representative_symbols=representative_symbols,
        )
        profile: dict[str, object] = {
            "display_name": display_name,
            "summary": subsystem.summary,
            "architectural_role": role_hint,
            "boundary_kind": _boundary_hint_for_subsystem(role_hint),
            "representative_module": representative_module_path,
            "representative_symbols": representative_symbols,
        }
        if not hide_internal_ids:
            profile["concept_id"] = subsystem.concept_id
        subsystem_profiles.append(profile)

    names = {str(profile["display_name"]) for profile in subsystem_profiles}
    relationship_hints: list[dict[str, str]] = []
    if "API" in names and "Engine" in names:
        relationship_hints.append(
            {
                "from_subsystem": "API",
                "to_subsystem": "Engine",
                "relationship": "fronts",
                "summary": "API presents the request-entry boundary while Engine handles downstream planning work.",
            }
        )
    if "Engine" in names and "Storage" in names:
        relationship_hints.append(
            {
                "from_subsystem": "Engine",
                "to_subsystem": "Storage",
                "relationship": "coordinates_with",
                "summary": "Engine acts as the coordination layer while Storage provides the persistence boundary for project state.",
            }
        )

    return {
        "rewrite_goal": (
            "Explain subsystem roles, boundaries, and relationships using representative"
            " modules and symbols. Do not foreground counts or directory-shape metadata."
        ),
        "subsystem_profiles": subsystem_profiles,
        "relationship_hints": relationship_hints,
    }


def _relevant_interval_payload(
    *,
    section: HistorySectionPlan,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
) -> dict[str, object]:
    concept_ids = set(section.concept_ids)
    algorithm_ids = set(section.algorithm_capsule_ids)
    dependency_ids = {
        link.reference for link in section.evidence_links if link.kind == "build_source"
    }
    capability_ids = {
        proposal.capability_id
        for proposal in checkpoint_model_enrichment.capability_proposals
        if concept_ids.intersection(
            {*proposal.related_subsystem_ids, *proposal.related_module_ids}
        )
    }
    design_note_ids = {
        anchor.note_id
        for anchor in checkpoint_model_enrichment.design_note_anchors
        if concept_ids.intersection(anchor.related_concept_ids)
    }
    insights = cast(
        list[dict[str, object]],
        [
            {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "summary": insight.summary,
                "significance": insight.significance,
            }
            for insight in interval_interpretation.insights
            if concept_ids.intersection(insight.related_subsystem_ids)
            or any(
                change_id in algorithm_ids for change_id in insight.related_change_ids
            )
            or section.section_id in {"design_notes_rationale", "interfaces"}
            and insight.kind in {"design_rationale", "interface_change"}
        ][:_MAX_INSIGHTS],
    )
    clues: list[dict[str, object]] = [
        {
            "clue_id": clue.clue_id,
            "text": clue.text,
            "source_kind": clue.source_kind,
            "confidence": clue.confidence,
        }
        for clue in interval_interpretation.rationale_clues[:_MAX_CLUES]
        if section.section_id == "design_notes_rationale" or clue.related_change_ids
    ]
    capabilities = cast(
        list[dict[str, object]],
        [
            {
                "capability_id": proposal.capability_id,
                "title": proposal.title,
                "summary": proposal.summary,
            }
            for proposal in checkpoint_model_enrichment.capability_proposals
            if proposal.capability_id in capability_ids
        ][:_MAX_CAPABILITIES],
    )
    design_notes = cast(
        list[dict[str, object]],
        [
            {
                "note_id": anchor.note_id,
                "title": anchor.title,
                "summary": anchor.summary,
            }
            for anchor in checkpoint_model_enrichment.design_note_anchors
            if anchor.note_id in design_note_ids
        ][:_MAX_DESIGN_NOTES],
    )
    return {
        "insights": insights,
        "rationale_clues": clues,
        "capabilities": capabilities,
        "design_notes": design_notes,
        "dependency_signals": sorted(dependency_ids),
    }


def _section_evidence(
    *,
    section: HistorySectionPlan,
    render_section: HistoryRenderedSection,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    semantic_context_map: HistorySemanticContextMap | None,
    baseline_body: str,
    algorithm_capsule_enrichments: list[HistoryAlgorithmCapsuleEnrichment] | None,
    interface_inventory: HistoryInterfaceInventory | None,
    dependency_landscape: HistoryDependencyLandscape | None,
    hide_internal_ids: bool = False,
) -> dict[str, object]:
    enrichments_by_id = {
        enrichment.capsule_id: enrichment
        for enrichment in algorithm_capsule_enrichments or []
    }
    concepts = [
        summary
        for summary in (
            _concept_summary(
                checkpoint_model,
                concept_id,
                hide_internal_ids=hide_internal_ids,
            )
            for concept_id in section.concept_ids
        )
        if summary is not None
    ][:_MAX_CONCEPT_SUMMARIES]
    algorithms = [
        summary
        for summary in (
            _capsule_summary(
                capsule_index,
                capsules,
                enrichments_by_id,
                capsule_id,
                hide_internal_ids=hide_internal_ids,
            )
            for capsule_id in render_section.algorithm_capsule_ids
        )
        if summary is not None
    ][:_MAX_ALGORITHM_SUMMARIES]
    dependencies = [
        summary
        for summary in (
            _dependency_summary(
                dependency_inventory,
                dependency_id,
                hide_internal_ids=hide_internal_ids,
            )
            for dependency_id in render_section.dependency_ids
        )
        if summary is not None
    ][:_MAX_DEPENDENCY_SUMMARIES]
    semantic_context: dict[str, list[dict[str, object]]] = {
        "context_nodes": [],
        "interfaces": [],
    }
    if semantic_context_map is not None and section.section_id in {
        "system_context",
        "interfaces",
    }:
        semantic_context = {
            "context_nodes": [
                {
                    "title": node.title,
                    "kind": node.kind,
                    **(
                        {}
                        if hide_internal_ids
                        else {"related_subsystem_ids": node.related_subsystem_ids}
                    ),
                }
                for node in semantic_context_map.context_nodes
            ],
            "interfaces": [
                {
                    "title": interface.title,
                    "kind": interface.kind,
                    **(
                        {}
                        if hide_internal_ids
                        else {
                            "provider_subsystem_ids": interface.provider_subsystem_ids,
                            "related_module_ids": interface.related_module_ids,
                        }
                    ),
                }
                for interface in semantic_context_map.interfaces
            ],
        }
    richer_interfaces = (
        []
        if interface_inventory is None or section.section_id != "interfaces"
        else [
            (
                {
                    "title": interface.title,
                    "summary": interface.summary,
                    "responsibility_titles": [
                        responsibility.title
                        for responsibility in interface.responsibilities
                    ],
                    "contract_titles": [
                        contract.title for contract in interface.cross_module_contracts
                    ],
                }
                if hide_internal_ids
                else {
                    "interface_id": interface.interface_id,
                    "title": interface.title,
                    "summary": interface.summary,
                    "responsibility_titles": [
                        responsibility.title
                        for responsibility in interface.responsibilities
                    ],
                    "contract_titles": [
                        contract.title for contract in interface.cross_module_contracts
                    ],
                }
            )
            for interface in interface_inventory.interfaces
        ]
    )
    dependency_patterns = (
        []
        if dependency_landscape is None
        or section.section_id
        not in {"dependencies", "build_development_infrastructure"}
        else [
            {
                "title": pattern.title,
                "summary": pattern.summary,
                **(
                    {}
                    if hide_internal_ids
                    else {"dependency_ids": pattern.dependency_ids}
                ),
            }
            for pattern in dependency_landscape.usage_patterns
        ]
    )
    return {
        "concepts": concepts,
        "algorithms": algorithms,
        "dependencies": dependencies,
        "interval_and_enrichment": _relevant_interval_payload(
            section=section,
            interval_interpretation=interval_interpretation,
            checkpoint_model_enrichment=checkpoint_model_enrichment,
        ),
        "semantic_context": semantic_context,
        "interface_inventory": richer_interfaces,
        "dependency_landscape_patterns": dependency_patterns,
        "architecture_focus": _architecture_focus(
            section=section,
            checkpoint_model=checkpoint_model,
            hide_internal_ids=hide_internal_ids,
        ),
        "baseline_section_body": baseline_body,
    }


def _validate_judgment(
    *,
    judgment: HistorySectionDraftJudgment,
    section: HistorySectionPlan,
    render_section: HistoryRenderedSection,
    checkpoint_model: HistoryCheckpointModel,
    interval_interpretation: HistoryIntervalInterpretation,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
    semantic_context_map: HistorySemanticContextMap | None,
    algorithm_capsule_enrichments: list[HistoryAlgorithmCapsuleEnrichment] | None,
    interface_inventory: HistoryInterfaceInventory | None,
    dependency_landscape: HistoryDependencyLandscape | None,
    baseline_body: str,
) -> HistorySectionDraft:
    if not judgment.markdown_body.strip():
        raise ValueError("section draft markdown_body must not be empty")
    known_concept_ids = {
        *[concept.concept_id for concept in checkpoint_model.subsystems],
        *[concept.concept_id for concept in checkpoint_model.modules],
        *[concept.concept_id for concept in checkpoint_model.dependencies],
    }
    if not set(judgment.supporting_concept_ids) <= known_concept_ids:
        raise ValueError("section draft referenced unknown concept ids")
    known_capsule_ids = {capsule.capsule_id for capsule in capsules}
    if not set(judgment.supporting_algorithm_capsule_ids) <= known_capsule_ids:
        raise ValueError("section draft referenced unknown algorithm capsule ids")
    known_insight_ids = {
        insight.insight_id for insight in interval_interpretation.insights
    }
    if not set(judgment.source_insight_ids) <= known_insight_ids:
        raise ValueError("section draft referenced unknown insight ids")
    known_capability_ids = {
        proposal.capability_id
        for proposal in checkpoint_model_enrichment.capability_proposals
    }
    if not set(judgment.source_capability_ids) <= known_capability_ids:
        raise ValueError("section draft referenced unknown capability ids")
    known_design_note_ids = {
        anchor.note_id for anchor in checkpoint_model_enrichment.design_note_anchors
    }
    if not set(judgment.source_design_note_ids) <= known_design_note_ids:
        raise ValueError("section draft referenced unknown design-note ids")
    known_evidence = _known_evidence_keys(
        section.evidence_links,
        *(subsystem.evidence_links for subsystem in checkpoint_model.subsystems),
        *(module.evidence_links for module in checkpoint_model.modules),
        *(dependency.evidence_links for dependency in checkpoint_model.dependencies),
        *(insight.evidence_links for insight in interval_interpretation.insights),
        *(clue.evidence_links for clue in interval_interpretation.rationale_clues),
        *(
            anchor.evidence_links
            for anchor in checkpoint_model_enrichment.design_note_anchors
        ),
    )
    if not all(
        (link.kind, link.reference) in known_evidence
        for link in judgment.evidence_links
    ):
        raise ValueError("section draft referenced unknown evidence links")
    return HistorySectionDraft(
        section_id=section.section_id,
        markdown_body=judgment.markdown_body.strip(),
        supporting_concept_ids=list(judgment.supporting_concept_ids),
        supporting_algorithm_capsule_ids=list(
            judgment.supporting_algorithm_capsule_ids
        ),
        source_insight_ids=list(judgment.source_insight_ids),
        source_capability_ids=list(judgment.source_capability_ids),
        source_design_note_ids=list(judgment.source_design_note_ids),
        evidence_links=dedupe_evidence(
            list(judgment.evidence_links),
            list(section.evidence_links),
        ),
    )


def _fallback_draft(
    *,
    section: HistorySectionPlan,
    render_section: HistoryRenderedSection,
    baseline_body: str,
) -> HistorySectionDraft:
    return HistorySectionDraft(
        section_id=section.section_id,
        markdown_body=baseline_body,
        supporting_concept_ids=list(section.concept_ids),
        supporting_algorithm_capsule_ids=list(render_section.algorithm_capsule_ids),
        evidence_links=list(section.evidence_links),
    )


def build_section_drafts(
    *,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    render_manifest: HistoryRenderManifest,
    baseline_markdown: str,
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
    targeted_rewrite_only: bool = False,
) -> tuple[HistorySectionDraftArtifact, str, HistoryRenderManifest]:
    """Build H14-01 section drafts plus assembled draft markdown and manifest."""

    del algorithm_capsule_enrichment_index  # manifest paths stay deterministic here
    preamble_lines, baseline_bodies = extract_section_bodies(
        baseline_markdown,
        render_manifest,
    )
    sections_by_id: dict[HistorySectionPlanId, HistorySectionPlan] = {
        section.section_id: section for section in section_outline.sections
    }
    rendered_by_id: dict[HistorySectionPlanId, HistoryRenderedSection] = {
        section.section_id: section for section in render_manifest.sections
    }
    drafts: list[HistorySectionDraft] = []
    had_failure = False
    for render_section in render_manifest.sections:
        section = sections_by_id[render_section.section_id]
        baseline_body = baseline_bodies.get(section.section_id, "")
        fallback = _fallback_draft(
            section=section,
            render_section=render_section,
            baseline_body=baseline_body,
        )
        if llm_client is None:
            drafts.append(fallback)
            continue
        if (
            targeted_rewrite_only
            and section.section_id not in _TARGETED_REWRITE_LLM_SECTION_IDS
        ):
            drafts.append(fallback)
            continue
        try:
            system_prompt, user_prompt = build_section_drafting_prompt(
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
                    "targeted_rewrite_only": targeted_rewrite_only,
                },
                supporting_evidence=_section_evidence(
                    section=section,
                    render_section=render_section,
                    checkpoint_model=checkpoint_model,
                    interval_interpretation=interval_interpretation,
                    checkpoint_model_enrichment=checkpoint_model_enrichment,
                    dependency_inventory=dependency_inventory,
                    capsule_index=capsule_index,
                    capsules=capsules,
                    semantic_context_map=semantic_context_map,
                    baseline_body=baseline_body,
                    algorithm_capsule_enrichments=algorithm_capsule_enrichments,
                    interface_inventory=interface_inventory,
                    dependency_landscape=dependency_landscape,
                    hide_internal_ids=targeted_rewrite_only,
                ),
            )
            response = llm_client.generate_structured(
                StructuredGenerationRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=HistorySectionDraftJudgment,
                    model_name=model_name,
                    temperature=temperature,
                )
            )
            judgment = HistorySectionDraftJudgment.model_validate(
                response.content.model_dump(mode="python")
            )
            drafts.append(
                _validate_judgment(
                    judgment=judgment,
                    section=section,
                    render_section=render_section,
                    checkpoint_model=checkpoint_model,
                    interval_interpretation=interval_interpretation,
                    checkpoint_model_enrichment=checkpoint_model_enrichment,
                    dependency_inventory=dependency_inventory,
                    capsule_index=capsule_index,
                    capsules=capsules,
                    semantic_context_map=semantic_context_map,
                    algorithm_capsule_enrichments=algorithm_capsule_enrichments,
                    interface_inventory=interface_inventory,
                    dependency_landscape=dependency_landscape,
                    baseline_body=baseline_body,
                )
            )
        except Exception:
            had_failure = True
            drafts.append(fallback)

    drafts.sort(key=lambda item: rendered_by_id[item.section_id].order)
    evaluation_status: HistoryDraftStatus = (
        "heuristic_only"
        if llm_client is None
        else "llm_failed" if had_failure else "scored"
    )
    artifact = HistorySectionDraftArtifact(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        evaluation_status=evaluation_status,
        sections=drafts,
    )
    draft_bodies: dict[HistorySectionPlanId, str] = {
        draft.section_id: draft.markdown_body for draft in drafts
    }
    draft_markdown = assemble_shadow_markdown(
        preamble_lines=preamble_lines,
        render_manifest=render_manifest,
        section_bodies=draft_bodies,
    )
    draft_manifest = clone_render_manifest(
        render_manifest,
        markdown_filename=(
            "checkpoint_targeted_rewrite_llm.md"
            if targeted_rewrite_only
            else "checkpoint_draft_llm.md"
        ),
    )
    return artifact, draft_markdown, draft_manifest


__all__ = [
    "assemble_shadow_markdown",
    "build_section_drafts",
    "checkpoint_draft_markdown_path",
    "clone_render_manifest",
    "checkpoint_targeted_rewrite_markdown_path",
    "extract_section_bodies",
    "render_manifest_targeted_rewrite_path",
    "render_manifest_draft_path",
    "section_drafts_path",
    "targeted_section_rewrites_path",
    "validation_report_targeted_rewrite_path",
    "validation_report_draft_path",
]
