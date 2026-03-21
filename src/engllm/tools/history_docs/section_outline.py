"""History-docs section-outline planning for H5."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TypeVar

from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryEvidenceLink,
    HistoryIntervalDeltaModel,
    HistoryModuleConcept,
    HistorySectionDepth,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySectionSignalKind,
    HistorySubsystemConcept,
)

_SECTION_TITLES: dict[HistorySectionPlanId, str] = {
    "introduction": "Introduction",
    "architectural_overview": "Architectural Overview",
    "subsystems_modules": "Subsystems and Modules",
    "algorithms_core_logic": "Algorithms and Core Logic",
    "dependencies": "Dependencies",
    "build_development_infrastructure": "Build and Development Infrastructure",
    "strategy_variants_design_alternatives": "Strategy Variants and Design Alternatives",
    "data_state_management": "Data and State Management",
    "error_handling_robustness": "Error Handling and Robustness",
    "performance_considerations": "Performance Considerations",
    "security_considerations": "Security Considerations",
    "design_notes_rationale": "Design Notes and Rationale",
    "limitations_constraints": "Limitations and Constraints",
}

_BUILD_INFRA_CATEGORIES = {
    "build_config",
    "dependency_manifest",
    "dependency_lockfile",
}

_TOKEN_SETS: dict[HistorySectionPlanId, tuple[str, ...]] = {
    "data_state_management": (
        "state",
        "store",
        "cache",
        "repository",
        "session",
        "model",
        "schema",
        "queue",
        "db",
        "database",
    ),
    "error_handling_robustness": (
        "error",
        "exception",
        "retry",
        "fallback",
        "guard",
        "validate",
        "timeout",
        "recover",
        "safe",
    ),
    "performance_considerations": (
        "cache",
        "batch",
        "pool",
        "stream",
        "async",
        "parallel",
        "perf",
        "benchmark",
        "profile",
    ),
    "security_considerations": (
        "auth",
        "oauth",
        "token",
        "secret",
        "credential",
        "encrypt",
        "decrypt",
        "hash",
        "secure",
        "permission",
        "acl",
        "tls",
        "jwt",
    ),
    "limitations_constraints": (
        "limit",
        "constraint",
        "unsupported",
        "experimental",
        "fallback",
        "max",
        "min",
        "timeout",
    ),
}

_InterfaceOrArchSignals = {"architectural", "interface"}
_RationaleSignals = {"architectural", "interface", "dependency"}
_ConceptT = TypeVar("_ConceptT")


def section_outline_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the tool-scoped section-outline artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "section_outline.json"


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _active_only(concepts: list[_ConceptT]) -> list[_ConceptT]:
    return [
        concept
        for concept in concepts
        if getattr(concept, "lifecycle_status", None) == "active"
    ]


def _concept_order_key(concept_id: str) -> tuple[int, str]:
    if concept_id.startswith("subsystem::"):
        return (0, concept_id)
    if concept_id.startswith("module::"):
        return (1, concept_id)
    if concept_id.startswith("dependency-source::"):
        return (2, concept_id)
    return (3, concept_id)


def _sorted_concept_ids(concept_ids: set[str]) -> list[str]:
    return sorted(concept_ids, key=_concept_order_key)


def _signal_list(
    *signals: HistorySectionSignalKind | None,
) -> list[HistorySectionSignalKind]:
    return sorted({signal for signal in signals if signal is not None})


def _commit_evidence_for_signals(
    delta_model: HistoryIntervalDeltaModel,
    signals: set[str],
) -> list[HistoryEvidenceLink]:
    return _dedupe_evidence(
        *[
            commit_delta.evidence_links
            for commit_delta in delta_model.commit_deltas
            if signals & set(commit_delta.signal_kinds)
        ]
    )


def _core_depth(score: int) -> HistorySectionDepth:
    if score <= 5:
        return "brief"
    if score <= 7:
        return "standard"
    return "deep"


def _optional_depth(score: int) -> HistorySectionDepth:
    if score <= 6:
        return "brief"
    if score <= 8:
        return "standard"
    return "deep"


def _search_terms(module: HistoryModuleConcept) -> list[str]:
    return [
        module.path.as_posix().lower(),
        *(value.lower() for value in module.functions),
        *(value.lower() for value in module.classes),
        *(value.lower() for value in module.imports),
        *(value.lower() for value in module.docstrings),
        *(value.lower() for value in module.symbol_names),
    ]


def _module_matches_tokens(
    module: HistoryModuleConcept,
    tokens: tuple[str, ...],
) -> bool:
    terms = _search_terms(module)
    return any(token in term for token in tokens for term in terms)


def _active_module_maps(
    checkpoint_model: HistoryCheckpointModel,
) -> tuple[
    list[HistorySubsystemConcept],
    list[HistoryModuleConcept],
    list[HistoryDependencyConcept],
    dict[str, HistorySubsystemConcept],
    dict[str, HistoryModuleConcept],
    dict[str, HistoryDependencyConcept],
    dict[Path, HistoryModuleConcept],
]:
    active_subsystems = sorted(
        _active_only(checkpoint_model.subsystems), key=lambda item: item.concept_id
    )
    active_modules = sorted(
        _active_only(checkpoint_model.modules), key=lambda item: item.concept_id
    )
    active_dependencies = sorted(
        _active_only(checkpoint_model.dependencies), key=lambda item: item.concept_id
    )
    return (
        active_subsystems,
        active_modules,
        active_dependencies,
        {item.concept_id: item for item in active_subsystems},
        {item.concept_id: item for item in active_modules},
        {item.concept_id: item for item in active_dependencies},
        {item.path: item for item in active_modules},
    )


def _plan_core_section(
    *,
    section_id: HistorySectionPlanId,
    concept_ids: list[str],
    evidence_links: list[HistoryEvidenceLink],
    evidence_score: int,
    trigger_signals: list[HistorySectionSignalKind],
) -> HistorySectionPlan:
    if section_id == "introduction":
        return HistorySectionPlan(
            section_id=section_id,
            title=_SECTION_TITLES[section_id],
            kind="core",
            status="included",
            confidence_score=100,
            evidence_score=10,
            depth="brief",
            concept_ids=[],
            evidence_links=[],
            trigger_signals=[],
            omission_reason=None,
        )

    return HistorySectionPlan(
        section_id=section_id,
        title=_SECTION_TITLES[section_id],
        kind="core",
        status="included",
        confidence_score=max(60, min(100, evidence_score * 10)),
        evidence_score=evidence_score,
        depth=_core_depth(evidence_score),
        concept_ids=concept_ids,
        evidence_links=evidence_links,
        trigger_signals=sorted(trigger_signals),
        omission_reason=None,
    )


def _plan_optional_section(
    *,
    section_id: HistorySectionPlanId,
    concept_ids: list[str],
    evidence_links: list[HistoryEvidenceLink],
    evidence_score: int,
    trigger_signals: list[HistorySectionSignalKind],
    special_case_included: bool = False,
) -> HistorySectionPlan:
    included = (
        evidence_score >= 5 and len(evidence_links) >= 2
    ) or special_case_included
    return HistorySectionPlan(
        section_id=section_id,
        title=_SECTION_TITLES[section_id],
        kind="optional",
        status="included" if included else "omitted",
        confidence_score=min(100, evidence_score * 10),
        evidence_score=evidence_score,
        depth=_optional_depth(evidence_score) if included else None,
        concept_ids=concept_ids,
        evidence_links=evidence_links,
        trigger_signals=sorted(trigger_signals),
        omission_reason=None if included else "insufficient_evidence",
    )


def build_section_outline(
    checkpoint_model: HistoryCheckpointModel,
    interval_delta_model: HistoryIntervalDeltaModel,
) -> HistorySectionOutline:
    """Build the H5 scored section outline for one checkpoint."""

    (
        active_subsystems,
        active_modules,
        active_dependencies,
        active_subsystems_by_id,
        _,
        active_dependencies_by_id,
        active_modules_by_path,
    ) = _active_module_maps(checkpoint_model)

    infra_dependencies = [
        dependency
        for dependency in active_dependencies
        if dependency.category in _BUILD_INFRA_CATEGORIES
    ]

    architectural_commit_evidence = _commit_evidence_for_signals(
        interval_delta_model, {"architectural"}
    )
    arch_or_interface_commit_evidence = _commit_evidence_for_signals(
        interval_delta_model, _InterfaceOrArchSignals
    )
    dependency_commit_evidence = _commit_evidence_for_signals(
        interval_delta_model, {"dependency"}
    )
    infrastructure_commit_evidence = _commit_evidence_for_signals(
        interval_delta_model, {"infrastructure"}
    )

    core_sections = [
        _plan_core_section(
            section_id="introduction",
            concept_ids=[],
            evidence_links=[],
            evidence_score=10,
            trigger_signals=[],
        ),
        _plan_core_section(
            section_id="architectural_overview",
            concept_ids=[item.concept_id for item in active_subsystems],
            evidence_links=_dedupe_evidence(
                *[item.evidence_links for item in active_subsystems],
                architectural_commit_evidence,
            ),
            evidence_score=(
                4
                + min(len(active_subsystems), 3)
                + (1 if architectural_commit_evidence else 0)
                + (1 if len(active_subsystems) >= 2 else 0)
            ),
            trigger_signals=_signal_list(
                "active_subsystems" if active_subsystems else None,
                "architectural_change" if architectural_commit_evidence else None,
            ),
        ),
        _plan_core_section(
            section_id="subsystems_modules",
            concept_ids=[item.concept_id for item in active_subsystems]
            + [item.concept_id for item in active_modules],
            evidence_links=_dedupe_evidence(
                *[item.evidence_links for item in active_subsystems],
                *[item.evidence_links for item in active_modules],
                arch_or_interface_commit_evidence,
            ),
            evidence_score=(
                4
                + min(len(active_subsystems), 2)
                + min(len(active_modules) // 2, 3)
                + (1 if arch_or_interface_commit_evidence else 0)
            ),
            trigger_signals=_signal_list(
                "active_subsystems" if active_subsystems else None,
                "active_modules" if active_modules else None,
                "architectural_change" if architectural_commit_evidence else None,
                (
                    "interface_change"
                    if any(
                        "interface" in commit_delta.signal_kinds
                        for commit_delta in interval_delta_model.commit_deltas
                    )
                    else None
                ),
            ),
        ),
        _plan_core_section(
            section_id="dependencies",
            concept_ids=[item.concept_id for item in active_dependencies],
            evidence_links=_dedupe_evidence(
                *[item.evidence_links for item in active_dependencies],
                dependency_commit_evidence,
            ),
            evidence_score=(
                4
                + min(len(active_dependencies), 3)
                + (1 if dependency_commit_evidence else 0)
            ),
            trigger_signals=_signal_list(
                "active_dependencies" if active_dependencies else None,
                "dependency_change" if dependency_commit_evidence else None,
            ),
        ),
        _plan_core_section(
            section_id="build_development_infrastructure",
            concept_ids=[item.concept_id for item in infra_dependencies],
            evidence_links=_dedupe_evidence(
                *[item.evidence_links for item in infra_dependencies],
                infrastructure_commit_evidence,
            ),
            evidence_score=(
                4
                + min(len(infra_dependencies), 3)
                + (1 if infrastructure_commit_evidence else 0)
            ),
            trigger_signals=_signal_list(
                "active_dependencies" if infra_dependencies else None,
                "infrastructure_change" if infrastructure_commit_evidence else None,
            ),
        ),
    ]

    token_matches: dict[HistorySectionPlanId, list[HistoryModuleConcept]] = {}
    for section_id, tokens in _TOKEN_SETS.items():
        token_matches[section_id] = [
            module
            for module in active_modules
            if _module_matches_tokens(module, tokens)
        ]

    optional_sections: list[HistorySectionPlan] = []

    variant_candidates = [
        candidate
        for candidate in interval_delta_model.algorithm_candidates
        if "variant_family" in candidate.signal_kinds
    ]
    variant_subsystem_counts = Counter(
        candidate.subsystem_id
        for candidate in variant_candidates
        if candidate.subsystem_id is not None
    )
    strategy_concept_ids: set[str] = set()
    strategy_evidence: list[HistoryEvidenceLink] = []
    for candidate in variant_candidates:
        if (
            candidate.subsystem_id is not None
            and candidate.subsystem_id in active_subsystems_by_id
        ):
            strategy_concept_ids.add(candidate.subsystem_id)
        if candidate.scope_path in active_modules_by_path:
            strategy_concept_ids.add(
                active_modules_by_path[candidate.scope_path].concept_id
            )
        strategy_evidence.extend(candidate.evidence_links)
    strategy_score = 0
    if variant_candidates:
        strategy_score += 4
    if any(len(candidate.variant_names) >= 2 for candidate in variant_candidates):
        strategy_score += 2
    if any(count >= 2 for count in variant_subsystem_counts.values()):
        strategy_score += 1
    strategy_triggers: set[HistorySectionSignalKind] = set()
    if variant_candidates:
        strategy_triggers.add("algorithm_candidate")
        strategy_triggers.add("variant_family")
    optional_sections.append(
        _plan_optional_section(
            section_id="strategy_variants_design_alternatives",
            concept_ids=_sorted_concept_ids(strategy_concept_ids),
            evidence_links=_dedupe_evidence(strategy_evidence),
            evidence_score=strategy_score,
            trigger_signals=sorted(strategy_triggers),
            special_case_included=any(
                len(candidate.variant_names) >= 2 for candidate in variant_candidates
            ),
        )
    )

    def token_section(
        section_id: HistorySectionPlanId,
        signal_kind: HistorySectionSignalKind,
    ) -> HistorySectionPlan:
        matched_modules = token_matches[section_id]
        concept_ids = {module.concept_id for module in matched_modules}
        concept_ids.update(
            module.subsystem_id
            for module in matched_modules
            if module.subsystem_id is not None
            and module.subsystem_id in active_subsystems_by_id
        )
        evidence = _dedupe_evidence(
            *[module.evidence_links for module in matched_modules],
            *[
                active_subsystems_by_id[module.subsystem_id].evidence_links
                for module in matched_modules
                if module.subsystem_id is not None
                and module.subsystem_id in active_subsystems_by_id
            ],
        )
        trigger_signals: set[HistorySectionSignalKind] = set()
        if matched_modules:
            trigger_signals.update({"active_modules", signal_kind})
        return _plan_optional_section(
            section_id=section_id,
            concept_ids=_sorted_concept_ids(concept_ids),
            evidence_links=evidence,
            evidence_score=min(len(matched_modules) * 2, 6),
            trigger_signals=sorted(trigger_signals),
        )

    optional_sections.append(
        token_section("data_state_management", "data_state_tokens")
    )

    robustness_modules = token_matches["error_handling_robustness"]
    robustness_concept_ids = {module.concept_id for module in robustness_modules}
    robustness_concept_ids.update(
        module.subsystem_id
        for module in robustness_modules
        if module.subsystem_id is not None
        and module.subsystem_id in active_subsystems_by_id
    )
    robustness_interface_candidates = [
        candidate
        for candidate in interval_delta_model.interface_changes
        if any(
            token in line.lower()
            for line in candidate.signature_changes
            for token in _TOKEN_SETS["error_handling_robustness"]
        )
    ]
    robustness_score = min(len(robustness_modules) * 2, 6) + min(
        len(robustness_interface_candidates), 2
    )
    robustness_triggers: set[HistorySectionSignalKind] = set()
    if robustness_modules:
        robustness_triggers.update({"active_modules", "robustness_tokens"})
    if robustness_interface_candidates:
        robustness_triggers.add("interface_change")
    optional_sections.append(
        _plan_optional_section(
            section_id="error_handling_robustness",
            concept_ids=_sorted_concept_ids(robustness_concept_ids),
            evidence_links=_dedupe_evidence(
                *[module.evidence_links for module in robustness_modules],
                *[
                    active_subsystems_by_id[module.subsystem_id].evidence_links
                    for module in robustness_modules
                    if module.subsystem_id is not None
                    and module.subsystem_id in active_subsystems_by_id
                ],
                *[
                    candidate.evidence_links
                    for candidate in robustness_interface_candidates
                ],
            ),
            evidence_score=robustness_score,
            trigger_signals=sorted(robustness_triggers),
        )
    )

    optional_sections.append(
        token_section("performance_considerations", "performance_tokens")
    )
    optional_sections.append(
        token_section("security_considerations", "security_tokens")
    )

    rationale_commit_deltas = [
        commit_delta
        for commit_delta in interval_delta_model.commit_deltas
        if _RationaleSignals & set(commit_delta.signal_kinds)
    ]
    rationale_concept_ids: set[str] = set()
    touched_active_module_ids: set[str] = set()
    for commit_delta in rationale_commit_deltas:
        for link in commit_delta.evidence_links:
            if link.kind == "subsystem" and link.reference in active_subsystems_by_id:
                rationale_concept_ids.add(link.reference)
            if link.kind == "file":
                file_path = Path(link.reference)
                if file_path in active_modules_by_path:
                    touched_active_module_ids.add(
                        active_modules_by_path[file_path].concept_id
                    )
    rationale_concept_ids.update(touched_active_module_ids)
    rationale_concept_ids.update(
        dependency.concept_id
        for dependency in active_dependencies
        if dependency.change_status in {"introduced", "modified", "retired", "observed"}
        and any(
            candidate.path == dependency.path
            for candidate in interval_delta_model.dependency_changes
        )
    )
    introduced_bonus = min(
        2,
        sum(
            any(candidate.status == "introduced" for candidate in candidates)
            for candidates in (
                interval_delta_model.subsystem_changes,
                interval_delta_model.interface_changes,
                interval_delta_model.dependency_changes,
            )
        ),
    )
    retired_bonus = min(
        2,
        sum(
            any(candidate.status == "retired" for candidate in candidates)
            for candidates in (
                interval_delta_model.subsystem_changes,
                interval_delta_model.interface_changes,
                interval_delta_model.dependency_changes,
            )
        ),
    )
    rationale_triggers: set[HistorySectionSignalKind] = set()
    if rationale_commit_deltas:
        rationale_triggers.add("rationale_change")
    if any("architectural" in delta.signal_kinds for delta in rationale_commit_deltas):
        rationale_triggers.add("architectural_change")
    if any("interface" in delta.signal_kinds for delta in rationale_commit_deltas):
        rationale_triggers.add("interface_change")
    if any("dependency" in delta.signal_kinds for delta in rationale_commit_deltas):
        rationale_triggers.add("dependency_change")
    optional_sections.append(
        _plan_optional_section(
            section_id="design_notes_rationale",
            concept_ids=_sorted_concept_ids(rationale_concept_ids),
            evidence_links=_dedupe_evidence(
                *[
                    active_subsystems_by_id[concept_id].evidence_links
                    for concept_id in rationale_concept_ids
                    if concept_id in active_subsystems_by_id
                ],
                *[
                    active_modules_by_path[
                        Path(concept_id.removeprefix("module::"))
                    ].evidence_links
                    for concept_id in rationale_concept_ids
                    if concept_id.startswith("module::")
                    and Path(concept_id.removeprefix("module::"))
                    in active_modules_by_path
                ],
                *[
                    active_dependencies_by_id[concept_id].evidence_links
                    for concept_id in rationale_concept_ids
                    if concept_id in active_dependencies_by_id
                ],
                *[
                    commit_delta.evidence_links
                    for commit_delta in rationale_commit_deltas
                ],
                *[
                    candidate.evidence_links
                    for candidate in interval_delta_model.subsystem_changes
                ],
                *[
                    candidate.evidence_links
                    for candidate in interval_delta_model.interface_changes
                ],
                *[
                    candidate.evidence_links
                    for candidate in interval_delta_model.dependency_changes
                ],
            ),
            evidence_score=min(len(rationale_commit_deltas) * 2, 6)
            + introduced_bonus
            + retired_bonus,
            trigger_signals=sorted(rationale_triggers),
        )
    )

    optional_sections.append(
        token_section("limitations_constraints", "limitations_tokens")
    )

    return HistorySectionOutline(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        sections=[*core_sections, *optional_sections],
    )


__all__ = ["build_section_outline", "section_outline_path"]
