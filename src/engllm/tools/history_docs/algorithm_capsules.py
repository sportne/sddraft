"""Deterministic H6 algorithm capsule builders and linkers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from engllm.tools.history_docs.models import (
    HistoryAlgorithmAssumption,
    HistoryAlgorithmCandidate,
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryAlgorithmCapsuleIndexEntry,
    HistoryAlgorithmCapsuleStatus,
    HistoryAlgorithmDataStructure,
    HistoryAlgorithmDataStructureKind,
    HistoryAlgorithmPhase,
    HistoryAlgorithmScopeKind,
    HistoryAlgorithmSharedAbstraction,
    HistoryAlgorithmSharedAbstractionKind,
    HistoryCheckpointModel,
    HistoryEvidenceLink,
    HistoryIntervalDeltaModel,
    HistoryModuleConcept,
    HistorySectionDepth,
    HistorySectionId,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySectionSignalKind,
    HistorySectionState,
    HistorySubsystemConcept,
)

_PHASE_TOKENS: tuple[str, ...] = (
    "parse",
    "load",
    "scan",
    "resolve",
    "validate",
    "plan",
    "build",
    "index",
    "merge",
    "execute",
    "run",
    "rank",
    "render",
    "save",
)
_DATA_STRUCTURE_TOKENS: tuple[str, ...] = (
    "state",
    "config",
    "request",
    "response",
    "context",
    "model",
    "record",
    "queue",
    "cache",
    "graph",
    "node",
    "edge",
)
_ASSUMPTION_TOKENS: tuple[str, ...] = (
    "must",
    "require",
    "assume",
    "unsupported",
    "fallback",
    "strict",
    "deterministic",
    "conservative",
    "only",
    "expects",
)
_CHECKPOINT_SECTION_TITLES: dict[HistorySectionId, str] = {
    "introduction": "Introduction",
    "architectural_overview": "Architectural Overview",
    "subsystems_modules": "Subsystems and Modules",
    "algorithms_core_logic": "Algorithms and Core Logic",
    "dependencies": "Dependencies",
    "build_development_infrastructure": "Build and Development Infrastructure",
}
_OUTLINE_SECTION_TITLES: dict[HistorySectionPlanId, str] = {
    "introduction": "Introduction",
    "architectural_overview": "Architectural Overview",
    "subsystems_modules": "Subsystems and Modules",
    "algorithms_core_logic": "Algorithms and Core Logic",
    "dependencies": "Dependencies",
    "build_development_infrastructure": "Build and Development Infrastructure",
    "strategy_variants_design_alternatives": (
        "Strategy Variants and Design Alternatives"
    ),
    "data_state_management": "Data and State Management",
    "error_handling_robustness": "Error Handling and Robustness",
    "performance_considerations": "Performance Considerations",
    "security_considerations": "Security Considerations",
    "design_notes_rationale": "Design Notes and Rationale",
    "limitations_constraints": "Limitations and Constraints",
}
_CHECKPOINT_SECTION_ORDER: tuple[HistorySectionId, ...] = (
    "introduction",
    "architectural_overview",
    "subsystems_modules",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
)
_OUTLINE_SECTION_ORDER: tuple[HistorySectionPlanId, ...] = (
    "introduction",
    "architectural_overview",
    "subsystems_modules",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
)
_FILE_OR_MODULE_PREFIX = "algorithm-capsule::file::"
_SUBSYSTEM_PREFIX = "algorithm-capsule::subsystem::"


@dataclass(slots=True)
class _CapsuleSeed:
    capsule_id: str
    scope_kind: HistoryAlgorithmScopeKind
    scope_path: Path
    subsystem_id: str | None = None
    candidates: list[HistoryAlgorithmCandidate] = field(default_factory=list)


def algorithm_capsule_dir(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the per-checkpoint algorithm capsule directory."""

    return tool_root / "checkpoints" / checkpoint_id / "algorithm_capsules"


def algorithm_capsule_index_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the per-checkpoint algorithm capsule index path."""

    return algorithm_capsule_dir(tool_root, checkpoint_id) / "index.json"


def algorithm_capsule_filename(capsule_id: str) -> str:
    """Return the deterministic artifact filename for one capsule id."""

    return f"{re.sub(r'[^A-Za-z0-9._-]', '_', capsule_id)}.json"


def _title_case(value: str) -> str:
    tokens = value.replace("_", " ").replace("-", " ").split()
    return " ".join(token.capitalize() for token in tokens) or "Root"


def _capsule_title(scope_kind: str, scope_path: Path) -> str:
    if scope_kind == "subsystem":
        leaf = scope_path.name or scope_path.as_posix()
        return f"Algorithm Cluster: {_title_case(leaf)}"
    return f"Algorithm Module: {_title_case(scope_path.stem)}"


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _active_maps(
    checkpoint_model: HistoryCheckpointModel,
) -> tuple[
    dict[str, HistorySubsystemConcept],
    dict[str, HistoryModuleConcept],
    dict[Path, HistoryModuleConcept],
]:
    active_subsystems = {
        concept.concept_id: concept
        for concept in checkpoint_model.subsystems
        if concept.lifecycle_status == "active"
    }
    active_modules = {
        concept.concept_id: concept
        for concept in checkpoint_model.modules
        if concept.lifecycle_status == "active"
    }
    active_modules_by_path = {
        concept.path: concept for concept in active_modules.values()
    }
    return active_subsystems, active_modules, active_modules_by_path


def _candidate_file_paths(candidate: HistoryAlgorithmCandidate) -> list[Path]:
    file_paths = {
        Path(link.reference) for link in candidate.evidence_links if link.kind == "file"
    }
    if candidate.scope_kind == "file":
        file_paths.add(candidate.scope_path)
    return sorted(file_paths, key=lambda item: item.as_posix())


def _resolve_related_module_ids(
    *,
    seed: _CapsuleSeed,
    active_subsystems: dict[str, HistorySubsystemConcept],
    active_modules: dict[str, HistoryModuleConcept],
    active_modules_by_path: dict[Path, HistoryModuleConcept],
) -> list[str]:
    resolved: set[str] = set()
    for candidate in seed.candidates:
        for file_path in _candidate_file_paths(candidate):
            module = active_modules_by_path.get(file_path)
            if module is not None:
                resolved.add(module.concept_id)
    if resolved:
        return sorted(resolved)
    if seed.scope_kind == "subsystem" and seed.subsystem_id in active_subsystems:
        subsystem = active_subsystems[seed.subsystem_id]
        return [
            module_id
            for module_id in subsystem.module_ids
            if module_id in active_modules
        ][:6]
    return []


def _name_evidence(
    modules: list[HistoryModuleConcept],
    *,
    name: str,
) -> list[HistoryEvidenceLink]:
    evidence: list[HistoryEvidenceLink] = []
    for module in modules:
        if name in {*(module.functions), *(module.classes), *(module.symbol_names)}:
            evidence.extend(module.evidence_links)
            evidence.append(
                HistoryEvidenceLink(
                    kind="symbol",
                    reference=f"{module.path.as_posix()}::{name}",
                )
            )
    return _dedupe_evidence(evidence)


def _shared_abstractions(
    modules: list[HistoryModuleConcept],
) -> list[HistoryAlgorithmSharedAbstraction]:
    if len(modules) < 2:
        return []

    name_to_modules: dict[str, set[str]] = defaultdict(set)
    function_names: set[str] = set()
    class_names: set[str] = set()
    for module in modules:
        for name in set(module.functions):
            if len(name) >= 3 and not name.startswith("_"):
                function_names.add(name)
                name_to_modules[name].add(module.concept_id)
        for name in set(module.classes):
            if len(name) >= 3 and not name.startswith("_"):
                class_names.add(name)
                name_to_modules[name].add(module.concept_id)
        for name in set(module.symbol_names):
            if len(name) >= 3 and not name.startswith("_"):
                name_to_modules[name].add(module.concept_id)

    abstractions: list[HistoryAlgorithmSharedAbstraction] = []
    for name in sorted(name_to_modules, key=lambda value: value.lower()):
        if len(name_to_modules[name]) < 2:
            continue
        kind: HistoryAlgorithmSharedAbstractionKind
        if name in function_names:
            kind = "function"
        elif name in class_names:
            kind = "class"
        else:
            kind = "symbol"
        abstractions.append(
            HistoryAlgorithmSharedAbstraction(
                name=name,
                kind=kind,
                evidence_links=_name_evidence(modules, name=name),
            )
        )
        if len(abstractions) >= 6:
            break
    return abstractions


def _data_structures(
    modules: list[HistoryModuleConcept],
) -> list[HistoryAlgorithmDataStructure]:
    names: dict[str, HistoryAlgorithmDataStructureKind] = {}
    for module in modules:
        for name in module.classes:
            if any(token in name.lower() for token in _DATA_STRUCTURE_TOKENS):
                names.setdefault(name, "class")
        for name in module.symbol_names:
            if any(token in name.lower() for token in _DATA_STRUCTURE_TOKENS):
                names.setdefault(name, "symbol")

    structures: list[HistoryAlgorithmDataStructure] = []
    for name in sorted(names, key=lambda value: value.lower())[:6]:
        structures.append(
            HistoryAlgorithmDataStructure(
                name=name,
                kind=names[name],
                evidence_links=_name_evidence(modules, name=name),
            )
        )
    return structures


def _phases(modules: list[HistoryModuleConcept]) -> list[HistoryAlgorithmPhase]:
    phases: list[HistoryAlgorithmPhase] = []
    for order, token in enumerate(_PHASE_TOKENS, start=1):
        matched_names = sorted(
            {
                name
                for module in modules
                for name in [*module.functions, *module.symbol_names]
                if token in name.lower()
            },
            key=lambda value: value.lower(),
        )
        if not matched_names:
            continue
        phases.append(
            HistoryAlgorithmPhase(
                phase_key=token,
                order=order,
                matched_names=matched_names,
                evidence_links=_dedupe_evidence(
                    *[_name_evidence(modules, name=name) for name in matched_names]
                ),
            )
        )
    return phases


def _normalize_line(value: str) -> str:
    return " ".join(value.strip().split())


def _assumptions(
    modules: list[HistoryModuleConcept],
    interval_delta_model: HistoryIntervalDeltaModel,
) -> list[HistoryAlgorithmAssumption]:
    module_by_path = {module.path: module for module in modules}
    entries: list[HistoryAlgorithmAssumption] = []
    seen: set[tuple[str, str]] = set()

    for module in modules:
        for line in "\n".join(module.docstrings).splitlines():
            normalized = _normalize_line(line)
            if not normalized:
                continue
            if not any(token in normalized.lower() for token in _ASSUMPTION_TOKENS):
                continue
            key = (normalized, "docstring")
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                HistoryAlgorithmAssumption(
                    text=normalized,
                    source_kind="docstring",
                    evidence_links=_dedupe_evidence(module.evidence_links),
                )
            )

    for candidate in interval_delta_model.interface_changes:
        if candidate.source_path not in module_by_path:
            continue
        for line in candidate.signature_changes:
            normalized = _normalize_line(line)
            if not normalized:
                continue
            if not any(token in normalized.lower() for token in _ASSUMPTION_TOKENS):
                continue
            key = (normalized, "signature_change")
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                HistoryAlgorithmAssumption(
                    text=normalized,
                    source_kind="signature_change",
                    evidence_links=_dedupe_evidence(candidate.evidence_links),
                )
            )

    return sorted(entries, key=lambda item: (item.text.lower(), item.source_kind))[:5]


def _build_capsule(
    *,
    seed: _CapsuleSeed,
    checkpoint_model: HistoryCheckpointModel,
    interval_delta_model: HistoryIntervalDeltaModel,
) -> HistoryAlgorithmCapsule:
    active_subsystems, active_modules, active_modules_by_path = _active_maps(
        checkpoint_model
    )
    related_subsystem_ids = (
        [seed.subsystem_id]
        if seed.subsystem_id is not None and seed.subsystem_id in active_subsystems
        else []
    )
    related_module_ids = _resolve_related_module_ids(
        seed=seed,
        active_subsystems=active_subsystems,
        active_modules=active_modules,
        active_modules_by_path=active_modules_by_path,
    )
    related_modules = [active_modules[module_id] for module_id in related_module_ids]

    signal_kinds = sorted(
        {signal for candidate in seed.candidates for signal in candidate.signal_kinds}
    )
    commit_ids = sorted(
        {
            commit_id
            for candidate in seed.candidates
            for commit_id in candidate.commit_ids
        }
    )
    changed_symbol_names = sorted(
        {
            name
            for candidate in seed.candidates
            for name in candidate.changed_symbol_names
        }
    )
    variant_names = sorted(
        {name for candidate in seed.candidates for name in candidate.variant_names}
    )
    evidence_links = _dedupe_evidence(
        *[candidate.evidence_links for candidate in seed.candidates]
    )
    status: HistoryAlgorithmCapsuleStatus = (
        "observed"
        if not interval_delta_model.previous_snapshot_available
        else (
            "introduced"
            if any(
                "introduced_module" in candidate.signal_kinds
                for candidate in seed.candidates
            )
            else "modified"
        )
    )

    return HistoryAlgorithmCapsule(
        capsule_id=seed.capsule_id,
        title=_capsule_title(seed.scope_kind, seed.scope_path),
        status=status,
        scope_kind=seed.scope_kind,
        scope_path=seed.scope_path,
        related_subsystem_ids=related_subsystem_ids,
        related_module_ids=related_module_ids,
        source_candidate_ids=sorted(
            candidate.candidate_id for candidate in seed.candidates
        ),
        commit_ids=commit_ids,
        changed_symbol_names=changed_symbol_names,
        variant_names=variant_names,
        signal_kinds=signal_kinds,
        shared_abstractions=_shared_abstractions(related_modules),
        data_structures=_data_structures(related_modules),
        phases=_phases(related_modules),
        assumptions=_assumptions(related_modules, interval_delta_model),
        evidence_links=evidence_links,
    )


def build_algorithm_capsules(
    checkpoint_model: HistoryCheckpointModel,
    interval_delta_model: HistoryIntervalDeltaModel,
) -> tuple[HistoryAlgorithmCapsuleIndex, list[HistoryAlgorithmCapsule]]:
    """Build deterministic H6 algorithm capsules from H3 candidates."""

    active_subsystems, _, _ = _active_maps(checkpoint_model)
    seeds: dict[str, _CapsuleSeed] = {}
    attached_candidate_ids: set[str] = set()

    for candidate in interval_delta_model.algorithm_candidates:
        if candidate.scope_kind != "subsystem" or candidate.subsystem_id is None:
            continue
        capsule_id = f"{_SUBSYSTEM_PREFIX}{candidate.subsystem_id}"
        seed = seeds.setdefault(
            capsule_id,
            _CapsuleSeed(
                capsule_id=capsule_id,
                scope_kind="subsystem",
                scope_path=candidate.scope_path,
                subsystem_id=candidate.subsystem_id,
            ),
        )
        seed.candidates.append(candidate)
        attached_candidate_ids.add(candidate.candidate_id)

    for candidate in interval_delta_model.algorithm_candidates:
        if candidate.scope_kind != "file" or candidate.subsystem_id is None:
            continue
        capsule_id = f"{_SUBSYSTEM_PREFIX}{candidate.subsystem_id}"
        if capsule_id not in seeds:
            continue
        seeds[capsule_id].candidates.append(candidate)
        attached_candidate_ids.add(candidate.candidate_id)

    for candidate in interval_delta_model.algorithm_candidates:
        if (
            candidate.scope_kind != "file"
            or candidate.candidate_id in attached_candidate_ids
        ):
            continue
        signal_kinds = set(candidate.signal_kinds)
        qualifies = {"introduced_module", "multi_symbol"} <= signal_kinds or len(
            candidate.changed_symbol_names
        ) >= 3
        if not qualifies:
            continue
        capsule_id = f"{_FILE_OR_MODULE_PREFIX}{candidate.scope_path.as_posix()}"
        seeds[capsule_id] = _CapsuleSeed(
            capsule_id=capsule_id,
            scope_kind="file",
            scope_path=candidate.scope_path,
            subsystem_id=(
                candidate.subsystem_id
                if candidate.subsystem_id in active_subsystems
                else None
            ),
            candidates=[candidate],
        )

    capsules = [
        _build_capsule(
            seed=seed,
            checkpoint_model=checkpoint_model,
            interval_delta_model=interval_delta_model,
        )
        for _, seed in sorted(seeds.items())
    ]
    index = HistoryAlgorithmCapsuleIndex(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        capsules=[
            HistoryAlgorithmCapsuleIndexEntry(
                capsule_id=capsule.capsule_id,
                title=capsule.title,
                status=capsule.status,
                scope_kind=capsule.scope_kind,
                scope_path=capsule.scope_path,
                artifact_path=Path("algorithm_capsules")
                / algorithm_capsule_filename(capsule.capsule_id),
            )
            for capsule in capsules
        ],
    )
    return index, capsules


def _overlapping_capsule_ids(
    capsules: list[HistoryAlgorithmCapsule],
    concept_ids: list[str],
) -> list[str]:
    concept_id_set = set(concept_ids)
    return [
        capsule.capsule_id
        for capsule in capsules
        if concept_id_set
        & set(capsule.related_subsystem_ids + capsule.related_module_ids)
    ]


def link_algorithm_capsules_to_checkpoint_model(
    checkpoint_model: HistoryCheckpointModel,
    capsules: list[HistoryAlgorithmCapsule],
) -> HistoryCheckpointModel:
    """Return a checkpoint model rewritten with H6 capsule links."""

    linked_model = checkpoint_model.model_copy(deep=True)
    linked_model.algorithm_capsule_ids = [capsule.capsule_id for capsule in capsules]

    subsystem_capsules: dict[str, list[str]] = defaultdict(list)
    module_capsules: dict[str, list[str]] = defaultdict(list)
    for capsule in capsules:
        for subsystem_id in capsule.related_subsystem_ids:
            subsystem_capsules[subsystem_id].append(capsule.capsule_id)
        for module_id in capsule.related_module_ids:
            module_capsules[module_id].append(capsule.capsule_id)

    for subsystem in linked_model.subsystems:
        subsystem.algorithm_capsule_ids = sorted(
            subsystem_capsules[subsystem.concept_id]
        )
    for module in linked_model.modules:
        module.algorithm_capsule_ids = sorted(module_capsules[module.concept_id])

    active_subsystems = {
        subsystem.concept_id
        for subsystem in linked_model.subsystems
        if subsystem.lifecycle_status == "active"
    }
    active_modules = {
        module.concept_id
        for module in linked_model.modules
        if module.lifecycle_status == "active"
    }
    algorithm_concept_ids = sorted(
        {
            subsystem_id
            for capsule in capsules
            for subsystem_id in capsule.related_subsystem_ids
            if subsystem_id in active_subsystems
        }
    ) + sorted(
        {
            module_id
            for capsule in capsules
            for module_id in capsule.related_module_ids
            if module_id in active_modules
        }
    )
    algorithm_evidence = _dedupe_evidence(
        *[capsule.evidence_links for capsule in capsules]
    )

    existing_sections: dict[HistorySectionId, HistorySectionState] = {
        section.section_id: section.model_copy(deep=True)
        for section in linked_model.sections
    }
    subsystems_modules = existing_sections.get("subsystems_modules")
    if subsystems_modules is not None:
        subsystems_modules.algorithm_capsule_ids = _overlapping_capsule_ids(
            capsules,
            subsystems_modules.concept_ids,
        )

    existing_sections["algorithms_core_logic"] = HistorySectionState(
        section_id="algorithms_core_logic",
        title=_CHECKPOINT_SECTION_TITLES["algorithms_core_logic"],
        concept_ids=algorithm_concept_ids,
        algorithm_capsule_ids=[capsule.capsule_id for capsule in capsules],
        evidence_links=algorithm_evidence,
    )
    linked_model.sections = [
        existing_sections.get(
            section_id,
            HistorySectionState(
                section_id=section_id,
                title=_CHECKPOINT_SECTION_TITLES[section_id],
            ),
        )
        for section_id in _CHECKPOINT_SECTION_ORDER
    ]
    return linked_model


def _algorithm_outline_plan(
    *,
    checkpoint_model: HistoryCheckpointModel,
    interval_delta_model: HistoryIntervalDeltaModel,
    capsules: list[HistoryAlgorithmCapsule],
) -> HistorySectionPlan:
    active_subsystems, active_modules, _ = _active_maps(checkpoint_model)
    concept_ids = sorted(
        {
            subsystem_id
            for capsule in capsules
            for subsystem_id in capsule.related_subsystem_ids
            if subsystem_id in active_subsystems
        }
    ) + sorted(
        {
            module_id
            for capsule in capsules
            for module_id in capsule.related_module_ids
            if module_id in active_modules
        }
    )
    has_algorithm_commit = any(
        "algorithm_candidate" in commit_delta.signal_kinds
        for commit_delta in interval_delta_model.commit_deltas
    )
    trigger_signals: list[HistorySectionSignalKind] = []
    if has_algorithm_commit:
        trigger_signals.append("algorithm_candidate")
    if any(capsule.variant_names for capsule in capsules):
        trigger_signals.append("variant_family")
    evidence_score = (
        4
        + min(len(capsules), 3)
        + (1 if has_algorithm_commit else 0)
        + (1 if any(capsule.variant_names for capsule in capsules) else 0)
    )
    if not capsules:
        return HistorySectionPlan(
            section_id="algorithms_core_logic",
            title=_OUTLINE_SECTION_TITLES["algorithms_core_logic"],
            kind="optional",
            status="omitted",
            confidence_score=min(100, evidence_score * 10),
            evidence_score=evidence_score,
            depth=None,
            concept_ids=[],
            algorithm_capsule_ids=[],
            evidence_links=[],
            trigger_signals=trigger_signals,
            omission_reason="insufficient_evidence",
        )

    if evidence_score <= 6:
        depth: HistorySectionDepth = "brief"
    elif evidence_score <= 8:
        depth = "standard"
    else:
        depth = "deep"
    return HistorySectionPlan(
        section_id="algorithms_core_logic",
        title=_OUTLINE_SECTION_TITLES["algorithms_core_logic"],
        kind="optional",
        status="included",
        confidence_score=min(100, evidence_score * 10),
        evidence_score=evidence_score,
        depth=depth,
        concept_ids=concept_ids,
        algorithm_capsule_ids=[capsule.capsule_id for capsule in capsules],
        evidence_links=_dedupe_evidence(
            *[capsule.evidence_links for capsule in capsules]
        ),
        trigger_signals=trigger_signals,
        omission_reason=None,
    )


def link_algorithm_capsules_to_section_outline(
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    interval_delta_model: HistoryIntervalDeltaModel,
    capsules: list[HistoryAlgorithmCapsule],
) -> HistorySectionOutline:
    """Return a scored section outline rewritten with H6 capsule links."""

    linked_outline = section_outline.model_copy(deep=True)
    sections: dict[HistorySectionPlanId, HistorySectionPlan] = {
        section.section_id: section.model_copy(deep=True)
        for section in linked_outline.sections
    }
    if "subsystems_modules" in sections:
        sections["subsystems_modules"].algorithm_capsule_ids = _overlapping_capsule_ids(
            capsules,
            sections["subsystems_modules"].concept_ids,
        )
    if "strategy_variants_design_alternatives" in sections:
        sections["strategy_variants_design_alternatives"].algorithm_capsule_ids = [
            capsule.capsule_id for capsule in capsules if capsule.variant_names
        ]
    sections["algorithms_core_logic"] = _algorithm_outline_plan(
        checkpoint_model=checkpoint_model,
        interval_delta_model=interval_delta_model,
        capsules=capsules,
    )
    linked_outline.sections = [
        sections.get(
            section_id,
            HistorySectionPlan(
                section_id=section_id,
                title=_OUTLINE_SECTION_TITLES[section_id],
                kind="optional",
                status="omitted",
                confidence_score=0,
                evidence_score=0,
                depth=None,
                omission_reason="insufficient_evidence",
            ),
        )
        for section_id in _OUTLINE_SECTION_ORDER
    ]
    return linked_outline


__all__ = [
    "algorithm_capsule_dir",
    "algorithm_capsule_filename",
    "algorithm_capsule_index_path",
    "build_algorithm_capsules",
    "link_algorithm_capsules_to_checkpoint_model",
    "link_algorithm_capsules_to_section_outline",
]
