"""History-docs checkpoint model builders for H4."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryConceptChangeStatus,
    HistoryConceptLifecycleStatus,
    HistoryDependencyConcept,
    HistoryEvidenceLink,
    HistoryIntervalDeltaModel,
    HistoryModuleConcept,
    HistorySectionId,
    HistorySectionState,
    HistorySnapshotStructuralModel,
    HistorySubsystemConcept,
)
from engllm.tools.history_docs.semantic_structure import HistorySubsystemGroupingView

_SECTION_TITLES: dict[HistorySectionId, str] = {
    "introduction": "Introduction",
    "architectural_overview": "Architectural Overview",
    "subsystems_modules": "Subsystems and Modules",
    "algorithms_core_logic": "Algorithms and Core Logic",
    "dependencies": "Dependencies",
    "build_development_infrastructure": "Build and Development Infrastructure",
}


def checkpoint_model_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the tool-scoped checkpoint model path."""

    return tool_root / "checkpoints" / checkpoint_id / "checkpoint_model.json"


def load_checkpoint_model(path: Path) -> HistoryCheckpointModel | None:
    """Load one checkpoint model if it exists."""

    if not path.exists():
        return None
    return HistoryCheckpointModel.model_validate_json(path.read_text(encoding="utf-8"))


def _module_concept_id(path: Path) -> str:
    return f"module::{path.as_posix()}"


def _dependency_concept_id(path: Path) -> str:
    return f"dependency-source::{path.as_posix()}"


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _subsystem_snapshot_evidence(
    candidate: HistorySubsystemConcept,
) -> list[HistoryEvidenceLink]:
    links = [
        HistoryEvidenceLink(kind="subsystem", reference=candidate.concept_id),
    ]
    links.extend(
        HistoryEvidenceLink(kind="file", reference=path.as_posix())
        for path in candidate.representative_files
    )
    return _dedupe_evidence(links)


def _module_snapshot_evidence(
    path: Path, symbol_names: list[str]
) -> list[HistoryEvidenceLink]:
    links = [HistoryEvidenceLink(kind="file", reference=path.as_posix())]
    links.extend(
        HistoryEvidenceLink(kind="symbol", reference=f"{path.as_posix()}::{name}")
        for name in symbol_names
    )
    return _dedupe_evidence(links)


def _dependency_snapshot_evidence(path: Path) -> list[HistoryEvidenceLink]:
    return _dedupe_evidence(
        [
            HistoryEvidenceLink(kind="build_source", reference=path.as_posix()),
            HistoryEvidenceLink(kind="file", reference=path.as_posix()),
        ]
    )


def _index_symbols(snapshot: HistorySnapshotStructuralModel) -> dict[Path, list[str]]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    for symbol in snapshot.symbol_summaries:
        by_path[symbol.source_path].add(symbol.qualified_name or symbol.name)
    return {
        path: sorted(names)
        for path, names in sorted(by_path.items(), key=lambda item: item[0].as_posix())
    }


def _module_subsystem_id(
    grouping: HistorySubsystemGroupingView | None,
    path: Path,
) -> str | None:
    if grouping is None:
        return None
    return grouping.module_subsystem_ids.get(path)


def _module_change_status(
    *,
    path: Path,
    previous_exists: bool,
    current_exists: bool,
    previous_model_available: bool,
    touched_files: set[Path],
) -> HistoryConceptChangeStatus:
    if not previous_model_available:
        return "observed"
    if current_exists and not previous_exists:
        return "introduced"
    if not current_exists and previous_exists:
        return "retired"
    if current_exists and previous_exists:
        return "modified" if path in touched_files else "unchanged"
    return "observed"


def _last_updated_checkpoint(
    *,
    checkpoint_id: str,
    previous_last_updated: str | None,
    change_status: HistoryConceptChangeStatus,
) -> str:
    if previous_last_updated is None:
        return checkpoint_id
    if change_status == "unchanged":
        return previous_last_updated
    return checkpoint_id


def _section(
    section_id: HistorySectionId,
    concept_ids: list[str],
    evidence_links: list[HistoryEvidenceLink],
) -> HistorySectionState:
    return HistorySectionState(
        section_id=section_id,
        title=_SECTION_TITLES[section_id],
        concept_ids=concept_ids,
        evidence_links=evidence_links,
    )


def _section_evidence(
    *groups: list[list[HistoryEvidenceLink]],
) -> list[HistoryEvidenceLink]:
    flattened: list[list[HistoryEvidenceLink]] = []
    for group in groups:
        flattened.extend(group)
    return _dedupe_evidence(*flattened)


def build_checkpoint_model(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    current_snapshot: HistorySnapshotStructuralModel,
    current_delta: HistoryIntervalDeltaModel,
    previous_model: HistoryCheckpointModel | None,
    current_grouping: HistorySubsystemGroupingView | None = None,
) -> HistoryCheckpointModel:
    """Build the H4 checkpoint model for one checkpoint."""

    previous_model_available = previous_model is not None
    current_symbols = _index_symbols(current_snapshot)

    previous_subsystems = {
        concept.concept_id: concept
        for concept in ([] if previous_model is None else previous_model.subsystems)
    }
    previous_modules = {
        concept.concept_id: concept
        for concept in ([] if previous_model is None else previous_model.modules)
    }
    previous_dependencies = {
        concept.concept_id: concept
        for concept in ([] if previous_model is None else previous_model.dependencies)
    }

    subsystem_changes = {
        candidate.candidate_id: candidate
        for candidate in current_delta.subsystem_changes
    }
    dependency_changes = {
        candidate.path: candidate
        for candidate in current_delta.dependency_changes
        if candidate.path is not None and candidate.dependency_kind == "build_source"
    }
    dependency_signal_subsystems: dict[str, set[Path]] = defaultdict(set)
    for candidate in current_delta.dependency_changes:
        if candidate.subsystem_id is not None:
            dependency_signal_subsystems[candidate.subsystem_id].update(
                candidate.file_paths
            )
    touched_files = {
        path
        for commit_delta in current_delta.commit_deltas
        for link in commit_delta.evidence_links
        if link.kind == "file"
        for path in [Path(link.reference)]
    }

    current_module_ids_by_subsystem: dict[str, list[str]] = defaultdict(list)
    current_modules_by_id: dict[str, HistoryModuleConcept] = {}
    current_module_paths = {summary.path for summary in current_snapshot.code_summaries}
    all_module_ids = sorted(
        {
            *(_module_concept_id(path) for path in current_module_paths),
            *previous_modules.keys(),
        }
    )
    code_summaries_by_path = {
        summary.path: summary for summary in current_snapshot.code_summaries
    }

    for module_id in all_module_ids:
        path = Path(module_id.removeprefix("module::"))
        current_summary = code_summaries_by_path.get(path)
        previous_module = previous_modules.get(module_id)
        current_exists = current_summary is not None
        previous_exists = previous_module is not None
        if not current_exists and not previous_exists:
            continue

        module_change_status = _module_change_status(
            path=path,
            previous_exists=previous_exists,
            current_exists=current_exists,
            previous_model_available=previous_model_available,
            touched_files=touched_files,
        )
        module_lifecycle_status: HistoryConceptLifecycleStatus = (
            "active" if current_exists else "retired"
        )
        if current_summary is None:
            assert previous_module is not None
            module_language = previous_module.language
            module_functions = previous_module.functions
            module_classes = previous_module.classes
            module_imports = previous_module.imports
            module_docstrings = previous_module.docstrings
        else:
            module_language = current_summary.language
            module_functions = current_summary.functions
            module_classes = current_summary.classes
            module_imports = current_summary.imports
            module_docstrings = current_summary.docstrings
        subsystem_id = (
            _module_subsystem_id(current_grouping, path)
            if current_exists
            else previous_module.subsystem_id if previous_module is not None else None
        )
        symbol_names = (
            current_symbols.get(path, [])
            if current_exists
            else previous_module.symbol_names if previous_module is not None else []
        )
        module = HistoryModuleConcept(
            concept_id=module_id,
            lifecycle_status=module_lifecycle_status,
            change_status=module_change_status,
            first_seen_checkpoint=(
                previous_module.first_seen_checkpoint
                if previous_module is not None
                else checkpoint_id
            ),
            last_updated_checkpoint=_last_updated_checkpoint(
                checkpoint_id=checkpoint_id,
                previous_last_updated=(
                    previous_module.last_updated_checkpoint
                    if previous_module is not None
                    else None
                ),
                change_status=module_change_status,
            ),
            path=path,
            subsystem_id=subsystem_id,
            language=module_language,
            functions=module_functions,
            classes=module_classes,
            imports=module_imports,
            docstrings=module_docstrings,
            symbol_names=symbol_names,
            evidence_links=_dedupe_evidence(
                _module_snapshot_evidence(path, symbol_names),
                [] if previous_module is None else previous_module.evidence_links,
            ),
        )
        current_modules_by_id[module_id] = module
        if module.lifecycle_status == "active" and subsystem_id is not None:
            current_module_ids_by_subsystem[subsystem_id].append(module_id)

    subsystem_concepts: list[HistorySubsystemConcept] = []
    current_subsystems = (
        {
            candidate.candidate_id: candidate
            for candidate in current_grouping.subsystem_candidates
        }
        if current_grouping is not None
        else {
            candidate.candidate_id: candidate
            for candidate in current_snapshot.subsystem_candidates
        }
    )
    all_subsystem_ids = sorted(
        {*current_subsystems.keys(), *previous_subsystems.keys()}
    )
    for subsystem_id in all_subsystem_ids:
        current_candidate = current_subsystems.get(subsystem_id)
        previous_subsystem = previous_subsystems.get(subsystem_id)
        current_exists = current_candidate is not None
        previous_exists = previous_subsystem is not None
        if not current_exists and not previous_exists:
            continue

        if not previous_model_available:
            subsystem_change_status: HistoryConceptChangeStatus = (
                subsystem_changes[subsystem_id].status
                if subsystem_id in subsystem_changes
                else "observed"
            )
        elif current_exists and not previous_exists:
            subsystem_change_status = "introduced"
        elif not current_exists and previous_exists:
            subsystem_change_status = "retired"
        elif subsystem_id in subsystem_changes:
            subsystem_change_status = subsystem_changes[subsystem_id].status
        else:
            subsystem_change_status = "unchanged"

        subsystem_lifecycle_status: HistoryConceptLifecycleStatus = (
            "active" if current_exists else "retired"
        )
        if current_candidate is None:
            assert previous_subsystem is not None
            subsystem_source_root = previous_subsystem.source_root
            subsystem_group_path = previous_subsystem.group_path
            subsystem_module_ids = previous_subsystem.module_ids
            subsystem_file_count = previous_subsystem.file_count
            subsystem_symbol_count = previous_subsystem.symbol_count
            subsystem_language_counts = previous_subsystem.language_counts
            subsystem_representative_files = previous_subsystem.representative_files
            subsystem_display_name = previous_subsystem.display_name
            subsystem_summary = previous_subsystem.summary
            subsystem_capability_labels = previous_subsystem.capability_labels
            subsystem_baseline_subsystem_ids = previous_subsystem.baseline_subsystem_ids
            subsystem_snapshot_evidence: list[HistoryEvidenceLink] = []
        else:
            subsystem_source_root = current_candidate.source_root
            subsystem_group_path = current_candidate.group_path
            subsystem_module_ids = sorted(
                current_module_ids_by_subsystem.get(subsystem_id, [])
            )
            subsystem_file_count = current_candidate.file_count
            subsystem_symbol_count = current_candidate.symbol_count
            subsystem_language_counts = current_candidate.language_counts
            subsystem_representative_files = current_candidate.representative_files
            subsystem_display_name = (
                None
                if current_grouping is None
                else current_grouping.display_names.get(subsystem_id)
            )
            subsystem_summary = (
                None
                if current_grouping is None
                else current_grouping.summaries.get(subsystem_id)
            )
            subsystem_capability_labels = (
                []
                if current_grouping is None
                else current_grouping.capability_labels.get(subsystem_id, [])
            )
            subsystem_baseline_subsystem_ids = (
                [subsystem_id]
                if current_grouping is None
                else current_grouping.baseline_subsystem_ids.get(
                    subsystem_id,
                    [subsystem_id],
                )
            )
            subsystem_snapshot_evidence = _subsystem_snapshot_evidence(
                HistorySubsystemConcept(
                    concept_id=subsystem_id,
                    lifecycle_status="active",
                    change_status="unchanged",
                    first_seen_checkpoint=checkpoint_id,
                    last_updated_checkpoint=checkpoint_id,
                    source_root=current_candidate.source_root,
                    group_path=current_candidate.group_path,
                    module_ids=[],
                    file_count=current_candidate.file_count,
                    symbol_count=current_candidate.symbol_count,
                    language_counts=current_candidate.language_counts,
                    representative_files=current_candidate.representative_files,
                    display_name=subsystem_display_name,
                    summary=subsystem_summary,
                    capability_labels=subsystem_capability_labels,
                    baseline_subsystem_ids=subsystem_baseline_subsystem_ids,
                )
            )
        subsystem_concepts.append(
            HistorySubsystemConcept(
                concept_id=subsystem_id,
                lifecycle_status=subsystem_lifecycle_status,
                change_status=subsystem_change_status,
                first_seen_checkpoint=(
                    previous_subsystem.first_seen_checkpoint
                    if previous_subsystem is not None
                    else checkpoint_id
                ),
                last_updated_checkpoint=_last_updated_checkpoint(
                    checkpoint_id=checkpoint_id,
                    previous_last_updated=(
                        previous_subsystem.last_updated_checkpoint
                        if previous_subsystem is not None
                        else None
                    ),
                    change_status=subsystem_change_status,
                ),
                source_root=subsystem_source_root,
                group_path=subsystem_group_path,
                module_ids=subsystem_module_ids,
                file_count=subsystem_file_count,
                symbol_count=subsystem_symbol_count,
                language_counts=subsystem_language_counts,
                representative_files=subsystem_representative_files,
                display_name=subsystem_display_name,
                summary=subsystem_summary,
                capability_labels=subsystem_capability_labels,
                baseline_subsystem_ids=subsystem_baseline_subsystem_ids,
                evidence_links=_dedupe_evidence(
                    subsystem_snapshot_evidence,
                    (
                        []
                        if previous_subsystem is None
                        else previous_subsystem.evidence_links
                    ),
                    (
                        []
                        if subsystem_id not in subsystem_changes
                        else subsystem_changes[subsystem_id].evidence_links
                    ),
                ),
            )
        )

    dependency_concepts: list[HistoryDependencyConcept] = []
    current_build_sources = {
        source.path: source for source in current_snapshot.build_sources
    }
    all_dependency_ids = sorted(
        {
            *(_dependency_concept_id(path) for path in current_build_sources),
            *previous_dependencies.keys(),
        }
    )
    active_module_paths = {
        concept.path
        for concept in current_modules_by_id.values()
        if concept.lifecycle_status == "active"
    }
    for dependency_id in all_dependency_ids:
        path = Path(dependency_id.removeprefix("dependency-source::"))
        current_dependency = current_build_sources.get(path)
        previous_dependency = previous_dependencies.get(dependency_id)
        current_exists = current_dependency is not None
        previous_exists = previous_dependency is not None
        if not current_exists and not previous_exists:
            continue

        dependency_change_status: HistoryConceptChangeStatus
        if not previous_model_available:
            dependency_change_status = (
                dependency_changes[path].status
                if path in dependency_changes
                else "observed"
            )
        elif current_exists and not previous_exists:
            dependency_change_status = "introduced"
        elif not current_exists and previous_exists:
            dependency_change_status = "retired"
        elif path in dependency_changes:
            dependency_change_status = dependency_changes[path].status
        else:
            dependency_change_status = "unchanged"

        related_subsystem_ids = sorted(
            subsystem_id
            for subsystem_id, touched in dependency_signal_subsystems.items()
            if touched & active_module_paths
        )
        if previous_dependency is not None:
            related_subsystem_ids = sorted(
                {*related_subsystem_ids, *previous_dependency.related_subsystem_ids}
            )
        if current_dependency is None:
            assert previous_dependency is not None
            dependency_ecosystem = previous_dependency.ecosystem
            dependency_category = previous_dependency.category
        else:
            dependency_ecosystem = current_dependency.ecosystem
            dependency_category = current_dependency.category

        dependency_concepts.append(
            HistoryDependencyConcept(
                concept_id=dependency_id,
                lifecycle_status="active" if current_exists else "retired",
                change_status=dependency_change_status,
                first_seen_checkpoint=(
                    previous_dependency.first_seen_checkpoint
                    if previous_dependency is not None
                    else checkpoint_id
                ),
                last_updated_checkpoint=_last_updated_checkpoint(
                    checkpoint_id=checkpoint_id,
                    previous_last_updated=(
                        previous_dependency.last_updated_checkpoint
                        if previous_dependency is not None
                        else None
                    ),
                    change_status=dependency_change_status,
                ),
                path=path,
                ecosystem=dependency_ecosystem,
                category=dependency_category,
                related_subsystem_ids=related_subsystem_ids,
                evidence_links=_dedupe_evidence(
                    (
                        []
                        if current_dependency is None
                        else _dependency_snapshot_evidence(path)
                    ),
                    (
                        []
                        if previous_dependency is None
                        else previous_dependency.evidence_links
                    ),
                    (
                        []
                        if path not in dependency_changes
                        else dependency_changes[path].evidence_links
                    ),
                ),
            )
        )

    subsystem_concepts = sorted(subsystem_concepts, key=lambda item: item.concept_id)
    module_concepts = sorted(
        current_modules_by_id.values(), key=lambda item: item.concept_id
    )
    dependency_concepts = sorted(dependency_concepts, key=lambda item: item.concept_id)

    active_subsystems = [
        item for item in subsystem_concepts if item.lifecycle_status == "active"
    ]
    active_modules = [
        item for item in module_concepts if item.lifecycle_status == "active"
    ]
    active_dependencies = [
        item for item in dependency_concepts if item.lifecycle_status == "active"
    ]
    infra_dependencies = [
        item
        for item in active_dependencies
        if item.category
        in {"build_config", "dependency_manifest", "dependency_lockfile"}
    ]

    sections = [
        _section("introduction", [], []),
        _section(
            "architectural_overview",
            [item.concept_id for item in active_subsystems],
            _section_evidence([item.evidence_links for item in active_subsystems]),
        ),
        _section(
            "subsystems_modules",
            [item.concept_id for item in active_subsystems]
            + [item.concept_id for item in active_modules],
            _section_evidence(
                [item.evidence_links for item in active_subsystems],
                [item.evidence_links for item in active_modules],
            ),
        ),
        _section(
            "dependencies",
            [item.concept_id for item in active_dependencies],
            _section_evidence([item.evidence_links for item in active_dependencies]),
        ),
        _section(
            "build_development_infrastructure",
            [item.concept_id for item in infra_dependencies],
            _section_evidence([item.evidence_links for item in infra_dependencies]),
        ),
    ]

    return HistoryCheckpointModel(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        previous_checkpoint_model_available=previous_model_available,
        subsystems=subsystem_concepts,
        modules=module_concepts,
        dependencies=dependency_concepts,
        sections=sections,
    )


__all__ = [
    "build_checkpoint_model",
    "checkpoint_model_path",
    "load_checkpoint_model",
]
