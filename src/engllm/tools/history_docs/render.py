"""Deterministic H8 markdown rendering for history-docs checkpoints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from engllm.domain.errors import RenderingError
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryModuleConcept,
    HistoryRenderedSection,
    HistoryRenderManifest,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySubsystemConcept,
)

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
_TITLE_CASE_IDS = {
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
}


def checkpoint_markdown_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the rendered checkpoint markdown artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "checkpoint.md"


def render_manifest_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the render-manifest artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "render_manifest.json"


def write_checkpoint_markdown(path: Path, content: str) -> Path:
    """Persist rendered checkpoint markdown."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RenderingError(f"Failed writing markdown to {path}: {exc}") from exc
    return path


def _title_case(value: str) -> str:
    tokens = value.replace("_", " ").replace("-", " ").split()
    return " ".join(token.capitalize() for token in tokens) or "Root"


def _section_title(section_id: str) -> str:
    return _title_case(section_id)


def _paragraph(lines: list[str], text: str) -> None:
    lines.append(text)
    lines.append("")


def _bullet_list(lines: list[str], bullets: list[str]) -> None:
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.append("")


def _stable_unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _module_terms(module: HistoryModuleConcept) -> list[str]:
    return [
        module.path.as_posix().lower(),
        *(value.lower() for value in module.functions),
        *(value.lower() for value in module.classes),
        *(value.lower() for value in module.imports),
        *(value.lower() for value in module.docstrings),
        *(value.lower() for value in module.symbol_names),
    ]


def _module_token_hits(
    module: HistoryModuleConcept,
    section_id: HistorySectionPlanId,
) -> list[str]:
    tokens = _TOKEN_SETS.get(section_id, ())
    terms = _module_terms(module)
    return sorted({token for token in tokens if any(token in term for term in terms)})


def _active_subsystems(
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistorySubsystemConcept]:
    return sorted(
        [
            concept
            for concept in checkpoint_model.subsystems
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    )


def _active_modules(
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistoryModuleConcept]:
    return sorted(
        [
            concept
            for concept in checkpoint_model.modules
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    )


def _active_dependencies(
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistoryDependencyConcept]:
    return sorted(
        [
            concept
            for concept in checkpoint_model.dependencies
            if concept.lifecycle_status == "active"
        ],
        key=lambda item: item.concept_id,
    )


def _subsystem_label(subsystem: HistorySubsystemConcept) -> str:
    if subsystem.display_name:
        return subsystem.display_name
    group_path = subsystem.group_path.as_posix()
    if group_path == ".":
        return subsystem.source_root.as_posix()
    return group_path


def _subsystem_scope_label(subsystem: HistorySubsystemConcept) -> str:
    group_path = subsystem.group_path.as_posix()
    if group_path == ".":
        return subsystem.source_root.as_posix()
    return group_path


def _module_map(
    modules: list[HistoryModuleConcept],
) -> tuple[dict[str, HistoryModuleConcept], dict[str, list[HistoryModuleConcept]]]:
    by_id = {module.concept_id: module for module in modules}
    by_subsystem: dict[str, list[HistoryModuleConcept]] = {}
    for module in modules:
        if module.subsystem_id is None:
            continue
        by_subsystem.setdefault(module.subsystem_id, []).append(module)
    for values in by_subsystem.values():
        values.sort(key=lambda item: item.path.as_posix())
    return by_id, by_subsystem


def _capsule_map(
    capsules: list[HistoryAlgorithmCapsule],
) -> dict[str, HistoryAlgorithmCapsule]:
    return {capsule.capsule_id: capsule for capsule in capsules}


def _dependency_entry_map(
    inventory: HistoryDependencyInventory,
) -> dict[str, HistoryDependencyEntry]:
    return {entry.dependency_id: entry for entry in inventory.entries}


def _dependency_entries_for_target(
    inventory: HistoryDependencyInventory,
    target: str,
) -> list[HistoryDependencyEntry]:
    return sorted(
        [entry for entry in inventory.entries if entry.section_target == target],
        key=lambda item: item.dependency_id,
    )


def _artifact_paths_for_section(
    *,
    dependency_ids: list[str],
    algorithm_capsule_ids: list[str],
    capsule_index: HistoryAlgorithmCapsuleIndex,
) -> list[Path]:
    paths = [Path("checkpoint_model.json"), Path("section_outline.json")]
    if dependency_ids:
        paths.append(Path("dependencies.json"))
    if algorithm_capsule_ids:
        paths.append(Path("algorithm_capsules") / "index.json")
        capsule_paths = {
            entry.artifact_path
            for entry in capsule_index.capsules
            if entry.capsule_id in set(algorithm_capsule_ids)
        }
        paths.extend(sorted(capsule_paths, key=lambda item: item.as_posix()))
    return _stable_unique_paths(paths)


def _ecosystem_summary(entries: list[HistoryDependencyEntry]) -> str:
    ecosystems = Counter(entry.ecosystem for entry in entries)
    if not ecosystems:
        return "no documented dependency ecosystems"
    parts = [
        f"{ecosystem} ({count})"
        for ecosystem, count in sorted(ecosystems.items(), key=lambda item: item[0])
    ]
    return ", ".join(parts)


def _summary_line_for_module(module: HistoryModuleConcept) -> str:
    return (
        f"`{module.path.as_posix()}` ({module.language}); "
        f"functions {len(module.functions)}, classes {len(module.classes)}, imports {len(module.imports)}"
    )


def _dependency_paragraph(value: str) -> str:
    text = value.strip() or "TBD"
    return text


def _render_introduction(
    *,
    workspace_id: str,
    checkpoint_model: HistoryCheckpointModel,
    active_subsystems: list[HistorySubsystemConcept],
    active_modules: list[HistoryModuleConcept],
    active_dependencies: list[HistoryDependencyConcept],
    capsules: list[HistoryAlgorithmCapsule],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    _paragraph(
        lines,
        (
            f"This document describes `{workspace_id}` as it exists at checkpoint "
            f"`{checkpoint_model.checkpoint_id}` on commit `{checkpoint_model.target_commit}`."
        ),
    )
    _paragraph(
        lines,
        (
            f"At this checkpoint, the documented system includes {len(active_subsystems)} active subsystems, "
            f"{len(active_modules)} active modules, {len(active_dependencies)} active dependency sources, "
            f"and {len(capsules)} algorithm capsules."
        ),
    )
    return lines, [], [], 0


def _render_architectural_overview(
    active_subsystems: list[HistorySubsystemConcept],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    _paragraph(
        lines,
        "The current architecture is organized around deterministic subsystem groupings derived from the checkpoint snapshot.",
    )
    bullets = [
        (
            f"`{_subsystem_label(subsystem)}` covering `{_subsystem_scope_label(subsystem)}`: "
            f"{subsystem.file_count} files, {subsystem.symbol_count} symbols, representative files "
            f"{', '.join(f'`{path.as_posix()}`' for path in subsystem.representative_files) or 'none'}"
            + (f"; summary: {subsystem.summary}" if subsystem.summary else "")
            + (
                f"; capability labels: {', '.join(f'`{label}`' for label in subsystem.capability_labels)}"
                if subsystem.capability_labels
                else ""
            )
            + "."
        )
        for subsystem in active_subsystems
    ]
    if bullets:
        _bullet_list(lines, bullets)
    else:
        _paragraph(
            lines,
            "This checkpoint did not yield any active subsystem groupings.",
        )
    return lines, [], [], 0


def _render_subsystems_modules(
    active_subsystems: list[HistorySubsystemConcept],
    modules_by_subsystem: dict[str, list[HistoryModuleConcept]],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    subheading_count = 0
    for subsystem in active_subsystems:
        lines.append(f"### {_subsystem_label(subsystem)}")
        lines.append("")
        subheading_count += 1
        languages = ", ".join(
            f"{language} ({count})"
            for language, count in sorted(subsystem.language_counts.items())
        )
        summary_text = (
            subsystem.summary
            if subsystem.summary
            else (
                f"This subsystem groups files under `{_subsystem_scope_label(subsystem)}` and currently spans "
                f"{subsystem.file_count} files, {subsystem.symbol_count} symbols, and languages {languages or 'unknown'}."
            )
        )
        _paragraph(lines, summary_text)
        if subsystem.capability_labels:
            _paragraph(
                lines,
                "Capability labels: "
                + ", ".join(f"`{label}`" for label in subsystem.capability_labels)
                + ".",
            )
        module_bullets = [
            _summary_line_for_module(module)
            for module in modules_by_subsystem.get(subsystem.concept_id, [])
        ]
        if module_bullets:
            _bullet_list(lines, module_bullets)
        else:
            _paragraph(
                lines, "No active modules are currently linked to this subsystem."
            )
    if not active_subsystems:
        _paragraph(
            lines,
            "This checkpoint did not yield any active subsystem groupings.",
        )
    return lines, [], [], subheading_count


def _render_algorithms_core_logic(
    capsules: list[HistoryAlgorithmCapsule],
    modules_by_id: dict[str, HistoryModuleConcept],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    subheading_count = 0
    rendered_capsule_ids: list[str] = []
    for capsule in sorted(capsules, key=lambda item: item.capsule_id):
        rendered_capsule_ids.append(capsule.capsule_id)
        lines.append(f"### {capsule.title}")
        lines.append("")
        subheading_count += 1
        bullets = [f"Scope: {capsule.scope_kind} `{capsule.scope_path.as_posix()}`."]
        if capsule.related_subsystem_ids:
            bullets.append(
                "Related subsystems: "
                + ", ".join(f"`{item}`" for item in capsule.related_subsystem_ids)
                + "."
            )
        if capsule.related_module_ids:
            bullets.append(
                "Related modules: "
                + ", ".join(
                    (
                        f"`{modules_by_id[module_id].path.as_posix()}`"
                        if module_id in modules_by_id
                        else f"`{module_id}`"
                    )
                    for module_id in capsule.related_module_ids
                )
                + "."
            )
        if capsule.changed_symbol_names:
            bullets.append(
                "Changed symbols: "
                + ", ".join(f"`{name}`" for name in capsule.changed_symbol_names)
                + "."
            )
        if capsule.variant_names:
            bullets.append(
                "Variant names: "
                + ", ".join(f"`{name}`" for name in capsule.variant_names)
                + "."
            )
        if capsule.phases:
            bullets.append(
                "Phases: "
                + "; ".join(
                    f"`{phase.phase_key}` ({', '.join(f'`{name}`' for name in phase.matched_names)})"
                    for phase in capsule.phases
                )
                + "."
            )
        if capsule.shared_abstractions:
            bullets.append(
                "Shared abstractions: "
                + ", ".join(
                    f"`{item.name}` ({item.kind})"
                    for item in capsule.shared_abstractions
                )
                + "."
            )
        if capsule.data_structures:
            bullets.append(
                "Data structures: "
                + ", ".join(
                    f"`{item.name}` ({item.kind})" for item in capsule.data_structures
                )
                + "."
            )
        if capsule.assumptions:
            bullets.append(
                "Assumptions: "
                + "; ".join(f"{item.text}" for item in capsule.assumptions)
                + "."
            )
        _bullet_list(lines, bullets)
    if not capsules:
        _paragraph(lines, "No algorithm capsules were emitted for this checkpoint.")
    return lines, rendered_capsule_ids, [], subheading_count


def _render_dependency_section(
    entries: list[HistoryDependencyEntry],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    subheading_count = 0
    _paragraph(
        lines,
        (
            f"This checkpoint documents {len(entries)} direct dependency entries across "
            f"{_ecosystem_summary(entries)}."
        ),
    )
    rendered_dependency_ids: list[str] = []
    for entry in entries:
        rendered_dependency_ids.append(entry.dependency_id)
        lines.append(f"### {entry.display_name}")
        lines.append("")
        subheading_count += 1
        _paragraph(lines, _dependency_paragraph(entry.general_description))
        _paragraph(lines, _dependency_paragraph(entry.project_usage_description))
    return lines, [], rendered_dependency_ids, subheading_count


def _render_build_infrastructure(
    *,
    active_dependencies: list[HistoryDependencyConcept],
    entries: list[HistoryDependencyEntry],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    subheading_count = 0
    _paragraph(
        lines,
        "The current build and development infrastructure is defined through the active build-source concepts and their documented tooling dependencies.",
    )
    build_sources = [
        dependency
        for dependency in active_dependencies
        if dependency.category in _BUILD_INFRA_CATEGORIES
    ]
    if build_sources:
        lines.append("### Build Sources")
        lines.append("")
        subheading_count += 1
        _bullet_list(
            lines,
            [
                (
                    f"`{dependency.path.as_posix()}` ({dependency.ecosystem}, {dependency.category})"
                )
                for dependency in build_sources
            ],
        )
    else:
        _paragraph(
            lines,
            "This checkpoint did not yield any active build-source concepts.",
        )

    rendered_dependency_ids: list[str] = []
    for entry in entries:
        rendered_dependency_ids.append(entry.dependency_id)
        lines.append(f"### {entry.display_name}")
        lines.append("")
        subheading_count += 1
        _paragraph(lines, _dependency_paragraph(entry.general_description))
        _paragraph(lines, _dependency_paragraph(entry.project_usage_description))
    return lines, [], rendered_dependency_ids, subheading_count


def _render_strategy_variants(
    capsules: list[HistoryAlgorithmCapsule],
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    subheading_count = 0
    rendered_capsule_ids = [
        capsule.capsule_id for capsule in capsules if capsule.variant_names
    ]
    for capsule in sorted(
        [capsule for capsule in capsules if capsule.variant_names],
        key=lambda item: item.capsule_id,
    ):
        lines.append(f"### {capsule.title}")
        lines.append("")
        subheading_count += 1
        _paragraph(
            lines,
            (
                f"The current algorithm cluster exposes variant names "
                f"{', '.join(f'`{name}`' for name in capsule.variant_names)} within scope `{capsule.scope_path.as_posix()}`."
            ),
        )
    if not rendered_capsule_ids:
        _paragraph(lines, "No variant families were identified for this checkpoint.")
    return lines, rendered_capsule_ids, [], subheading_count


def _render_token_section(
    *,
    section_id: HistorySectionPlanId,
    active_modules: list[HistoryModuleConcept],
    modules_by_id: dict[str, HistoryModuleConcept],
) -> tuple[list[str], list[str], list[str], int]:
    del modules_by_id
    lines: list[str] = []
    matched_modules = [
        module for module in active_modules if _module_token_hits(module, section_id)
    ]
    if not matched_modules:
        _paragraph(lines, "No strong evidence was carried into this rendered section.")
        return lines, [], [], 0
    _paragraph(
        lines,
        f"This section highlights active modules whose current metadata matches the {_section_title(section_id).lower()} signal set.",
    )
    _bullet_list(
        lines,
        [
            (
                f"`{module.path.as_posix()}`: matched signals "
                f"{', '.join(f'`{token}`' for token in _module_token_hits(module, section_id))}."
            )
            for module in matched_modules
        ],
    )
    return lines, [], [], 0


def _render_design_notes_rationale(
    checkpoint_model: HistoryCheckpointModel,
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    active_subsystems = [
        subsystem
        for subsystem in checkpoint_model.subsystems
        if subsystem.lifecycle_status == "active"
        and subsystem.change_status in {"introduced", "modified", "observed"}
    ]
    active_modules = [
        module
        for module in checkpoint_model.modules
        if module.lifecycle_status == "active"
        and module.change_status in {"introduced", "modified", "observed"}
    ]
    active_dependencies = [
        dependency
        for dependency in checkpoint_model.dependencies
        if dependency.lifecycle_status == "active"
        and dependency.change_status in {"introduced", "modified", "observed"}
    ]
    bullets: list[str] = []
    for subsystem in active_subsystems[:3]:
        bullets.append(
            f"Subsystem `{_subsystem_label(subsystem)}` acts as a current architectural coordination point."
        )
    for module in active_modules[:3]:
        bullets.append(
            f"Module `{module.path.as_posix()}` remains an active design focal point for the current checkpoint."
        )
    for dependency in active_dependencies[:3]:
        bullets.append(
            f"Dependency source `{dependency.path.as_posix()}` anchors current dependency or build configuration."
        )
    if bullets:
        _paragraph(
            lines,
            "The current checkpoint highlights a small set of concepts that act as present-state design anchors.",
        )
        _bullet_list(lines, bullets)
    else:
        _paragraph(
            lines, "No strong design-note anchors were identified for this checkpoint."
        )
    return lines, [], [], 0


def _render_limitations_constraints(
    *,
    active_modules: list[HistoryModuleConcept],
    dependency_inventory: HistoryDependencyInventory,
) -> tuple[list[str], list[str], list[str], int]:
    lines: list[str] = []
    matched_modules = [
        module
        for module in active_modules
        if _module_token_hits(module, "limitations_constraints")
    ]
    bullets = [
        (
            f"`{module.path.as_posix()}`: matched signals "
            f"{', '.join(f'`{token}`' for token in _module_token_hits(module, 'limitations_constraints'))}."
        )
        for module in matched_modules
    ]
    bullets.extend(
        f"`{warning.source_path.as_posix()}`: {warning.message}"
        for warning in dependency_inventory.warnings
    )
    if bullets:
        _paragraph(
            lines,
            "The current checkpoint carries explicit limitation or constraint signals in module metadata and dependency parsing warnings.",
        )
        _bullet_list(lines, bullets)
    else:
        _paragraph(
            lines,
            "No explicit limitation or constraint signals were identified for this checkpoint.",
        )
    return lines, [], [], 0


def _render_section_content(
    *,
    section: HistorySectionPlan,
    workspace_id: str,
    checkpoint_model: HistoryCheckpointModel,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
) -> tuple[list[str], list[str], list[str], int]:
    active_subsystems = _active_subsystems(checkpoint_model)
    active_modules = _active_modules(checkpoint_model)
    active_dependencies = _active_dependencies(checkpoint_model)
    modules_by_id, modules_by_subsystem = _module_map(active_modules)

    if section.section_id == "introduction":
        return _render_introduction(
            workspace_id=workspace_id,
            checkpoint_model=checkpoint_model,
            active_subsystems=active_subsystems,
            active_modules=active_modules,
            active_dependencies=active_dependencies,
            capsules=capsules,
        )
    if section.section_id == "architectural_overview":
        return _render_architectural_overview(active_subsystems)
    if section.section_id == "subsystems_modules":
        return _render_subsystems_modules(active_subsystems, modules_by_subsystem)
    if section.section_id == "algorithms_core_logic":
        return _render_algorithms_core_logic(capsules, modules_by_id)
    if section.section_id == "dependencies":
        return _render_dependency_section(
            _dependency_entries_for_target(dependency_inventory, "dependencies")
        )
    if section.section_id == "build_development_infrastructure":
        return _render_build_infrastructure(
            active_dependencies=active_dependencies,
            entries=_dependency_entries_for_target(
                dependency_inventory,
                "build_development_infrastructure",
            ),
        )
    if section.section_id == "strategy_variants_design_alternatives":
        return _render_strategy_variants(capsules)
    if section.section_id in _TOKEN_SETS:
        return _render_token_section(
            section_id=section.section_id,
            active_modules=active_modules,
            modules_by_id=modules_by_id,
        )
    if section.section_id == "design_notes_rationale":
        return _render_design_notes_rationale(checkpoint_model)
    if section.section_id == "limitations_constraints":
        return _render_limitations_constraints(
            active_modules=active_modules,
            dependency_inventory=dependency_inventory,
        )
    return [], [], [], 0


def render_checkpoint_markdown(
    *,
    workspace_id: str,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    capsules: list[HistoryAlgorithmCapsule],
) -> tuple[str, HistoryRenderManifest]:
    """Render deterministic checkpoint markdown and its debug manifest."""

    lines = [f"# {workspace_id} Documentation", ""]
    _bullet_list(
        lines,
        [
            f"Checkpoint ID: `{checkpoint_model.checkpoint_id}`",
            f"Target Commit: `{checkpoint_model.target_commit}`",
            (
                f"Previous Checkpoint: `{checkpoint_model.previous_checkpoint_commit}`"
                if checkpoint_model.previous_checkpoint_commit is not None
                else "Previous Checkpoint: `initial`"
            ),
        ],
    )

    rendered_sections: list[HistoryRenderedSection] = []
    included_sections = [
        section for section in section_outline.sections if section.status == "included"
    ]
    entry_map = _dependency_entry_map(dependency_inventory)
    capsule_map = _capsule_map(capsules)

    for order, section in enumerate(included_sections, start=1):
        lines.append(f"## {section.title}")
        lines.append("")
        (
            section_lines,
            rendered_capsule_ids,
            rendered_dependency_ids,
            subheading_count,
        ) = _render_section_content(
            section=section,
            workspace_id=workspace_id,
            checkpoint_model=checkpoint_model,
            dependency_inventory=dependency_inventory,
            capsule_index=capsule_index,
            capsules=capsules,
        )
        lines.extend(section_lines)
        if section_lines and section_lines[-1] != "":
            lines.append("")

        manifest_capsules = sorted(
            {
                *section.algorithm_capsule_ids,
                *rendered_capsule_ids,
            }
            & set(capsule_map),
        )
        manifest_dependencies = sorted(
            {
                *rendered_dependency_ids,
                *[
                    dependency_id
                    for dependency_id in rendered_dependency_ids
                    if dependency_id in entry_map
                ],
            }
        )
        rendered_sections.append(
            HistoryRenderedSection(
                section_id=section.section_id,
                title=section.title,
                order=order,
                kind=section.kind,
                concept_ids=list(section.concept_ids),
                algorithm_capsule_ids=manifest_capsules,
                dependency_ids=manifest_dependencies,
                source_artifact_paths=_artifact_paths_for_section(
                    dependency_ids=manifest_dependencies,
                    algorithm_capsule_ids=manifest_capsules,
                    capsule_index=capsule_index,
                ),
                subheading_count=subheading_count,
            )
        )

    markdown = "\n".join(lines).strip() + "\n"
    manifest = HistoryRenderManifest(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        markdown_path=Path("checkpoint.md"),
        sections=rendered_sections,
    )
    return markdown, manifest


__all__ = [
    "checkpoint_markdown_path",
    "render_checkpoint_markdown",
    "render_manifest_path",
    "write_checkpoint_markdown",
]
