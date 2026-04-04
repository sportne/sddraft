"""Shadow dependency narrative helpers for history-docs quality recovery."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from engllm.tools.history_docs.models import (
    HistoryDependencyEntry,
    HistoryDependencyNarrativeRenderStyle,
    HistoryDependencyNarrativeShadow,
    HistoryDependencyNarrativeShadowEntry,
    HistoryDependencyRole,
)

_GENERAL_KNOWLEDGE_BY_NAME: dict[str, str] = {
    "requests": "requests is a Python HTTP client library commonly used for outbound web requests.",
    "pydantic": "pydantic is a Python data validation and settings library centered on typed models.",
    "pytest": "pytest is a Python test runner and assertion framework.",
    "react": "react is a JavaScript library for building component-based user interfaces.",
    "eslint": "eslint is a JavaScript and TypeScript linting tool.",
}
_GENERAL_KNOWLEDGE_BY_TOKEN: list[tuple[str, str]] = [
    (
        "pytest",
        "This package is commonly used to define and run automated Python tests.",
    ),
    (
        "eslint",
        "This package is commonly used to lint JavaScript and TypeScript source code.",
    ),
    (
        "react",
        "This package is commonly used to build reactive component-based web interfaces.",
    ),
    (
        "request",
        "This package is commonly used to issue HTTP requests to remote services.",
    ),
    ("http", "This package is commonly used to communicate with HTTP services."),
    ("schema", "This package is commonly used to validate or define structured data."),
    ("test", "This package is commonly used in automated test workflows."),
]
_GROUP_TITLE_BY_ROLE: dict[HistoryDependencyRole, str] = {
    "build": "Build Tooling",
    "development": "Development Tooling",
    "test": "Test Tooling",
    "plugin": "Plugin Tooling",
    "toolchain": "Toolchain Dependencies",
}
_GROUPABLE_ROLES = {"build", "development", "test", "plugin", "toolchain"}


def dependency_narratives_shadow_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the shadow dependency-narrative artifact path."""

    return (
        tool_root / "checkpoints" / checkpoint_id / "dependency_narratives_shadow.json"
    )


def _clean_shadow_text(value: str) -> str:
    text = value.strip()
    if not text:
        return "TBD"
    lowered = text.lower()
    if lowered.startswith("tbd - "):
        candidate = text[6:].strip()
        return candidate or "TBD"
    if lowered.startswith("tbd:"):
        candidate = text[4:].strip()
        return candidate or "TBD"
    return text


def _package_general_knowledge(entry: HistoryDependencyEntry) -> str:
    normalized = entry.normalized_name
    if normalized in _GENERAL_KNOWLEDGE_BY_NAME:
        return _GENERAL_KNOWLEDGE_BY_NAME[normalized]
    for token, description in _GENERAL_KNOWLEDGE_BY_TOKEN:
        if token in normalized:
            return description
    if entry.ecosystem == "python":
        return f"{entry.display_name} is a Python package used in this ecosystem."
    if entry.ecosystem in {"javascript", "typescript"}:
        return f"{entry.display_name} is a {entry.ecosystem} package used in this ecosystem."
    return f"{entry.display_name} is a direct dependency declared for this project."


def _default_group_title(entry: HistoryDependencyEntry) -> str | None:
    role = next((role for role in entry.scope_roles if role in _GROUPABLE_ROLES), None)
    if role is None:
        return None
    prefix = (
        "Project " if entry.section_target == "build_development_infrastructure" else ""
    )
    base = _GROUP_TITLE_BY_ROLE[role]
    if entry.ecosystem not in {"unknown", ""}:
        return f"{prefix}{entry.ecosystem.capitalize()} {base}"
    return f"{prefix}{base}"


def _subsystem_text(
    subsystem_ids: list[str],
    subsystem_display_names: dict[str, str],
) -> str:
    labels: list[str] = []
    for subsystem_id in subsystem_ids[:4]:
        label = subsystem_display_names.get(subsystem_id) or subsystem_id
        labels.append(f"`{label}`")
    return ", ".join(labels)


def _build_shadow_entry(
    entry: HistoryDependencyEntry,
    *,
    subsystem_display_names: dict[str, str],
) -> HistoryDependencyNarrativeShadowEntry:
    cleaned_general = _clean_shadow_text(entry.general_description)
    general_basis = entry.general_description_basis
    if cleaned_general == "TBD":
        cleaned_general = _package_general_knowledge(entry)
        general_basis = "package_general_knowledge"
    elif entry.general_description.strip().lower().startswith(("tbd - ", "tbd:")):
        general_basis = "package_general_knowledge"
    elif general_basis == "tbd":
        general_basis = "project_evidence"

    cleaned_project_usage = _clean_shadow_text(entry.project_usage_description)
    project_basis = entry.project_usage_basis
    if cleaned_project_usage == "TBD":
        if (
            entry.usage_signals
            or entry.related_module_ids
            or entry.related_subsystem_ids
        ):
            evidence_parts: list[str] = []
            if entry.usage_signals:
                evidence_parts.append(
                    "Observed usage signals include "
                    + ", ".join(f"`{signal}`" for signal in entry.usage_signals[:4])
                )
            if entry.related_module_ids:
                evidence_parts.append(
                    "linked modules: "
                    + ", ".join(
                        f"`{module_id}`" for module_id in entry.related_module_ids[:4]
                    )
                )
            if entry.related_subsystem_ids:
                evidence_parts.append(
                    "linked subsystems: "
                    + _subsystem_text(
                        entry.related_subsystem_ids,
                        subsystem_display_names,
                    )
                )
            cleaned_project_usage = "; ".join(evidence_parts) + "."
            project_basis = "project_evidence"
        else:
            cleaned_project_usage = "Project-specific usage is not strongly evidenced by the current manifests and import signals."
            project_basis = "tbd"
    elif project_basis == "tbd":
        project_basis = "project_evidence"

    render_style: HistoryDependencyNarrativeRenderStyle = "standard"
    group_title = None
    if (
        entry.section_target == "build_development_infrastructure"
        and project_basis == "tbd"
        and any(role in _GROUPABLE_ROLES for role in entry.scope_roles)
    ):
        render_style = "grouped_tooling"
        group_title = _default_group_title(entry)

    return HistoryDependencyNarrativeShadowEntry(
        dependency_id=entry.dependency_id,
        display_name=entry.display_name,
        ecosystem=entry.ecosystem,
        section_target=entry.section_target,
        scope_roles=list(entry.scope_roles),
        render_style=render_style,
        group_title=group_title,
        general_description=cleaned_general,
        general_description_basis=general_basis,
        project_usage_description=cleaned_project_usage,
        project_usage_basis=project_basis,
        uncertainty=list(entry.uncertainty),
        confidence=entry.confidence,
        related_subsystem_ids=list(entry.related_subsystem_ids),
        related_module_ids=list(entry.related_module_ids),
        usage_signals=list(entry.usage_signals),
    )


def build_dependency_narratives_shadow(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    entries: list[HistoryDependencyEntry],
    subsystem_display_names: dict[str, str] | None = None,
) -> HistoryDependencyNarrativeShadow:
    """Build the shadow dependency narrative artifact deterministically."""

    resolved_subsystem_display_names = subsystem_display_names or {}
    shadow_entries = sorted(
        (
            _build_shadow_entry(
                entry,
                subsystem_display_names=resolved_subsystem_display_names,
            )
            for entry in entries
        ),
        key=lambda item: (item.section_target, item.render_style, item.dependency_id),
    )
    return HistoryDependencyNarrativeShadow(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        entries=shadow_entries,
    )


def grouped_tooling_entries(
    shadow: HistoryDependencyNarrativeShadow,
    *,
    section_target: str,
) -> dict[str, list[HistoryDependencyNarrativeShadowEntry]]:
    """Return grouped shadow tooling entries for one rendered section target."""

    grouped: dict[str, list[HistoryDependencyNarrativeShadowEntry]] = defaultdict(list)
    for entry in shadow.entries:
        if (
            entry.section_target != section_target
            or entry.render_style != "grouped_tooling"
        ):
            continue
        grouped[entry.group_title or "Tooling"].append(entry)
    return {
        title: sorted(values, key=lambda item: item.dependency_id)
        for title, values in sorted(grouped.items(), key=lambda item: item[0])
    }


__all__ = [
    "build_dependency_narratives_shadow",
    "dependency_narratives_shadow_path",
    "grouped_tooling_entries",
]
