"""Deterministic H9 validation for rendered history-docs checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryDependencyInventory,
    HistoryRenderManifest,
    HistorySectionOutline,
    HistoryValidationCheckId,
    HistoryValidationFinding,
    HistoryValidationReport,
    HistoryValidationSeverity,
)

_RELEASE_NOTE_PHRASES = (
    "since last",
    "new this quarter",
    "in this release",
    "previous version",
    "changed since",
    "recent change",
    "new this checkpoint",
)
_FILLER_LINES = {
    "No strong evidence was carried into this rendered section.",
    "No variant families were identified for this checkpoint.",
    "No algorithm capsules were emitted for this checkpoint.",
    "No active subsystems were identified for this checkpoint.",
    "No active build-source concepts were identified for this checkpoint.",
    "No strong design-note anchors were identified for this checkpoint.",
    "No explicit limitation or constraint signals were identified for this checkpoint.",
}


@dataclass(frozen=True, slots=True)
class _SectionBlock:
    title: str
    heading_line: int
    body_start_line: int
    body_lines: tuple[str, ...]


def validation_report_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H9 validation report artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "validation_report.json"


def _heading_blocks(
    lines: list[str], prefix: str, *, offset: int = 0
) -> list[_SectionBlock]:
    blocks: list[_SectionBlock] = []
    positions = [
        (index, line[len(prefix) :].strip())
        for index, line in enumerate(lines)
        if line.startswith(prefix)
    ]
    for position, title in positions:
        next_position = next(
            (
                other_position
                for other_position, _ in positions
                if other_position > position
            ),
            len(lines),
        )
        blocks.append(
            _SectionBlock(
                title=title,
                heading_line=offset + position + 1,
                body_start_line=offset + position + 2,
                body_lines=tuple(lines[position + 1 : next_position]),
            )
        )
    return blocks


def _top_level_sections(markdown: str) -> tuple[list[str], list[_SectionBlock]]:
    lines = markdown.splitlines()
    return lines, _heading_blocks(lines, "## ")


def _subheading_blocks(section: _SectionBlock) -> list[_SectionBlock]:
    return _heading_blocks(
        list(section.body_lines),
        "### ",
        offset=section.body_start_line - 1,
    )


def _paragraphs(body_lines: tuple[str, ...]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in body_lines:
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _active_concept_ids(checkpoint_model: HistoryCheckpointModel) -> set[str]:
    concept_ids: set[str] = set()
    concept_ids.update(
        concept.concept_id
        for concept in checkpoint_model.subsystems
        if concept.lifecycle_status == "active"
    )
    concept_ids.update(
        concept.concept_id
        for concept in checkpoint_model.modules
        if concept.lifecycle_status == "active"
    )
    concept_ids.update(
        concept.concept_id
        for concept in checkpoint_model.dependencies
        if concept.lifecycle_status == "active"
    )
    return concept_ids


def _finding(
    *,
    check_id: HistoryValidationCheckId,
    severity: HistoryValidationSeverity,
    message: str,
    section_id: str | None = None,
    artifact_path: Path | None = None,
    reference: str | None = None,
    line_number: int | None = None,
) -> HistoryValidationFinding:
    return HistoryValidationFinding(
        check_id=check_id,
        severity=severity,
        message=message,
        section_id=section_id,
        artifact_path=artifact_path,
        reference=reference,
        line_number=line_number,
    )


def _validate_included_and_omitted_sections(
    *,
    outline: HistorySectionOutline,
    manifest: HistoryRenderManifest,
    markdown_sections: list[_SectionBlock],
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    included_sections = [
        section for section in outline.sections if section.status == "included"
    ]
    omitted_sections = [
        section for section in outline.sections if section.status == "omitted"
    ]
    manifest_by_id = {
        section.section_id: index for index, section in enumerate(manifest.sections)
    }
    markdown_by_title = {
        section.title: index for index, section in enumerate(markdown_sections)
    }

    for expected_index, section in enumerate(included_sections):
        manifest_index = manifest_by_id.get(section.section_id)
        markdown_index = markdown_by_title.get(section.title)
        if manifest_index != expected_index or markdown_index != expected_index:
            findings.append(
                _finding(
                    check_id="included_section_missing",
                    severity="error",
                    message=(
                        f"Included section {section.section_id!r} must appear exactly once in "
                        "render manifest and markdown in outline order."
                    ),
                    section_id=section.section_id,
                    artifact_path=Path("checkpoint.md"),
                    reference=section.title,
                    line_number=(
                        markdown_sections[markdown_index].heading_line
                        if markdown_index is not None
                        and markdown_index < len(markdown_sections)
                        else None
                    ),
                )
            )

    markdown_titles = {section.title for section in markdown_sections}
    for section in omitted_sections:
        if section.title in markdown_titles:
            findings.append(
                _finding(
                    check_id="omitted_section_rendered",
                    severity="error",
                    message=(
                        f"Omitted section {section.section_id!r} must not be rendered in markdown."
                    ),
                    section_id=section.section_id,
                    artifact_path=Path("checkpoint.md"),
                    reference=section.title,
                )
            )

    manifest_titles = [section.title for section in manifest.sections]
    markdown_titles_in_order = [section.title for section in markdown_sections]
    if manifest_titles != markdown_titles_in_order:
        findings.append(
            _finding(
                check_id="render_manifest_mismatch",
                severity="error",
                message=(
                    "Render manifest section titles must match markdown top-level "
                    "section headings exactly."
                ),
                artifact_path=Path("render_manifest.json"),
                reference=" -> ".join(markdown_titles_in_order),
            )
        )

    return findings


def _validate_manifest_artifacts(
    *,
    checkpoint_dir: Path,
    render_manifest: HistoryRenderManifest,
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    for section in render_manifest.sections:
        for source_path in section.source_artifact_paths:
            if not (checkpoint_dir / source_path).exists():
                findings.append(
                    _finding(
                        check_id="missing_source_artifact",
                        severity="error",
                        message=(
                            f"Referenced source artifact {source_path.as_posix()} is missing."
                        ),
                        section_id=section.section_id,
                        artifact_path=source_path,
                        reference=source_path.as_posix(),
                    )
                )
    return findings


def _validate_references(
    *,
    checkpoint_model: HistoryCheckpointModel,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    render_manifest: HistoryRenderManifest,
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    active_concepts = _active_concept_ids(checkpoint_model)
    dependency_entries = {
        entry.dependency_id: entry for entry in dependency_inventory.entries
    }
    capsule_ids = {entry.capsule_id for entry in capsule_index.capsules}

    for section in render_manifest.sections:
        for concept_id in section.concept_ids:
            if concept_id not in active_concepts:
                findings.append(
                    _finding(
                        check_id="unknown_concept_reference",
                        severity="error",
                        message=f"Rendered concept reference {concept_id!r} is not active.",
                        section_id=section.section_id,
                        artifact_path=Path("render_manifest.json"),
                        reference=concept_id,
                    )
                )
        for dependency_id in section.dependency_ids:
            entry = dependency_entries.get(dependency_id)
            if entry is None or entry.section_target != section.section_id:
                findings.append(
                    _finding(
                        check_id="unknown_dependency_reference",
                        severity="error",
                        message=(
                            f"Rendered dependency reference {dependency_id!r} is missing or "
                            "does not match the rendered section target."
                        ),
                        section_id=section.section_id,
                        artifact_path=Path("render_manifest.json"),
                        reference=dependency_id,
                    )
                )
        for capsule_id in section.algorithm_capsule_ids:
            if capsule_id not in capsule_ids:
                findings.append(
                    _finding(
                        check_id="unknown_algorithm_capsule_reference",
                        severity="error",
                        message=(
                            f"Rendered algorithm capsule reference {capsule_id!r} is missing "
                            "from the capsule index."
                        ),
                        section_id=section.section_id,
                        artifact_path=Path("render_manifest.json"),
                        reference=capsule_id,
                    )
                )
    return findings


def _matching_subheadings(
    section_block: _SectionBlock,
    title: str,
) -> list[_SectionBlock]:
    return [
        subheading
        for subheading in _subheading_blocks(section_block)
        if subheading.title == title
    ]


def _validate_dependency_subsections(
    *,
    markdown_sections: list[_SectionBlock],
    render_manifest: HistoryRenderManifest,
    dependency_inventory: HistoryDependencyInventory,
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    dependency_entries = {
        entry.dependency_id: entry for entry in dependency_inventory.entries
    }
    markdown_by_title = {section.title: section for section in markdown_sections}
    dependency_sections = {
        "dependencies",
        "build_development_infrastructure",
    }

    for rendered_section in render_manifest.sections:
        if rendered_section.section_id not in dependency_sections:
            continue
        section_block = markdown_by_title.get(rendered_section.title)
        if section_block is None:
            continue
        for dependency_id in rendered_section.dependency_ids:
            entry = dependency_entries.get(dependency_id)
            if entry is None:
                continue
            matches = _matching_subheadings(section_block, entry.display_name)
            if len(matches) != 1:
                findings.append(
                    _finding(
                        check_id="dependency_subsection_shape_invalid",
                        severity="error",
                        message=(
                            f"Dependency subsection heading {entry.display_name!r} must appear "
                            "exactly once."
                        ),
                        section_id=rendered_section.section_id,
                        artifact_path=Path("checkpoint.md"),
                        reference=dependency_id,
                    )
                )
                continue

            subsection = matches[0]
            paragraphs = _paragraphs(subsection.body_lines)
            if (
                any(line.lstrip().startswith("- ") for line in subsection.body_lines)
                or len(paragraphs) != 2
            ):
                findings.append(
                    _finding(
                        check_id="dependency_subsection_shape_invalid",
                        severity="error",
                        message=(
                            f"Dependency subsection {entry.display_name!r} must contain "
                            "exactly two non-empty paragraphs and no bullets."
                        ),
                        section_id=rendered_section.section_id,
                        artifact_path=Path("checkpoint.md"),
                        reference=dependency_id,
                        line_number=subsection.heading_line,
                    )
                )
                continue

            for paragraph in paragraphs:
                if paragraph == "TBD":
                    findings.append(
                        _finding(
                            check_id="dependency_summary_tbd",
                            severity="warning",
                            message=(
                                f"Dependency subsection {entry.display_name!r} contains TBD "
                                "summary content."
                            ),
                            section_id=rendered_section.section_id,
                            artifact_path=Path("checkpoint.md"),
                            reference=dependency_id,
                            line_number=subsection.heading_line,
                        )
                    )
                    break

    return findings


def _validate_release_note_phrases(
    markdown_lines: list[str],
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    first_section_index = next(
        (index for index, line in enumerate(markdown_lines) if line.startswith("## ")),
        None,
    )
    if first_section_index is None:
        return findings

    for index, line in enumerate(
        markdown_lines[first_section_index:], start=first_section_index + 1
    ):
        lowered = line.lower()
        for phrase in _RELEASE_NOTE_PHRASES:
            if phrase in lowered:
                findings.append(
                    _finding(
                        check_id="release_note_phrase",
                        severity="warning",
                        message=f"Rendered markdown contains release-note phrasing: {phrase!r}.",
                        artifact_path=Path("checkpoint.md"),
                        reference=phrase,
                        line_number=index,
                    )
                )
                break
    return findings


def _validate_filler_lines(
    *,
    markdown_sections: list[_SectionBlock],
    outline: HistorySectionOutline,
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    outline_by_title = {section.title: section for section in outline.sections}
    for section_block in markdown_sections:
        section = outline_by_title.get(section_block.title)
        if section is None or section.section_id == "introduction":
            continue
        for offset, line in enumerate(
            section_block.body_lines, start=section_block.body_start_line
        ):
            stripped = line.strip()
            if stripped not in _FILLER_LINES:
                continue
            if section.kind == "core":
                findings.append(
                    _finding(
                        check_id="weak_core_section",
                        severity="error",
                        message=(
                            f"Included core section {section.section_id!r} rendered weak filler "
                            "content."
                        ),
                        section_id=section.section_id,
                        artifact_path=Path("checkpoint.md"),
                        reference=stripped,
                        line_number=offset,
                    )
                )
            else:
                findings.append(
                    _finding(
                        check_id="weak_optional_section",
                        severity="warning",
                        message=(
                            f"Included optional section {section.section_id!r} rendered weak "
                            "filler content."
                        ),
                        section_id=section.section_id,
                        artifact_path=Path("checkpoint.md"),
                        reference=stripped,
                        line_number=offset,
                    )
                )
    return findings


def _validate_algorithm_capsules(
    *,
    markdown_sections: list[_SectionBlock],
    render_manifest: HistoryRenderManifest,
    capsule_index: HistoryAlgorithmCapsuleIndex,
) -> list[HistoryValidationFinding]:
    findings: list[HistoryValidationFinding] = []
    render_section = next(
        (
            section
            for section in render_manifest.sections
            if section.section_id == "algorithms_core_logic"
        ),
        None,
    )
    if render_section is None:
        return findings

    section_block = next(
        (
            section
            for section in markdown_sections
            if section.title == render_section.title
        ),
        None,
    )
    if section_block is None:
        return findings

    title_by_capsule_id = {
        entry.capsule_id: entry.title for entry in capsule_index.capsules
    }
    for capsule_id in render_section.algorithm_capsule_ids:
        title = title_by_capsule_id.get(capsule_id)
        if title is None:
            continue
        matches = _matching_subheadings(section_block, title)
        if len(matches) != 1:
            continue
        subsection = matches[0]
        bullet_lines = [
            line.strip()
            for line in subsection.body_lines
            if line.lstrip().startswith("- ")
        ]
        if len(bullet_lines) == 1 and bullet_lines[0].startswith("- Scope:"):
            findings.append(
                _finding(
                    check_id="algorithm_capsule_thin",
                    severity="warning",
                    message=(
                        f"Algorithm capsule subsection {title!r} only rendered a scope bullet."
                    ),
                    section_id="algorithms_core_logic",
                    artifact_path=Path("checkpoint.md"),
                    reference=capsule_id,
                    line_number=subsection.heading_line,
                )
            )
    return findings


def validate_checkpoint_render(
    *,
    checkpoint_dir: Path,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    dependency_inventory: HistoryDependencyInventory,
    capsule_index: HistoryAlgorithmCapsuleIndex,
    markdown: str,
    render_manifest: HistoryRenderManifest,
    markdown_filename: str = "checkpoint.md",
    render_manifest_filename: str = "render_manifest.json",
) -> HistoryValidationReport:
    """Validate final checkpoint markdown against structured render artifacts."""

    markdown_lines, markdown_sections = _top_level_sections(markdown)
    findings: list[HistoryValidationFinding] = []
    findings.extend(
        _validate_included_and_omitted_sections(
            outline=section_outline,
            manifest=render_manifest,
            markdown_sections=markdown_sections,
        )
    )
    findings.extend(
        _validate_manifest_artifacts(
            checkpoint_dir=checkpoint_dir,
            render_manifest=render_manifest,
        )
    )
    findings.extend(
        _validate_references(
            checkpoint_model=checkpoint_model,
            dependency_inventory=dependency_inventory,
            capsule_index=capsule_index,
            render_manifest=render_manifest,
        )
    )
    findings.extend(
        _validate_dependency_subsections(
            markdown_sections=markdown_sections,
            render_manifest=render_manifest,
            dependency_inventory=dependency_inventory,
        )
    )
    findings.extend(_validate_release_note_phrases(markdown_lines))
    findings.extend(
        _validate_filler_lines(
            markdown_sections=markdown_sections,
            outline=section_outline,
        )
    )
    findings.extend(
        _validate_algorithm_capsules(
            markdown_sections=markdown_sections,
            render_manifest=render_manifest,
            capsule_index=capsule_index,
        )
    )

    findings.sort(
        key=lambda finding: (
            finding.severity,
            finding.check_id,
            finding.section_id or "",
            finding.reference or "",
            finding.line_number or 0,
            (finding.artifact_path or Path("")).as_posix(),
        )
    )
    error_count = sum(finding.severity == "error" for finding in findings)
    warning_count = sum(finding.severity == "warning" for finding in findings)
    report = HistoryValidationReport(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        markdown_path=Path(markdown_filename),
        render_manifest_path=Path(render_manifest_filename),
        error_count=error_count,
        warning_count=warning_count,
        findings=findings,
    )
    for finding in report.findings:
        if finding.artifact_path == Path("checkpoint.md"):
            finding.artifact_path = Path(markdown_filename)
        elif finding.artifact_path == Path("render_manifest.json"):
            finding.artifact_path = Path(render_manifest_filename)
    return report


__all__ = [
    "validate_checkpoint_render",
    "validation_report_path",
]
