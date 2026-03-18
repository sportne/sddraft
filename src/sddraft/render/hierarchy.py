"""Hierarchy documentation renderers."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.errors import RenderingError
from sddraft.domain.models import (
    DirectorySummaryDoc,
    DirectorySummaryRecord,
    FileSummaryDoc,
    FileSummaryRecord,
    HierarchyDocArtifact,
)


def file_doc_relative_path(file_path: Path) -> Path:
    """Return the markdown path used for one file summary document."""

    suffix = f"{file_path.suffix}.md" if file_path.suffix else ".md"
    return file_path.with_suffix(suffix)


def directory_doc_relative_path(directory_path: Path) -> Path:
    """Return the markdown path used for one directory summary document."""

    if directory_path == Path("."):
        return Path("_directory.md")
    return directory_path / "_directory.md"


def render_file_summary_markdown(summary: FileSummaryDoc) -> str:
    """Render one file summary markdown document."""

    lines = [f"# File Summary: {summary.path.as_posix()}", ""]
    lines.append(summary.summary)
    lines.append("")

    if summary.functions:
        lines.append("## Functions")
        lines.extend(f"- {name}" for name in summary.functions)
        lines.append("")

    if summary.classes:
        lines.append("## Classes")
        lines.extend(f"- {name}" for name in summary.classes)
        lines.append("")

    if summary.imports:
        lines.append("## Imports")
        lines.extend(f"- {name}" for name in summary.imports)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_directory_summary_markdown(summary: DirectorySummaryDoc) -> str:
    """Render one directory summary markdown document."""

    title_path = summary.path.as_posix() if summary.path != Path(".") else "."
    lines = [f"# Directory Summary: {title_path}", ""]
    lines.append(summary.summary)
    lines.append("")

    lines.append("## Subtree At A Glance")
    lines.append(f"- Descendant Files: {summary.subtree_rollup.descendant_file_count}")
    lines.append(
        f"- Descendant Directories: {summary.subtree_rollup.descendant_directory_count}"
    )
    if summary.subtree_rollup.language_counts:
        language_parts = ", ".join(
            f"{language}={count}"
            for language, count in sorted(
                summary.subtree_rollup.language_counts.items()
            )
        )
        lines.append(f"- Language Distribution: {language_parts}")
    else:
        lines.append("- Language Distribution: None")
    if summary.subtree_rollup.key_topics:
        lines.append(f"- Key Topics: {', '.join(summary.subtree_rollup.key_topics)}")
    else:
        lines.append("- Key Topics: None")
    if summary.subtree_rollup.representative_files:
        lines.append("- Representative Files:")
        lines.extend(
            f"  - {path.as_posix()}"
            for path in summary.subtree_rollup.representative_files
        )
    else:
        lines.append("- Representative Files: None")
    lines.append("")

    lines.append("## Local Files")
    if summary.local_files:
        lines.extend(f"- {path.as_posix()}" for path in summary.local_files)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Child Directories")
    if summary.child_directories:
        lines.extend(f"- {path.as_posix()}" for path in summary.child_directories)
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_file_summary_markdown(
    *,
    hierarchy_root: Path,
    summary: FileSummaryDoc | FileSummaryRecord,
) -> Path:
    """Write one file summary markdown document and return the path."""

    path = hierarchy_root / file_doc_relative_path(summary.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = FileSummaryDoc(
        node_id=summary.node_id,
        path=summary.path,
        language=summary.language,
        summary=summary.summary,
        functions=summary.functions,
        classes=summary.classes,
        imports=summary.imports,
        evidence_refs=summary.evidence_refs,
        missing_information=summary.missing_information,
        confidence=summary.confidence,
    )
    path.write_text(render_file_summary_markdown(doc), encoding="utf-8")
    return path


def write_directory_summary_markdown(
    *,
    hierarchy_root: Path,
    summary: DirectorySummaryDoc | DirectorySummaryRecord,
) -> Path:
    """Write one directory summary markdown document and return the path."""

    path = hierarchy_root / directory_doc_relative_path(summary.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = DirectorySummaryDoc(
        node_id=summary.node_id,
        path=summary.path,
        summary=summary.summary,
        local_files=summary.local_files,
        child_directories=summary.child_directories,
        subtree_rollup=summary.subtree_rollup,
        evidence_refs=summary.evidence_refs,
        missing_information=summary.missing_information,
        confidence=summary.confidence,
    )
    path.write_text(render_directory_summary_markdown(doc), encoding="utf-8")
    return path


def write_hierarchy_markdown(
    hierarchy_root: Path, artifact: HierarchyDocArtifact
) -> dict[str, Path]:
    """Write hierarchy markdown docs and return node-id to doc path mapping."""

    node_docs: dict[str, Path] = {}

    try:
        hierarchy_root.mkdir(parents=True, exist_ok=True)

        for file_item in sorted(
            artifact.file_summaries, key=lambda entry: entry.path.as_posix()
        ):
            path = write_file_summary_markdown(
                hierarchy_root=hierarchy_root,
                summary=file_item,
            )
            node_docs[file_item.node_id] = path

        for directory_item in sorted(
            artifact.directory_summaries, key=lambda entry: entry.path.as_posix()
        ):
            path = write_directory_summary_markdown(
                hierarchy_root=hierarchy_root,
                summary=directory_item,
            )
            node_docs[directory_item.node_id] = path
    except OSError as exc:
        raise RenderingError(
            f"Failed writing hierarchy markdown to {hierarchy_root}: {exc}"
        ) from exc

    return node_docs
