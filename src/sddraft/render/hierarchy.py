"""Hierarchy documentation renderers."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.errors import RenderingError
from sddraft.domain.models import (
    DirectorySummaryDoc,
    FileSummaryDoc,
    HierarchyDocArtifact,
)


def _file_doc_relative_path(file_path: Path) -> Path:
    suffix = f"{file_path.suffix}.md" if file_path.suffix else ".md"
    return file_path.with_suffix(suffix)


def _directory_doc_relative_path(directory_path: Path) -> Path:
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
            path = hierarchy_root / _file_doc_relative_path(file_item.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_file_summary_markdown(file_item), encoding="utf-8")
            node_docs[file_item.node_id] = path

        for directory_item in sorted(
            artifact.directory_summaries, key=lambda entry: entry.path.as_posix()
        ):
            path = hierarchy_root / _directory_doc_relative_path(directory_item.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                render_directory_summary_markdown(directory_item), encoding="utf-8"
            )
            node_docs[directory_item.node_id] = path
    except OSError as exc:
        raise RenderingError(
            f"Failed writing hierarchy markdown to {hierarchy_root}: {exc}"
        ) from exc

    return node_docs
