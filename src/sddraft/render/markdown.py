"""Markdown renderers."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.errors import RenderingError
from sddraft.domain.models import SDDDocument


def render_sdd_markdown(document: SDDDocument) -> str:
    """Render structured SDD document to markdown."""

    lines = [f"# {document.title}", ""]
    for section in document.sections:
        lines.append(f"## {section.section_id} {section.title}")
        lines.append(section.content)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_markdown(path: Path, content: str) -> Path:
    """Persist markdown content."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RenderingError(f"Failed writing markdown to {path}: {exc}") from exc
    return path
