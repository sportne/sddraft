"""User-facing renderers for the SDD tool."""

from __future__ import annotations

from pathlib import Path

from engllm.domain.errors import RenderingError
from engllm.tools.sdd.models import SDDDocument, UpdateProposalReport


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


def render_update_report_markdown(report: UpdateProposalReport) -> str:
    """Render update proposals to markdown report."""

    lines = [f"# Update Proposals ({report.commit_range})", ""]
    lines.append("## Impacted Sections")
    if report.impacted_sections:
        lines.extend(f"- {section}" for section in report.impacted_sections)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Proposals")
    if not report.proposals:
        lines.append("No proposals generated.")
    else:
        for proposal in report.proposals:
            lines.append(f"### {proposal.section_id} {proposal.title}")
            lines.append(f"Priority: {proposal.review_priority}")
            lines.append("")
            lines.append("#### Rationale")
            lines.append(proposal.rationale)
            lines.append("")
            lines.append("#### Proposed Text")
            lines.append(proposal.proposed_text)
            lines.append("")
            if proposal.uncertainty_list:
                lines.append("#### Uncertainty")
                lines.extend(f"- {item}" for item in proposal.uncertainty_list)
                lines.append("")

    return "\n".join(lines).strip() + "\n"
