"""Human-readable report renderers."""

from __future__ import annotations

from sddraft.domain.models import QueryAnswer, UpdateProposalReport


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


def render_query_answer_text(answer: QueryAnswer) -> str:
    """Render grounded Q&A response for CLI output."""

    lines = ["Answer:", answer.answer, ""]

    if answer.citations:
        lines.append("Citations:")
        for citation in answer.citations:
            location = citation.source_path.as_posix()
            if citation.line_start is not None and citation.line_end is not None:
                location = f"{location}:{citation.line_start}-{citation.line_end}"
            lines.append(f"- {location} [{citation.chunk_id}]")
        lines.append("")

    if answer.uncertainty:
        lines.append("Uncertainty:")
        lines.extend(f"- {item}" for item in answer.uncertainty)
        lines.append("")

    if answer.missing_information:
        lines.append("Missing Information:")
        lines.extend(f"- {item}" for item in answer.missing_information)
        lines.append("")

    lines.append(f"Confidence: {answer.confidence:.2f}")
    return "\n".join(lines)
