"""CLI rendering for ask answers."""

from __future__ import annotations

from engllm.tools.ask.models import QueryAnswer


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
