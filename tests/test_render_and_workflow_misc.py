"""Additional tests for renderers and minor workflow branches."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from sddraft.domain.errors import RenderingError
from sddraft.domain.models import (
    Citation,
    DirectorySummaryDoc,
    FileDiffSummary,
    QueryAnswer,
    SectionUpdateProposal,
    SubtreeRollup,
    UpdateProposalReport,
)
from sddraft.render.hierarchy import render_directory_summary_markdown
from sddraft.render.json_artifacts import write_json_model
from sddraft.render.markdown import render_sdd_markdown, write_markdown
from sddraft.render.reports import (
    render_query_answer_text,
    render_update_report_markdown,
)
from sddraft.workflows.inspect_diff import inspect_diff


class _SimpleModel(QueryAnswer):
    pass


def test_render_report_and_query_text_paths() -> None:
    empty_report = UpdateProposalReport(
        commit_range="HEAD~1..HEAD",
        impacted_sections=[],
        proposals=[],
    )
    md = render_update_report_markdown(empty_report)
    assert "- None" in md
    assert "No proposals generated." in md

    proposal_report = UpdateProposalReport(
        commit_range="HEAD~1..HEAD",
        impacted_sections=["Interface Design"],
        proposals=[
            SectionUpdateProposal(
                section_id="3",
                title="Interface Design",
                existing_text="old",
                proposed_text="new",
                rationale="changed",
                uncertainty_list=["TBD"],
                review_priority="high",
                evidence_refs=[],
            )
        ],
    )
    md2 = render_update_report_markdown(proposal_report)
    assert "### 3 Interface Design" in md2
    assert "#### Uncertainty" in md2

    answer = QueryAnswer(
        answer="x",
        citations=[
            Citation(
                chunk_id="c",
                source_path=Path("src/a.py"),
                line_start=1,
                line_end=2,
                quote="q",
            )
        ],
        uncertainty=["u"],
        missing_information=["m"],
        confidence=0.5,
    )
    rendered = render_query_answer_text(answer)
    assert "Citations:" in rendered
    assert "Uncertainty:" in rendered
    assert "Missing Information:" in rendered


def test_render_directory_summary_includes_subtree_rollup_section() -> None:
    summary = DirectorySummaryDoc(
        node_id="dir::src",
        path=Path("src"),
        summary="Source subtree summary.",
        local_files=[Path("src/app.py")],
        child_directories=[Path("src/sub")],
        subtree_rollup=SubtreeRollup(
            descendant_file_count=3,
            descendant_directory_count=1,
            language_counts={"python": 2, "unknown": 1},
            key_topics=["workflow", "analysis"],
            representative_files=[Path("src/app.py"), Path("src/sub/worker.py")],
        ),
    )
    rendered = render_directory_summary_markdown(summary)
    assert "## Subtree At A Glance" in rendered
    assert "Descendant Files: 3" in rendered
    assert "Language Distribution: python=2, unknown=1" in rendered
    assert "src/sub/worker.py" in rendered


def test_write_render_error_paths(tmp_path: Path, monkeypatch) -> None:
    model = _SimpleModel(answer="x", citations=[], confidence=0.1)

    original_write_text = Path.write_text

    def _boom(self: Path, data: str, encoding: str = "utf-8") -> int:
        raise OSError("boom")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(RenderingError):
        write_json_model(tmp_path / "a.json", model)
    with pytest.raises(RenderingError):
        write_markdown(tmp_path / "a.md", "text")

    monkeypatch.setattr(Path, "write_text", original_write_text)


def test_render_sdd_markdown_and_inspect_diff(monkeypatch) -> None:
    inspect_module = importlib.import_module("sddraft.workflows.inspect_diff")
    from sddraft.domain.models import SDDDocument, SectionDraft

    doc = SDDDocument(
        csc_id="X",
        title="Title",
        sections=[SectionDraft(section_id="1", title="Scope", content="Body")],
    )
    md = render_sdd_markdown(doc)
    assert "# Title" in md
    assert "## 1 Scope" in md

    monkeypatch.setattr(
        inspect_module, "get_git_diff", lambda commit_range, repo_root: ""
    )
    monkeypatch.setattr(
        inspect_module,
        "parse_diff",
        lambda diff_text: [FileDiffSummary(path=Path("a.py"), language="python")],
    )

    result = inspect_diff("HEAD~1..HEAD", Path("."))
    assert result.impact.changed_files
