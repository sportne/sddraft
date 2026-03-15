"""Additional tests for commit impact classification branches."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.commit_impact import build_commit_impact
from sddraft.domain.models import FileDiffSummary


def test_build_commit_impact_covers_all_classification_paths() -> None:
    diffs = [
        FileDiffSummary(path=Path("a.py"), signature_changes=["def x()"], language="python"),
        FileDiffSummary(path=Path("b.py"), dependency_changes=["import os"], language="python"),
        FileDiffSummary(path=Path("c.py"), comment_only=True, language="python"),
        FileDiffSummary(path=Path("d.py"), added_lines=1, language="python"),
        FileDiffSummary(path=Path("e.py"), language="python"),
    ]

    impact = build_commit_impact("HEAD~2..HEAD", diffs)
    assert set(impact.change_kinds) == {
        "interface_change",
        "dependency_change",
        "documentation_only",
        "logic_change",
        "unknown",
    }
    assert "Interface Design" in impact.impacted_sections
    assert "Design Overview" in impact.impacted_sections
    assert "Referenced Documents" in impact.impacted_sections
    assert "Detailed Design" in impact.impacted_sections


def test_build_commit_impact_handles_no_files() -> None:
    impact = build_commit_impact("HEAD~1..HEAD", [])
    assert impact.changed_files == []
    assert impact.change_kinds == []
    assert impact.impacted_sections == []
    assert "No changed files" in impact.summary
