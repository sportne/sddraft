"""Diff inspection workflow."""

from __future__ import annotations

from pathlib import Path

from engllm.core.analysis.commit_impact import build_commit_impact
from engllm.core.repo.diff_parser import get_git_diff, parse_diff
from engllm.domain.models import InspectDiffResult


def inspect_diff(commit_range: str, repo_root: Path) -> InspectDiffResult:
    """Inspect git diff and return normalized impact model."""

    raw_diff = get_git_diff(commit_range=commit_range, repo_root=repo_root)
    file_diffs = parse_diff(raw_diff)
    impact = build_commit_impact(commit_range=commit_range, file_diffs=file_diffs)
    return InspectDiffResult(impact=impact, raw_diff=raw_diff)
