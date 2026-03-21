"""History-docs tool model namespace."""

from __future__ import annotations

from pathlib import Path

from engllm.domain.models import DomainModel


class HistoryBuildResult(DomainModel):
    """Result for one H1 history-docs build run."""

    workspace_id: str
    checkpoint_id: str
    target_commit: str
    previous_checkpoint_commit: str | None = None
    previous_checkpoint_source: str
    commit_count: int
    checkpoint_plan_path: Path
    intervals_path: Path


__all__ = ["HistoryBuildResult"]
