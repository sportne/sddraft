"""Shared history-docs manifest models and persistence helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from engllm.domain.models import DomainModel

PreviousCheckpointSource = Literal["initial", "artifact", "explicit_override"]


class HistoryCommitSummary(DomainModel):
    """Deterministic summary for one commit inside an interval."""

    sha: str
    short_sha: str
    timestamp: str
    subject: str


class HistoryCheckpoint(DomainModel):
    """One persisted checkpoint summary record."""

    checkpoint_id: str
    target_commit: str
    target_commit_short: str
    target_commit_timestamp: str
    target_commit_subject: str
    tree_sha: str
    previous_checkpoint_commit: str | None = None
    previous_checkpoint_source: PreviousCheckpointSource


class HistoryInterval(DomainModel):
    """One persisted commit window for a checkpoint."""

    checkpoint_id: str
    start_commit: str | None = None
    end_commit: str
    commit_count: int
    commits: list[HistoryCommitSummary] = Field(default_factory=list)


class HistoryCheckpointPlan(DomainModel):
    """Authoritative registry of built history-docs checkpoints."""

    workspace_id: str
    repo_root: Path
    latest_updated_timestamp: str | None = None
    checkpoints: list[HistoryCheckpoint] = Field(default_factory=list)


def history_artifact_root(shared_root: Path) -> Path:
    """Return the shared history artifact root for one workspace."""

    return shared_root / "history"


def default_checkpoint_plan_path(shared_root: Path) -> Path:
    """Return the default checkpoint plan path for one workspace."""

    return history_artifact_root(shared_root) / "checkpoint_plan.json"


def default_intervals_path(shared_root: Path) -> Path:
    """Return the default intervals JSONL path for one workspace."""

    return history_artifact_root(shared_root) / "intervals.jsonl"


def checkpoint_id_for(timestamp: str, short_sha: str) -> str:
    """Build the canonical checkpoint identifier for one target commit."""

    commit_date = datetime.fromisoformat(timestamp).date().isoformat()
    return f"{commit_date}-{short_sha[:7]}"


def sort_checkpoints(checkpoints: list[HistoryCheckpoint]) -> list[HistoryCheckpoint]:
    """Return checkpoints in canonical timestamp/SHA order."""

    return sorted(
        checkpoints,
        key=lambda item: (item.target_commit_timestamp, item.target_commit),
    )


def normalize_checkpoint_plan(plan: HistoryCheckpointPlan) -> HistoryCheckpointPlan:
    """Return a canonically ordered checkpoint plan."""

    ordered = sort_checkpoints(plan.checkpoints)
    latest_timestamp = ordered[-1].target_commit_timestamp if ordered else None
    return HistoryCheckpointPlan(
        workspace_id=plan.workspace_id,
        repo_root=plan.repo_root.resolve(),
        latest_updated_timestamp=latest_timestamp,
        checkpoints=ordered,
    )


def load_checkpoint_plan(path: Path) -> HistoryCheckpointPlan | None:
    """Load a checkpoint plan if it exists."""

    if not path.exists():
        return None
    return normalize_checkpoint_plan(
        HistoryCheckpointPlan.model_validate_json(path.read_text(encoding="utf-8"))
    )


def save_checkpoint_plan(plan: HistoryCheckpointPlan, path: Path) -> None:
    """Write a canonical checkpoint plan JSON file."""

    normalized = normalize_checkpoint_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")


def load_intervals(path: Path) -> list[HistoryInterval]:
    """Load persisted history intervals from JSONL."""

    if not path.exists():
        return []
    intervals: list[HistoryInterval] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            intervals.append(HistoryInterval.model_validate_json(line))
    return intervals


def save_intervals(intervals: list[HistoryInterval], path: Path) -> None:
    """Rewrite canonical history intervals as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [interval.model_dump_json() for interval in intervals]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
