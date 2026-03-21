"""History-docs H1 workflow: checkpoint selection and interval manifests."""

from __future__ import annotations

from pathlib import Path

from engllm.core.analysis.history import (
    HistoryCheckpoint,
    HistoryCheckpointPlan,
    HistoryCommitSummary,
    HistoryInterval,
    PreviousCheckpointSource,
    checkpoint_id_for,
    default_checkpoint_plan_path,
    default_intervals_path,
    load_checkpoint_plan,
    load_intervals,
    normalize_checkpoint_plan,
    save_checkpoint_plan,
    save_intervals,
)
from engllm.core.repo.history import (
    get_commit_metadata,
    is_strict_ancestor,
    iter_interval_commits,
    resolve_commit,
)
from engllm.core.workspaces import build_workspace_context
from engllm.domain.errors import AnalysisError, GitError
from engllm.domain.models import ProjectConfig
from engllm.tools.history_docs.models import HistoryBuildResult


def _resolve_workspace_id(repo_root: Path, workspace_id: str | None) -> str:
    return workspace_id or repo_root.resolve().name


def _select_previous_from_artifacts(
    checkpoints: list[HistoryCheckpoint],
    *,
    repo_root: Path,
    target_commit: str,
) -> str | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.target_commit == target_commit:
            continue
        if is_strict_ancestor(repo_root, checkpoint.target_commit, target_commit):
            return checkpoint.target_commit
    return None


def _validate_previous_boundary(
    repo_root: Path,
    *,
    target_commit: str,
    previous_commit: str,
) -> str:
    resolved_previous = resolve_commit(repo_root, previous_commit)
    if resolved_previous == target_commit:
        raise GitError(
            "Previous checkpoint commit must be a strict ancestor of the target commit"
        )
    if not is_strict_ancestor(repo_root, resolved_previous, target_commit):
        raise GitError(
            f"Previous checkpoint commit {resolved_previous} is not an ancestor of {target_commit}"
        )
    return resolved_previous


def build_history_docs_checkpoint(
    *,
    project_config: ProjectConfig,
    repo_root: Path,
    checkpoint_commit: str,
    previous_checkpoint_commit: str | None = None,
    workspace_id: str | None = None,
) -> HistoryBuildResult:
    """Persist H1 checkpoint and interval manifests for one target commit."""

    resolved_repo_root = repo_root.resolve()
    target_metadata = get_commit_metadata(resolved_repo_root, checkpoint_commit)
    resolved_workspace_id = _resolve_workspace_id(resolved_repo_root, workspace_id)
    workspace = build_workspace_context(
        output_root=project_config.workspace.output_root,
        workspace_id=resolved_workspace_id,
        kind="repo",
        repo_root=resolved_repo_root,
    )
    checkpoint_plan_path = default_checkpoint_plan_path(workspace.shared_root)
    intervals_path = default_intervals_path(workspace.shared_root)

    existing_plan = load_checkpoint_plan(checkpoint_plan_path)
    if (
        existing_plan is not None
        and existing_plan.repo_root.resolve() != resolved_repo_root
    ):
        raise AnalysisError(
            "Existing history checkpoint plan repo root does not match this run"
        )

    if previous_checkpoint_commit is not None:
        resolved_previous = _validate_previous_boundary(
            resolved_repo_root,
            target_commit=target_metadata.sha,
            previous_commit=previous_checkpoint_commit,
        )
        previous_source: PreviousCheckpointSource = "explicit_override"
    else:
        resolved_previous = None
        if existing_plan is not None:
            resolved_previous = _select_previous_from_artifacts(
                existing_plan.checkpoints,
                repo_root=resolved_repo_root,
                target_commit=target_metadata.sha,
            )
        previous_source = "artifact" if resolved_previous is not None else "initial"

    interval_commits = iter_interval_commits(
        resolved_repo_root,
        target_commit=target_metadata.sha,
        previous_commit=resolved_previous,
    )
    checkpoint_id = checkpoint_id_for(
        target_metadata.timestamp, target_metadata.short_sha
    )
    checkpoint = HistoryCheckpoint(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        target_commit_short=target_metadata.short_sha,
        target_commit_timestamp=target_metadata.timestamp,
        target_commit_subject=target_metadata.subject,
        tree_sha=target_metadata.tree_sha,
        previous_checkpoint_commit=resolved_previous,
        previous_checkpoint_source=previous_source,
    )
    interval = HistoryInterval(
        checkpoint_id=checkpoint_id,
        start_commit=resolved_previous,
        end_commit=target_metadata.sha,
        commit_count=len(interval_commits),
        commits=[
            HistoryCommitSummary(
                sha=commit.sha,
                short_sha=commit.short_sha,
                timestamp=commit.timestamp,
                subject=commit.subject,
            )
            for commit in interval_commits
        ],
    )

    prior_checkpoints = (
        existing_plan.checkpoints[:] if existing_plan is not None else []
    )
    checkpoint_by_commit = {
        item.target_commit: item
        for item in prior_checkpoints
        if item.target_commit != checkpoint.target_commit
    }
    checkpoint_by_commit[checkpoint.target_commit] = checkpoint
    normalized_plan = normalize_checkpoint_plan(
        HistoryCheckpointPlan(
            workspace_id=resolved_workspace_id,
            repo_root=resolved_repo_root,
            checkpoints=list(checkpoint_by_commit.values()),
        )
    )

    prior_intervals = load_intervals(intervals_path)
    interval_by_checkpoint = {
        item.checkpoint_id: item
        for item in prior_intervals
        if item.checkpoint_id != interval.checkpoint_id
    }
    interval_by_checkpoint[interval.checkpoint_id] = interval
    ordered_intervals = [
        interval_by_checkpoint[item.checkpoint_id]
        for item in normalized_plan.checkpoints
        if item.checkpoint_id in interval_by_checkpoint
    ]

    save_checkpoint_plan(normalized_plan, checkpoint_plan_path)
    save_intervals(ordered_intervals, intervals_path)

    return HistoryBuildResult(
        workspace_id=resolved_workspace_id,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        previous_checkpoint_source=previous_source,
        commit_count=len(interval_commits),
        checkpoint_plan_path=checkpoint_plan_path,
        intervals_path=intervals_path,
    )
