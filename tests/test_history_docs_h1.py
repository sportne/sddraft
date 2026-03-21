"""Tests for history-docs H1 checkpoint traversal and manifest persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.analysis.history import (
    checkpoint_id_for,
    load_checkpoint_plan,
    load_intervals,
)
from engllm.core.repo.history import (
    get_commit_metadata,
    iter_interval_commits,
    resolve_commit,
)
from engllm.domain.errors import GitError
from engllm.domain.models import ProjectConfig
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from tests.history_docs_helpers import create_forked_repo as _create_forked_repo
from tests.history_docs_helpers import create_linear_repo as _create_linear_repo
from tests.history_docs_helpers import history_paths as _history_paths
from tests.history_docs_helpers import write_project_config as _write_project_config


def test_git_history_helpers_resolve_metadata_and_interval_order(
    tmp_path: Path,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)

    resolved = resolve_commit(repo_root, "HEAD")
    metadata = get_commit_metadata(repo_root, commits[-1])
    all_commits = iter_interval_commits(repo_root, target_commit=commits[-1])
    final_interval = iter_interval_commits(
        repo_root,
        target_commit=commits[-1],
        previous_commit=commits[-2],
    )

    assert resolved == commits[-1]
    assert metadata.sha == commits[-1]
    assert metadata.subject == "add third step"
    assert len(metadata.tree_sha) == 40
    assert [item.subject for item in all_commits] == [
        "initial commit",
        "add second step",
        "add third step",
    ]
    assert [item.subject for item in final_interval] == ["add third step"]


def test_checkpoint_id_for_uses_commit_date_and_short_sha() -> None:
    assert (
        checkpoint_id_for("2024-05-06T12:00:00+00:00", "abcdef123456")
        == "2024-05-06-abcdef1"
    )


def test_build_history_docs_checkpoint_initial_run_writes_manifests(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[1],
    )
    checkpoint_plan_path, intervals_path = _history_paths(
        sample_project_config.workspace.output_root,
        repo_root.name,
    )
    plan = load_checkpoint_plan(checkpoint_plan_path)
    intervals = load_intervals(intervals_path)

    assert result.workspace_id == repo_root.name
    assert result.previous_checkpoint_commit is None
    assert result.previous_checkpoint_source == "initial"
    assert result.commit_count == 2
    assert result.checkpoint_plan_path == checkpoint_plan_path
    assert result.intervals_path == intervals_path
    assert plan is not None
    assert len(plan.checkpoints) == 1
    assert plan.checkpoints[0].target_commit == commits[1]
    assert plan.checkpoints[0].previous_checkpoint_commit is None
    assert intervals[0].start_commit is None
    assert intervals[0].end_commit == commits[1]
    assert intervals[0].commit_count == 2
    assert [item.subject for item in intervals[0].commits] == [
        "initial commit",
        "add second step",
    ]


def test_build_history_docs_checkpoint_uses_latest_ancestor_from_artifacts(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[0],
    )
    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[1],
    )
    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[2],
    )

    checkpoint_plan_path, intervals_path = _history_paths(
        sample_project_config.workspace.output_root,
        repo_root.name,
    )
    plan = load_checkpoint_plan(checkpoint_plan_path)
    intervals = load_intervals(intervals_path)

    assert result.previous_checkpoint_commit == commits[1]
    assert result.previous_checkpoint_source == "artifact"
    assert plan is not None
    assert [item.target_commit for item in plan.checkpoints] == commits
    assert [item.end_commit for item in intervals] == commits
    assert intervals[-1].start_commit == commits[1]
    assert [item.subject for item in intervals[-1].commits] == ["add third step"]


def test_build_history_docs_checkpoint_explicit_previous_override_wins(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[0],
    )
    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[1],
    )
    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[2],
        previous_checkpoint_commit=commits[0],
    )

    _, intervals_path = _history_paths(
        sample_project_config.workspace.output_root,
        repo_root.name,
    )
    intervals = load_intervals(intervals_path)

    assert result.previous_checkpoint_commit == commits[0]
    assert result.previous_checkpoint_source == "explicit_override"
    assert intervals[-1].start_commit == commits[0]
    assert intervals[-1].commit_count == 2
    assert [item.subject for item in intervals[-1].commits] == [
        "add second step",
        "add third step",
    ]


def test_build_history_docs_checkpoint_rejects_equal_previous(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    with pytest.raises(GitError, match="strict ancestor"):
        build_history_docs_checkpoint(
            project_config=sample_project_config,
            repo_root=repo_root,
            checkpoint_commit=commits[1],
            previous_checkpoint_commit=commits[1],
        )


def test_build_history_docs_checkpoint_rejects_non_ancestor_previous(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, main_tip, feature_tip = _create_forked_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    with pytest.raises(GitError, match="is not an ancestor"):
        build_history_docs_checkpoint(
            project_config=sample_project_config,
            repo_root=repo_root,
            checkpoint_commit=main_tip,
            previous_checkpoint_commit=feature_tip,
        )


def test_build_history_docs_checkpoint_rerun_is_idempotent(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[1],
    )
    checkpoint_plan_path, intervals_path = _history_paths(
        sample_project_config.workspace.output_root,
        repo_root.name,
    )
    first_plan = checkpoint_plan_path.read_text(encoding="utf-8")
    first_intervals = intervals_path.read_text(encoding="utf-8")

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[1],
    )

    assert checkpoint_plan_path.read_text(encoding="utf-8") == first_plan
    assert intervals_path.read_text(encoding="utf-8") == first_intervals


def test_build_history_docs_checkpoint_workspace_override_changes_artifact_location(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits[0],
        workspace_id="custom-workspace",
    )

    assert result.workspace_id == "custom-workspace"
    assert result.checkpoint_plan_path == (
        tmp_path
        / "artifacts"
        / "workspaces"
        / "custom-workspace"
        / "shared"
        / "history"
        / "checkpoint_plan.json"
    )


def test_history_docs_cli_requires_checkpoint_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root, _ = _create_linear_repo(tmp_path)
    config_path = tmp_path / "project.yaml"
    _write_project_config(config_path, tmp_path / "artifacts")

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
        ]
    )

    assert rc == 2
    assert "checkpoint-commit" in capsys.readouterr().err


def test_history_docs_cli_build_succeeds_and_invalid_rev_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    config_path = tmp_path / "project.yaml"
    _write_project_config(config_path, tmp_path / "artifacts")

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            commits[1],
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "Built history checkpoint" in output
    assert "Checkpoint plan:" in output
    assert "Intervals:" in output

    rc_invalid = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            "not-a-real-rev",
            "--workspace-id",
            "cli-custom",
        ]
    )
    invalid_output = capsys.readouterr().out

    assert rc_invalid == 2
    assert "Error:" in invalid_output
    assert "not-a-real-rev" in invalid_output


def test_history_docs_cli_workspace_override_changes_artifact_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root, commits = _create_linear_repo(tmp_path)
    config_path = tmp_path / "project.yaml"
    _write_project_config(config_path, tmp_path / "artifacts")

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            commits[0],
            "--workspace-id",
            "cli-history",
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "cli-history" in output
    assert (
        tmp_path
        / "artifacts"
        / "workspaces"
        / "cli-history"
        / "shared"
        / "history"
        / "checkpoint_plan.json"
    ).exists()
