"""Shared git-history test helpers for history-docs workflows."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def git(
    repo_root: Path,
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> str:
    """Run one git command inside a temporary test repository."""

    full_env = os.environ.copy()
    full_env.setdefault("GIT_AUTHOR_NAME", "EngLLM Test")
    full_env.setdefault("GIT_AUTHOR_EMAIL", "engllm@example.com")
    full_env.setdefault("GIT_COMMITTER_NAME", "EngLLM Test")
    full_env.setdefault("GIT_COMMITTER_EMAIL", "engllm@example.com")
    if env is not None:
        full_env.update(env)
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=check,
        env=full_env,
    )
    if check:
        return result.stdout.strip()
    return (result.stdout + result.stderr).strip()


def init_repo(tmp_path: Path) -> Path:
    """Create and initialize a temporary git repository."""

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    git(repo_root, "init")
    return repo_root


def commit_file(
    repo_root: Path,
    relative_path: str,
    content: str,
    *,
    message: str,
    timestamp: str,
) -> str:
    """Write one file and commit it with a fixed timestamp."""

    file_path = repo_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    git(repo_root, "add", relative_path)
    env = {
        "GIT_AUTHOR_DATE": timestamp,
        "GIT_COMMITTER_DATE": timestamp,
    }
    git(repo_root, "commit", "-m", message, env=env)
    return git(repo_root, "rev-parse", "HEAD")


def create_linear_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    """Create a simple linear three-commit test repository."""

    repo_root = init_repo(tmp_path)
    commits = [
        commit_file(
            repo_root,
            "src/app.py",
            "print('one')\n",
            message="initial commit",
            timestamp="2024-01-01T10:00:00+00:00",
        ),
        commit_file(
            repo_root,
            "src/app.py",
            "print('two')\n",
            message="add second step",
            timestamp="2024-02-01T10:00:00+00:00",
        ),
        commit_file(
            repo_root,
            "src/app.py",
            "print('three')\n",
            message="add third step",
            timestamp="2024-03-01T10:00:00+00:00",
        ),
    ]
    return repo_root, commits


def create_forked_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a repo with diverging branch tips for ancestor tests."""

    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "src/app.py",
        "print('base')\n",
        message="base commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    current_branch = git(repo_root, "branch", "--show-current")
    main_tip = commit_file(
        repo_root,
        "src/app.py",
        "print('main')\n",
        message="main change",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    git(repo_root, "checkout", "-b", "feature", base)
    feature_tip = commit_file(
        repo_root,
        "src/app.py",
        "print('feature')\n",
        message="feature change",
        timestamp="2024-01-03T10:00:00+00:00",
    )
    git(repo_root, "checkout", current_branch)
    return repo_root, main_tip, feature_tip


def create_merge_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a repo with one merge commit for first-parent diff tests."""

    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "src/app.py",
        "def base() -> int:\n    return 1\n",
        message="base commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    current_branch = git(repo_root, "branch", "--show-current")
    git(repo_root, "checkout", "-b", "feature", base)
    commit_file(
        repo_root,
        "src/feature.py",
        "def feature() -> int:\n    return 2\n",
        message="feature change",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    git(repo_root, "checkout", current_branch)
    commit_file(
        repo_root,
        "src/app.py",
        "def base() -> int:\n    return 3\n",
        message="main change",
        timestamp="2024-01-03T10:00:00+00:00",
    )
    git(
        repo_root,
        "merge",
        "--no-ff",
        "feature",
        "-m",
        "merge feature",
        env={
            "GIT_AUTHOR_DATE": "2024-01-04T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-04T10:00:00+00:00",
        },
    )
    return repo_root, git(repo_root, "rev-parse", "HEAD")


def history_paths(output_root: Path, workspace_id: str) -> tuple[Path, Path]:
    """Return shared H1 artifact paths for one workspace."""

    history_root = output_root / "workspaces" / workspace_id / "shared" / "history"
    return history_root / "checkpoint_plan.json", history_root / "intervals.jsonl"


def snapshot_manifest_path(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> Path:
    """Return the shared H2 snapshot manifest path for one checkpoint."""

    return (
        output_root
        / "workspaces"
        / workspace_id
        / "shared"
        / "history"
        / "checkpoints"
        / checkpoint_id
        / "snapshot_manifest.json"
    )


def snapshot_structural_model_path(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> Path:
    """Return the tool-scoped H2 structural model path for one checkpoint."""

    return (
        output_root
        / "workspaces"
        / workspace_id
        / "tools"
        / "history_docs"
        / "checkpoints"
        / checkpoint_id
        / "snapshot_structural_model.json"
    )


def interval_delta_model_path(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> Path:
    """Return the tool-scoped H3 interval-delta model path for one checkpoint."""

    return (
        output_root
        / "workspaces"
        / workspace_id
        / "tools"
        / "history_docs"
        / "checkpoints"
        / checkpoint_id
        / "interval_delta_model.json"
    )


def checkpoint_model_path(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> Path:
    """Return the tool-scoped H4 checkpoint model path for one checkpoint."""

    return (
        output_root
        / "workspaces"
        / workspace_id
        / "tools"
        / "history_docs"
        / "checkpoints"
        / checkpoint_id
        / "checkpoint_model.json"
    )


def section_outline_path(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> Path:
    """Return the tool-scoped H5 section outline path for one checkpoint."""

    return (
        output_root
        / "workspaces"
        / workspace_id
        / "tools"
        / "history_docs"
        / "checkpoints"
        / checkpoint_id
        / "section_outline.json"
    )


def write_project_config(
    path: Path,
    output_root: Path,
    *,
    source_roots: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    """Write a minimal project config for history-docs CLI tests."""

    roots = source_roots or ["repo/src"]
    excludes = exclude or []
    lines = [
        "project_name: Example",
        "workspace:",
        f"  output_root: {output_root.as_posix()}",
        "sources:",
        "  roots:",
    ]
    lines.extend(f"    - {root}" for root in roots)
    if excludes:
        lines.append("  exclude:")
        lines.extend(f"    - {item}" for item in excludes)
    lines.extend(
        [
            "tools:",
            "  sdd:",
            "    template: unused-template.yaml",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
