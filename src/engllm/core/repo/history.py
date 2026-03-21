"""Read-only git history helpers for history-docs workflows."""

from __future__ import annotations

import subprocess
from pathlib import Path

from engllm.domain.errors import GitError
from engllm.domain.models import DomainModel


class GitCommitSummary(DomainModel):
    """Deterministic summary for one git commit."""

    sha: str
    short_sha: str
    timestamp: str
    subject: str


class GitCommitMetadata(GitCommitSummary):
    """Expanded metadata for a specific git commit."""

    tree_sha: str


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment issue
        raise GitError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        joined_args = " ".join(args)
        raise GitError(f"git {joined_args} failed: {message}") from exc
    return result.stdout.strip()


def resolve_commit(repo_root: Path, rev: str) -> str:
    """Resolve any rev-parseable revision to a full commit SHA."""

    return _run_git(repo_root, "rev-parse", f"{rev}^{{commit}}")


def get_commit_metadata(repo_root: Path, rev: str) -> GitCommitMetadata:
    """Return deterministic metadata for one commit revision."""

    payload = _run_git(
        repo_root,
        "show",
        "-s",
        "--format=%H%x00%h%x00%cI%x00%s%x00%T",
        rev,
    )
    fields = payload.split("\x00")
    if len(fields) != 5:
        raise GitError(f"Unexpected git metadata payload for revision {rev!r}")
    return GitCommitMetadata(
        sha=fields[0],
        short_sha=fields[1],
        timestamp=fields[2],
        subject=fields[3],
        tree_sha=fields[4],
    )


def is_strict_ancestor(repo_root: Path, ancestor_rev: str, descendant_rev: str) -> bool:
    """Return True when ``ancestor_rev`` is a strict ancestor of ``descendant_rev``."""

    ancestor_sha = resolve_commit(repo_root, ancestor_rev)
    descendant_sha = resolve_commit(repo_root, descendant_rev)
    if ancestor_sha == descendant_sha:
        return False

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    message = (result.stderr or result.stdout).strip()
    raise GitError(f"git merge-base --is-ancestor failed: {message}")


def iter_interval_commits(
    repo_root: Path,
    *,
    target_commit: str,
    previous_commit: str | None = None,
) -> list[GitCommitSummary]:
    """Return ordered commit summaries for one checkpoint interval."""

    args = [
        "log",
        "--format=%H%x00%h%x00%cI%x00%s",
        "--reverse",
        "--topo-order",
    ]
    if previous_commit is None:
        args.append(target_commit)
    else:
        args.extend(["--ancestry-path", f"{previous_commit}..{target_commit}"])

    output = _run_git(repo_root, *args)
    if not output:
        return []

    commits: list[GitCommitSummary] = []
    for line in output.splitlines():
        fields = line.split("\x00")
        if len(fields) != 4:
            raise GitError("Unexpected git log payload while building interval commits")
        commits.append(
            GitCommitSummary(
                sha=fields[0],
                short_sha=fields[1],
                timestamp=fields[2],
                subject=fields[3],
            )
        )
    return commits
