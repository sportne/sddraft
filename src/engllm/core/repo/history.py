"""Read-only git history helpers for history-docs workflows."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path
from typing import Literal

from engllm.domain.errors import GitError
from engllm.domain.models import DomainModel

_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


class GitCommitSummary(DomainModel):
    """Deterministic summary for one git commit."""

    sha: str
    short_sha: str
    timestamp: str
    subject: str


class GitCommitMetadata(GitCommitSummary):
    """Expanded metadata for a specific git commit."""

    tree_sha: str


class GitCommitDiffSpec(DomainModel):
    """Deterministic first-parent diff description for one commit."""

    commit_sha: str
    parent_commit: str | None = None
    base_rev: str
    diff_basis: Literal["root", "first_parent"]


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


def iter_first_parent_commits(
    repo_root: Path,
    *,
    target_commit: str,
    previous_commit: str | None = None,
) -> list[GitCommitSummary]:
    """Return chronological first-parent commit summaries up to one target."""

    args = [
        "log",
        "--format=%H%x00%h%x00%cI%x00%s",
        "--reverse",
        "--topo-order",
        "--first-parent",
    ]
    if previous_commit is None:
        args.append(target_commit)
    else:
        args.append(f"{previous_commit}..{target_commit}")

    output = _run_git(repo_root, *args)
    if not output:
        return []

    commits: list[GitCommitSummary] = []
    for line in output.splitlines():
        fields = line.split("\x00")
        if len(fields) != 4:
            raise GitError(
                "Unexpected git log payload while building first-parent commits"
            )
        commits.append(
            GitCommitSummary(
                sha=fields[0],
                short_sha=fields[1],
                timestamp=fields[2],
                subject=fields[3],
            )
        )
    return commits


def export_commit_snapshot(
    repo_root: Path,
    *,
    target_commit: str,
    destination_root: Path,
) -> Path:
    """Export one commit tree into a disposable snapshot directory."""

    destination_root.mkdir(parents=True, exist_ok=True)
    if not _run_git(repo_root, "ls-tree", "--name-only", target_commit):
        return destination_root

    archive_path = destination_root.parent / "snapshot.tar"
    try:
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                "--output",
                str(archive_path),
                target_commit,
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment issue
        raise GitError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise GitError(f"git archive failed: {message}") from exc

    if archive_path.stat().st_size == 0:
        archive_path.unlink(missing_ok=True)
        return destination_root
    try:
        with tarfile.open(archive_path, mode="r") as archive:
            try:
                archive.extractall(destination_root, filter="data")
            except TypeError:
                archive.extractall(destination_root)
    except (tarfile.TarError, OSError) as exc:
        raise GitError(
            f"Failed to extract git archive for {target_commit}: {exc}"
        ) from exc
    finally:
        archive_path.unlink(missing_ok=True)

    return destination_root


def read_file_at_commit(
    repo_root: Path,
    rev: str,
    file_path: Path,
) -> str:
    """Return the UTF-8 text of one tracked file as it existed at a commit."""

    try:
        result = subprocess.run(
            ["git", "show", f"{rev}:{file_path.as_posix()}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment issue
        raise GitError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raw_message = exc.stderr or exc.stdout
        message = (
            raw_message.decode("utf-8", errors="replace").strip()
            if raw_message is not None
            else str(exc).strip()
        )
        raise GitError(
            f"git show failed for {file_path.as_posix()} at {rev}: {message}"
        ) from exc
    return result.stdout.decode("utf-8")


def get_commit_parents(repo_root: Path, rev: str) -> list[str]:
    """Return the parent commit SHAs for one commit."""

    commit_sha = resolve_commit(repo_root, rev)
    payload = _run_git(repo_root, "rev-list", "--parents", "-n", "1", commit_sha)
    fields = payload.split()
    if not fields or fields[0] != commit_sha:
        raise GitError(f"Unexpected parent payload for revision {rev!r}")
    return fields[1:]


def list_reachable_tags_by_commit(
    repo_root: Path,
    *,
    target_commit: str,
    commit_shas: list[str],
) -> dict[str, list[str]]:
    """Return reachable tag names keyed by commit SHA on the analyzed history."""

    commit_set = set(commit_shas)
    tag_output = _run_git(repo_root, "tag", "--merged", target_commit)
    if not tag_output:
        return {}

    tags_by_commit: dict[str, list[str]] = {}
    for tag_name in sorted(tag_output.splitlines()):
        if not tag_name:
            continue
        tag_commit = resolve_commit(repo_root, tag_name)
        if tag_commit not in commit_set:
            continue
        tags_by_commit.setdefault(tag_commit, []).append(tag_name)
    return {key: sorted(value) for key, value in tags_by_commit.items()}


def list_tree_paths_at_commit(
    repo_root: Path,
    rev: str,
    *,
    prefix: Path | None = None,
) -> list[Path]:
    """Return tracked file paths at one commit, optionally filtered by prefix."""

    args = ["ls-tree", "-r", "--name-only", rev]
    if prefix is not None:
        args.extend(["--", prefix.as_posix()])
    output = _run_git(repo_root, *args)
    if not output:
        return []
    return sorted(Path(line) for line in output.splitlines() if line)


def describe_commit_diff(repo_root: Path, rev: str) -> GitCommitDiffSpec:
    """Describe the deterministic diff basis for one commit."""

    commit_sha = resolve_commit(repo_root, rev)
    payload = _run_git(repo_root, "rev-list", "--parents", "-n", "1", commit_sha)
    fields = payload.split()
    if not fields or fields[0] != commit_sha:
        raise GitError(f"Unexpected parent payload for revision {rev!r}")
    if len(fields) == 1:
        return GitCommitDiffSpec(
            commit_sha=commit_sha,
            parent_commit=None,
            base_rev=_EMPTY_TREE_SHA,
            diff_basis="root",
        )
    return GitCommitDiffSpec(
        commit_sha=commit_sha,
        parent_commit=fields[1],
        base_rev=fields[1],
        diff_basis="first_parent",
    )
