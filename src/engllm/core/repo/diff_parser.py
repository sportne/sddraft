"""Git diff extraction and normalization."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from engllm.core.repo.language_analyzers import detect_language, get_analyzer_for_path
from engllm.domain.errors import GitError
from engllm.domain.models import FileDiffSummary


def get_git_diff(commit_range: str, repo_root: Path) -> str:
    """Return unified git diff text for a commit/range."""

    command = ["git", "diff", "--unified=0", commit_range]
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "unknown git error"
        raise GitError(
            f"Failed to run git diff for '{commit_range}': {stderr}"
        ) from exc
    return proc.stdout


def get_git_diff_between(base_rev: str, target_rev: str, repo_root: Path) -> str:
    """Return unified git diff text between two revisions."""

    command = ["git", "diff", "--unified=0", base_rev, target_rev]
    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "unknown git error"
        raise GitError(
            f"Failed to run git diff for '{base_rev}..{target_rev}': {stderr}"
        ) from exc
    return proc.stdout


def extract_changed_symbol_names(changed_lines: list[str]) -> set[str]:
    """Extract likely changed symbol names from signature-like diff lines."""

    names: set[str] = set()
    for line in changed_lines:
        stripped = line.strip()
        for pattern in (
            r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)",
        ):
            match = re.search(pattern, stripped)
            if match:
                names.add(match.group(1))
        call_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
        if call_match:
            names.add(call_match.group(1))
    return names


def _finalize_file_summary(
    path: str, changed_lines: list[str], added: int, removed: int
) -> FileDiffSummary:
    path_obj = Path(path)
    analyzer = get_analyzer_for_path(path_obj)

    signature_changes = analyzer.signature_changes(changed_lines)
    dependency_changes = analyzer.dependency_changes(changed_lines)
    comment_only = bool(changed_lines) and all(
        analyzer.is_comment_line(line) for line in changed_lines
    )

    return FileDiffSummary(
        path=path_obj,
        language=detect_language(path_obj),
        added_lines=added,
        removed_lines=removed,
        signature_changes=signature_changes,
        dependency_changes=dependency_changes,
        comment_only=comment_only,
    )


def parse_diff(diff_text: str) -> list[FileDiffSummary]:
    """Parse a unified diff into structured per-file summaries."""

    summaries: list[FileDiffSummary] = []

    current_path: str | None = None
    current_added = 0
    current_removed = 0
    current_changed_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_path, current_added, current_removed, current_changed_lines
        if current_path is None:
            return
        summaries.append(
            _finalize_file_summary(
                path=current_path,
                changed_lines=current_changed_lines,
                added=current_added,
                removed=current_removed,
            )
        )
        current_path = None
        current_added = 0
        current_removed = 0
        current_changed_lines = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush_current()
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                current_path = b_path[2:] if b_path.startswith("b/") else b_path
            continue

        if line.startswith("+++ "):
            candidate = line[4:].strip()
            if candidate != "/dev/null":
                current_path = (
                    candidate[2:] if candidate.startswith("b/") else candidate
                )
            continue

        if (
            line.startswith("@@")
            or line.startswith("index ")
            or line.startswith("--- ")
        ):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current_added += 1
            current_changed_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            current_removed += 1
            current_changed_lines.append(line[1:])

    flush_current()
    return sorted(summaries, key=lambda item: item.path.as_posix())
