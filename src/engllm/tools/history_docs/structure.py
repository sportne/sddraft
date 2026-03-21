"""Shared history-docs structural helpers for H2/H3."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from engllm.tools.history_docs.models import HistorySubsystemCandidate

_MAX_REPRESENTATIVE_FILES = 12


def normalize_relative_path(path: Path) -> Path:
    """Normalize an empty relative path to ``.``."""

    return Path(".") if path in {Path(""), Path(".")} else path


def owning_source_root(file_path: Path, analyzed_roots: list[Path]) -> Path:
    """Return the most specific analyzed source root that owns ``file_path``."""

    ordered_roots = sorted(
        analyzed_roots,
        key=lambda item: (-len(item.parts), item.as_posix()),
    )
    for root in ordered_roots:
        if root == Path("."):
            return root
        try:
            file_path.relative_to(root)
            return root
        except ValueError:
            continue
    return Path(".")


def subsystem_candidate_id(
    *,
    source_root: Path,
    group_path: Path,
) -> str:
    """Return the canonical subsystem candidate identifier."""

    if group_path == source_root:
        group_relative = Path(".")
    elif source_root == Path("."):
        group_relative = group_path
    else:
        group_relative = group_path.relative_to(source_root)
    return f"subsystem::{source_root.as_posix()}::{group_relative.as_posix()}"


def subsystem_candidate_id_for_file(
    *,
    file_path: Path,
    analyzed_roots: list[Path],
) -> str:
    """Return the deterministic subsystem candidate id for one file path."""

    source_root = owning_source_root(file_path, analyzed_roots)
    relative_under_root = (
        file_path if source_root == Path(".") else file_path.relative_to(source_root)
    )
    if len(relative_under_root.parts) <= 1:
        group_path = source_root
    else:
        first_segment = Path(relative_under_root.parts[0])
        group_path = (
            first_segment if source_root == Path(".") else source_root / first_segment
        )
    return subsystem_candidate_id(source_root=source_root, group_path=group_path)


def subsystem_is_root_scope(candidate_id: str) -> bool:
    """Return whether a subsystem candidate id represents the root scope."""

    return candidate_id.endswith("::.")


def build_subsystem_candidates(
    *,
    files: list[Path],
    symbol_counts_by_path: Mapping[Path, int],
    language_by_path: Mapping[Path, str],
    analyzed_roots: list[Path],
) -> list[HistorySubsystemCandidate]:
    """Group files into deterministic subsystem candidates."""

    grouped_files: dict[tuple[Path, Path], list[Path]] = defaultdict(list)
    for file_path in sorted(files, key=lambda item: item.as_posix()):
        source_root = owning_source_root(file_path, analyzed_roots)
        relative_under_root = (
            file_path
            if source_root == Path(".")
            else file_path.relative_to(source_root)
        )
        if len(relative_under_root.parts) <= 1:
            group_path = source_root
        else:
            first_segment = Path(relative_under_root.parts[0])
            group_path = (
                first_segment
                if source_root == Path(".")
                else source_root / first_segment
            )
        grouped_files[(source_root, group_path)].append(file_path)

    candidates: list[HistorySubsystemCandidate] = []
    for (source_root, group_path), grouped in sorted(
        grouped_files.items(),
        key=lambda item: (item[0][0].as_posix(), item[0][1].as_posix()),
    ):
        language_counts: Counter[str] = Counter()
        symbol_count = 0
        for file_path in grouped:
            language_counts[language_by_path[file_path]] += 1
            symbol_count += symbol_counts_by_path.get(file_path, 0)
        candidates.append(
            HistorySubsystemCandidate(
                candidate_id=subsystem_candidate_id(
                    source_root=source_root,
                    group_path=group_path,
                ),
                source_root=source_root,
                group_path=group_path,
                file_count=len(grouped),
                symbol_count=symbol_count,
                language_counts={
                    key: language_counts[key] for key in sorted(language_counts)
                },
                representative_files=grouped[:_MAX_REPRESENTATIVE_FILES],
            )
        )
    return candidates


__all__ = [
    "build_subsystem_candidates",
    "normalize_relative_path",
    "owning_source_root",
    "subsystem_candidate_id",
    "subsystem_candidate_id_for_file",
    "subsystem_is_root_scope",
]
