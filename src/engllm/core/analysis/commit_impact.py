"""Commit impact classification and section mapping."""

from __future__ import annotations

from engllm.domain.models import CommitImpact, FileDiffSummary


def _classify_file_change(file_diff: FileDiffSummary) -> str:
    if file_diff.signature_changes:
        return "interface_change"
    if file_diff.dependency_changes:
        return "dependency_change"
    if file_diff.comment_only:
        return "documentation_only"
    if file_diff.added_lines or file_diff.removed_lines:
        return "logic_change"
    return "unknown"


def _map_change_to_sections(change_kind: str) -> list[str]:
    mapping = {
        "interface_change": ["Interface Design"],
        "logic_change": ["Detailed Design"],
        "dependency_change": ["Design Overview"],
        "documentation_only": ["Referenced Documents"],
        "unknown": ["Detailed Design"],
    }
    return mapping.get(change_kind, ["Detailed Design"])


def build_commit_impact(
    commit_range: str,
    file_diffs: list[FileDiffSummary],
) -> CommitImpact:
    """Build normalized commit impact from file-level diff summaries."""

    change_kinds = sorted({_classify_file_change(diff) for diff in file_diffs})
    impacted_sections = sorted(
        {section for kind in change_kinds for section in _map_change_to_sections(kind)}
    )

    if not file_diffs:
        summary = f"No changed files detected for commit range '{commit_range}'."
    else:
        summary = (
            f"Detected {len(file_diffs)} changed files across commit range "
            f"'{commit_range}' with change types: {', '.join(change_kinds)}."
        )

    return CommitImpact(
        commit_range=commit_range,
        changed_files=file_diffs,
        change_kinds=change_kinds,
        impacted_sections=impacted_sections,
        summary=summary,
    )
