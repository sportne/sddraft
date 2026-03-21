"""Deterministic prompt builders for shared platform summarization tasks."""

from __future__ import annotations

import json

from engllm.domain.models import (
    CodeUnitSummary,
    DirectorySummaryDoc,
    FileSummaryDoc,
    SubtreeRollup,
    SymbolSummary,
)
from engllm.prompts.core.templates import (
    DIRECTORY_SUMMARY_SYSTEM_PROMPT,
    FILE_SUMMARY_SYSTEM_PROMPT,
)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_file_summary_prompt(
    *,
    code_summary: CodeUnitSummary,
    symbols: list[SymbolSummary],
    code_excerpt: str,
) -> tuple[str, str]:
    """Return system and user prompts for file summary generation."""

    user_prompt = (
        "Generate a concise summary for one source file.\n"
        f"Code Summary:\n{_json(code_summary.model_dump(mode='json'))}\n\n"
        f"Symbols:\n{_json([item.model_dump(mode='json') for item in symbols])}\n\n"
        f"Code Excerpt:\n{code_excerpt or 'TBD'}\n"
    )
    return FILE_SUMMARY_SYSTEM_PROMPT, user_prompt


def build_directory_summary_prompt(
    *,
    directory_path: str,
    local_files: list[FileSummaryDoc],
    child_directories: list[DirectorySummaryDoc],
    subtree_rollup: SubtreeRollup,
) -> tuple[str, str]:
    """Return prompts for recursive directory subtree summary generation."""

    role = "project root" if directory_path == "." else "directory subtree"
    root_instruction = (
        "This path is the repository root. Write a project-level overview.\n\n"
        if directory_path == "."
        else ""
    )

    user_prompt = (
        "Generate a recursive subtree summary for this directory.\n"
        f"Directory Path:\n{directory_path}\n\n"
        f"Directory Role:\n{role}\n\n"
        f"{root_instruction}"
        f"Local File Summaries:\n{_json([item.model_dump(mode='json') for item in local_files])}\n\n"
        f"Child Directory Summaries:\n{_json([item.model_dump(mode='json') for item in child_directories])}\n"
        f"\nSubtree Rollup:\n{_json(subtree_rollup.model_dump(mode='json'))}\n"
    )
    return DIRECTORY_SUMMARY_SYSTEM_PROMPT, user_prompt
