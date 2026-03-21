"""Deterministic prompt builders for the history-docs tool."""

from __future__ import annotations

import json

from engllm.prompts.history_docs.templates import DEPENDENCY_SUMMARY_SYSTEM_PROMPT
from engllm.tools.history_docs.models import HistoryDependencyEntry


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_dependency_summary_prompt(
    entry: HistoryDependencyEntry,
) -> tuple[str, str]:
    """Return prompts for one dependency summary generation request."""

    user_prompt = (
        "Document this direct dependency using only the provided evidence.\n"
        f"Dependency Evidence:\n{_json(entry.model_dump(mode='json'))}\n"
    )
    return DEPENDENCY_SUMMARY_SYSTEM_PROMPT, user_prompt
