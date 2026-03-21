"""Deterministic prompt builders for the ask tool."""

from __future__ import annotations

import json

from engllm.prompts.ask.templates import (
    INTENSIVE_SCREENING_SYSTEM_PROMPT,
    QUERY_SYSTEM_PROMPT,
)
from engllm.tools.ask.models import IntensiveCorpusChunk, QueryEvidencePack


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_query_prompt(pack: QueryEvidencePack) -> tuple[str, str]:
    """Return system and user prompts for grounded Q&A."""

    user_prompt = (
        f"Question:\n{pack.request.question}\n\n"
        f"Session History:\n{_json(pack.request.session_history)}\n\n"
        f"Primary Chunks:\n{_json([chunk.model_dump(mode='json') for chunk in pack.primary_chunks])}\n\n"
        f"Selected Chunks:\n{_json([chunk.model_dump(mode='json') for chunk in pack.chunks])}\n\n"
        f"Citations:\n{_json([citation.model_dump(mode='json') for citation in pack.citations])}\n\n"
        f"Related Files:\n{_json([path.as_posix() for path in pack.related_files])}\n\n"
        f"Related Symbols:\n{_json(pack.related_symbols)}\n\n"
        f"Related Sections:\n{_json(pack.related_sections)}\n\n"
        f"Related Commits:\n{_json(pack.related_commits)}\n\n"
        f"Inclusion Reasons:\n{_json([reason.model_dump(mode='json') for reason in pack.inclusion_reasons])}\n"
    )
    return QUERY_SYSTEM_PROMPT, user_prompt


def build_intensive_screening_prompt(
    *,
    question: str,
    session_history: list[str],
    chunk: IntensiveCorpusChunk,
) -> tuple[str, str]:
    """Return prompts for one intensive corpus chunk screening pass."""

    user_prompt = (
        "Screen this structured repository chunk for relevance to the question.\n"
        f"Screening Input:\n{_json({'question': question, 'session_history': session_history, 'chunk': chunk.model_dump(mode='json')})}\n"
    )
    return INTENSIVE_SCREENING_SYSTEM_PROMPT, user_prompt
