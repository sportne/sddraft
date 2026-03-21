"""Question-answering tool for repository evidence."""

from __future__ import annotations

from typing import Any

__all__ = ["answer_question"]


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily to avoid prompt/workflow import cycles."""

    if name == "answer_question":
        from engllm.tools.ask.ask import answer_question

        return answer_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
