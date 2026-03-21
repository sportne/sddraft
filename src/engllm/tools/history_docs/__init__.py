"""History-walk documentation tool."""

from __future__ import annotations

from typing import Any

__all__ = ["build_history_docs_checkpoint"]


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily to avoid unnecessary import coupling."""

    if name == "build_history_docs_checkpoint":
        from engllm.tools.history_docs.build import build_history_docs_checkpoint

        return build_history_docs_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
