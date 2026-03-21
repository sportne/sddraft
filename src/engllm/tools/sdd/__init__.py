"""Software design documentation tool."""

from __future__ import annotations

from typing import Any

__all__ = ["generate_sdd", "propose_updates"]


def __getattr__(name: str) -> Any:
    """Resolve package exports lazily to avoid prompt/workflow import cycles."""

    if name == "generate_sdd":
        from engllm.tools.sdd.generate import generate_sdd

        return generate_sdd
    if name == "propose_updates":
        from engllm.tools.sdd.propose_updates import propose_updates

        return propose_updates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
