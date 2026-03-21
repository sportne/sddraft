"""Internal tool registration models used by the CLI router."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCommand:
    """One executable CLI command exposed by a registered tool."""

    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """CLI-visible registration for one EngLLM tool namespace."""

    name: str
    help: str
    commands: tuple[ToolCommand, ...]
