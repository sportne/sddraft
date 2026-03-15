"""Import smoke tests for package modules."""

from __future__ import annotations

import sys
from importlib import import_module, reload

import pytest

MODULES = (
    "sddraft",
    "sddraft.config.loader",
    "sddraft.repo.scanner",
    "sddraft.analysis.retrieval",
    "sddraft.analysis.hierarchy",
    "sddraft.prompts.builders",
    "sddraft.llm.base",
    "sddraft.workflows.generate",
    "sddraft.workflows.hierarchy_docs",
    "sddraft.render.markdown",
    "sddraft.render.hierarchy",
    "sddraft.cli.main",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_importable(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    if "." in module_name:
        sys.modules.pop(module_name.split(".", 1)[0], None)
    module = import_module(module_name)
    reload(module)
