"""Import smoke tests for package modules."""

from __future__ import annotations

import sys
from importlib import import_module, reload

import pytest

MODULES = (
    "engllm",
    "engllm.core.config.loader",
    "engllm.core.repo.scanner",
    "engllm.core.analysis.retrieval",
    "engllm.core.analysis.hierarchy",
    "engllm.prompts.ask.builders",
    "engllm.prompts.core.builders",
    "engllm.prompts.sdd.builders",
    "engllm.llm.base",
    "engllm.tools.sdd.generate",
    "engllm.tools.ask.ask",
    "engllm.core.hierarchy_docs",
    "engllm.tools.sdd.markdown",
    "engllm.core.render.hierarchy",
    "engllm.cli.main",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_importable(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    if "." in module_name:
        sys.modules.pop(module_name.split(".", 1)[0], None)
    module = import_module(module_name)
    reload(module)
