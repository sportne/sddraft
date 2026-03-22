"""Import smoke tests for package modules."""

from __future__ import annotations

import sys
from importlib import import_module, reload

import pytest

MODULES = (
    "engllm",
    "engllm.core.config.loader",
    "engllm.core.repo.scanner",
    "engllm.core.repo.history",
    "engllm.core.analysis.retrieval",
    "engllm.core.analysis.history",
    "engllm.core.analysis.hierarchy",
    "engllm.prompts.ask.builders",
    "engllm.prompts.core.builders",
    "engllm.prompts.history_docs.builders",
    "engllm.prompts.history_docs",
    "engllm.prompts.sdd.builders",
    "engllm.llm.base",
    "engllm.tools.sdd.generate",
    "engllm.tools.ask.ask",
    "engllm.tools.history_docs",
    "engllm.tools.history_docs.benchmark",
    "engllm.tools.history_docs.build",
    "engllm.tools.history_docs.algorithm_capsules",
    "engllm.tools.history_docs.checkpoint_model",
    "engllm.tools.history_docs.dependencies",
    "engllm.tools.history_docs.delta",
    "engllm.tools.history_docs.render",
    "engllm.tools.history_docs.semantic_planner",
    "engllm.tools.history_docs.semantic_structure",
    "engllm.tools.history_docs.section_outline",
    "engllm.tools.history_docs.structure",
    "engllm.tools.history_docs.validation",
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


def test_lazy_tool_package_exports() -> None:
    for module_name in (
        "engllm.tools.ask.ask",
        "engllm.tools.ask",
        "engllm.tools.sdd.generate",
        "engllm.tools.sdd.propose_updates",
        "engllm.tools.sdd",
        "engllm.tools.history_docs.build",
        "engllm.tools.history_docs",
    ):
        sys.modules.pop(module_name, None)

    ask_package = import_module("engllm.tools.ask")
    sdd_package = import_module("engllm.tools.sdd")
    history_package = import_module("engllm.tools.history_docs")

    assert callable(ask_package.answer_question)
    assert callable(sdd_package.generate_sdd)
    assert callable(sdd_package.propose_updates)
    assert callable(history_package.build_history_docs_checkpoint)

    with pytest.raises(AttributeError):
        _ = ask_package.missing_export  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _ = sdd_package.missing_export  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        _ = history_package.missing_export  # type: ignore[attr-defined]
