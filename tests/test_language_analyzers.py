"""Tests for tree-sitter language analyzers and routing."""

from __future__ import annotations

from pathlib import Path

from sddraft.repo.language_analyzers import (
    detect_language,
    get_analyzer_for_path,
)


def test_detect_language_router() -> None:
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("A.java")) == "java"
    assert detect_language(Path("a.cpp")) == "cpp"
    assert detect_language(Path("a.hpp")) == "cpp"
    assert detect_language(Path("notes.txt")) == "unknown"


def test_python_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("module.py"))
    summary, interfaces = analyzer.analyze(
        Path("module.py"),
        "import os\n\nclass Service:\n    def run(self):\n        pass\n\ndef plan():\n    return 1\n",
    )

    assert summary.language == "python"
    assert "Service" in summary.classes
    assert "plan" in summary.functions
    assert any("import os" in item for item in summary.imports)

    names = {item.name for item in interfaces}
    assert "Service" in names
    assert "plan" in names


def test_java_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("Main.java"))
    summary, interfaces = analyzer.analyze(
        Path("Main.java"),
        "package demo;\nimport java.util.List;\npublic class Main {\n  public int run(int x) { return x; }\n}\n",
    )

    assert summary.language == "java"
    assert "Main" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("import") for item in summary.imports)

    names = {item.name for item in interfaces}
    assert "Main" in names
    assert "run" in names


def test_cpp_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("core.cpp"))
    summary, interfaces = analyzer.analyze(
        Path("core.cpp"),
        "#include <vector>\nclass Core {};\nint run(int x) { return x; }\n",
    )

    assert summary.language == "cpp"
    assert "Core" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("#include") for item in summary.imports)

    names = {item.name for item in interfaces}
    assert "Core" in names
    assert "run" in names
