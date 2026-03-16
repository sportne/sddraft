"""Tests for tree-sitter language analyzers and routing."""

from __future__ import annotations

from pathlib import Path

from sddraft.repo.language_analyzers import (
    detect_language,
    get_analyzer_for_language,
    get_analyzer_for_path,
)


def test_detect_language_router() -> None:
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("A.java")) == "java"
    assert detect_language(Path("a.cpp")) == "cpp"
    assert detect_language(Path("a.hpp")) == "cpp"
    assert detect_language(Path("a.ts")) == "typescript"
    assert detect_language(Path("a.js")) == "javascript"
    assert detect_language(Path("a.go")) == "go"
    assert detect_language(Path("a.rs")) == "rust"
    assert detect_language(Path("a.cs")) == "csharp"
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


def test_javascript_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("app.js"))
    summary, interfaces = analyzer.analyze(
        Path("app.js"),
        "import x from 'lib';\nclass App {}\nfunction run() { return 1; }\n",
    )

    assert summary.language == "javascript"
    assert "App" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("import") for item in summary.imports)
    assert {item.name for item in interfaces} >= {"App", "run"}


def test_typescript_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("app.ts"))
    summary, interfaces = analyzer.analyze(
        Path("app.ts"),
        "import {X} from './x';\ninterface Api {}\nexport function run(x: number): number { return x; }\n",
    )

    assert summary.language == "typescript"
    assert "Api" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("import") for item in summary.imports)
    assert {item.name for item in interfaces} >= {"Api", "run"}


def test_go_rust_and_csharp_analyzers_extract_symbols() -> None:
    go_analyzer = get_analyzer_for_path(Path("main.go"))
    go_summary, _ = go_analyzer.analyze(
        Path("main.go"),
        'package main\nimport "fmt"\ntype Service struct{}\nfunc Run() {}\n',
    )
    assert go_summary.language == "go"
    assert "Service" in go_summary.classes
    assert "Run" in go_summary.functions

    rust_analyzer = get_analyzer_for_path(Path("lib.rs"))
    rust_summary, _ = rust_analyzer.analyze(
        Path("lib.rs"),
        "use std::fmt;\nstruct Service;\nfn run() {}\n",
    )
    assert rust_summary.language == "rust"
    assert "Service" in rust_summary.classes
    assert "run" in rust_summary.functions

    csharp_analyzer = get_analyzer_for_path(Path("Program.cs"))
    csharp_summary, _ = csharp_analyzer.analyze(
        Path("Program.cs"),
        "using System;\nclass Program { public static void Run() {} }\n",
    )
    assert csharp_summary.language == "csharp"
    assert "Program" in csharp_summary.classes
    assert "Run" in csharp_summary.functions


def test_unknown_analyzer_behavior() -> None:
    analyzer = get_analyzer_for_path(Path("README.md"))
    assert analyzer.language == "unknown"
    assert analyzer.is_comment_line("# comment")
    assert analyzer.signature_changes(["fn run()", "xyz"]) == ["fn run()"]
    assert analyzer.dependency_changes(["use std::fmt;", "abc"]) == ["use std::fmt;"]
    assert get_analyzer_for_language("unknown").language == "unknown"
    summary, interfaces = analyzer.analyze(
        Path("README.md"),
        "import os\ninclude common.mk\njust text\n",
    )
    assert summary.language == "unknown"
    assert summary.functions == []
    assert summary.classes == []
    assert summary.imports == ["import os", "include common.mk"]
    assert interfaces == []
