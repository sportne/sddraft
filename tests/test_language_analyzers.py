"""Tests for tree-sitter language analyzers and symbol-first contracts."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.models import SymbolSummary
from sddraft.repo.language_analyzers import (
    detect_language,
    get_analyzer_for_language,
    get_analyzer_for_path,
)


def _symbol_names(symbols: list[SymbolSummary]) -> set[str]:
    return {item.name for item in symbols}


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


def test_python_analyzer_extracts_symbols_with_owner_and_span() -> None:
    analyzer = get_analyzer_for_path(Path("module.py"))
    summary, symbols = analyzer.analyze(
        Path("module.py"),
        (
            "import os\n\n"
            "class Service:\n"
            "    def run(self):\n"
            "        pass\n\n"
            "def plan():\n"
            "    return 1\n"
        ),
    )

    assert summary.language == "python"
    assert "Service" in summary.classes
    assert "plan" in summary.functions
    assert any("import os" in item for item in summary.imports)
    assert _symbol_names(symbols) >= {"Service", "run", "plan"}

    run_symbol = next(item for item in symbols if item.name == "run")
    assert run_symbol.kind == "method"
    assert run_symbol.owner_qualified_name == "Service"
    assert run_symbol.line_start is not None


def test_java_analyzer_extracts_symbols_with_owner_and_span() -> None:
    analyzer = get_analyzer_for_path(Path("Main.java"))
    summary, symbols = analyzer.analyze(
        Path("Main.java"),
        (
            "package demo;\n"
            "import java.util.List;\n"
            "public class Main {\n"
            "  public int run(int x) { return x; }\n"
            "}\n"
        ),
    )

    assert summary.language == "java"
    assert "Main" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("import") for item in summary.imports)
    assert _symbol_names(symbols) >= {"Main", "run"}

    run_symbol = next(item for item in symbols if item.name == "run")
    assert run_symbol.kind == "method"
    assert run_symbol.owner_qualified_name == "Main"
    assert run_symbol.line_start is not None


def test_cpp_analyzer_extracts_symbols() -> None:
    analyzer = get_analyzer_for_path(Path("core.cpp"))
    summary, symbols = analyzer.analyze(
        Path("core.cpp"),
        "#include <vector>\nclass Core {};\nint run(int x) { return x; }\n",
    )

    assert summary.language == "cpp"
    assert "Core" in summary.classes
    assert "run" in summary.functions
    assert any(item.startswith("#include") for item in summary.imports)
    assert _symbol_names(symbols) >= {"Core", "run"}
    assert any(item.line_start is not None for item in symbols)


def test_javascript_and_typescript_analyzers_extract_symbols() -> None:
    js_analyzer = get_analyzer_for_path(Path("app.js"))
    js_summary, js_symbols = js_analyzer.analyze(
        Path("app.js"),
        "import x from 'lib';\nclass App { run() { return 1; } }\nfunction plan() { return 1; }\n",
    )

    assert js_summary.language == "javascript"
    assert _symbol_names(js_symbols) >= {"App", "run", "plan"}
    run_js = next(item for item in js_symbols if item.name == "run")
    assert run_js.owner_qualified_name == "App"

    ts_analyzer = get_analyzer_for_path(Path("app.ts"))
    ts_summary, ts_symbols = ts_analyzer.analyze(
        Path("app.ts"),
        (
            "import {X} from './x';\n"
            "interface Api { run(x: number): number; }\n"
            "export function execute(x: number): number { return x; }\n"
        ),
    )
    assert ts_summary.language == "typescript"
    assert _symbol_names(ts_symbols) >= {"Api", "run", "execute"}
    assert any(item.kind == "interface" and item.name == "Api" for item in ts_symbols)


def test_go_rust_and_csharp_analyzers_extract_symbols() -> None:
    go_analyzer = get_analyzer_for_path(Path("main.go"))
    go_summary, go_symbols = go_analyzer.analyze(
        Path("main.go"),
        'package main\nimport "fmt"\ntype Service struct{}\nfunc (s *Service) Run() {}\n',
    )
    assert go_summary.language == "go"
    assert "Service" in go_summary.classes
    assert "Run" in go_summary.functions
    run_go = next(item for item in go_symbols if item.name == "Run")
    assert run_go.owner_qualified_name == "Service"

    rust_analyzer = get_analyzer_for_path(Path("lib.rs"))
    rust_summary, rust_symbols = rust_analyzer.analyze(
        Path("lib.rs"),
        "use std::fmt;\nstruct Service;\nimpl Service { fn run() {} }\n",
    )
    assert rust_summary.language == "rust"
    assert "Service" in rust_summary.classes
    assert "run" in rust_summary.functions
    run_rust = next(item for item in rust_symbols if item.name == "run")
    assert run_rust.owner_qualified_name == "Service"

    csharp_analyzer = get_analyzer_for_path(Path("Program.cs"))
    csharp_summary, csharp_symbols = csharp_analyzer.analyze(
        Path("Program.cs"),
        "using System;\nclass Program { public static void Run() {} }\n",
    )
    assert csharp_summary.language == "csharp"
    assert "Program" in csharp_summary.classes
    assert "Run" in csharp_summary.functions
    run_cs = next(item for item in csharp_symbols if item.name == "Run")
    assert run_cs.owner_qualified_name == "Program"


def test_symbol_fidelity_inventory_matrix() -> None:
    # Known conservative gaps are documented here explicitly:
    # - C++: owner resolution for namespace/class methods is best-effort string parsing.
    # - Go: method receiver owner names are parsed conservatively from receiver text.
    # - Rust: impl/type ownership is best-effort and may miss complex trait impl forms.
    # - C#: generic/partial nested type ownership is conservative.
    fixtures: list[tuple[Path, str, set[str]]] = [
        (
            Path("a.py"),
            "class A:\n    def m(self):\n        pass\n",
            {"class", "method"},
        ),
        (
            Path("A.java"),
            "public class A { int m() { return 1; } }\n",
            {"class", "method"},
        ),
        (Path("a.cpp"), "class A {};\nint f() { return 1; }\n", {"class", "function"}),
        (
            Path("a.js"),
            "class A { m() { return 1; } }\nfunction f() { return 1; }\n",
            {"class", "method", "function"},
        ),
        (
            Path("a.ts"),
            "interface A { m(): number }\nfunction f(): number { return 1; }\n",
            {"interface", "function"},
        ),
        (Path("a.go"), "type A struct{}\nfunc (a *A) M() {}\n", {"class", "method"}),
        (Path("a.rs"), "struct A;\nimpl A { fn m() {} }\n", {"struct", "method"}),
        (Path("A.cs"), "class A { void M() {} }\n", {"class", "method"}),
    ]

    for path, source_text, expected_kinds in fixtures:
        analyzer = get_analyzer_for_path(path)
        summary, symbols = analyzer.analyze(path, source_text)
        assert summary.language == detect_language(path)
        symbol_kinds = {item.kind for item in symbols}
        assert expected_kinds.issubset(symbol_kinds)
        assert all(
            item.line_start is not None for item in symbols if item.kind != "module"
        )


def test_unknown_analyzer_behavior() -> None:
    analyzer = get_analyzer_for_path(Path("README.md"))
    assert analyzer.language == "unknown"
    assert analyzer.is_comment_line("# comment")
    assert analyzer.signature_changes(["fn run()", "xyz"]) == ["fn run()"]
    assert analyzer.dependency_changes(["use std::fmt;", "abc"]) == ["use std::fmt;"]
    assert get_analyzer_for_language("unknown").language == "unknown"
    summary, symbols = analyzer.analyze(
        Path("README.md"),
        "import os\ninclude common.mk\njust text\n",
    )
    assert summary.language == "unknown"
    assert summary.functions == []
    assert summary.classes == []
    assert summary.imports == ["import os", "include common.mk"]
    assert symbols == []
