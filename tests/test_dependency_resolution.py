"""Unit tests for language-aware dependency normalization and resolution."""

from __future__ import annotations

import json
from pathlib import Path

from sddraft.analysis.dependency_resolution import (
    dependency_reason_payload,
    resolve_dependency_records,
)
from sddraft.domain.models import CodeUnitSummary, SymbolSummary


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolve_dependency_records_multilanguage_repo_local_edges(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "helper.py", "def helper():\n    return 1\n")
    _write(tmp_path / "src" / "module.py", "from helper import helper\n")

    _write(tmp_path / "src" / "java" / "Util.java", "package demo;\nclass Util {}\n")
    _write(tmp_path / "src" / "java" / "Main.java", "package demo;\n")

    _write(tmp_path / "src" / "js" / "lib.js", "export function dep() {}\n")
    _write(tmp_path / "src" / "js" / "caller.js", "import { dep } from './lib.js';\n")

    _write(tmp_path / "src" / "ts" / "dep.ts", "export const dep = 1;\n")
    _write(tmp_path / "src" / "ts" / "caller.ts", "import { dep } from './dep';\n")

    _write(tmp_path / "src" / "go" / "gohelper.go", "package gomain\n")
    _write(tmp_path / "src" / "go" / "main.go", 'package gomain\nimport "./gohelper"\n')

    _write(tmp_path / "src" / "rust" / "util.rs", "pub fn run() {}\n")
    _write(tmp_path / "src" / "rust" / "lib.rs", "use crate::util::run;\n")

    _write(
        tmp_path / "src" / "cs" / "Service.cs",
        "namespace Demo.Core { public class Service {} }\n",
    )
    _write(
        tmp_path / "src" / "cs" / "Program.cs",
        "using Demo.Core;\nnamespace Demo.App { public class Program {} }\n",
    )

    _write(tmp_path / "src" / "cpp" / "core.hpp", "#pragma once\n")
    _write(tmp_path / "src" / "cpp" / "main.cpp", '#include "core.hpp"\n')

    files = [
        Path("src/helper.py"),
        Path("src/module.py"),
        Path("src/java/Util.java"),
        Path("src/java/Main.java"),
        Path("src/js/lib.js"),
        Path("src/js/caller.js"),
        Path("src/ts/dep.ts"),
        Path("src/ts/caller.ts"),
        Path("src/go/gohelper.go"),
        Path("src/go/main.go"),
        Path("src/rust/util.rs"),
        Path("src/rust/lib.rs"),
        Path("src/cs/Service.cs"),
        Path("src/cs/Program.cs"),
        Path("src/cpp/core.hpp"),
        Path("src/cpp/main.cpp"),
    ]

    code_summaries = [
        CodeUnitSummary(
            path=Path("src/module.py"),
            language="python",
            imports=["from helper import helper"],
        ),
        CodeUnitSummary(
            path=Path("src/helper.py"),
            language="python",
        ),
        CodeUnitSummary(
            path=Path("src/java/Util.java"),
            language="java",
            imports=["package demo;"],
        ),
        CodeUnitSummary(
            path=Path("src/java/Main.java"),
            language="java",
            imports=["package demo;", "import demo.Util;"],
        ),
        CodeUnitSummary(
            path=Path("src/js/caller.js"),
            language="javascript",
            imports=["import { dep } from './lib.js';"],
        ),
        CodeUnitSummary(path=Path("src/js/lib.js"), language="javascript"),
        CodeUnitSummary(
            path=Path("src/ts/caller.ts"),
            language="typescript",
            imports=["import { dep } from './dep';"],
        ),
        CodeUnitSummary(path=Path("src/ts/dep.ts"), language="typescript"),
        CodeUnitSummary(
            path=Path("src/go/main.go"),
            language="go",
            imports=['"./gohelper"'],
        ),
        CodeUnitSummary(path=Path("src/go/gohelper.go"), language="go"),
        CodeUnitSummary(
            path=Path("src/rust/lib.rs"),
            language="rust",
            imports=["use crate::util::run;"],
        ),
        CodeUnitSummary(path=Path("src/rust/util.rs"), language="rust"),
        CodeUnitSummary(
            path=Path("src/cs/Program.cs"),
            language="csharp",
            imports=["using Demo.Core;"],
        ),
        CodeUnitSummary(path=Path("src/cs/Service.cs"), language="csharp"),
        CodeUnitSummary(
            path=Path("src/cpp/main.cpp"),
            language="cpp",
            imports=['#include "core.hpp"'],
        ),
        CodeUnitSummary(path=Path("src/cpp/core.hpp"), language="cpp"),
    ]
    symbol_summaries = [
        SymbolSummary(
            name="Util",
            qualified_name="Util",
            kind="class",
            language="java",
            source_path=Path("src/java/Util.java"),
        )
    ]

    records = resolve_dependency_records(
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        files=files,
        repo_root=tmp_path,
    )
    resolved = {
        (item.source_path.as_posix(), item.target_path.as_posix()): item
        for item in records
        if item.target_path is not None
    }

    expected = {
        ("src/module.py", "src/helper.py"),
        ("src/java/Main.java", "src/java/Util.java"),
        ("src/js/caller.js", "src/js/lib.js"),
        ("src/ts/caller.ts", "src/ts/dep.ts"),
        ("src/go/main.go", "src/go/gohelper.go"),
        ("src/rust/lib.rs", "src/rust/util.rs"),
        ("src/cs/Program.cs", "src/cs/Service.cs"),
        ("src/cpp/main.cpp", "src/cpp/core.hpp"),
    }
    assert expected.issubset(set(resolved))

    # Rust symbol import path still resolves to module file conservatively.
    rust_record = resolved[("src/rust/lib.rs", "src/rust/util.rs")]
    assert rust_record.dependency_kind == "use"
    assert rust_record.resolution_status == "resolved_exact"

    # Edge reason payload stays machine-inspectable and deterministic.
    js_reason = json.loads(
        dependency_reason_payload(resolved[("src/js/caller.js", "src/js/lib.js")])
    )
    assert js_reason["language"] == "javascript"
    assert js_reason["kind"] == "import"
    assert js_reason["normalized"] == "./lib.js"
    assert js_reason["target"] == "src/js/lib.js"


def test_resolve_dependency_records_keeps_ambiguous_or_external_unresolved(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "a" / "common.hpp", "#pragma once\n")
    _write(tmp_path / "src" / "b" / "common.hpp", "#pragma once\n")
    _write(tmp_path / "src" / "cpp" / "main.cpp", '#include "common.hpp"\n')

    code_summaries = [
        CodeUnitSummary(
            path=Path("src/cpp/main.cpp"),
            language="cpp",
            imports=['#include "common.hpp"', "#include <vector>"],
        ),
        CodeUnitSummary(
            path=Path("src/external.java"),
            language="java",
            imports=["import java.util.List;"],
        ),
    ]
    files = [
        Path("src/a/common.hpp"),
        Path("src/b/common.hpp"),
        Path("src/cpp/main.cpp"),
    ]

    records = resolve_dependency_records(
        code_summaries=code_summaries,
        symbol_summaries=[],
        files=files,
        repo_root=tmp_path,
    )

    unresolved = [
        item
        for item in records
        if item.target_path is None and item.resolution_status == "unresolved"
    ]
    assert any(
        item.language == "cpp" and item.normalized_key == "common.hpp"
        for item in unresolved
    )
    assert any(
        item.language == "cpp" and item.normalized_key == "vector"
        for item in unresolved
    )
    assert any(
        item.language == "java" and item.normalized_key == "java.util.List"
        for item in unresolved
    )
