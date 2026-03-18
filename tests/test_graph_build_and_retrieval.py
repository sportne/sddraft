"""Tests for engineering graph artifacts and graph-aware ask retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from sddraft.analysis.graph_retrieval import (
    AnchorSet,
    GraphChunkCandidate,
    rerank_evidence,
)
from sddraft.analysis.retrieval import ScoredChunk
from sddraft.analysis.symbol_inventory import build_symbol_inventory
from sddraft.domain.models import KnowledgeChunk, QueryRequest
from sddraft.llm.mock import MockLLMClient
from sddraft.repo.scanner import scan_repository
from sddraft.workflows.ask import answer_question
from sddraft.workflows.generate import generate_sdd


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _imports_with_reasons(graph_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    edges = _read_jsonl(graph_root / "edges.jsonl")
    values: dict[tuple[str, str], dict[str, object]] = {}
    for edge in edges:
        if edge.get("edge_type") != "imports":
            continue
        reason_text = str(edge.get("reason") or "{}")
        reason = json.loads(reason_text)
        values[(str(edge["source_id"]), str(edge["target_id"]))] = reason
    return values


def test_generate_writes_deterministic_graph_artifacts(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "helper.py").write_text(
        "def helper() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (src_dir / "module.py").write_text(
        (
            "from helper import helper\n\n"
            "class Planner:\n"
            "    def run(self) -> int:\n"
            "        return helper()\n\n"
            "def compute_distance(x: int, y: int) -> int:\n"
            "    return x + y\n"
        ),
        encoding="utf-8",
    )

    llm = MockLLMClient()
    first = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )

    assert first.graph_manifest_path is not None
    assert first.graph_store_path is not None
    graph_root = first.graph_store_path
    manifest = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v1-engineering-graph-jsonl"

    nodes_path = graph_root / "nodes.jsonl"
    edges_path = graph_root / "edges.jsonl"
    assert nodes_path.exists()
    assert edges_path.exists()
    assert (graph_root / "symbol_index.json").exists()
    assert (graph_root / "adjacency.json").exists()

    nodes = _read_jsonl(nodes_path)
    edges = _read_jsonl(edges_path)
    node_types = {str(row["node_type"]) for row in nodes}
    edge_types = {str(row["edge_type"]) for row in edges}

    assert {"directory", "file", "symbol", "chunk", "sdd_section"}.issubset(node_types)
    assert {"contains", "defines", "references", "documents", "imports"}.issubset(
        edge_types
    )

    node_snapshot = nodes_path.read_text(encoding="utf-8")
    edge_snapshot = edges_path.read_text(encoding="utf-8")

    second = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )
    assert second.graph_store_path is not None
    assert (second.graph_store_path / "nodes.jsonl").read_text(
        encoding="utf-8"
    ) == node_snapshot
    assert (second.graph_store_path / "edges.jsonl").read_text(
        encoding="utf-8"
    ) == edge_snapshot


def test_generate_writes_multilanguage_import_edges_with_metadata(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "helper.py").write_text(
        "def helper() -> int:\n    return 1\n", encoding="utf-8"
    )
    (src_dir / "module.py").write_text(
        "from helper import helper\n\ndef run() -> int:\n    return helper()\n",
        encoding="utf-8",
    )

    (src_dir / "java").mkdir()
    (src_dir / "java" / "Util.java").write_text(
        "package demo;\npublic class Util {}\n",
        encoding="utf-8",
    )
    (src_dir / "java" / "Main.java").write_text(
        "package demo;\nimport demo.Util;\npublic class Main { Util util; }\n",
        encoding="utf-8",
    )

    (src_dir / "js").mkdir()
    (src_dir / "js" / "lib.js").write_text(
        "export function dep() { return 1; }\n",
        encoding="utf-8",
    )
    (src_dir / "js" / "caller.js").write_text(
        "import { dep } from './lib.js';\nexport function call() { return dep(); }\n",
        encoding="utf-8",
    )

    (src_dir / "ts").mkdir()
    (src_dir / "ts" / "dep.ts").write_text(
        "export const dep = () => 1;\n",
        encoding="utf-8",
    )
    (src_dir / "ts" / "caller.ts").write_text(
        "import { dep } from './dep';\nexport const call = () => dep();\n",
        encoding="utf-8",
    )

    (src_dir / "go").mkdir()
    (src_dir / "go" / "gohelper.go").write_text(
        "package gomain\nfunc Helper() int { return 1 }\n",
        encoding="utf-8",
    )
    (src_dir / "go" / "main.go").write_text(
        'package gomain\nimport "./gohelper"\nfunc Run() int { return gohelper.Helper() }\n',
        encoding="utf-8",
    )

    (src_dir / "rust").mkdir()
    (src_dir / "rust" / "util.rs").write_text(
        "pub fn run() -> i32 { 1 }\n", encoding="utf-8"
    )
    (src_dir / "rust" / "lib.rs").write_text(
        "mod util;\nuse crate::util::run;\npub fn call() -> i32 { run() }\n",
        encoding="utf-8",
    )

    (src_dir / "cs").mkdir()
    (src_dir / "cs" / "Service.cs").write_text(
        "namespace Demo.Core { public class Service {} }\n",
        encoding="utf-8",
    )
    (src_dir / "cs" / "Program.cs").write_text(
        "using Demo.Core;\nnamespace Demo.App { public class Program {} }\n",
        encoding="utf-8",
    )

    (src_dir / "cpp").mkdir()
    (src_dir / "cpp" / "core.hpp").write_text(
        "#pragma once\nint core();\n", encoding="utf-8"
    )
    (src_dir / "cpp" / "main.cpp").write_text(
        '#include "core.hpp"\nint run() { return core(); }\n',
        encoding="utf-8",
    )

    # Unresolved external dependencies should not become repo-local edges.
    (src_dir / "external.java").write_text(
        "import java.util.List;\npublic class External {}\n",
        encoding="utf-8",
    )
    (src_dir / "external.cpp").write_text(
        "#include <vector>\nint x() { return 1; }\n",
        encoding="utf-8",
    )

    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=MockLLMClient(),
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )
    assert result.graph_store_path is not None
    imports = _imports_with_reasons(result.graph_store_path)

    expected_pairs = {
        ("file::src/module.py", "file::src/helper.py"),
        ("file::src/java/Main.java", "file::src/java/Util.java"),
        ("file::src/js/caller.js", "file::src/js/lib.js"),
        ("file::src/ts/caller.ts", "file::src/ts/dep.ts"),
        ("file::src/go/main.go", "file::src/go/gohelper.go"),
        ("file::src/rust/lib.rs", "file::src/rust/util.rs"),
        ("file::src/cs/Program.cs", "file::src/cs/Service.cs"),
        ("file::src/cpp/main.cpp", "file::src/cpp/core.hpp"),
    }
    assert expected_pairs.issubset(set(imports))

    # Reason payload is inspectable structured JSON.
    js_reason = imports[("file::src/js/caller.js", "file::src/js/lib.js")]
    assert js_reason["language"] == "javascript"
    assert js_reason["kind"] == "import"
    assert js_reason["resolution"] in {"resolved_exact", "resolved_heuristic"}
    assert js_reason["normalized"] == "./lib.js"

    cpp_reason = imports[("file::src/cpp/main.cpp", "file::src/cpp/core.hpp")]
    assert cpp_reason["language"] == "cpp"
    assert cpp_reason["kind"] == "include"
    assert cpp_reason["normalized"] == "core.hpp"


def test_symbol_inventory_extracts_python_symbols_with_spans(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        (
            "class Planner:\n"
            "    def run(self) -> int:\n"
            "        return 1\n\n"
            "def compute_distance(x: int, y: int) -> int:\n"
            "    return x + y\n"
        ),
        encoding="utf-8",
    )

    scan_result = scan_repository(
        project_config=sample_project_config,
        csc=sample_csc,
        repo_root=tmp_path,
    )
    symbols = build_symbol_inventory(scan_result=scan_result)

    names = {item.name for item in symbols}
    assert "Planner" in names
    assert "compute_distance" in names
    assert any(
        item.line_start is not None for item in symbols if item.name == "Planner"
    )
    assert any(
        item.line_start is not None
        for item in symbols
        if item.name == "compute_distance"
    )


def test_symbol_ids_stable_when_only_spans_change(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    module_path = src_dir / "module.py"
    module_path.write_text(
        (
            "class Planner:\n"
            "    def run(self) -> int:\n"
            "        return 1\n\n"
            "def compute_distance(x: int, y: int) -> int:\n"
            "    return x + y\n"
        ),
        encoding="utf-8",
    )

    first_scan = scan_repository(
        project_config=sample_project_config,
        csc=sample_csc,
        repo_root=tmp_path,
    )
    first_ids = {
        item.symbol_id for item in build_symbol_inventory(scan_result=first_scan)
    }

    module_path.write_text(
        (
            "\n\n"
            "class Planner:\n"
            "    def run(self) -> int:\n"
            "        return 1\n\n"
            "def compute_distance(x: int, y: int) -> int:\n"
            "    return x + y\n"
        ),
        encoding="utf-8",
    )

    second_scan = scan_repository(
        project_config=sample_project_config,
        csc=sample_csc,
        repo_root=tmp_path,
    )
    second_ids = {
        item.symbol_id for item in build_symbol_inventory(scan_result=second_scan)
    }
    assert first_ids == second_ids


def test_graph_build_uses_symbol_owners_for_parent_edges(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "class Planner:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    (src_dir / "Main.java").write_text(
        "public class Main { int run() { return 1; } }\n",
        encoding="utf-8",
    )

    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=MockLLMClient(),
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )
    assert result.graph_store_path is not None
    edges = _read_jsonl(result.graph_store_path / "edges.jsonl")
    parent_pairs = {
        (str(item["source_id"]), str(item["target_id"]))
        for item in edges
        if item.get("edge_type") == "parent_of"
    }

    assert (
        "sym::src/module.py::class::Planner",
        "sym::src/module.py::method::Planner.run",
    ) in parent_pairs
    assert (
        "sym::src/Main.java::class::Main",
        "sym::src/Main.java::method::Main.run",
    ) in parent_pairs


def test_ask_graph_fallback_and_graph_augmented_selection(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: int, y: int) -> int:\n    return x + y\n",
        encoding="utf-8",
    )
    (src_dir / "caller.py").write_text(
        (
            "from module import compute_distance\n\n"
            "def caller() -> int:\n"
            "    return compute_distance(1, 2)\n"
        ),
        encoding="utf-8",
    )

    llm = MockLLMClient()

    no_graph_result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=False,
    )
    fallback_answer = answer_question(
        request=QueryRequest(question="What does caller depend on?", top_k=1),
        index_path=no_graph_result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
        graph_enabled=True,
        graph_depth=1,
        graph_top_k=4,
    )
    assert any(
        "Engineering graph store unavailable" in msg
        for msg in fallback_answer.answer.uncertainty
    )

    graph_result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )

    ask_result = answer_question(
        request=QueryRequest(question="What does caller depend on?", top_k=1),
        index_path=graph_result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
        graph_enabled=True,
        graph_depth=1,
        graph_top_k=4,
    )

    assert any(
        chunk.source_path == Path("src/module.py")
        for chunk in ask_result.evidence_pack.chunks
    )
    assert ask_result.evidence_pack.inclusion_reasons
    assert any(
        reason.source == "graph"
        for reason in ask_result.evidence_pack.inclusion_reasons
    )
    assert Path("src/module.py") in ask_result.evidence_pack.related_files


def test_rerank_is_deterministic_with_tie_breaks() -> None:
    chunk_a = generate_chunk("a", Path("src/a.py"), "def alpha():\n    return 1")
    chunk_b = generate_chunk("b", Path("src/b.py"), "def beta():\n    return 2")

    result = rerank_evidence(
        lexical_candidates=[
            ScoredChunk(chunk=chunk_a, score=1.0),
            ScoredChunk(chunk=chunk_b, score=1.0),
        ],
        graph_candidates=[
            GraphChunkCandidate(chunk=chunk_b, graph_score=0.5, reason="graph:file:b")
        ],
        anchors=AnchorSet(
            node_ids=set(),
            file_paths=set(),
            symbol_names=set(),
            section_ids=set(),
        ),
        intent="implementation",
        top_k=2,
    )

    assert [item.chunk_id for item in result.chunks] == ["b", "a"]
    assert result.reasons[0].chunk_id == "b"


def generate_chunk(chunk_id: str, source_path: Path, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_type="code",
        source_path=source_path,
        text=text,
        line_start=1,
        line_end=2,
    )
