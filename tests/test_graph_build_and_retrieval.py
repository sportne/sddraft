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
    symbols = build_symbol_inventory(scan_result=scan_result, repo_root=tmp_path)

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
