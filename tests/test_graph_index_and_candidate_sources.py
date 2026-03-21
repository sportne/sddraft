"""Unit tests for graph index loading and graph candidate orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sddraft.analysis.graph_index import default_graph_manifest_path, load_graph_store
from sddraft.analysis.graph_models import (
    GraphEdgeRecord,
    GraphManifest,
    GraphNodeRecord,
    GraphStore,
    GraphSymbolRecord,
    chunk_node_id,
    commit_node_id,
    edge_record_id,
    file_node_id,
    section_node_id,
    symbol_node_id,
)
from sddraft.analysis.graph_retrieval import (
    AnchorSet,
    GraphExpansionCandidateSource,
    HierarchyCandidateSource,
    LexicalCandidateSource,
    SourceContext,
    VectorCandidateSource,
    collect_graph_candidates,
    collect_text_candidates,
    expand_graph_neighbors,
    flatten_text_candidates,
    infer_query_intent,
    preferred_edge_types,
    rerank_evidence,
)
from sddraft.analysis.retrieval import ScoredChunk
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import KnowledgeChunk
from sddraft.render.json_artifacts import write_json_model


def _chunk(
    *,
    chunk_id: str,
    source_path: Path,
    text: str,
    source_type: str = "code",
    section_id: str | None = None,
    line_start: int | None = 1,
    line_end: int | None = 1,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_type=source_type,
        source_path=source_path,
        text=text,
        section_id=section_id,
        line_start=line_start,
        line_end=line_end,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _seed_graph_files(base: Path) -> Path:
    graph_root = base / "graph"
    graph_root.mkdir(parents=True, exist_ok=True)
    (base / "retrieval").mkdir(parents=True, exist_ok=True)

    nodes = [
        GraphNodeRecord(
            node_id=file_node_id(Path("src/main.py")),
            node_type="file",
            label="main.py",
            path=Path("src/main.py"),
            language="python",
        ),
        GraphNodeRecord(
            node_id=symbol_node_id(Path("src/main.py"), "function", "compute"),
            node_type="symbol",
            label="compute",
            path=Path("src/main.py"),
            language="python",
            symbol_kind="function",
            symbol_name="compute",
            qualified_name="compute",
            line_start=2,
            line_end=2,
        ),
        GraphNodeRecord(
            node_id=chunk_node_id("seed"),
            node_type="chunk",
            label="seed",
            path=Path("src/main.py"),
            chunk_id="seed",
            line_start=1,
            line_end=3,
        ),
    ]
    edges = [
        GraphEdgeRecord(
            edge_id=edge_record_id(
                "contains", file_node_id(Path("src/main.py")), chunk_node_id("seed")
            ),
            edge_type="contains",
            source_id=file_node_id(Path("src/main.py")),
            target_id=chunk_node_id("seed"),
        ),
        GraphEdgeRecord(
            edge_id=edge_record_id(
                "defines",
                file_node_id(Path("src/main.py")),
                symbol_node_id(Path("src/main.py"), "function", "compute"),
            ),
            edge_type="defines",
            source_id=file_node_id(Path("src/main.py")),
            target_id=symbol_node_id(Path("src/main.py"), "function", "compute"),
        ),
    ]

    _write_jsonl(graph_root / "nodes.jsonl", [n.model_dump(mode="json") for n in nodes])
    _write_jsonl(graph_root / "edges.jsonl", [e.model_dump(mode="json") for e in edges])
    (graph_root / "adjacency.json").write_text("{invalid json", encoding="utf-8")
    (graph_root / "symbol_index.json").write_text("{invalid json", encoding="utf-8")

    manifest = GraphManifest(
        csc_id="NAV_CTRL",
        source_retrieval_manifest=Path("retrieval/manifest.json"),
        nodes_path=Path("nodes.jsonl"),
        edges_path=Path("edges.jsonl"),
        symbol_index_path=Path("symbol_index.json"),
        adjacency_path=Path("adjacency.json"),
        node_counts={
            "directory": 0,
            "file": 1,
            "symbol": 1,
            "chunk": 1,
            "sdd_section": 0,
            "commit": 0,
        },
        edge_counts={
            "contains": 1,
            "defines": 1,
            "references": 0,
            "documents": 0,
            "parent_of": 0,
            "imports": 0,
            "changed_in": 0,
            "impacts_section": 0,
        },
    )
    write_json_model(graph_root / "manifest.json", manifest)
    return graph_root


class _FakeEngine:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = chunks

    def search_scored(self, query: str, top_k: int) -> list[ScoredChunk]:
        _ = query
        ranked = [
            ScoredChunk(chunk=item, score=float(idx + 1))
            for idx, item in enumerate(self._chunks)
        ]
        return ranked[:top_k]

    def load_chunks_by_chunk_ids(
        self, chunk_ids: set[str], *, limit: int
    ) -> list[KnowledgeChunk]:
        return [item for item in self._chunks if item.chunk_id in chunk_ids][:limit]

    def load_chunks_by_source_paths(
        self, paths: set[Path], *, limit: int
    ) -> list[KnowledgeChunk]:
        return [item for item in self._chunks if item.source_path in paths][:limit]

    def load_chunks_by_section_ids(
        self, section_ids: set[str], *, limit: int
    ) -> list[KnowledgeChunk]:
        return [
            item
            for item in self._chunks
            if item.section_id is not None and item.section_id in section_ids
        ][:limit]


def _build_store_for_retrieval() -> (
    tuple[GraphStore, list[KnowledgeChunk], _FakeEngine]
):
    seed = _chunk(
        chunk_id="seed",
        source_path=Path("src/main.py"),
        text="compute calls helper",
        line_start=1,
        line_end=3,
    )
    extra = _chunk(
        chunk_id="extra",
        source_path=Path("src/main.py"),
        text="compute implementation details",
        line_start=2,
        line_end=4,
    )
    helper_chunk = _chunk(
        chunk_id="helper_chunk",
        source_path=Path("src/helper.py"),
        text="helper handles requests",
        line_start=1,
        line_end=3,
    )
    section_chunk = _chunk(
        chunk_id="section_chunk",
        source_path=Path("docs/sdd.md"),
        text="Section 2 documents helper module.",
        source_type="sdd_section",
        section_id="2",
        line_start=1,
        line_end=2,
    )
    chunks = [seed, extra, helper_chunk, section_chunk]

    main_file = file_node_id(Path("src/main.py"))
    helper_file = file_node_id(Path("src/helper.py"))
    section_2 = section_node_id("2")
    commit_id = commit_node_id("HEAD~1..HEAD")
    seed_node = chunk_node_id("seed")
    extra_node = chunk_node_id("extra")
    compute_sym = symbol_node_id(Path("src/main.py"), "function", "compute")
    helper_sym = symbol_node_id(Path("src/helper.py"), "function", "helper")

    nodes = {
        main_file: GraphNodeRecord(
            node_id=main_file,
            node_type="file",
            label="main.py",
            path=Path("src/main.py"),
            language="python",
        ),
        helper_file: GraphNodeRecord(
            node_id=helper_file,
            node_type="file",
            label="helper.py",
            path=Path("src/helper.py"),
            language="python",
        ),
        section_2: GraphNodeRecord(
            node_id=section_2,
            node_type="sdd_section",
            label="2 Design Overview",
            section_id="2",
        ),
        commit_id: GraphNodeRecord(
            node_id=commit_id,
            node_type="commit",
            label="HEAD~1..HEAD",
        ),
        seed_node: GraphNodeRecord(
            node_id=seed_node,
            node_type="chunk",
            label="seed",
            path=Path("src/main.py"),
            chunk_id="seed",
            line_start=1,
            line_end=3,
        ),
        extra_node: GraphNodeRecord(
            node_id=extra_node,
            node_type="chunk",
            label="extra",
            path=Path("src/main.py"),
            chunk_id="extra",
            line_start=2,
            line_end=4,
        ),
        compute_sym: GraphNodeRecord(
            node_id=compute_sym,
            node_type="symbol",
            label="compute",
            path=Path("src/main.py"),
            language="python",
            symbol_kind="function",
            symbol_name="compute",
            line_start=2,
            line_end=2,
        ),
        helper_sym: GraphNodeRecord(
            node_id=helper_sym,
            node_type="symbol",
            label="helper",
            path=Path("src/helper.py"),
            language="python",
            symbol_kind="function",
            symbol_name="helper",
        ),
    }

    edges = {
        edge_record_id("contains", main_file, seed_node): GraphEdgeRecord(
            edge_id=edge_record_id("contains", main_file, seed_node),
            edge_type="contains",
            source_id=main_file,
            target_id=seed_node,
        ),
        edge_record_id("contains", main_file, extra_node): GraphEdgeRecord(
            edge_id=edge_record_id("contains", main_file, extra_node),
            edge_type="contains",
            source_id=main_file,
            target_id=extra_node,
        ),
        edge_record_id(
            "contains", helper_file, chunk_node_id("helper_chunk")
        ): GraphEdgeRecord(
            edge_id=edge_record_id(
                "contains", helper_file, chunk_node_id("helper_chunk")
            ),
            edge_type="contains",
            source_id=helper_file,
            target_id=chunk_node_id("helper_chunk"),
        ),
        edge_record_id("references", seed_node, compute_sym): GraphEdgeRecord(
            edge_id=edge_record_id("references", seed_node, compute_sym),
            edge_type="references",
            source_id=seed_node,
            target_id=compute_sym,
        ),
        edge_record_id("references", seed_node, helper_sym): GraphEdgeRecord(
            edge_id=edge_record_id("references", seed_node, helper_sym),
            edge_type="references",
            source_id=seed_node,
            target_id=helper_sym,
        ),
        edge_record_id("documents", section_2, main_file): GraphEdgeRecord(
            edge_id=edge_record_id("documents", section_2, main_file),
            edge_type="documents",
            source_id=section_2,
            target_id=main_file,
        ),
        edge_record_id("documents", section_2, helper_file): GraphEdgeRecord(
            edge_id=edge_record_id("documents", section_2, helper_file),
            edge_type="documents",
            source_id=section_2,
            target_id=helper_file,
        ),
        edge_record_id("changed_in", main_file, commit_id): GraphEdgeRecord(
            edge_id=edge_record_id("changed_in", main_file, commit_id),
            edge_type="changed_in",
            source_id=main_file,
            target_id=commit_id,
        ),
        edge_record_id("changed_in", helper_sym, commit_id): GraphEdgeRecord(
            edge_id=edge_record_id("changed_in", helper_sym, commit_id),
            edge_type="changed_in",
            source_id=helper_sym,
            target_id=commit_id,
        ),
        edge_record_id("impacts_section", main_file, section_2): GraphEdgeRecord(
            edge_id=edge_record_id("impacts_section", main_file, section_2),
            edge_type="impacts_section",
            source_id=main_file,
            target_id=section_2,
        ),
    }

    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for edge in edges.values():
        outgoing.setdefault(edge.source_id, []).append(edge.edge_id)
        incoming.setdefault(edge.target_id, []).append(edge.edge_id)

    store = GraphStore(
        manifest_path=Path("graph/manifest.json"),
        manifest=GraphManifest(
            csc_id="NAV_CTRL",
            source_retrieval_manifest=Path("retrieval/manifest.json"),
            nodes_path=Path("nodes.jsonl"),
            edges_path=Path("edges.jsonl"),
            symbol_index_path=Path("symbol_index.json"),
            adjacency_path=Path("adjacency.json"),
        ),
        nodes_by_id=nodes,
        edges_by_id=edges,
        outgoing={key: sorted(value) for key, value in outgoing.items()},
        incoming={key: sorted(value) for key, value in incoming.items()},
        symbols_by_name={"compute": [compute_sym], "helper": [helper_sym]},
        symbol_records={
            compute_sym: GraphSymbolRecord(
                symbol_id=compute_sym,
                name="compute",
                qualified_name="compute",
                kind="function",
                language="python",
                file_path=Path("src/main.py"),
                line_start=2,
                line_end=2,
            ),
            helper_sym: GraphSymbolRecord(
                symbol_id=helper_sym,
                name="helper",
                qualified_name="helper",
                kind="function",
                language="python",
                file_path=Path("src/helper.py"),
            ),
        },
    )
    return store, chunks, _FakeEngine(chunks)


def test_default_graph_manifest_path_and_load_fallbacks(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "NAV_CTRL"
    graph_root = _seed_graph_files(root)

    retrieval_dir = root / "retrieval"
    graph_dir = root / "graph"
    retrieval_manifest = retrieval_dir / "manifest.json"
    graph_manifest = graph_dir / "manifest.json"
    retrieval_manifest.write_text("{}", encoding="utf-8")

    assert default_graph_manifest_path(retrieval_dir) == graph_manifest
    assert default_graph_manifest_path(graph_dir) == graph_manifest
    assert default_graph_manifest_path(retrieval_manifest) == graph_manifest
    assert default_graph_manifest_path(graph_manifest) == graph_manifest
    assert default_graph_manifest_path(root) == graph_manifest

    store = load_graph_store(retrieval_dir)
    assert store.manifest_path == graph_manifest
    assert file_node_id(Path("src/main.py")) in store.nodes_by_id
    assert store.outgoing[file_node_id(Path("src/main.py"))]
    assert "compute" in store.symbols_by_name
    assert any(item.name == "compute" for item in store.symbol_records.values())

    assert graph_root.exists()


def test_load_graph_store_error_paths(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="Graph manifest not found"):
        load_graph_store(tmp_path / "missing")

    root = tmp_path / "artifacts" / "NAV_CTRL"
    graph_root = _seed_graph_files(root)

    (graph_root / "nodes.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(AnalysisError, match="Invalid JSONL row"):
        load_graph_store(graph_root)

    _write_jsonl(
        graph_root / "nodes.jsonl",
        [
            GraphNodeRecord(
                node_id=file_node_id(Path("src/main.py")),
                node_type="file",
                label="main.py",
                path=Path("src/main.py"),
                language="python",
            ).model_dump(mode="json")
        ],
    )
    (graph_root / "edges.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(AnalysisError, match="Expected object row"):
        load_graph_store(graph_root)


def test_graph_candidate_collection_and_rerank() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    seed = chunks[0]

    lexical_source = LexicalCandidateSource(engine)  # type: ignore[arg-type]
    lexical = lexical_source.collect(query="where is helper documented?", top_k=2)
    assert [item.chunk.chunk_id for item in lexical] == ["seed", "extra"]

    assert VectorCandidateSource().collect(query="x", top_k=3) == []

    graph_source = GraphExpansionCandidateSource(
        engine=engine,  # type: ignore[arg-type]
        store=store,
        depth=2,
        top_k=3,
    )
    graph_candidates, anchors, intent = graph_source.collect(
        query="Which section documents helper implementation?",
        seed_chunks=[seed],
    )
    assert intent == "documentation"
    assert Path("src/main.py") in anchors.file_paths
    assert "helper" in anchors.symbol_names
    assert any(item.chunk.chunk_id == "section_chunk" for item in graph_candidates)
    assert any(item.chunk.chunk_id == "helper_chunk" for item in graph_candidates)

    reranked = rerank_evidence(
        lexical_candidates=lexical,
        graph_candidates=graph_candidates,
        anchors=anchors,
        intent=intent,
        top_k=3,
    )
    assert reranked.chunks
    assert reranked.reasons
    assert any(item.source == "graph" for item in reranked.reasons)
    assert "2" in reranked.related_sections
    assert "helper [src/helper.py]" in reranked.related_symbols
    assert all("implementation" not in symbol for symbol in reranked.related_symbols)
    assert any(
        reason.graph_paths for reason in reranked.reasons if reason.source == "graph"
    )
    graph_reason = next(
        reason for reason in reranked.reasons if reason.source == "graph"
    )
    assert any(
        path.edge_type in {"references", "documents", "contains"}
        for path in graph_reason.graph_paths
    )


def test_graph_retrieval_edge_cases() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    seed = chunks[0]

    assert infer_query_intent("How do modules depend on imports?") == "dependency"
    assert infer_query_intent("What changed in HEAD~1..HEAD?") == "change_impact"
    assert infer_query_intent("Show architecture overview") == "architecture"
    assert infer_query_intent("Which section is documented?") == "documentation"
    assert infer_query_intent("Where is compute implemented?") == "implementation"

    assert "changed_in" in preferred_edge_types("change_impact")
    assert "impacts_section" in preferred_edge_types("change_impact")
    assert "imports" in preferred_edge_types("dependency")
    assert "documents" in preferred_edge_types("documentation")
    assert "parent_of" in preferred_edge_types("architecture")

    empty_hits = expand_graph_neighbors(
        store=store,
        anchors=AnchorSet(
            node_ids=set(),
            file_paths=set(),
            symbol_ids=set(),
            symbol_labels=set(),
            symbol_names=set(),
            section_ids=set(),
        ),
        depth=1,
        edge_filter=preferred_edge_types("implementation"),
        limit=5,
    )
    assert empty_hits == []

    candidates, anchors, intent = collect_graph_candidates(
        query="Which section documents helper?",
        engine=engine,  # type: ignore[arg-type]
        store=store,
        seed_chunks=[seed],
        depth=2,
        top_k=2,
    )
    assert candidates
    assert intent == "documentation"
    assert anchors.node_ids

    empty_rank = rerank_evidence(
        lexical_candidates=[],
        graph_candidates=[],
        anchors=anchors,
        intent="implementation",
        top_k=0,
    )
    assert empty_rank.chunks == []
    assert empty_rank.reasons == []


def test_collect_graph_candidates_attaches_symbol_context() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    seed = chunks[0]

    candidates, _, intent = collect_graph_candidates(
        query="Which section documents helper implementation?",
        engine=engine,  # type: ignore[arg-type]
        store=store,
        seed_chunks=[seed],
        depth=2,
        top_k=4,
    )

    assert intent == "documentation"
    helper_candidate = next(
        item for item in candidates if item.chunk.chunk_id == "helper_chunk"
    )
    assert "helper [src/helper.py]" in helper_candidate.related_symbols
    assert helper_candidate.graph_paths
    assert any(
        path.edge_type in {"references", "contains", "documents"}
        for path in helper_candidate.graph_paths
    )


def test_collect_graph_candidates_surfaces_commit_context() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    seed = chunks[0]

    candidates, anchors, intent = collect_graph_candidates(
        query="Which sections were impacted by HEAD~1..HEAD?",
        engine=engine,  # type: ignore[arg-type]
        store=store,
        seed_chunks=[seed],
        depth=2,
        top_k=4,
    )

    assert intent == "change_impact"
    assert anchors.commit_ids == {commit_node_id("HEAD~1..HEAD")}
    assert any("HEAD~1..HEAD" in item.related_commits for item in candidates)

    reranked = rerank_evidence(
        lexical_candidates=LexicalCandidateSource(engine).collect(
            query="Which sections were impacted by HEAD~1..HEAD?",
            top_k=2,
        ),
        graph_candidates=candidates,
        anchors=anchors,
        intent=intent,
        top_k=3,
    )

    assert reranked.related_commits == ["HEAD~1..HEAD"]
    assert any(
        reason.graph_paths for reason in reranked.reasons if reason.source == "graph"
    )


def test_change_impact_queries_without_commit_nodes_remain_deterministic() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    filtered_nodes = {
        node_id: node
        for node_id, node in store.nodes_by_id.items()
        if node.node_type != "commit"
    }
    filtered_edges = {
        edge_id: edge
        for edge_id, edge in store.edges_by_id.items()
        if edge.edge_type not in {"changed_in", "impacts_section"}
    }
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for edge in filtered_edges.values():
        outgoing.setdefault(edge.source_id, []).append(edge.edge_id)
        incoming.setdefault(edge.target_id, []).append(edge.edge_id)
    no_commit_store = GraphStore(
        manifest_path=store.manifest_path,
        manifest=store.manifest,
        nodes_by_id=filtered_nodes,
        edges_by_id=filtered_edges,
        outgoing={key: sorted(value) for key, value in outgoing.items()},
        incoming={key: sorted(value) for key, value in incoming.items()},
        symbols_by_name=store.symbols_by_name,
        symbol_records=store.symbol_records,
    )

    candidates, anchors, intent = collect_graph_candidates(
        query="What changed in HEAD~1..HEAD?",
        engine=engine,  # type: ignore[arg-type]
        store=no_commit_store,
        seed_chunks=[chunks[0]],
        depth=2,
        top_k=3,
    )

    assert intent == "change_impact"
    assert anchors.commit_ids == set()
    assert all(not item.related_commits for item in candidates)


def test_rerank_biases_documentation_and_architecture_sources() -> None:
    code_chunk = _chunk(
        chunk_id="code",
        source_path=Path("src/main.py"),
        text="compute helper",
        source_type="code",
    )
    sdd_chunk = _chunk(
        chunk_id="section",
        source_path=Path("docs/sdd.md"),
        text="section text",
        source_type="sdd_section",
        section_id="2",
    )
    review_chunk = _chunk(
        chunk_id="review",
        source_path=Path("artifacts/review.json"),
        text="review text",
        source_type="review_artifact",
    )
    directory_chunk = _chunk(
        chunk_id="dir",
        source_path=Path("hierarchy/src/_directory.md"),
        text="directory summary",
        source_type="directory_summary",
    )

    anchors = AnchorSet(
        node_ids=set(),
        file_paths=set(),
        symbol_ids=set(),
        symbol_labels=set(),
        symbol_names=set(),
        section_ids=set(),
    )
    lexical = [
        ScoredChunk(chunk=code_chunk, score=1.0),
        ScoredChunk(chunk=sdd_chunk, score=1.0),
        ScoredChunk(chunk=review_chunk, score=1.0),
        ScoredChunk(chunk=directory_chunk, score=1.0),
    ]

    documentation_rank = rerank_evidence(
        lexical_candidates=lexical,
        graph_candidates=[],
        anchors=anchors,
        intent="documentation",
        top_k=4,
    )
    assert documentation_rank.chunks[0].chunk_id == "section"

    architecture_rank = rerank_evidence(
        lexical_candidates=lexical,
        graph_candidates=[],
        anchors=anchors,
        intent="architecture",
        top_k=4,
    )
    assert architecture_rank.chunks[0].chunk_id == "dir"


def test_collect_text_candidates_and_flatten_are_deterministic() -> None:
    store, chunks, engine = _build_store_for_retrieval()
    seed = chunks[0]
    lexical_rows = LexicalCandidateSource(engine).collect_candidates(
        SourceContext(
            query="where is helper documented?",
            top_k=2,
            request_top_k=2,
            seed_chunks=[seed],
            lexical_scored=[],
        )
    )
    lexical_scored, _, _ = flatten_text_candidates(lexical_rows)
    context = SourceContext(
        query="where is helper documented?",
        top_k=2,
        request_top_k=2,
        seed_chunks=chunks,
        lexical_scored=lexical_scored,
    )
    merged = collect_text_candidates(
        sources=[
            LexicalCandidateSource(engine),
            HierarchyCandidateSource(),
            VectorCandidateSource(),
        ],
        context=context,
    )
    lexical, hierarchy, vector = flatten_text_candidates(merged)
    assert lexical
    assert hierarchy
    assert vector == []
    # Stable merge ordering by chunk id then source.
    assert [item.scored_chunk.chunk.chunk_id for item in merged] == sorted(
        item.scored_chunk.chunk.chunk_id for item in merged
    )


def test_rerank_accepts_mixed_source_scored_chunks() -> None:
    chunk = _chunk(
        chunk_id="mixed",
        source_path=Path("src/main.py"),
        text="helper computes output",
        source_type="code",
    )
    lexical = ScoredChunk(chunk=chunk, score=1.0)
    hierarchy = ScoredChunk(chunk=chunk, score=0.0)
    vector = ScoredChunk(chunk=chunk, score=0.0)
    result = rerank_evidence(
        lexical_candidates=[lexical, hierarchy, vector],
        graph_candidates=[],
        anchors=AnchorSet(
            node_ids=set(),
            file_paths=set(),
            symbol_ids=set(),
            symbol_labels=set(),
            symbol_names={"helper"},
            section_ids=set(),
        ),
        intent="implementation",
        top_k=1,
    )
    assert result.chunks
    assert result.reasons[0].chunk_id == "mixed"
