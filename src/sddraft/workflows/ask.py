"""Grounded Q&A workflow."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.graph_index import default_graph_manifest_path, load_graph_store
from sddraft.analysis.graph_retrieval import (
    GraphExpansionCandidateSource,
    LexicalCandidateSource,
    rerank_evidence,
)
from sddraft.analysis.hierarchy import (
    default_hierarchy_index_path,
    expand_chunks_with_hierarchy,
    load_hierarchy_index,
)
from sddraft.analysis.metrics import RunMetricsCollector
from sddraft.analysis.retrieval import (
    ScoredChunk,
    open_query_engine,
    resolve_retrieval_store_path,
    to_citations,
)
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    AskResult,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_query_prompt
from sddraft.render.json_artifacts import write_json_model


def answer_question(
    request: QueryRequest,
    index_path: Path,
    llm_client: LLMClient,
    model_name: str,
    temperature: float = 0.2,
    graph_enabled: bool = True,
    graph_depth: int = 1,
    graph_top_k: int = 12,
) -> AskResult:
    """Retrieve evidence and generate a grounded structured answer."""

    # Stage 1: always start with lexical retrieval so behavior is predictable
    # even when hierarchy/graph artifacts are missing.
    retrieval_root = resolve_retrieval_store_path(index_path)
    metrics_collector = RunMetricsCollector(csc_id=retrieval_root.parent.name)
    metrics_collector.start("retrieve")
    engine = open_query_engine(retrieval_root)
    lexical_scored = LexicalCandidateSource(engine).collect(
        query=request.question, top_k=request.top_k
    )
    primary_chunks = [item.chunk for item in lexical_scored]
    chunks = list(primary_chunks)

    hierarchy_fallback_note: str | None = None
    hierarchy_path = default_hierarchy_index_path(retrieval_root)
    if hierarchy_path.exists():
        try:
            # Stage 2: expand lexical evidence with nearby hierarchy context.
            hierarchy_index = load_hierarchy_index(hierarchy_path)
            chunks = expand_chunks_with_hierarchy(
                initial_chunks=chunks,
                retrieval_index=None,
                hierarchy_index=hierarchy_index,
                top_k=request.top_k,
                load_chunks_by_node_ids=lambda node_ids, limit: (
                    engine.load_chunks_by_node_ids(node_ids, limit=limit)
                ),
            )
        except (AnalysisError, OSError):
            hierarchy_fallback_note = (
                "Hierarchy store unavailable or unreadable; used lexical evidence only."
            )
    else:
        hierarchy_fallback_note = (
            "Hierarchy store unavailable; used lexical evidence only."
        )
    graph_fallback_note: str | None = None
    rerank_reasons = []
    related_files: list[Path] = []
    related_symbols: list[str] = []
    related_sections: list[str] = []

    if graph_enabled:
        graph_manifest_path = default_graph_manifest_path(retrieval_root)
        if graph_manifest_path.exists():
            try:
                # Stage 3: graph expansion + reranking for richer cross-file context.
                graph_store = load_graph_store(graph_manifest_path)
                lexical_chunk_ids = {item.chunk.chunk_id for item in lexical_scored}
                lexical_seed = [
                    *lexical_scored,
                    *[
                        ScoredChunk(chunk=item, score=0.0)
                        for item in chunks
                        if item.chunk_id not in lexical_chunk_ids
                    ],
                ]
                graph_source = GraphExpansionCandidateSource(
                    engine=engine,
                    store=graph_store,
                    depth=graph_depth,
                    top_k=max(graph_top_k, request.top_k),
                )
                graph_candidates, anchors, intent = graph_source.collect(
                    query=request.question,
                    seed_chunks=chunks,
                )
                rerank = rerank_evidence(
                    lexical_candidates=lexical_seed,
                    graph_candidates=graph_candidates,
                    anchors=anchors,
                    intent=intent,
                    top_k=max(graph_top_k, request.top_k),
                )
                chunks = rerank.chunks
                rerank_reasons = rerank.reasons
                related_files = rerank.related_files
                related_symbols = rerank.related_symbols
                related_sections = rerank.related_sections
            except (AnalysisError, OSError, ValueError):
                graph_fallback_note = (
                    "Engineering graph store unavailable or unreadable; "
                    "used lexical/hierarchy evidence only."
                )
        else:
            graph_fallback_note = "Engineering graph store unavailable; used lexical/hierarchy evidence only."

    metrics_collector.finish(chunks_loaded=len(chunks))

    citations = to_citations(chunks)

    evidence_pack = QueryEvidencePack(
        request=request,
        chunks=chunks,
        citations=citations,
        primary_chunks=primary_chunks,
        related_files=related_files,
        related_symbols=related_symbols,
        related_sections=related_sections,
        inclusion_reasons=rerank_reasons,
    )

    metrics_collector.start("answer")
    # Stage 4: prompt + structured model response generation.
    system_prompt, user_prompt = build_query_prompt(evidence_pack)
    response = llm_client.generate_structured(
        StructuredGenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=QueryAnswer,
            model_name=model_name,
            temperature=temperature,
        )
    )
    metrics_collector.finish(chunks_loaded=len(citations))

    answer = QueryAnswer.model_validate(response.content.model_dump(mode="json"))
    if not answer.citations:
        answer = answer.model_copy(update={"citations": citations})

    if not answer.missing_information and answer.answer.strip().upper().startswith(
        "TBD"
    ):
        answer = answer.model_copy(update={"missing_information": ["TBD"]})

    if hierarchy_fallback_note and hierarchy_fallback_note not in answer.uncertainty:
        answer = answer.model_copy(
            update={"uncertainty": [*answer.uncertainty, hierarchy_fallback_note]}
        )
    if graph_fallback_note and graph_fallback_note not in answer.uncertainty:
        answer = answer.model_copy(
            update={"uncertainty": [*answer.uncertainty, graph_fallback_note]}
        )

    run_metrics_path = retrieval_root.parent / "run_metrics_ask.json"
    write_json_model(run_metrics_path, metrics_collector.metrics)

    return AskResult(answer=answer, evidence_pack=evidence_pack)
