"""Grounded Q&A workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sddraft.analysis.graph_index import default_graph_manifest_path, load_graph_store
from sddraft.analysis.graph_retrieval import (
    GraphExpansionCandidateSource,
    HierarchyCandidateSource,
    LexicalCandidateSource,
    SourceContext,
    TextCandidateSource,
    VectorCandidateSource,
    collect_text_candidates,
    flatten_text_candidates,
    rerank_evidence,
)
from sddraft.analysis.hierarchy import (
    default_hierarchy_index_path,
    expand_chunks_with_hierarchy,
    load_hierarchy_index,
)
from sddraft.analysis.metrics import RunMetricsCollector
from sddraft.analysis.retrieval import (
    open_query_engine,
    resolve_retrieval_store_path,
    to_citations,
)
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    AskMode,
    AskResult,
    ProjectConfig,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_query_prompt
from sddraft.render.json_artifacts import write_json_model
from sddraft.workflows.intensive_ask import answer_question_intensive


def answer_question(
    request: QueryRequest,
    index_path: Path,
    llm_client: LLMClient,
    model_name: str,
    temperature: float = 0.2,
    mode: AskMode = "standard",
    project_config: ProjectConfig | None = None,
    repo_root: Path | None = None,
    graph_enabled: bool = True,
    graph_depth: int = 1,
    graph_top_k: int = 12,
    vector_enabled: bool = False,
    vector_top_k: int = 8,
    intensive_chunk_tokens: int = 8192,
    intensive_max_selected_excerpts: int = 12,
    progress_callback: Callable[[str], None] | None = None,
) -> AskResult:
    """Retrieve evidence and generate a grounded structured answer."""

    if mode == "intensive":
        return answer_question_intensive(
            request=request,
            index_path=index_path,
            project_config=project_config,
            repo_root=repo_root,
            llm_client=llm_client,
            model_name=model_name,
            temperature=temperature,
            chunk_tokens=intensive_chunk_tokens,
            max_selected_excerpts=intensive_max_selected_excerpts,
            progress_callback=progress_callback,
        )

    # Stage 1: always start with lexical retrieval so behavior is predictable
    # even when hierarchy/graph artifacts are missing.
    retrieval_root = resolve_retrieval_store_path(index_path)
    metrics_collector = RunMetricsCollector(csc_id=retrieval_root.parent.name)
    metrics_collector.start("retrieve")
    engine = open_query_engine(retrieval_root)
    lexical_source = LexicalCandidateSource(engine)
    lexical_context = SourceContext(
        query=request.question,
        top_k=request.top_k,
        request_top_k=request.top_k,
        seed_chunks=[],
        lexical_scored=[],
    )
    lexical_candidates = collect_text_candidates(
        sources=[lexical_source],
        context=lexical_context,
    )
    lexical_scored, _, _ = flatten_text_candidates(lexical_candidates)
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
    vector_fallback_note: str | None = None
    rerank_reasons = []
    related_files: list[Path] = []
    related_symbols: list[str] = []
    related_sections: list[str] = []
    related_commits: list[str] = []

    if graph_enabled:
        graph_manifest_path = default_graph_manifest_path(retrieval_root)
        if graph_manifest_path.exists():
            try:
                # Stage 3: graph expansion + reranking for richer cross-file context.
                graph_store = load_graph_store(graph_manifest_path)
                text_context = SourceContext(
                    query=request.question,
                    top_k=max(graph_top_k, request.top_k),
                    request_top_k=request.top_k,
                    seed_chunks=chunks,
                    lexical_scored=lexical_scored,
                )
                text_sources: list[TextCandidateSource] = [
                    lexical_source,
                    HierarchyCandidateSource(),
                ]
                if vector_enabled:
                    text_sources.append(VectorCandidateSource())
                text_candidates = collect_text_candidates(
                    sources=text_sources,
                    context=text_context,
                )
                lexical_seed, hierarchy_seed, vector_seed = flatten_text_candidates(
                    text_candidates
                )
                combined_seed = [*lexical_seed, *hierarchy_seed, *vector_seed]
                graph_source = GraphExpansionCandidateSource(
                    engine=engine,
                    store=graph_store,
                    depth=graph_depth,
                    top_k=max(graph_top_k, request.top_k),
                )
                graph_result = graph_source.collect_result(text_context)
                rerank = rerank_evidence(
                    lexical_candidates=combined_seed,
                    graph_candidates=graph_result.candidates,
                    anchors=graph_result.anchors,
                    intent=graph_result.intent,
                    top_k=max(graph_top_k, request.top_k),
                )
                chunks = rerank.chunks
                rerank_reasons = rerank.reasons
                related_files = rerank.related_files
                related_symbols = rerank.related_symbols
                related_sections = rerank.related_sections
                related_commits = rerank.related_commits
                if vector_enabled and not vector_seed:
                    vector_fallback_note = (
                        "Vector source enabled but no vector backend is configured yet; "
                        "used lexical/hierarchy/graph evidence only."
                    )
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
        related_commits=related_commits,
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
    if vector_fallback_note and vector_fallback_note not in answer.uncertainty:
        answer = answer.model_copy(
            update={"uncertainty": [*answer.uncertainty, vector_fallback_note]}
        )

    run_metrics_path = retrieval_root.parent / "run_metrics_ask.json"
    write_json_model(run_metrics_path, metrics_collector.metrics)

    return AskResult(answer=answer, evidence_pack=evidence_pack)
