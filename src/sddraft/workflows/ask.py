"""Grounded Q&A workflow."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.hierarchy import (
    default_hierarchy_index_path,
    expand_chunks_with_hierarchy,
    load_hierarchy_index,
)
from sddraft.analysis.retrieval import BM25Retriever, load_retrieval_index, to_citations
from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    AskResult,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
)
from sddraft.llm.base import LLMClient, StructuredGenerationRequest
from sddraft.prompts.builders import build_query_prompt


def answer_question(
    request: QueryRequest,
    index_path: Path,
    llm_client: LLMClient,
    model_name: str,
    temperature: float = 0.2,
) -> AskResult:
    """Retrieve evidence and generate a grounded structured answer."""

    index = load_retrieval_index(index_path)
    retriever = BM25Retriever(index)
    chunks = retriever.search(request.question, top_k=request.top_k)

    hierarchy_fallback_note: str | None = None
    hierarchy_path = default_hierarchy_index_path(index_path)
    if hierarchy_path.exists():
        try:
            if hierarchy_path.stat().st_mtime < index_path.stat().st_mtime:
                hierarchy_fallback_note = (
                    "Hierarchy index appears stale; used lexical evidence only."
                )
            else:
                hierarchy_index = load_hierarchy_index(hierarchy_path)
                chunks = expand_chunks_with_hierarchy(
                    initial_chunks=chunks,
                    retrieval_index=index,
                    hierarchy_index=hierarchy_index,
                    top_k=request.top_k,
                )
        except (AnalysisError, OSError):
            hierarchy_fallback_note = (
                "Hierarchy index unavailable or unreadable; used lexical evidence only."
            )
    else:
        hierarchy_fallback_note = (
            "Hierarchy index unavailable; used lexical evidence only."
        )

    citations = to_citations(chunks)

    evidence_pack = QueryEvidencePack(
        request=request,
        chunks=chunks,
        citations=citations,
    )

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

    return AskResult(answer=answer, evidence_pack=evidence_pack)
