"""Grounded Q&A workflow."""

from __future__ import annotations

from pathlib import Path

from sddraft.analysis.retrieval import BM25Retriever, load_retrieval_index, to_citations
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

    return AskResult(answer=answer, evidence_pack=evidence_pack)
