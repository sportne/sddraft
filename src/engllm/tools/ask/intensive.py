"""Intensive ask workflow helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from engllm.core.analysis.intensive_corpus import (
    default_intensive_corpus_root,
    default_intensive_runs_root,
    iter_intensive_corpus_chunks,
    prepare_intensive_corpus,
)
from engllm.core.analysis.metrics import RunMetricsCollector
from engllm.core.analysis.retrieval import resolve_retrieval_store_path, to_citations
from engllm.core.render.json_artifacts import write_json_model
from engllm.core.workspaces import tool_root_from_shared_artifact
from engllm.domain.errors import ConfigError, RenderingError
from engllm.domain.models import (
    KnowledgeChunk,
    ProjectConfig,
)
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.ask.builders import (
    build_intensive_screening_prompt,
    build_query_prompt,
)
from engllm.tools.ask.models import (
    AskResult,
    ChunkInclusionReason,
    IntensiveChunkScreening,
    IntensiveCorpusChunk,
    IntensiveRunManifest,
    IntensiveScreeningExcerpt,
    IntensiveSelectedExcerpt,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
)


@dataclass(frozen=True, slots=True)
class _ScreenedExcerptCandidate:
    """Normalized excerpt candidate collected from chunk screening."""

    source_path: Path
    line_start: int
    line_end: int
    reason: str
    screening_score: float
    source_chunk_id: str


def answer_question_intensive(
    *,
    request: QueryRequest,
    index_path: Path,
    project_config: ProjectConfig | None,
    repo_root: Path | None,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    chunk_tokens: int,
    max_selected_excerpts: int,
    progress_callback: Callable[[str], None] | None = None,
) -> AskResult:
    """Answer a question by screening a structured cross-file corpus."""

    if project_config is None:
        raise ConfigError(
            "Intensive ask mode requires --config so repo scope can be resolved."
        )

    retrieval_root = resolve_retrieval_store_path(index_path)
    ask_root = tool_root_from_shared_artifact(retrieval_root, "ask")
    resolved_repo_root = (repo_root or Path(".")).resolve()
    csc_id = retrieval_root.parent.parent.name
    question_hash = _question_hash(request)
    run_root = default_intensive_runs_root(ask_root) / question_hash
    run_root.mkdir(parents=True, exist_ok=True)
    screenings_path = run_root / "screenings.jsonl"
    selected_excerpts_path = run_root / "selected_excerpts.json"
    run_manifest_path = run_root / "manifest.json"
    run_metrics_path = run_root / "run_metrics.json"

    metrics_collector = RunMetricsCollector(csc_id=csc_id)

    if progress_callback is not None:
        progress_callback(
            "intensive: discovering files and preparing structured corpus"
        )
    metrics_collector.start("prepare_corpus")
    corpus_manifest, corpus_reused = prepare_intensive_corpus(
        project_config=project_config,
        repo_root=resolved_repo_root,
        output_root=ask_root,
        csc_id=csc_id,
        chunk_tokens=chunk_tokens,
    )
    metrics_collector.finish(
        files_seen=corpus_manifest.file_count,
        chunks_written=0 if corpus_reused else corpus_manifest.chunk_count,
        chunks_loaded=corpus_manifest.chunk_count if corpus_reused else 0,
    )

    corpus_manifest_path = default_intensive_corpus_root(ask_root) / "manifest.json"
    chunk_list = list(iter_intensive_corpus_chunks(corpus_manifest_path))

    if progress_callback is not None:
        mode = "reused" if corpus_reused else "built"
        progress_callback(
            f"intensive: {mode} corpus with {corpus_manifest.chunk_count} chunks"
        )

    metrics_collector.start("screen_chunks")
    screened_candidates: list[_ScreenedExcerptCandidate] = []
    relevant_chunk_count = 0
    _write_screenings(
        screenings_path,
        _iter_screenings(
            chunks=chunk_list,
            request=request,
            llm_client=llm_client,
            model_name=model_name,
            temperature=temperature,
            progress_callback=progress_callback,
            screened_candidates=screened_candidates,
        ),
    )
    relevant_chunk_count = len({item.source_chunk_id for item in screened_candidates})
    metrics_collector.finish(chunks_loaded=len(chunk_list))

    if progress_callback is not None:
        progress_callback("intensive: merging and ranking selected excerpts")
    selected_excerpts = _merge_and_rank_excerpts(
        screened_candidates,
        max_selected_excerpts=max_selected_excerpts,
        repo_root=resolved_repo_root,
    )
    _write_json_list(selected_excerpts_path, selected_excerpts)

    metrics_collector.start("answer")
    answer, evidence_pack = _build_answer_from_selected_excerpts(
        request=request,
        selected_excerpts=selected_excerpts,
        llm_client=llm_client,
        model_name=model_name,
        temperature=temperature,
        progress_callback=progress_callback,
    )
    metrics_collector.finish(chunks_loaded=len(evidence_pack.citations))

    run_manifest = IntensiveRunManifest(
        question_hash=question_hash,
        question=request.question,
        corpus_manifest_path=Path("../../corpus/manifest.json"),
        screenings_path=Path("screenings.jsonl"),
        selected_excerpts_path=Path("selected_excerpts.json"),
        model_name=model_name,
        temperature=temperature,
        chunk_tokens=chunk_tokens,
        max_selected_excerpts=max_selected_excerpts,
        total_chunks_screened=len(chunk_list),
        relevant_chunk_count=relevant_chunk_count,
        selected_excerpt_count=len(selected_excerpts),
        corpus_reused=corpus_reused,
    )
    write_json_model(run_manifest_path, run_manifest)
    write_json_model(run_metrics_path, metrics_collector.metrics)
    write_json_model(ask_root / "run_metrics_ask.json", metrics_collector.metrics)

    return AskResult(answer=answer, evidence_pack=evidence_pack)


def _iter_screenings(
    *,
    chunks: list[IntensiveCorpusChunk],
    request: QueryRequest,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    progress_callback: Callable[[str], None] | None,
    screened_candidates: list[_ScreenedExcerptCandidate],
) -> Iterator[IntensiveChunkScreening]:
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if progress_callback is not None:
            progress_callback(f"intensive: screening chunk {index}/{total}")
        system_prompt, user_prompt = build_intensive_screening_prompt(
            question=request.question,
            session_history=request.session_history,
            chunk=chunk,
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=IntensiveChunkScreening,
                model_name=model_name,
                temperature=temperature,
            )
        )
        screening = IntensiveChunkScreening.model_validate(
            response.content.model_dump(mode="json")
        )
        valid_excerpts = _validate_screening_excerpts(
            chunk, screening.selected_excerpts
        )
        normalized = screening.model_copy(
            update={
                "chunk_id": chunk.chunk_id,
                "selected_excerpts": valid_excerpts,
                "is_relevant": screening.is_relevant or bool(valid_excerpts),
            }
        )
        if normalized.selected_excerpts:
            for excerpt in normalized.selected_excerpts:
                screened_candidates.append(
                    _ScreenedExcerptCandidate(
                        source_path=excerpt.source_path,
                        line_start=excerpt.line_start,
                        line_end=excerpt.line_end,
                        reason=excerpt.reason,
                        screening_score=normalized.relevance_score,
                        source_chunk_id=chunk.chunk_id,
                    )
                )
        yield normalized


def _validate_screening_excerpts(
    chunk: IntensiveCorpusChunk,
    excerpts: list[IntensiveScreeningExcerpt],
) -> list[IntensiveScreeningExcerpt]:
    """Keep only excerpts that stay within the provided structured segments."""

    bounded: list[IntensiveScreeningExcerpt] = []
    segments_by_path: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for segment in chunk.segments:
        segments_by_path[segment.source_path].append(
            (segment.line_start, segment.line_end)
        )

    for excerpt in excerpts:
        ranges = segments_by_path.get(excerpt.source_path, [])
        for segment_start, segment_end in ranges:
            if excerpt.line_end < segment_start or excerpt.line_start > segment_end:
                continue
            clipped_start = max(excerpt.line_start, segment_start)
            clipped_end = min(excerpt.line_end, segment_end)
            if clipped_start > clipped_end:
                continue
            bounded.append(
                excerpt.model_copy(
                    update={"line_start": clipped_start, "line_end": clipped_end}
                )
            )
            break
    return bounded


def _merge_and_rank_excerpts(
    candidates: list[_ScreenedExcerptCandidate],
    *,
    max_selected_excerpts: int,
    repo_root: Path,
) -> list[IntensiveSelectedExcerpt]:
    by_path: dict[Path, list[_ScreenedExcerptCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate.source_path].append(candidate)

    merged: list[IntensiveSelectedExcerpt] = []
    for source_path in sorted(by_path, key=lambda item: item.as_posix()):
        ordered = sorted(
            by_path[source_path],
            key=lambda item: (
                item.line_start,
                item.line_end,
                -item.screening_score,
                item.reason,
                item.source_chunk_id,
            ),
        )
        active: _ScreenedExcerptCandidate | None = None
        active_chunk_ids: set[str] = set()
        active_reasons: list[str] = []
        for candidate in ordered:
            if active is None:
                active = candidate
                active_chunk_ids = {candidate.source_chunk_id}
                active_reasons = [candidate.reason]
                continue
            if candidate.line_start <= active.line_end + 1:
                active = _ScreenedExcerptCandidate(
                    source_path=active.source_path,
                    line_start=active.line_start,
                    line_end=max(active.line_end, candidate.line_end),
                    reason=active.reason,
                    screening_score=max(
                        active.screening_score, candidate.screening_score
                    ),
                    source_chunk_id=active.source_chunk_id,
                )
                active_chunk_ids.add(candidate.source_chunk_id)
                active_reasons.append(candidate.reason)
                continue
            merged.append(
                _selected_excerpt_from_active(
                    active,
                    active_chunk_ids,
                    active_reasons,
                    repo_root=repo_root,
                )
            )
            active = candidate
            active_chunk_ids = {candidate.source_chunk_id}
            active_reasons = [candidate.reason]
        if active is not None:
            merged.append(
                _selected_excerpt_from_active(
                    active,
                    active_chunk_ids,
                    active_reasons,
                    repo_root=repo_root,
                )
            )

    ranked = sorted(
        merged,
        key=lambda item: (
            -item.screening_score,
            item.source_path.as_posix(),
            item.line_start,
            item.line_end,
            item.excerpt_id,
        ),
    )
    return ranked[:max_selected_excerpts]


def _selected_excerpt_from_active(
    active: _ScreenedExcerptCandidate,
    chunk_ids: set[str],
    reasons: list[str],
    *,
    repo_root: Path,
) -> IntensiveSelectedExcerpt:
    text = _read_excerpt_text(
        repo_root=repo_root,
        source_path=active.source_path,
        line_start=active.line_start,
        line_end=active.line_end,
    )
    excerpt_id = f"intensive::{active.source_path.as_posix()}::{active.line_start}-{active.line_end}"
    reason = (
        "; ".join(dict.fromkeys(reason for reason in reasons if reason))
        or "Relevant excerpt"
    )
    return IntensiveSelectedExcerpt(
        excerpt_id=excerpt_id,
        source_path=active.source_path,
        line_start=active.line_start,
        line_end=active.line_end,
        text=text,
        reason=reason,
        screening_score=active.screening_score,
        source_chunk_ids=sorted(chunk_ids),
    )


def _read_excerpt_text(
    *,
    repo_root: Path,
    source_path: Path,
    line_start: int,
    line_end: int,
) -> str:
    absolute_path = repo_root / source_path
    try:
        lines = absolute_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    start_index = max(line_start - 1, 0)
    end_index = min(line_end, len(lines))
    return "\n".join(lines[start_index:end_index]).strip()


def _build_answer_from_selected_excerpts(
    *,
    request: QueryRequest,
    selected_excerpts: list[IntensiveSelectedExcerpt],
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    progress_callback: Callable[[str], None] | None,
) -> tuple[QueryAnswer, QueryEvidencePack]:
    chunks = [
        KnowledgeChunk(
            chunk_id=excerpt.excerpt_id,
            source_type="code",
            source_path=excerpt.source_path,
            text=excerpt.text,
            line_start=excerpt.line_start,
            line_end=excerpt.line_end,
            metadata={
                "screening_score": f"{excerpt.screening_score:.3f}",
                "screening_reason": excerpt.reason,
            },
        )
        for excerpt in selected_excerpts
    ]
    citations = to_citations(chunks)
    inclusion_reasons = [
        ChunkInclusionReason(
            chunk_id=excerpt.excerpt_id,
            source="intensive",
            final_score=excerpt.screening_score,
            reason=excerpt.reason,
        )
        for excerpt in selected_excerpts
    ]
    evidence_pack = QueryEvidencePack(
        request=request,
        chunks=chunks,
        citations=citations,
        primary_chunks=chunks,
        related_files=sorted({excerpt.source_path for excerpt in selected_excerpts}),
        inclusion_reasons=inclusion_reasons,
    )

    if not selected_excerpts:
        answer = QueryAnswer(
            answer=(
                "TBD: Intensive screening did not identify relevant evidence in the "
                "configured codebase."
            ),
            citations=[],
            uncertainty=[
                "Intensive screening completed, but no relevant excerpts were selected."
            ],
            missing_information=["TBD"],
            confidence=0.0,
        )
        return answer, evidence_pack

    if progress_callback is not None:
        progress_callback("intensive: generating final answer from selected excerpts")
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
    return answer, evidence_pack


def _question_hash(request: QueryRequest) -> str:
    encoded = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_screenings(
    path: Path, screenings: Iterable[IntensiveChunkScreening]
) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            for screening in screenings:
                handle.write(screening.model_dump_json())
                handle.write("\n")
    except OSError as exc:
        raise RenderingError(
            f"Failed writing intensive screening results to {path}: {exc}"
        ) from exc


def _write_json_list(path: Path, values: list[IntensiveSelectedExcerpt]) -> None:
    try:
        path.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in values],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RenderingError(
            f"Failed writing intensive selected excerpts to {path}: {exc}"
        ) from exc
