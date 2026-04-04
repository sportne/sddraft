"""H14-02 shadow draft review helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_draft_review_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDraftReview,
    HistoryDraftReviewJudgment,
    HistoryDraftReviewStatus,
    HistorySectionDraftArtifact,
    HistorySectionPlanId,
    HistorySemanticContextMap,
    HistoryValidationReport,
)


def draft_review_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H14-02 draft-review artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "draft_review.json"


def summarize_draft_review(review: HistoryDraftReview) -> dict[str, object]:
    """Return a compact H14 review summary for benchmarks and prompts."""

    finding_counts_by_kind = Counter(finding.kind for finding in review.findings)
    finding_counts_by_severity = Counter(
        finding.severity for finding in review.findings
    )
    return {
        "evaluation_status": review.evaluation_status,
        "strength_count": len(review.strengths),
        "finding_count": len(review.findings),
        "finding_counts_by_kind": dict(sorted(finding_counts_by_kind.items())),
        "finding_counts_by_severity": dict(sorted(finding_counts_by_severity.items())),
        "recommended_repair_section_ids": list(review.recommended_repair_section_ids),
        "unrepairable_notes": list(review.unrepairable_notes),
    }


_SECTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dependencies", ("dependency", "package", "library", "runtime")),
    (
        "build_development_infrastructure",
        ("build", "tooling", "test", "lint", "ci", "infra"),
    ),
    (
        "interfaces",
        ("interface", "api", "endpoint", "contract", "consumer", "provider"),
    ),
    (
        "system_context",
        ("context", "external", "system boundary", "runtime environment"),
    ),
    ("subsystems_modules", ("subsystem", "module", "responsibility")),
    (
        "architectural_overview",
        ("architecture", "architectural", "structure", "relationship"),
    ),
)


def _infer_section_id(
    *,
    summary: str,
    revision_goal: str,
    allowed_section_ids: set[str],
) -> str | None:
    combined = f"{summary} {revision_goal}".lower()
    for section_id, keywords in _SECTION_KEYWORDS:
        if section_id in allowed_section_ids and any(
            keyword in combined for keyword in keywords
        ):
            return section_id
    return None


def _map_review_findings_to_sections(
    review: HistoryDraftReview,
    *,
    allowed_section_ids: Sequence[str],
) -> HistoryDraftReview:
    allowed_set = set(allowed_section_ids)
    mapped_findings = []
    unrepairable_notes: list[str] = list(review.unrepairable_notes)
    for finding in review.findings:
        section_id = finding.section_id
        if section_id is None:
            section_id = _infer_section_id(
                summary=finding.summary,
                revision_goal=finding.revision_goal,
                allowed_section_ids=allowed_set,
            )
            if section_id is None:
                unrepairable_notes.append(
                    f"Finding {finding.finding_id} remained unrepairable with current finding granularity."
                )
        mapped_findings.append(finding.model_copy(update={"section_id": section_id}))
    recommended = [
        section_id
        for section_id in review.recommended_repair_section_ids
        if section_id in allowed_set
    ]
    for finding in mapped_findings:
        if finding.section_id is not None and finding.section_id not in recommended:
            recommended.append(cast(HistorySectionPlanId, finding.section_id))
    return review.model_copy(
        update={
            "findings": mapped_findings,
            "recommended_repair_section_ids": recommended,
            "unrepairable_notes": unrepairable_notes,
        }
    )


def _fallback_review(
    *,
    checkpoint_model: HistoryCheckpointModel,
    evaluation_status: HistoryDraftReviewStatus,
) -> HistoryDraftReview:
    return HistoryDraftReview(
        checkpoint_id=checkpoint_model.checkpoint_id,
        target_commit=checkpoint_model.target_commit,
        previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
        evaluation_status=evaluation_status,
        strengths=[],
        findings=[],
        recommended_repair_section_ids=[],
    )


def build_draft_review(
    *,
    checkpoint_model: HistoryCheckpointModel,
    draft_artifact: HistorySectionDraftArtifact,
    draft_markdown: str,
    draft_validation_report: HistoryValidationReport,
    semantic_context_map: HistorySemanticContextMap | None,
    checkpoint_model_enrichment: HistoryCheckpointModelEnrichment,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> HistoryDraftReview:
    """Build H14-02 draft review."""

    section_ids = [section.section_id for section in draft_artifact.sections]
    fallback = _fallback_review(
        checkpoint_model=checkpoint_model,
        evaluation_status="heuristic_only" if llm_client is None else "llm_failed",
    )
    if llm_client is None:
        return fallback
    try:
        system_prompt, user_prompt = build_draft_review_prompt(
            checkpoint_context={
                "checkpoint_id": checkpoint_model.checkpoint_id,
                "target_commit": checkpoint_model.target_commit,
                "previous_checkpoint_commit": checkpoint_model.previous_checkpoint_commit,
            },
            draft_summary={
                "evaluation_status": draft_artifact.evaluation_status,
                "section_ids": section_ids,
                "section_count": len(section_ids),
            },
            validation_summary={
                "error_count": draft_validation_report.error_count,
                "warning_count": draft_validation_report.warning_count,
                "findings": [
                    {
                        "check_id": finding.check_id,
                        "severity": finding.severity,
                        "section_id": finding.section_id,
                        "reference": finding.reference,
                    }
                    for finding in draft_validation_report.findings
                ],
            },
            evidence_summary={
                "subsystem_count": len(checkpoint_model.subsystems),
                "module_count": len(checkpoint_model.modules),
                "dependency_concept_count": len(checkpoint_model.dependencies),
                "capability_proposal_titles": [
                    proposal.title
                    for proposal in checkpoint_model_enrichment.capability_proposals
                ],
                "design_note_titles": [
                    anchor.title
                    for anchor in checkpoint_model_enrichment.design_note_anchors
                ],
                "context_node_titles": (
                    []
                    if semantic_context_map is None
                    else [node.title for node in semantic_context_map.context_nodes]
                ),
                "interface_titles": (
                    []
                    if semantic_context_map is None
                    else [
                        interface.title for interface in semantic_context_map.interfaces
                    ]
                ),
            },
            markdown=draft_markdown,
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryDraftReviewJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryDraftReviewJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        if not all(
            finding.section_id is None or finding.section_id in set(section_ids)
            for finding in judgment.findings
        ):
            raise ValueError("draft review referenced unknown section ids")
        if not set(judgment.recommended_repair_section_ids) <= set(section_ids):
            raise ValueError("draft review recommended unknown repair sections")
        review = HistoryDraftReview(
            checkpoint_id=checkpoint_model.checkpoint_id,
            target_commit=checkpoint_model.target_commit,
            previous_checkpoint_commit=checkpoint_model.previous_checkpoint_commit,
            evaluation_status="scored",
            strengths=list(judgment.strengths),
            findings=sorted(
                judgment.findings,
                key=lambda item: (
                    item.section_id or "",
                    item.kind,
                    item.severity,
                    item.finding_id,
                ),
            ),
            recommended_repair_section_ids=list(
                judgment.recommended_repair_section_ids
            ),
            unrepairable_notes=[],
        )
        return _map_review_findings_to_sections(
            review,
            allowed_section_ids=section_ids,
        )
    except Exception:
        return fallback


__all__ = [
    "build_draft_review",
    "draft_review_path",
    "summarize_draft_review",
]
