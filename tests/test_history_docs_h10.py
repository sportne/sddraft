"""Tests for history-docs H10 benchmark and evaluation harness."""

from __future__ import annotations

from pathlib import Path

import engllm.tools.history_docs.benchmark as benchmark
from engllm.domain.errors import LLMError
from engllm.domain.models import (
    ProjectConfig,
    SDDToolDefaults,
    SourcesConfig,
    ToolDefaults,
    WorkspaceConfig,
)
from engllm.llm.base import StructuredGenerationRequest
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import build_history_docs_quality_judge_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDesignNoteAnchor,
    HistoryDocsBenchmarkCase,
    HistoryDocsBenchmarkExpectation,
    HistoryDocsQualityJudgmentEnvelope,
    HistoryDocsQualityReport,
    HistoryDocsRubricScore,
    HistoryLLMSectionOutline,
    HistoryRenderedSection,
    HistoryRenderManifest,
    HistorySectionPlan,
    HistorySubsystemConcept,
    HistorySubsystemConceptEnrichment,
    HistoryValidationReport,
)
from tests.history_docs_helpers import (
    benchmark_case_manifest_path,
    benchmark_comparison_report_path,
    benchmark_quality_report_path,
    benchmark_suite_manifest_path,
)


class _FailOnceJudgeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self._calls = 0
        self._delegate = MockLLMClient(
            canned={HistoryDocsQualityJudgmentEnvelope.__name__: payload}
        )

    def generate_structured(self, request: StructuredGenerationRequest):
        self._calls += 1
        if self._calls == 1:
            raise LLMError("judge unavailable")
        return self._delegate.generate_structured(request)


def _project_config(output_root: Path) -> ProjectConfig:
    return ProjectConfig(
        project_name="Benchmark Test",
        workspace=WorkspaceConfig(output_root=output_root),
        sources=SourcesConfig(roots=[Path("src")]),
        tools=ToolDefaults(sdd=SDDToolDefaults(template=Path("unused-template.yaml"))),
    )


def _rubric_scores(
    *,
    score: int,
    expectation_id: str = "exp-1",
    section_id: str = "introduction",
) -> list[HistoryDocsRubricScore]:
    dimensions = (
        "coverage",
        "coherence",
        "specificity",
        "algorithm_understanding",
        "dependency_understanding",
        "rationale_capture",
        "present_state_tone",
    )
    return [
        HistoryDocsRubricScore(
            dimension=dimension,
            score=score,
            rationale=f"{dimension} rationale",
            matched_expectation_ids=[expectation_id],
            cited_section_ids=[section_id],
        )
        for dimension in dimensions
    ]


def _canned_quality_payload(score: int = 4) -> dict[str, object]:
    return {
        "rubric_scores": [
            score_item.model_dump(mode="python")
            for score_item in _rubric_scores(score=score)
        ],
        "strengths": ["Strong section coverage."],
        "weaknesses": ["Could be more specific about rationale."],
        "unsupported_claim_risks": [],
        "tbd_overuse": False,
        "evaluator_notes": ["Mock benchmark review."],
        "uncertainty": ["Limited to fixture evidence."],
    }


def _loose_quality_payload(score: int = 4) -> dict[str, object]:
    return {
        "coverage": {"score": score, "rationale": "Coverage is strong."},
        "coherence": {"score": score, "rationale": "Coherence is strong."},
        "specificity": {"score": score, "rationale": "Specificity is strong."},
        "algorithm_understanding": {
            "score": score,
            "rationale": "Algorithm understanding is strong.",
        },
        "dependency_understanding": {
            "score": score,
            "rationale": "Dependency understanding is strong.",
        },
        "rationale_capture": {
            "score": score,
            "rationale": "Rationale capture is strong.",
        },
        "present_state_tone": {
            "score": score,
            "rationale": "Present-state tone is strong.",
        },
        "strengths": ["Loose-shape evaluation complete."],
        "weaknesses": [],
        "unsupported_claim_risks": [],
        "tbd_overuse": False,
        "evaluator_notes": [],
        "uncertainty": [],
    }


def test_default_benchmark_cases_cover_required_focus_tags(tmp_path: Path) -> None:
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=tmp_path / "artifacts",
    )

    assert [case.manifest.case_id for case in cases] == [
        "small_linear",
        "medium_mixed",
        "algorithm_heavy_variants",
        "dependency_heavy",
        "architecture_heavy",
    ]
    assert benchmark.validate_benchmark_focus_coverage(cases) == [
        "algorithm-heavy",
        "architecture-heavy",
        "dependency-heavy",
        "medium",
        "small",
    ]


def test_benchmark_paths_are_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"
    suite_id = "demo"
    case_id = "small_linear"

    assert (
        benchmark.benchmark_suite_workspace_id(suite_id)
        == "history-docs-benchmark-demo"
    )
    assert benchmark.benchmark_suite_manifest_path(
        output_root, suite_id
    ) == benchmark_suite_manifest_path(output_root, suite_id)
    assert benchmark.benchmark_case_manifest_path(
        output_root, suite_id, case_id
    ) == benchmark_case_manifest_path(output_root, suite_id, case_id)
    assert benchmark.benchmark_quality_report_path(
        output_root, suite_id, case_id, "baseline"
    ) == benchmark_quality_report_path(output_root, suite_id, case_id, "baseline")
    assert benchmark.benchmark_comparison_report_path(
        output_root, suite_id, case_id
    ) == benchmark_comparison_report_path(output_root, suite_id, case_id)


def test_quality_judge_prompt_includes_expectations_and_markdown(
    tmp_path: Path,
) -> None:
    case = HistoryDocsBenchmarkCase(
        case_id="case-1",
        title="Case",
        description="Prompt coverage case.",
        focus_tags=["small"],
        builder_name="builder",
        project_config=_project_config(tmp_path / "artifacts"),
        target_commit="a" * 40,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="exp-1",
                kind="section_presence",
                description="Expect the introduction section.",
                required_section_ids=["introduction"],
            )
        ],
    )
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        subsystems=[
            HistorySubsystemConcept(
                concept_id="semantic-subsystem::api",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                source_root=Path("src"),
                group_path=Path("src/app"),
                module_ids=["module::src/app/api/router.py"],
                file_count=1,
                symbol_count=2,
                language_counts={"python": 1},
                representative_files=[Path("src/app/api/router.py")],
                display_name="API Layer",
                capability_labels=["Request Handling"],
                baseline_subsystem_ids=["subsystem::src::app"],
            )
        ],
        modules=[],
        dependencies=[],
        sections=[],
    )
    render_manifest = HistoryRenderManifest(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        markdown_path=Path("checkpoint.md"),
        sections=[
            HistoryRenderedSection(
                section_id="introduction",
                title="Introduction",
                order=1,
                kind="core",
            )
        ],
    )
    validation_report = HistoryValidationReport(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        markdown_path=Path("checkpoint.md"),
        render_manifest_path=Path("render_manifest.json"),
    )
    enrichment = HistoryCheckpointModelEnrichment(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        evaluation_status="scored",
        subsystem_enrichments=[
            HistorySubsystemConceptEnrichment(
                concept_id="semantic-subsystem::api",
                display_name="API Layer",
                summary="Owns request handling.",
                capability_labels=["Request Handling"],
            )
        ],
        design_note_anchors=[
            HistoryDesignNoteAnchor(
                note_id="design-note::api-contract",
                title="API Contract",
                summary="Current boundary remains strict.",
            )
        ],
    )
    llm_outline = HistoryLLMSectionOutline(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        evaluation_status="scored",
        sections=[
            HistorySectionPlan(
                section_id="introduction",
                title="Introduction",
                kind="core",
                status="included",
                confidence_score=90,
                evidence_score=7,
                depth="standard",
                planning_rationale="Intro remains foundational.",
            ),
            HistorySectionPlan(
                section_id="interfaces",
                title="Interfaces",
                kind="optional",
                status="omitted",
                confidence_score=30,
                evidence_score=2,
                planning_rationale="Interface evidence stayed weak.",
                omission_reason="llm_planner_omitted",
            ),
        ],
    )

    system_prompt, user_prompt = build_history_docs_quality_judge_prompt(
        case=case,
        markdown="# Example Documentation\n\n## Introduction\n\nCurrent design summary.",
        render_manifest=render_manifest,
        validation_report=validation_report,
        checkpoint_model=checkpoint_model,
        checkpoint_model_enrichment=enrichment,
        llm_section_outline=llm_outline,
    )

    assert "coverage" in system_prompt
    assert "exp-1" in user_prompt
    assert "Current design summary." in user_prompt
    assert "Introduction" in user_prompt
    assert "Structure Summary" in user_prompt
    assert "Model Enrichment Summary" in user_prompt
    assert "Section Planning Summary" in user_prompt
    assert "API Layer" in user_prompt
    assert "Request Handling" in user_prompt
    assert "API Contract" in user_prompt
    assert "Intro remains foundational." in user_prompt


def test_compare_history_docs_quality_reports_uses_stable_tie_breaks() -> None:
    baseline = HistoryDocsQualityReport(
        case_id="case-1",
        variant_id="baseline",
        checkpoint_id="checkpoint-1",
        evaluation_status="scored",
        rubric_scores=_rubric_scores(score=4),
        overall_score=4.0,
    )
    candidate = HistoryDocsQualityReport(
        case_id="case-1",
        variant_id="aaa-candidate",
        checkpoint_id="checkpoint-1",
        evaluation_status="scored",
        rubric_scores=_rubric_scores(score=4),
        overall_score=4.0,
    )

    comparison = benchmark.compare_history_docs_quality_reports(
        case_id="case-1",
        baseline_report=baseline,
        candidate_report=candidate,
    )

    assert comparison.preferred_variant_id == "aaa-candidate"
    assert comparison.overall_delta == 0.0
    assert len(comparison.per_dimension_deltas) == 7
    assert comparison.comparison_notes == [
        "Tie resolved by stable variant-id ordering after score and risk tie-breaks."
    ]


def test_evaluate_history_docs_quality_normalizes_loose_provider_payload(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    case = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )[0]
    build_result = benchmark.baseline_history_docs_benchmark_variant().runner(
        case,
        "benchmark-normalization",
    )

    report = benchmark.evaluate_history_docs_quality(
        case=case,
        variant_id="baseline",
        build_result=build_result,
        llm_client=MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _loose_quality_payload(3)
            }
        ),
    )

    assert report.evaluation_status == "scored"
    assert report.overall_score == 3.0
    assert len(report.rubric_scores) == 7


def test_evaluate_history_docs_quality_normalizes_alias_fields_and_missing_dimensions(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    case = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )[0]
    build_result = benchmark.baseline_history_docs_benchmark_variant().runner(
        case,
        "benchmark-alias-normalization",
    )

    report = benchmark.evaluate_history_docs_quality(
        case=case,
        variant_id="baseline",
        build_result=build_result,
        llm_client=MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: {
                    "scores": {
                        "coverage": {
                            "rating": "4",
                            "reason": "Coverage is strong.",
                            "expectation_ids": ["exp-1"],
                            "section_ids": ["architectural_overview"],
                        },
                        "coherence": 3,
                        "specificity": {"value": 2, "notes": "Specific enough."},
                        "algorithm_understanding": True,
                        "dependency_understanding": "5 strong",
                        "rationale_capture": {"score": 2, "summary": "Some rationale."},
                    },
                    "strengths": ["Alias payload normalized."],
                    "weaknesses": [],
                    "unsupported_claim_risks": [],
                    "tbd_overuse": False,
                    "evaluator_notes": [],
                    "uncertainty": [],
                }
            }
        ),
    )

    by_dimension = {score.dimension: score for score in report.rubric_scores}
    assert report.evaluation_status == "scored"
    assert by_dimension["coverage"].score == 4
    assert by_dimension["coherence"].score == 3
    assert by_dimension["algorithm_understanding"].score == 1
    assert by_dimension["dependency_understanding"].score == 5
    assert by_dimension["present_state_tone"].score == 0
    assert any("omitted rubric dimensions" in note for note in report.evaluator_notes)


def test_evaluate_history_docs_quality_marks_reports_failed_when_scores_are_unusable(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    case = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )[0]
    build_result = benchmark.baseline_history_docs_benchmark_variant().runner(
        case,
        "benchmark-invalid-scores",
    )

    report = benchmark.evaluate_history_docs_quality(
        case=case,
        variant_id="baseline",
        build_result=build_result,
        llm_client=MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: {
                    "strengths": ["No usable rubric scores returned."],
                    "weaknesses": [],
                    "unsupported_claim_risks": [],
                    "tbd_overuse": False,
                    "evaluator_notes": [],
                    "uncertainty": [],
                }
            }
        ),
    )

    assert report.evaluation_status == "llm_failed"
    assert report.overall_score == 0.0
    assert "recognizable rubric scores" in (report.failure_note or "")


def test_run_history_docs_benchmark_suite_writes_suite_and_quality_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )
    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="demo",
        output_root=output_root,
        cases=cases,
        llm_client_factory=lambda config: MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _canned_quality_payload()
            }
        ),
    )

    suite_path = benchmark_suite_manifest_path(output_root, "demo")
    quality_path = benchmark_quality_report_path(
        output_root,
        "demo",
        "small_linear",
        "baseline",
    )
    comparison_path = benchmark_comparison_report_path(
        output_root,
        "demo",
        "small_linear",
    )

    assert suite_report.case_ids == [
        "small_linear",
        "medium_mixed",
        "algorithm_heavy_variants",
        "dependency_heavy",
        "architecture_heavy",
    ]
    assert suite_report.variant_ids == ["baseline"]
    assert suite_report.failed_evaluation_count == 0
    assert suite_report.average_score_by_variant["baseline"] == 4.0
    assert suite_path.exists()
    assert quality_path.exists()
    assert comparison_path.exists()
    report = HistoryDocsQualityReport.model_validate_json(
        quality_path.read_text(encoding="utf-8")
    )
    assert report.evaluation_status == "scored"
    assert report.overall_score == 4.0


def test_run_history_docs_benchmark_suite_persists_llm_failed_reports_and_continues(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )
    baseline = benchmark.baseline_history_docs_benchmark_variant()
    shadow = benchmark.HistoryDocsBenchmarkVariant(
        variant_id="shadow",
        runner=baseline.runner,
    )

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="llm-failure",
        output_root=output_root,
        cases=cases,
        variant_runners=[baseline, shadow],
        llm_client_factory=lambda config: _FailOnceJudgeClient(
            _canned_quality_payload(score=3)
        ),
    )

    baseline_report = HistoryDocsQualityReport.model_validate_json(
        benchmark_quality_report_path(
            output_root,
            "llm-failure",
            "small_linear",
            "baseline",
        ).read_text(encoding="utf-8")
    )
    shadow_report = HistoryDocsQualityReport.model_validate_json(
        benchmark_quality_report_path(
            output_root,
            "llm-failure",
            "small_linear",
            "shadow",
        ).read_text(encoding="utf-8")
    )

    assert suite_report.failed_evaluation_count == 5
    assert baseline_report.evaluation_status == "llm_failed"
    assert shadow_report.evaluation_status == "scored"
    assert shadow_report.overall_score == 3.0


def test_h12_enriched_model_variant_runs_through_benchmark_suite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="h12-enriched",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            benchmark.semantic_structure_context_benchmark_variant(),
            benchmark.semantic_structure_context_enriched_model_benchmark_variant(),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _canned_quality_payload(
                    score=4
                )
            }
        ),
    )

    assert suite_report.variant_ids == [
        "semantic-structure-context",
        "semantic-structure-context-enriched-model",
    ]
    assert benchmark_quality_report_path(
        output_root,
        "h12-enriched",
        cases[0].manifest.case_id,
        "semantic-structure-context-enriched-model",
    ).exists()


def test_h12_llm_section_planning_variant_runs_through_benchmark_suite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="h12-section-planning",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            benchmark.semantic_structure_context_benchmark_variant(),
            benchmark.semantic_structure_context_llm_section_planning_benchmark_variant(),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _canned_quality_payload(
                    score=4
                )
            }
        ),
    )

    assert suite_report.variant_ids == [
        "semantic-structure-context",
        "semantic-structure-context-llm-section-planning",
    ]
    assert benchmark_quality_report_path(
        output_root,
        "h12-section-planning",
        cases[0].manifest.case_id,
        "semantic-structure-context-llm-section-planning",
    ).exists()


def test_h13_variants_run_through_benchmark_suite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="h13-variants",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            benchmark.semantic_structure_context_benchmark_variant(),
            benchmark.semantic_structure_context_enriched_algorithms_benchmark_variant(),
            benchmark.semantic_structure_context_interface_inventory_benchmark_variant(),
            benchmark.semantic_structure_context_dependency_landscape_benchmark_variant(),
            benchmark.semantic_structure_context_h13_full_benchmark_variant(),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _canned_quality_payload(
                    score=4
                )
            }
        ),
    )

    assert suite_report.variant_ids == [
        "semantic-structure-context",
        "semantic-structure-context-enriched-algorithms",
        "semantic-structure-context-interface-inventory",
        "semantic-structure-context-dependency-landscape",
        "semantic-structure-context-h13-full",
    ]
    assert benchmark_quality_report_path(
        output_root,
        "h13-variants",
        cases[0].manifest.case_id,
        "semantic-structure-context-h13-full",
    ).exists()


def test_h14_variants_run_through_benchmark_suite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="h14-variants",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            benchmark.semantic_structure_context_benchmark_variant(),
            benchmark.semantic_structure_context_llm_draft_benchmark_variant(),
            benchmark.semantic_structure_context_llm_repair_benchmark_variant(),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={
                HistoryDocsQualityJudgmentEnvelope.__name__: _canned_quality_payload(
                    score=4
                )
            }
        ),
    )

    assert suite_report.variant_ids == [
        "semantic-structure-context",
        "semantic-structure-context-llm-draft",
        "semantic-structure-context-llm-repair",
    ]
    assert benchmark_quality_report_path(
        output_root,
        "h14-variants",
        cases[0].manifest.case_id,
        "semantic-structure-context-llm-repair",
    ).exists()
