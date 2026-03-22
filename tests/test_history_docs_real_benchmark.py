"""Tests for the internal real-repo benchmark runner and promotion gates."""

from __future__ import annotations

from pathlib import Path

from engllm.llm.mock import MockLLMClient
from engllm.tools.history_docs import benchmark
from engllm.tools.history_docs.models import (
    HistoryDocsBenchmarkCaseComparisonReport,
    HistoryDocsQualityJudgmentEnvelope,
    HistoryDocsQualityReport,
    HistoryDocsRubricDelta,
    HistoryDocsRubricScore,
    HistoryDocsVariantComparison,
)
from engllm.tools.history_docs.real_benchmark import (
    build_promotion_gate_report,
    prepare_real_history_docs_benchmark_cases,
)
from engllm.tools.history_docs.real_benchmark import (
    promotion_gate_report_path as build_promotion_gate_report_path,
)
from tests.history_docs_helpers import commit_file, git, promotion_gate_report_path

_DIMENSIONS = (
    "coverage",
    "coherence",
    "specificity",
    "algorithm_understanding",
    "dependency_understanding",
    "rationale_capture",
    "present_state_tone",
)
_REAL_REPOS = [
    "kleuw",
    "python-sonarqube-api",
    "aoa-agent-demo",
    "learn-blazor",
    "cpp-helper-libs",
]


def _quality_payload(score: int) -> dict[str, object]:
    return {
        "rubric_scores": [
            {
                "dimension": dimension,
                "score": score,
                "rationale": f"{dimension} matched the synthetic benchmark evidence.",
                "matched_expectation_ids": ["expectation-1"],
                "cited_section_ids": ["architectural_overview"],
            }
            for dimension in _DIMENSIONS
        ],
        "strengths": ["Synthetic evaluation complete."],
        "weaknesses": [],
        "unsupported_claim_risks": [],
        "tbd_overuse": False,
        "evaluator_notes": [],
        "uncertainty": [],
    }


def _rubric_scores(score: int) -> list[HistoryDocsRubricScore]:
    return [
        HistoryDocsRubricScore(
            dimension=dimension,
            score=score,
            rationale=f"Synthetic {dimension} score.",
            matched_expectation_ids=["expectation-1"],
            cited_section_ids=["architectural_overview"],
        )
        for dimension in _DIMENSIONS
    ]


def _create_real_repo(parent_dir: Path, repo_name: str) -> None:
    repo_root = parent_dir / repo_name
    repo_root.mkdir(parents=True, exist_ok=True)
    git(repo_root, "init")
    first = commit_file(
        repo_root,
        "src/app.py" if repo_name != "cpp-helper-libs" else "libs/core.cpp",
        (
            "print('one')\n"
            if repo_name != "cpp-helper-libs"
            else "int one() { return 1; }\n"
        ),
        message="initial commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    if repo_name == "learn-blazor":
        git(repo_root, "tag", "v0.1.0", first)
    commit_file(
        repo_root,
        "src/api.py" if repo_name != "cpp-helper-libs" else "libs/api.cpp",
        (
            "def api() -> str:\n    return 'ok'\n"
            if repo_name != "cpp-helper-libs"
            else "int api() { return 2; }\n"
        ),
        message="add boundary",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/storage.py" if repo_name != "cpp-helper-libs" else "libs/storage.cpp",
        (
            "class Repository:\n    pass\n"
            if repo_name != "cpp-helper-libs"
            else "int storage() { return 3; }\n"
        ),
        message="add storage",
        timestamp="2024-03-01T10:00:00+00:00",
    )


def test_prepare_real_history_docs_benchmark_cases_is_deterministic(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "siblings"
    parent_dir.mkdir()
    for repo_name in _REAL_REPOS:
        _create_real_repo(parent_dir, repo_name)

    first_cases = prepare_real_history_docs_benchmark_cases(
        parent_dir=parent_dir,
        output_root=tmp_path / "artifacts",
        provider="mock",
        model_name="mock-engllm",
        temperature=0.2,
    )
    second_cases = prepare_real_history_docs_benchmark_cases(
        parent_dir=parent_dir,
        output_root=tmp_path / "artifacts",
        provider="mock",
        model_name="mock-engllm",
        temperature=0.2,
    )

    assert [case.manifest.case_id for case in first_cases] == _REAL_REPOS
    assert [case.manifest.case_id for case in second_cases] == _REAL_REPOS
    assert first_cases[0].manifest.project_config.llm.provider == "mock"
    assert first_cases[0].manifest.project_config.sources.roots == [Path("src")]
    learn_blazor_case = next(
        case for case in first_cases if case.manifest.case_id == "learn-blazor"
    )
    assert learn_blazor_case.manifest.previous_checkpoint_commit is not None


def test_prepare_real_history_docs_benchmark_cases_fails_fast_for_missing_repo(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "siblings"
    parent_dir.mkdir()

    try:
        prepare_real_history_docs_benchmark_cases(
            parent_dir=parent_dir,
            output_root=tmp_path / "artifacts",
            repo_names=["kleuw"],
        )
    except ValueError as exc:
        assert "kleuw" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected missing sibling repo preparation to fail")


def test_run_history_docs_benchmark_suite_continues_when_one_variant_build_fails(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = benchmark.build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )

    def _fail(_case: benchmark.PreparedHistoryDocsBenchmarkCase, _workspace_id: str):
        raise RuntimeError("boom")

    suite_report = benchmark.run_history_docs_benchmark_suite(
        suite_id="continue-on-build-failure",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            benchmark.baseline_history_docs_benchmark_variant(),
            benchmark.HistoryDocsBenchmarkVariant(variant_id="broken", runner=_fail),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={HistoryDocsQualityJudgmentEnvelope.__name__: _quality_payload(4)}
        ),
    )

    broken_report = HistoryDocsQualityReport.model_validate_json(
        benchmark.benchmark_quality_report_path(
            output_root,
            "continue-on-build-failure",
            "small_linear",
            "broken",
        ).read_text(encoding="utf-8")
    )
    assert suite_report.variant_ids == ["baseline", "broken"]
    assert broken_report.build_failed is True
    assert broken_report.evaluation_status == "llm_failed"


def test_promotion_gate_report_is_deterministic_and_uses_three_way_comparisons(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    suite_id = "real-shadow"
    parent_dir = tmp_path / "siblings"
    parent_dir.mkdir()
    for repo_name in _REAL_REPOS:
        _create_real_repo(parent_dir, repo_name)

    cases = prepare_real_history_docs_benchmark_cases(
        parent_dir=parent_dir,
        output_root=output_root,
        provider="mock",
        model_name="mock-engllm",
        temperature=0.2,
    )

    assert build_promotion_gate_report_path(
        output_root, suite_id
    ) == promotion_gate_report_path(
        output_root,
        suite_id,
    )

    for case in cases:
        for variant_id, score in (
            ("baseline", 3),
            ("semantic-clustering", 4),
            ("semantic-structure-context", 5),
        ):
            report = HistoryDocsQualityReport(
                case_id=case.manifest.case_id,
                variant_id=variant_id,
                checkpoint_id=f"checkpoint::{case.manifest.case_id}::{variant_id}",
                build_failed=False,
                evaluation_status="scored",
                validation_error_count=0,
                validation_warning_count=0,
                rubric_scores=_rubric_scores(score),
                overall_score=float(score),
                strengths=["Synthetic pass."],
                weaknesses=[],
                unsupported_claim_risks=[] if variant_id != "baseline" else ["minor"],
                tbd_overuse=False,
                evaluator_notes=[],
                uncertainty=[],
            )
            path = benchmark.benchmark_quality_report_path(
                output_root,
                suite_id,
                case.manifest.case_id,
                variant_id,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        comparison = HistoryDocsBenchmarkCaseComparisonReport(
            case_id=case.manifest.case_id,
            baseline_variant_id="baseline",
            quality_report_paths={},
            comparisons=[
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="baseline",
                    candidate_variant_id="semantic-clustering",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=3,
                            candidate_score=4,
                            delta=1,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=1.0,
                    preferred_variant_id="semantic-clustering",
                ),
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="baseline",
                    candidate_variant_id="semantic-structure-context",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=3,
                            candidate_score=5,
                            delta=2,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=2.0,
                    preferred_variant_id="semantic-structure-context",
                ),
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="semantic-clustering",
                    candidate_variant_id="semantic-structure-context",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=4,
                            candidate_score=5,
                            delta=1,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=1.0,
                    preferred_variant_id="semantic-structure-context",
                ),
            ],
        )
        comparison_path = benchmark.benchmark_comparison_report_path(
            output_root,
            suite_id,
            case.manifest.case_id,
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            comparison.model_dump_json(indent=2),
            encoding="utf-8",
        )

    gate_report = build_promotion_gate_report(
        output_root=output_root,
        suite_id=suite_id,
        cases=cases,
        provider="ollama",
        model_name="qwen2.5:14b-instruct-q4_K_M",
        temperature=0.2,
    )

    assert gate_report.average_score_by_variant == {
        "baseline": 3.0,
        "semantic-clustering": 4.0,
        "semantic-structure-context": 5.0,
    }
    assert gate_report.win_counts_by_variant["semantic-clustering"] == 5
    assert gate_report.win_counts_by_variant["semantic-structure-context"] == 5
    assert all(verdict.passed for verdict in gate_report.gate_verdicts)


def test_promotion_gate_report_records_failure_reasons(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    suite_id = "real-shadow-failures"
    parent_dir = tmp_path / "siblings"
    parent_dir.mkdir()
    for repo_name in _REAL_REPOS:
        _create_real_repo(parent_dir, repo_name)

    cases = prepare_real_history_docs_benchmark_cases(
        parent_dir=parent_dir,
        output_root=output_root,
        provider="mock",
        model_name="mock-engllm",
        temperature=0.2,
    )

    for case in cases:
        reports = [
            HistoryDocsQualityReport(
                case_id=case.manifest.case_id,
                variant_id="baseline",
                checkpoint_id=f"checkpoint::{case.manifest.case_id}::baseline",
                build_failed=False,
                evaluation_status="scored",
                validation_error_count=0,
                validation_warning_count=0,
                rubric_scores=_rubric_scores(4),
                overall_score=4.0,
                strengths=["Baseline stable."],
                weaknesses=[],
                unsupported_claim_risks=[],
                tbd_overuse=False,
            ),
            HistoryDocsQualityReport(
                case_id=case.manifest.case_id,
                variant_id="semantic-clustering",
                checkpoint_id=f"checkpoint::{case.manifest.case_id}::semantic",
                build_failed=False,
                evaluation_status="llm_failed",
                validation_error_count=0,
                validation_warning_count=0,
                rubric_scores=_rubric_scores(3),
                overall_score=3.0,
                strengths=[],
                weaknesses=["Semantic clustering failed to score cleanly."],
                unsupported_claim_risks=["extra risk"],
                tbd_overuse=False,
            ),
            HistoryDocsQualityReport(
                case_id=case.manifest.case_id,
                variant_id="semantic-structure-context",
                checkpoint_id=f"checkpoint::{case.manifest.case_id}::context",
                build_failed=False,
                evaluation_status="scored",
                validation_error_count=1,
                validation_warning_count=0,
                rubric_scores=[
                    HistoryDocsRubricScore(
                        dimension=dimension,
                        score=2,
                        rationale=f"{dimension} regressed.",
                    )
                    for dimension in _DIMENSIONS
                ],
                overall_score=2.0,
                strengths=[],
                weaknesses=["Context variant underperformed."],
                unsupported_claim_risks=[],
                tbd_overuse=False,
            ),
        ]
        for report in reports:
            quality_path = benchmark.benchmark_quality_report_path(
                output_root,
                suite_id,
                case.manifest.case_id,
                report.variant_id,
            )
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            quality_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

        comparison = HistoryDocsBenchmarkCaseComparisonReport(
            case_id=case.manifest.case_id,
            baseline_variant_id="baseline",
            quality_report_paths={},
            comparisons=[
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="baseline",
                    candidate_variant_id="semantic-clustering",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=4,
                            candidate_score=3,
                            delta=-1,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=-1.0,
                    preferred_variant_id="baseline",
                ),
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="baseline",
                    candidate_variant_id="semantic-structure-context",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=4,
                            candidate_score=2,
                            delta=-2,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=-2.0,
                    preferred_variant_id="baseline",
                ),
                HistoryDocsVariantComparison(
                    case_id=case.manifest.case_id,
                    baseline_variant_id="semantic-clustering",
                    candidate_variant_id="semantic-structure-context",
                    per_dimension_deltas=[
                        HistoryDocsRubricDelta(
                            dimension=dimension,
                            baseline_score=3,
                            candidate_score=2,
                            delta=-1,
                        )
                        for dimension in _DIMENSIONS
                    ],
                    overall_delta=-1.0,
                    preferred_variant_id="semantic-clustering",
                ),
            ],
        )
        comparison_path = benchmark.benchmark_comparison_report_path(
            output_root,
            suite_id,
            case.manifest.case_id,
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            comparison.model_dump_json(indent=2),
            encoding="utf-8",
        )

    gate_report = build_promotion_gate_report(
        output_root=output_root,
        suite_id=suite_id,
        cases=cases,
        provider="mock",
        model_name="mock-engllm",
        temperature=0.2,
    )

    semantic_verdict = gate_report.gate_verdicts[0]
    context_verdict = gate_report.gate_verdicts[1]

    assert semantic_verdict.passed is False
    assert any("below the 0.25 gate" in reason for reason in semantic_verdict.reasons)
    assert any(
        "won 0/5 baseline comparisons" in reason for reason in semantic_verdict.reasons
    )
    assert any(
        "failed builds/evaluations" in reason for reason in semantic_verdict.reasons
    )
    assert any(
        "unsupported-claim risk" in reason for reason in semantic_verdict.reasons
    )

    assert context_verdict.passed is False
    assert any(
        "did not exceed baseline average score" in reason
        for reason in context_verdict.reasons
    )
    assert any(
        "won 0/5 baseline comparisons" in reason for reason in context_verdict.reasons
    )
    assert any(
        "produced validation errors" in reason for reason in context_verdict.reasons
    )
    assert any(
        "worse than semantic clustering" in reason for reason in context_verdict.reasons
    )
