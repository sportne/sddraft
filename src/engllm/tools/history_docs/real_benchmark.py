"""Internal real-repo benchmark runner for history-docs shadow variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from engllm.core.render.json_artifacts import write_json_model
from engllm.core.repo.history import (
    get_commit_metadata,
    get_commit_parents,
    iter_first_parent_commits,
    list_reachable_tags_by_commit,
)
from engllm.domain.models import (
    LLMConfig,
    ProjectConfig,
    SDDToolDefaults,
    SourcesConfig,
    ToolDefaults,
    WorkspaceConfig,
)
from engllm.llm.factory import create_llm_client
from engllm.tools.history_docs.benchmark import (
    PreparedHistoryDocsBenchmarkCase,
    baseline_history_docs_benchmark_variant,
    benchmark_comparison_report_path,
    benchmark_quality_report_path,
    run_history_docs_benchmark_suite,
    semantic_history_docs_benchmark_variant,
    semantic_structure_context_benchmark_variant,
)
from engllm.tools.history_docs.models import (
    HistoryDocsBenchmarkCase,
    HistoryDocsBenchmarkCaseComparisonReport,
    HistoryDocsBenchmarkExpectation,
    HistoryDocsBenchmarkFocusTag,
    HistoryDocsPromotionGateReport,
    HistoryDocsPromotionGateVerdict,
    HistoryDocsQualityReport,
)

_DEFAULT_REAL_REPOS = [
    "kleuw",
    "python-sonarqube-api",
    "aoa-agent-demo",
    "learn-blazor",
    "cpp-helper-libs",
]
_DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_K_M"
_DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class _RealRepoSpec:
    repo_name: str
    source_roots: tuple[str, ...]
    title: str
    description: str
    focus_tags: tuple[HistoryDocsBenchmarkFocusTag, ...]
    expectations: tuple[HistoryDocsBenchmarkExpectation, ...]


_REAL_REPO_SPECS: dict[str, _RealRepoSpec] = {
    "kleuw": _RealRepoSpec(
        repo_name="kleuw",
        source_roots=("src",),
        title="Kleuw",
        description="A sibling repository used to test semantic subsystem naming and interface clarity across a Python source tree.",
        focus_tags=("medium",),
        expectations=(
            HistoryDocsBenchmarkExpectation(
                expectation_id="kleuw-architecture",
                kind="architectural_distinction",
                description="The document should surface meaningful subsystem distinctions rather than only directory-shape labels.",
                keywords=["subsystem", "service", "client", "ui"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="kleuw-interfaces",
                kind="interface_distinction",
                description="The document should distinguish external or boundary-facing surfaces when the evidence supports it.",
                required_section_ids=["interfaces"],
                keywords=["interface", "api", "cli", "surface"],
            ),
        ),
    ),
    "python-sonarqube-api": _RealRepoSpec(
        repo_name="python-sonarqube-api",
        source_roots=("src",),
        title="python-sonarqube-api",
        description="A sibling repository that should reward library-interface and external-system boundary clarity.",
        focus_tags=("dependency-heavy",),
        expectations=(
            HistoryDocsBenchmarkExpectation(
                expectation_id="sonarqube-dependencies",
                kind="dependency_understanding",
                description="The document should explain core dependency roles without inventing unsupported integration details.",
                keywords=["sonarqube", "api", "client", "http"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="sonarqube-context",
                kind="system_context_signal",
                description="The document should describe the system boundary and adjacent external context when evidence exists.",
                required_section_ids=["system_context"],
                keywords=["system", "external", "context"],
            ),
        ),
    ),
    "aoa-agent-demo": _RealRepoSpec(
        repo_name="aoa-agent-demo",
        source_roots=("src",),
        title="aoa-agent-demo",
        description="A sibling repository used to test capability decomposition and agent-facing interface extraction.",
        focus_tags=("algorithm-heavy",),
        expectations=(
            HistoryDocsBenchmarkExpectation(
                expectation_id="aoa-architecture",
                kind="architectural_distinction",
                description="The document should distinguish major semantic responsibilities rather than flattening the repo into one cluster.",
                keywords=["agent", "workflow", "planner", "adapter"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="aoa-interfaces",
                kind="interface_distinction",
                description="The document should identify important interfaces or boundaries when evidence supports them.",
                required_section_ids=["interfaces"],
                keywords=["interface", "request", "response", "protocol"],
            ),
        ),
    ),
    "learn-blazor": _RealRepoSpec(
        repo_name="learn-blazor",
        source_roots=("src",),
        title="learn-blazor",
        description="A sibling repository used to test client/server/shared system-context clarity.",
        focus_tags=("small", "architecture-heavy"),
        expectations=(
            HistoryDocsBenchmarkExpectation(
                expectation_id="blazor-context",
                kind="system_context_signal",
                description="The document should describe the system boundary and neighboring context nodes for the Blazor application.",
                required_section_ids=["system_context"],
                keywords=["system", "client", "server", "shared"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="blazor-interfaces",
                kind="interface_distinction",
                description="The document should surface relevant UI or HTTP-facing interfaces when supported by the evidence.",
                required_section_ids=["interfaces"],
                keywords=["http", "api", "component", "interface"],
            ),
        ),
    ),
    "cpp-helper-libs": _RealRepoSpec(
        repo_name="cpp-helper-libs",
        source_roots=("libs",),
        title="cpp-helper-libs",
        description="A sibling repository used to test semantic clustering over many first-party helper libraries and algorithm-heavy surfaces.",
        focus_tags=("medium", "architecture-heavy"),
        expectations=(
            HistoryDocsBenchmarkExpectation(
                expectation_id="cpp-architecture",
                kind="architectural_distinction",
                description="The document should distinguish meaningful helper-library responsibilities instead of only mirroring directory names.",
                keywords=["library", "helper", "core", "algorithm"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="cpp-context",
                kind="system_context_signal",
                description="The document should describe system context clearly when the helper libraries expose strong boundary signals.",
                required_section_ids=["system_context"],
                keywords=["context", "runtime", "system"],
            ),
        ),
    ),
}


def promotion_gate_report_path(output_root: Path, suite_id: str) -> Path:
    """Return the promotion-gate report path for one real benchmark suite."""

    return (
        output_root
        / "workspaces"
        / f"history-docs-benchmark-{suite_id}"
        / "tools"
        / "history_docs"
        / "benchmarks"
        / suite_id
        / "promotion_gate_report.json"
    )


def _default_project_config(
    output_root: Path,
    *,
    source_roots: tuple[str, ...],
    provider: str,
    model_name: str,
    temperature: float,
) -> ProjectConfig:
    return ProjectConfig(
        project_name="History Docs Real Benchmark",
        workspace=WorkspaceConfig(output_root=output_root),
        sources=SourcesConfig(roots=[Path(root) for root in source_roots]),
        tools=ToolDefaults(
            sdd=SDDToolDefaults(template=Path("unused-template.yaml")),
        ),
        llm=LLMConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
        ),
    )


def _select_previous_checkpoint_commit(
    repo_root: Path, target_commit: str
) -> str | None:
    commits = iter_first_parent_commits(repo_root, target_commit=target_commit)
    if len(commits) <= 1:
        return None
    ancestor_commits = commits[:-1]
    tags_by_commit = list_reachable_tags_by_commit(
        repo_root,
        target_commit=target_commit,
        commit_shas=[commit.sha for commit in ancestor_commits],
    )
    for commit in reversed(ancestor_commits):
        if tags_by_commit.get(commit.sha):
            return commit.sha
    for commit in reversed(ancestor_commits):
        if len(get_commit_parents(repo_root, commit.sha)) > 1:
            return commit.sha
    index = max(0, len(commits) - 21)
    return commits[index].sha if index < len(commits) - 1 else commits[0].sha


def prepare_real_history_docs_benchmark_cases(
    *,
    parent_dir: Path,
    output_root: Path,
    repo_names: list[str] | None = None,
    provider: str = "ollama",
    model_name: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> list[PreparedHistoryDocsBenchmarkCase]:
    """Prepare H10 benchmark cases for sibling repositories."""

    selected_names = repo_names or list(_DEFAULT_REAL_REPOS)
    cases: list[PreparedHistoryDocsBenchmarkCase] = []
    for repo_name in selected_names:
        spec = _REAL_REPO_SPECS.get(repo_name)
        if spec is None:
            raise ValueError(f"Unsupported real benchmark repo {repo_name!r}")
        repo_root = (parent_dir / repo_name).resolve()
        if not repo_root.exists() or not repo_root.is_dir():
            raise ValueError(
                f"Sibling repository {repo_name!r} was not found under {parent_dir}"
            )
        if not (repo_root / ".git").exists():
            raise ValueError(f"Sibling path {repo_root} is not a git repository")
        target_commit = get_commit_metadata(repo_root, "HEAD").sha
        previous_checkpoint_commit = _select_previous_checkpoint_commit(
            repo_root,
            target_commit,
        )
        manifest = HistoryDocsBenchmarkCase(
            case_id=repo_name,
            title=spec.title,
            description=spec.description,
            focus_tags=list(spec.focus_tags),
            builder_name=f"real_repo::{repo_name}",
            project_config=_default_project_config(
                output_root,
                source_roots=spec.source_roots,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
            ),
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            expectations=list(spec.expectations),
        )
        cases.append(
            PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)
        )
    return cases


def _load_quality_report(path: Path) -> HistoryDocsQualityReport:
    return HistoryDocsQualityReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _load_comparison_report(path: Path) -> HistoryDocsBenchmarkCaseComparisonReport:
    return HistoryDocsBenchmarkCaseComparisonReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _average_dimension_score(
    reports: list[HistoryDocsQualityReport],
    dimension: str,
) -> float:
    values = [
        next(
            score.score
            for score in report.rubric_scores
            if score.dimension == dimension
        )
        for report in reports
    ]
    return round(fmean(values), 3) if values else 0.0


def build_promotion_gate_report(
    *,
    output_root: Path,
    suite_id: str,
    cases: list[PreparedHistoryDocsBenchmarkCase],
    provider: str,
    model_name: str,
    temperature: float,
) -> HistoryDocsPromotionGateReport:
    """Build a deterministic promotion-gate report from suite artifacts."""

    variant_ids = ["baseline", "semantic-clustering", "semantic-structure-context"]
    quality_reports_by_variant: dict[str, list[HistoryDocsQualityReport]] = {
        variant_id: [] for variant_id in variant_ids
    }
    win_counts_by_variant = {variant_id: 0 for variant_id in variant_ids}

    for case in cases:
        for variant_id in variant_ids:
            quality_reports_by_variant[variant_id].append(
                _load_quality_report(
                    benchmark_quality_report_path(
                        output_root,
                        suite_id,
                        case.manifest.case_id,
                        variant_id,
                    )
                )
            )
        comparison_report = _load_comparison_report(
            benchmark_comparison_report_path(
                output_root,
                suite_id,
                case.manifest.case_id,
            )
        )
        for comparison in comparison_report.comparisons:
            if comparison.baseline_variant_id != "baseline":
                continue
            win_counts_by_variant[comparison.preferred_variant_id] += 1

    average_score_by_variant = {
        variant_id: (
            round(
                fmean(report.overall_score for report in reports),
                3,
            )
            if reports
            else 0.0
        )
        for variant_id, reports in quality_reports_by_variant.items()
    }
    unsupported_claim_risk_totals = {
        variant_id: sum(len(report.unsupported_claim_risks) for report in reports)
        for variant_id, reports in quality_reports_by_variant.items()
    }
    failed_build_or_evaluation_count_by_variant = {
        variant_id: sum(
            report.build_failed or report.evaluation_status != "scored"
            for report in reports
        )
        for variant_id, reports in quality_reports_by_variant.items()
    }
    validation_error_count_by_variant = {
        variant_id: sum(report.validation_error_count for report in reports)
        for variant_id, reports in quality_reports_by_variant.items()
    }

    semantic_reports = quality_reports_by_variant["semantic-clustering"]
    context_reports = quality_reports_by_variant["semantic-structure-context"]

    semantic_wins_against_baseline = 0
    context_wins_against_baseline = 0
    context_wins_against_semantic = 0
    for case in cases:
        comparison_report = _load_comparison_report(
            benchmark_comparison_report_path(
                output_root,
                suite_id,
                case.manifest.case_id,
            )
        )
        for comparison in comparison_report.comparisons:
            pair = (comparison.baseline_variant_id, comparison.candidate_variant_id)
            if pair == ("baseline", "semantic-clustering"):
                semantic_wins_against_baseline += (
                    comparison.preferred_variant_id == "semantic-clustering"
                )
            elif pair == ("baseline", "semantic-structure-context"):
                context_wins_against_baseline += (
                    comparison.preferred_variant_id == "semantic-structure-context"
                )
            elif pair == ("semantic-clustering", "semantic-structure-context"):
                context_wins_against_semantic += (
                    comparison.preferred_variant_id == "semantic-structure-context"
                )

    semantic_reasons: list[str] = []
    semantic_pass = True
    semantic_delta = round(
        average_score_by_variant["semantic-clustering"]
        - average_score_by_variant["baseline"],
        3,
    )
    if semantic_delta < 0.25:
        semantic_pass = False
        semantic_reasons.append(
            f"Average score delta vs baseline was {semantic_delta}, below the 0.25 gate."
        )
    if semantic_wins_against_baseline < 3:
        semantic_pass = False
        semantic_reasons.append(
            f"Semantic clustering won {semantic_wins_against_baseline}/5 baseline comparisons."
        )
    if (
        failed_build_or_evaluation_count_by_variant["semantic-clustering"]
        > failed_build_or_evaluation_count_by_variant["baseline"]
    ):
        semantic_pass = False
        semantic_reasons.append(
            "Semantic clustering increased failed builds/evaluations relative to baseline."
        )
    if (
        unsupported_claim_risk_totals["semantic-clustering"]
        > unsupported_claim_risk_totals["baseline"]
    ):
        semantic_pass = False
        semantic_reasons.append(
            "Semantic clustering increased unsupported-claim risk relative to baseline."
        )
    if semantic_pass:
        semantic_reasons.append(
            "Semantic clustering met all promotion-gate thresholds."
        )

    context_reasons: list[str] = []
    context_pass = True
    context_delta_vs_semantic = round(
        average_score_by_variant["semantic-structure-context"]
        - average_score_by_variant["semantic-clustering"],
        3,
    )
    if context_delta_vs_semantic < 0.25:
        context_pass = False
        context_reasons.append(
            f"Average score delta vs semantic clustering was {context_delta_vs_semantic}, below the 0.25 gate."
        )
    if (
        average_score_by_variant["semantic-structure-context"]
        <= average_score_by_variant["baseline"]
    ):
        context_pass = False
        context_reasons.append(
            "Semantic structure plus context did not exceed baseline average score."
        )
    if context_wins_against_baseline < 3:
        context_pass = False
        context_reasons.append(
            f"Semantic structure plus context won {context_wins_against_baseline}/5 baseline comparisons."
        )
    if validation_error_count_by_variant["semantic-structure-context"] > 0:
        context_pass = False
        context_reasons.append(
            "Semantic structure plus context produced validation errors."
        )
    for dimension in ("coverage", "coherence", "present_state_tone"):
        if _average_dimension_score(
            context_reports, dimension
        ) < _average_dimension_score(
            semantic_reports,
            dimension,
        ):
            context_pass = False
            context_reasons.append(
                f"Average {dimension} score was worse than semantic clustering."
            )
    if context_wins_against_semantic < 3:
        context_pass = False
        context_reasons.append(
            f"Semantic structure plus context won {context_wins_against_semantic}/5 comparisons against semantic clustering."
        )
    if context_pass:
        context_reasons.append(
            "Semantic structure plus context met all promotion-gate thresholds."
        )

    return HistoryDocsPromotionGateReport(
        suite_id=suite_id,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        repo_case_ids=[case.manifest.case_id for case in cases],
        average_score_by_variant=average_score_by_variant,
        win_counts_by_variant=win_counts_by_variant,
        unsupported_claim_risk_totals=unsupported_claim_risk_totals,
        failed_build_or_evaluation_count_by_variant=failed_build_or_evaluation_count_by_variant,
        validation_error_count_by_variant=validation_error_count_by_variant,
        gate_verdicts=[
            HistoryDocsPromotionGateVerdict(
                phase_id="H11-02",
                candidate_variant_id="semantic-clustering",
                baseline_variant_ids=["baseline"],
                passed=semantic_pass,
                reasons=semantic_reasons,
            ),
            HistoryDocsPromotionGateVerdict(
                phase_id="H11-03",
                candidate_variant_id="semantic-structure-context",
                baseline_variant_ids=["baseline", "semantic-clustering"],
                passed=context_pass,
                reasons=context_reasons,
            ),
        ],
    )


def run_real_repo_history_docs_benchmark_suite(
    *,
    parent_dir: Path,
    output_root: Path,
    suite_id: str,
    repo_names: list[str] | None = None,
    provider: str = "ollama",
    model_name: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> HistoryDocsPromotionGateReport:
    """Run the internal real-repo history-docs benchmark suite."""

    cases = prepare_real_history_docs_benchmark_cases(
        parent_dir=parent_dir,
        output_root=output_root,
        repo_names=repo_names,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )
    run_history_docs_benchmark_suite(
        suite_id=suite_id,
        output_root=output_root,
        cases=cases,
        variant_runners=[
            baseline_history_docs_benchmark_variant(),
            semantic_history_docs_benchmark_variant(),
            semantic_structure_context_benchmark_variant(),
        ],
        llm_client_factory=create_llm_client,
        progress_callback=lambda message: print(message, flush=True),
    )
    gate_report = build_promotion_gate_report(
        output_root=output_root,
        suite_id=suite_id,
        cases=cases,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
    )
    write_json_model(promotion_gate_report_path(output_root, suite_id), gate_report)
    return gate_report


def main(argv: list[str] | None = None) -> int:
    """Run the internal real-repo benchmark suite via ``python -m``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=Path(".."))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--suite-id", default="real-repo-ollama-shadow")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        default=None,
        help="Sibling repository name to include. May be repeated.",
    )
    args = parser.parse_args(argv)

    gate_report = run_real_repo_history_docs_benchmark_suite(
        parent_dir=args.parent_dir.resolve(),
        output_root=args.output_root.resolve(),
        suite_id=args.suite_id,
        repo_names=args.repos,
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
    )
    print(
        f"Promotion gate report: {promotion_gate_report_path(args.output_root.resolve(), args.suite_id)}",
        flush=True,
    )
    for verdict in gate_report.gate_verdicts:
        print(
            f"{verdict.phase_id} {verdict.candidate_variant_id}: {'PASS' if verdict.passed else 'FAIL'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
