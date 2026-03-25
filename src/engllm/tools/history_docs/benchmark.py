"""Internal benchmark and evaluation harness for history-docs."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import cast

from engllm.core.render.json_artifacts import write_json_model
from engllm.core.workspaces import build_workspace_context, tool_artifact_root
from engllm.domain.models import (
    LLMConfig,
    ProjectConfig,
    SDDToolDefaults,
    SourcesConfig,
    ToolDefaults,
    WorkspaceConfig,
)
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.llm.factory import create_llm_client
from engllm.prompts.history_docs.builders import build_history_docs_quality_judge_prompt
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryBuildResult,
    HistoryCheckpointModel,
    HistoryDocsBenchmarkCase,
    HistoryDocsBenchmarkCaseComparisonReport,
    HistoryDocsBenchmarkCaseReportRef,
    HistoryDocsBenchmarkExpectation,
    HistoryDocsBenchmarkFocusTag,
    HistoryDocsBenchmarkSuiteReport,
    HistoryDocsQualityJudgment,
    HistoryDocsQualityJudgmentEnvelope,
    HistoryDocsQualityReport,
    HistoryDocsRubricDelta,
    HistoryDocsRubricDimension,
    HistoryDocsRubricScore,
    HistoryDocsVariantComparison,
    HistoryRenderManifest,
    HistorySectionPlanId,
    HistorySemanticContextMap,
    HistoryValidationReport,
)

ProgressCallback = Callable[[str], None]
HistoryDocsBenchmarkVariantRunner = Callable[
    ["PreparedHistoryDocsBenchmarkCase", str],
    HistoryBuildResult,
]
_REQUIRED_FOCUS_TAGS = {
    "small",
    "medium",
    "algorithm-heavy",
    "dependency-heavy",
    "architecture-heavy",
}
_RUBRIC_DIMENSIONS: tuple[HistoryDocsRubricDimension, ...] = (
    "coverage",
    "coherence",
    "specificity",
    "algorithm_understanding",
    "dependency_understanding",
    "rationale_capture",
    "present_state_tone",
)
_VALID_SECTION_IDS = {
    "introduction",
    "architectural_overview",
    "system_context",
    "subsystems_modules",
    "interfaces",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
}


@dataclass(frozen=True)
class PreparedHistoryDocsBenchmarkCase:
    """Prepared benchmark case with a resolved repo and case manifest."""

    manifest: HistoryDocsBenchmarkCase
    repo_root: Path


@dataclass(frozen=True)
class HistoryDocsBenchmarkVariant:
    """One variant runner plugged into the H10 suite."""

    variant_id: str
    runner: HistoryDocsBenchmarkVariantRunner


@dataclass(frozen=True)
class _LoadedBenchmarkArtifacts:
    """Loaded render-time artifacts for one quality evaluation."""

    checkpoint_model: HistoryCheckpointModel
    render_manifest: HistoryRenderManifest
    validation_report: HistoryValidationReport
    markdown: str
    semantic_context_map: HistorySemanticContextMap | None = None


def _progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def benchmark_suite_workspace_id(suite_id: str) -> str:
    """Return the dedicated workspace id for one benchmark suite."""

    return f"history-docs-benchmark-{suite_id}"


def benchmark_suite_root(output_root: Path, suite_id: str) -> Path:
    """Return the suite artifact root for one benchmark run."""

    context = build_workspace_context(
        output_root=output_root,
        workspace_id=benchmark_suite_workspace_id(suite_id),
        kind="benchmark",
        repo_root=output_root,
    )
    return tool_artifact_root(context, "history_docs") / "benchmarks" / suite_id


def benchmark_suite_manifest_path(output_root: Path, suite_id: str) -> Path:
    """Return the top-level suite manifest path."""

    return benchmark_suite_root(output_root, suite_id) / "suite_manifest.json"


def benchmark_case_root(output_root: Path, suite_id: str, case_id: str) -> Path:
    """Return the artifact root for one benchmark case."""

    return benchmark_suite_root(output_root, suite_id) / "cases" / case_id


def benchmark_case_manifest_path(
    output_root: Path, suite_id: str, case_id: str
) -> Path:
    """Return the persisted case manifest path for one benchmark case."""

    return benchmark_case_root(output_root, suite_id, case_id) / "case_manifest.json"


def benchmark_quality_report_path(
    output_root: Path,
    suite_id: str,
    case_id: str,
    variant_id: str,
) -> Path:
    """Return the per-variant quality report path."""

    return (
        benchmark_case_root(output_root, suite_id, case_id)
        / variant_id
        / "quality_report.json"
    )


def benchmark_comparison_report_path(
    output_root: Path,
    suite_id: str,
    case_id: str,
) -> Path:
    """Return the per-case comparison report path."""

    return (
        benchmark_case_root(output_root, suite_id, case_id) / "comparison_report.json"
    )


def _default_project_config(
    output_root: Path, *, source_roots: list[str]
) -> ProjectConfig:
    return ProjectConfig(
        project_name="History Docs Benchmark",
        workspace=WorkspaceConfig(output_root=output_root),
        sources=SourcesConfig(roots=[Path(root) for root in source_roots]),
        tools=ToolDefaults(
            sdd=SDDToolDefaults(template=Path("unused-template.yaml")),
        ),
        llm=LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2),
    )


def _git(
    repo_root: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> str:
    full_env = os.environ.copy()
    full_env.setdefault("GIT_AUTHOR_NAME", "EngLLM Benchmark")
    full_env.setdefault("GIT_AUTHOR_EMAIL", "engllm@example.com")
    full_env.setdefault("GIT_COMMITTER_NAME", "EngLLM Benchmark")
    full_env.setdefault("GIT_COMMITTER_EMAIL", "engllm@example.com")
    if env is not None:
        full_env.update(env)
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=full_env,
    )
    return result.stdout.strip()


def _init_repo(base_root: Path, case_id: str) -> Path:
    repo_root = base_root / case_id
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init")
    return repo_root


def _commit_file(
    repo_root: Path,
    relative_path: str,
    content: str,
    *,
    message: str,
    timestamp: str,
) -> str:
    file_path = repo_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    _git(repo_root, "add", relative_path)
    _git(
        repo_root,
        "commit",
        "-m",
        message,
        env={
            "GIT_AUTHOR_DATE": timestamp,
            "GIT_COMMITTER_DATE": timestamp,
        },
    )
    return _git(repo_root, "rev-parse", "HEAD")


def _prepare_small_linear_case(
    base_root: Path, output_root: Path
) -> PreparedHistoryDocsBenchmarkCase:
    repo_root = _init_repo(base_root, "small_linear")
    first = _commit_file(
        repo_root,
        "pyproject.toml",
        '[project]\nname = "small-linear"\ndependencies = ["requests>=2"]\n',
        message="add project metadata",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    _commit_file(
        repo_root,
        "src/app.py",
        "import requests\n\n\ndef load_state() -> str:\n    return requests.__name__\n",
        message="add app loader",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    target = _commit_file(
        repo_root,
        "src/app.py",
        "import requests\n\n\ndef load_state() -> str:\n    return requests.__name__\n\n\ndef validate_request(value: str) -> bool:\n    return bool(value)\n",
        message="add validation step",
        timestamp="2024-03-01T10:00:00+00:00",
    )
    manifest = HistoryDocsBenchmarkCase(
        case_id="small_linear",
        title="Small Linear Python Service",
        description="A compact linear history with one Python subsystem and one direct runtime dependency.",
        focus_tags=["small"],
        builder_name="small_linear",
        project_config=_default_project_config(output_root, source_roots=["src"]),
        target_commit=target,
        previous_checkpoint_commit=first,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="small-sections",
                kind="section_presence",
                description="The document should cover the core architectural and dependency sections.",
                required_section_ids=[
                    "introduction",
                    "architectural_overview",
                    "dependencies",
                ],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="small-dependency",
                kind="dependency_understanding",
                description="The dependency discussion should mention the requests HTTP client dependency without inventing extra behavior.",
                keywords=["requests"],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="small-tone",
                kind="present_state_tone",
                description="The write-up should read as a standalone present-state document rather than a release note.",
            ),
        ],
    )
    return PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)


def _prepare_medium_mixed_case(
    base_root: Path, output_root: Path
) -> PreparedHistoryDocsBenchmarkCase:
    repo_root = _init_repo(base_root, "medium_mixed")
    _commit_file(
        repo_root,
        "package.json",
        '{"dependencies":{"lit":"^3.0.0"},"devDependencies":{"vitest":"^1.0.0"}}\n',
        message="add package metadata",
        timestamp="2024-01-05T10:00:00+00:00",
    )
    first = _commit_file(
        repo_root,
        "src/core/service.py",
        'def build_message(name: str) -> str:\n    return f"hello {name}"\n',
        message="add core service",
        timestamp="2024-02-05T10:00:00+00:00",
    )
    _commit_file(
        repo_root,
        "web/ui.js",
        "export function renderMessage(name) {\n  return `hello ${name}`;\n}\n",
        message="add ui renderer",
        timestamp="2024-03-05T10:00:00+00:00",
    )
    target = _commit_file(
        repo_root,
        "src/support/cache.py",
        "class SessionCache:\n    pass\n",
        message="add support cache",
        timestamp="2024-04-05T10:00:00+00:00",
    )
    manifest = HistoryDocsBenchmarkCase(
        case_id="medium_mixed",
        title="Medium Mixed Python and Web Workspace",
        description="A medium-sized mixed-language checkpoint with backend and web modules plus build metadata.",
        focus_tags=["medium"],
        builder_name="medium_mixed",
        project_config=_default_project_config(
            output_root, source_roots=["src", "web"]
        ),
        target_commit=target,
        previous_checkpoint_commit=first,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="medium-sections",
                kind="section_presence",
                description="The document should include architectural and subsystem coverage for the mixed workspace.",
                required_section_ids=[
                    "architectural_overview",
                    "subsystems_modules",
                ],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="medium-architecture",
                kind="architectural_distinction",
                description="The document should distinguish backend, web, and support responsibilities with meaningful subsystem naming rather than flattening them into one undifferentiated blob.",
                keywords=["core", "web", "support", "backend", "ui", "cache"],
            ),
        ],
    )
    return PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)


def _prepare_algorithm_heavy_case(
    base_root: Path, output_root: Path
) -> PreparedHistoryDocsBenchmarkCase:
    repo_root = _init_repo(base_root, "algorithm_heavy_variants")
    base = _commit_file(
        repo_root,
        "src/strategies/base.py",
        "class RequestContext:\n    pass\n",
        message="add strategy base",
        timestamp="2024-01-10T10:00:00+00:00",
    )
    _commit_file(
        repo_root,
        "src/strategies/http_adapter.py",
        '"""HTTP adapter must remain deterministic."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add http adapter",
        timestamp="2024-02-10T10:00:00+00:00",
    )
    target = _commit_file(
        repo_root,
        "src/strategies/grpc_adapter.py",
        '"""gRPC adapter only supports strict mode."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add grpc adapter",
        timestamp="2024-03-10T10:00:00+00:00",
    )
    manifest = HistoryDocsBenchmarkCase(
        case_id="algorithm_heavy_variants",
        title="Algorithm Heavy Variant Cluster",
        description="A strategy-heavy checkpoint where sibling adapters should surface as one algorithm family rather than isolated files.",
        focus_tags=["algorithm-heavy"],
        builder_name="algorithm_heavy_variants",
        project_config=_default_project_config(output_root, source_roots=["src"]),
        target_commit=target,
        previous_checkpoint_commit=base,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="algorithm-section",
                kind="section_presence",
                description="The rendered document should include algorithm-focused coverage when variant families are present.",
                required_section_ids=[
                    "algorithms_core_logic",
                    "strategy_variants_design_alternatives",
                ],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="algorithm-variants",
                kind="algorithm_signal",
                description="The document should recognize a variant family rather than describing the adapters as unrelated modules.",
                keywords=["adapter", "variant", "strict"],
            ),
        ],
    )
    return PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)


def _prepare_dependency_heavy_case(
    base_root: Path, output_root: Path
) -> PreparedHistoryDocsBenchmarkCase:
    repo_root = _init_repo(base_root, "dependency_heavy")
    _commit_file(
        repo_root,
        "pyproject.toml",
        '[project]\nname = "dependency-heavy"\ndependencies = ["requests>=2", "pydantic>=2"]\n[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
        message="add python deps",
        timestamp="2024-01-15T10:00:00+00:00",
    )
    first = _commit_file(
        repo_root,
        "package.json",
        '{"dependencies":{"react":"^18.0.0"},"devDependencies":{"eslint":"^9.0.0"}}\n',
        message="add web deps",
        timestamp="2024-02-15T10:00:00+00:00",
    )
    _commit_file(
        repo_root,
        "src/server.py",
        "import requests\nfrom pydantic import BaseModel\n\n\nclass RequestModel(BaseModel):\n    value: str\n\n\ndef fetch() -> str:\n    return requests.__name__\n",
        message="add python service",
        timestamp="2024-03-15T10:00:00+00:00",
    )
    target = _commit_file(
        repo_root,
        "web/app.js",
        'import React from "react";\n\nexport function App() {\n  return React.createElement("div", null, "demo");\n}\n',
        message="add web entry",
        timestamp="2024-04-15T10:00:00+00:00",
    )
    manifest = HistoryDocsBenchmarkCase(
        case_id="dependency_heavy",
        title="Dependency Heavy Multi-Manifest Project",
        description="A checkpoint with multiple manifest families and both runtime and development dependencies.",
        focus_tags=["dependency-heavy"],
        builder_name="dependency_heavy",
        project_config=_default_project_config(
            output_root, source_roots=["src", "web"]
        ),
        target_commit=target,
        previous_checkpoint_commit=first,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="dependency-sections",
                kind="section_presence",
                description="The document should clearly render dependency and build/development infrastructure coverage.",
                required_section_ids=[
                    "dependencies",
                    "build_development_infrastructure",
                ],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="dependency-understanding",
                kind="dependency_understanding",
                description="The dependency write-up should recognize both runtime libraries and development tooling without collapsing them together.",
                keywords=["requests", "pydantic", "react", "eslint"],
            ),
        ],
    )
    return PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)


def _prepare_architecture_heavy_case(
    base_root: Path, output_root: Path
) -> PreparedHistoryDocsBenchmarkCase:
    repo_root = _init_repo(base_root, "architecture_heavy")
    first = _commit_file(
        repo_root,
        "src/api/router.py",
        "def route_request(path: str) -> str:\n    return path\n",
        message="add api router",
        timestamp="2024-01-20T10:00:00+00:00",
    )
    _commit_file(
        repo_root,
        "src/engine/planner.py",
        'def build_plan() -> list[str]:\n    return ["plan"]\n',
        message="add planner engine",
        timestamp="2024-02-20T10:00:00+00:00",
    )
    target = _commit_file(
        repo_root,
        "src/storage/repository.py",
        "class StateRepository:\n    pass\n",
        message="add storage subsystem",
        timestamp="2024-03-20T10:00:00+00:00",
    )
    manifest = HistoryDocsBenchmarkCase(
        case_id="architecture_heavy",
        title="Architecture Heavy Multi-Subsystem Project",
        description="A checkpoint with clearly separated API, engine, and storage areas that should surface as distinct architectural units.",
        focus_tags=["architecture-heavy"],
        builder_name="architecture_heavy",
        project_config=_default_project_config(output_root, source_roots=["src"]),
        target_commit=target,
        previous_checkpoint_commit=first,
        expectations=[
            HistoryDocsBenchmarkExpectation(
                expectation_id="architecture-sections",
                kind="section_presence",
                description="The document should spend real attention on architecture and subsystem structure.",
                required_section_ids=[
                    "architectural_overview",
                    "subsystems_modules",
                ],
            ),
            HistoryDocsBenchmarkExpectation(
                expectation_id="architecture-distinction",
                kind="architectural_distinction",
                description="The document should distinguish API, engine, and storage roles with clear subsystem or capability framing instead of merging them into generic modules.",
                keywords=[
                    "api",
                    "engine",
                    "storage",
                    "router",
                    "planner",
                    "repository",
                ],
            ),
        ],
    )
    return PreparedHistoryDocsBenchmarkCase(manifest=manifest, repo_root=repo_root)


_CASE_BUILDERS: dict[
    str,
    Callable[[Path, Path], PreparedHistoryDocsBenchmarkCase],
] = {
    "small_linear": _prepare_small_linear_case,
    "medium_mixed": _prepare_medium_mixed_case,
    "algorithm_heavy_variants": _prepare_algorithm_heavy_case,
    "dependency_heavy": _prepare_dependency_heavy_case,
    "architecture_heavy": _prepare_architecture_heavy_case,
}


def build_default_history_docs_benchmark_cases(
    *,
    base_root: Path,
    output_root: Path,
) -> list[PreparedHistoryDocsBenchmarkCase]:
    """Return the default H10 benchmark corpus with prepared fixture repos."""

    return [
        _CASE_BUILDERS[case_id](base_root, output_root)
        for case_id in (
            "small_linear",
            "medium_mixed",
            "algorithm_heavy_variants",
            "dependency_heavy",
            "architecture_heavy",
        )
    ]


def validate_benchmark_focus_coverage(
    cases: list[PreparedHistoryDocsBenchmarkCase],
) -> list[HistoryDocsBenchmarkFocusTag]:
    """Return the sorted focus coverage tags after enforcing the H10 minimum set."""

    covered = {tag for case in cases for tag in case.manifest.focus_tags}
    missing = sorted(_REQUIRED_FOCUS_TAGS - covered)
    if missing:
        raise ValueError(
            "Benchmark suite is missing required focus tags: " + ", ".join(missing)
        )
    return sorted(covered)


def baseline_history_docs_benchmark_variant() -> HistoryDocsBenchmarkVariant:
    """Return the baseline variant wired to the current history-docs build flow."""

    def _run(
        prepared_case: PreparedHistoryDocsBenchmarkCase,
        workspace_id: str,
    ) -> HistoryBuildResult:
        return build_history_docs_checkpoint(
            project_config=prepared_case.manifest.project_config,
            repo_root=prepared_case.repo_root,
            checkpoint_commit=prepared_case.manifest.target_commit,
            previous_checkpoint_commit=prepared_case.manifest.previous_checkpoint_commit,
            workspace_id=workspace_id,
            subsystem_grouping_mode="path",
            experimental_section_mode="default",
        )

    return HistoryDocsBenchmarkVariant(variant_id="baseline", runner=_run)


def semantic_history_docs_benchmark_variant(
    *,
    llm_client_builder: (
        Callable[
            [PreparedHistoryDocsBenchmarkCase],
            LLMClient | None,
        ]
        | None
    ) = None,
) -> HistoryDocsBenchmarkVariant:
    """Return the H11 semantic-clustering benchmark variant."""

    def _run(
        prepared_case: PreparedHistoryDocsBenchmarkCase,
        workspace_id: str,
    ) -> HistoryBuildResult:
        return build_history_docs_checkpoint(
            project_config=prepared_case.manifest.project_config,
            repo_root=prepared_case.repo_root,
            checkpoint_commit=prepared_case.manifest.target_commit,
            previous_checkpoint_commit=prepared_case.manifest.previous_checkpoint_commit,
            workspace_id=workspace_id,
            subsystem_grouping_mode="semantic",
            experimental_section_mode="default",
            llm_client_override=(
                None
                if llm_client_builder is None
                else llm_client_builder(prepared_case)
            ),
        )

    return HistoryDocsBenchmarkVariant(variant_id="semantic-clustering", runner=_run)


def semantic_structure_context_benchmark_variant(
    *,
    llm_client_builder: (
        Callable[
            [PreparedHistoryDocsBenchmarkCase],
            LLMClient | None,
        ]
        | None
    ) = None,
) -> HistoryDocsBenchmarkVariant:
    """Return the H11-03 semantic structure plus context benchmark variant."""

    def _run(
        prepared_case: PreparedHistoryDocsBenchmarkCase,
        workspace_id: str,
    ) -> HistoryBuildResult:
        return build_history_docs_checkpoint(
            project_config=prepared_case.manifest.project_config,
            repo_root=prepared_case.repo_root,
            checkpoint_commit=prepared_case.manifest.target_commit,
            previous_checkpoint_commit=prepared_case.manifest.previous_checkpoint_commit,
            workspace_id=workspace_id,
            subsystem_grouping_mode="semantic",
            experimental_section_mode="semantic_context",
            llm_client_override=(
                None
                if llm_client_builder is None
                else llm_client_builder(prepared_case)
            ),
        )

    return HistoryDocsBenchmarkVariant(
        variant_id="semantic-structure-context",
        runner=_run,
    )


def _empty_rubric_scores() -> list[HistoryDocsRubricScore]:
    return [
        HistoryDocsRubricScore(
            dimension=dimension,
            score=0,
            rationale="Evaluation did not complete.",
        )
        for dimension in _RUBRIC_DIMENSIONS
    ]


def _load_benchmark_artifacts(
    build_result: HistoryBuildResult,
) -> _LoadedBenchmarkArtifacts:
    checkpoint_model_path = getattr(build_result, "checkpoint_model_path", None)
    render_manifest_path = getattr(build_result, "render_manifest_path", None)
    validation_report_path = getattr(build_result, "validation_report_path", None)
    checkpoint_markdown_path = getattr(build_result, "checkpoint_markdown_path", None)
    if (
        checkpoint_model_path is None
        or render_manifest_path is None
        or validation_report_path is None
        or checkpoint_markdown_path is None
    ):
        raise ValueError(
            "History-docs benchmark evaluation requires H4-H9 artifact paths"
        )

    return _LoadedBenchmarkArtifacts(
        checkpoint_model=HistoryCheckpointModel.model_validate_json(
            Path(checkpoint_model_path).read_text(encoding="utf-8")
        ),
        render_manifest=HistoryRenderManifest.model_validate_json(
            Path(render_manifest_path).read_text(encoding="utf-8")
        ),
        validation_report=HistoryValidationReport.model_validate_json(
            Path(validation_report_path).read_text(encoding="utf-8")
        ),
        markdown=Path(checkpoint_markdown_path).read_text(encoding="utf-8"),
        semantic_context_map=(
            None
            if build_result.semantic_context_map_path is None
            else HistorySemanticContextMap.model_validate_json(
                Path(build_result.semantic_context_map_path).read_text(encoding="utf-8")
            )
        ),
    )


def _overall_score(scores: list[HistoryDocsRubricScore]) -> float:
    return round(fmean(score.score for score in scores), 3)


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _coerce_score_value(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, min(5, int(round(value))))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return max(0, min(5, int(stripped)))
        if stripped and stripped[0].isdigit():
            return max(0, min(5, int(stripped[0])))
    return None


def _coerce_rubric_score(
    dimension: HistoryDocsRubricDimension,
    payload: object,
) -> HistoryDocsRubricScore | None:
    if isinstance(payload, HistoryDocsRubricScore):
        return payload
    if isinstance(payload, dict):
        score = _coerce_score_value(
            payload.get("score", payload.get("rating", payload.get("value")))
        )
        if score is None:
            return None
        rationale = str(
            payload.get("rationale")
            or payload.get("reason")
            or payload.get("summary")
            or payload.get("notes")
            or "No rationale provided."
        ).strip()
        cited_section_ids = cast(
            list[HistorySectionPlanId],
            [
                section_id
                for section_id in _coerce_string_list(
                    payload.get("cited_section_ids")
                    or payload.get("section_ids")
                    or payload.get("cited_sections")
                )
                if section_id in _VALID_SECTION_IDS
            ],
        )
        return HistoryDocsRubricScore(
            dimension=dimension,
            score=score,
            rationale=rationale or "No rationale provided.",
            matched_expectation_ids=_coerce_string_list(
                payload.get("matched_expectation_ids")
                or payload.get("expectation_ids")
                or payload.get("matched_expectations")
            ),
            cited_section_ids=cited_section_ids,
        )
    score = _coerce_score_value(payload)
    if score is None:
        return None
    return HistoryDocsRubricScore(
        dimension=dimension,
        score=score,
        rationale="No rationale provided.",
    )


def _normalize_quality_judgment(
    raw_judgment: HistoryDocsQualityJudgmentEnvelope,
) -> HistoryDocsQualityJudgment:
    normalized_scores: dict[HistoryDocsRubricDimension, HistoryDocsRubricScore] = {}
    normalization_notes: list[str] = []

    def remember(dimension: str, payload: object) -> None:
        if dimension not in _RUBRIC_DIMENSIONS or dimension in normalized_scores:
            return
        rubric_dimension = cast(HistoryDocsRubricDimension, dimension)
        score = _coerce_rubric_score(rubric_dimension, payload)
        if score is not None:
            normalized_scores[rubric_dimension] = score

    for payload in (raw_judgment.rubric_scores, raw_judgment.scores):
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, HistoryDocsRubricScore):
                    remember(item.dimension, item)
                elif isinstance(item, dict):
                    remember(str(item.get("dimension", "")).strip(), item)
        elif isinstance(payload, dict):
            for dimension, item in payload.items():
                remember(str(dimension).strip(), item)

    for dimension in _RUBRIC_DIMENSIONS:
        remember(dimension, getattr(raw_judgment, dimension, None))

    for dimension, payload in (raw_judgment.model_extra or {}).items():
        remember(str(dimension).strip(), payload)

    missing_dimensions = [
        dimension
        for dimension in _RUBRIC_DIMENSIONS
        if dimension not in normalized_scores
    ]
    if missing_dimensions and not normalized_scores:
        raise ValueError(
            "Evaluator response did not include any recognizable rubric scores"
        )
    if missing_dimensions:
        normalization_notes.append(
            "Evaluator omitted rubric dimensions: "
            + ", ".join(missing_dimensions)
            + ". Filled with conservative zero scores."
        )
        for dimension in missing_dimensions:
            normalized_scores[dimension] = HistoryDocsRubricScore(
                dimension=dimension,
                score=0,
                rationale="Evaluator omitted this rubric dimension.",
            )

    return HistoryDocsQualityJudgment(
        rubric_scores=[
            normalized_scores[dimension] for dimension in _RUBRIC_DIMENSIONS
        ],
        strengths=raw_judgment.strengths,
        weaknesses=raw_judgment.weaknesses,
        unsupported_claim_risks=raw_judgment.unsupported_claim_risks,
        tbd_overuse=raw_judgment.tbd_overuse,
        evaluator_notes=[*raw_judgment.evaluator_notes, *normalization_notes],
        uncertainty=[*raw_judgment.uncertainty, *normalization_notes],
    )


def _failed_quality_report(
    *,
    case: PreparedHistoryDocsBenchmarkCase,
    variant_id: str,
    checkpoint_id: str,
    failure_note: str,
    build_failed: bool,
) -> HistoryDocsQualityReport:
    return HistoryDocsQualityReport(
        case_id=case.manifest.case_id,
        variant_id=variant_id,
        checkpoint_id=checkpoint_id,
        build_failed=build_failed,
        evaluation_status="llm_failed",
        validation_error_count=0,
        validation_warning_count=0,
        rubric_scores=_empty_rubric_scores(),
        overall_score=0.0,
        weaknesses=[
            (
                "Variant build failed before evaluation completed."
                if build_failed
                else "Quality evaluation failed before scoring completed."
            )
        ],
        evaluator_notes=[failure_note],
        uncertainty=[failure_note],
        failure_note=failure_note,
    )


def evaluate_history_docs_quality(
    *,
    case: PreparedHistoryDocsBenchmarkCase,
    variant_id: str,
    build_result: HistoryBuildResult,
    llm_client: LLMClient,
) -> HistoryDocsQualityReport:
    """Evaluate one rendered checkpoint document through the H10 judge."""

    checkpoint_id = getattr(build_result, "checkpoint_id", "unknown")
    try:
        artifacts = _load_benchmark_artifacts(build_result)
        system_prompt, user_prompt = build_history_docs_quality_judge_prompt(
            case=case.manifest,
            markdown=artifacts.markdown,
            render_manifest=artifacts.render_manifest,
            validation_report=artifacts.validation_report,
            checkpoint_model=artifacts.checkpoint_model,
            semantic_context_map=artifacts.semantic_context_map,
        )
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryDocsQualityJudgmentEnvelope,
                model_name=case.manifest.project_config.llm.model_name,
                temperature=case.manifest.project_config.llm.temperature,
            )
        )
        judgment = _normalize_quality_judgment(
            HistoryDocsQualityJudgmentEnvelope.model_validate(
                response.content.model_dump(mode="python")
            )
        )
        return HistoryDocsQualityReport(
            case_id=case.manifest.case_id,
            variant_id=variant_id,
            checkpoint_id=checkpoint_id,
            build_failed=False,
            evaluation_status="scored",
            validation_error_count=artifacts.validation_report.error_count,
            validation_warning_count=artifacts.validation_report.warning_count,
            rubric_scores=judgment.rubric_scores,
            overall_score=_overall_score(judgment.rubric_scores),
            strengths=judgment.strengths,
            weaknesses=judgment.weaknesses,
            unsupported_claim_risks=judgment.unsupported_claim_risks,
            tbd_overuse=judgment.tbd_overuse,
            evaluator_notes=judgment.evaluator_notes,
            uncertainty=judgment.uncertainty,
        )
    except Exception as exc:
        return _failed_quality_report(
            case=case,
            variant_id=variant_id,
            checkpoint_id=checkpoint_id,
            build_failed=False,
            failure_note=f"Evaluator failure: {type(exc).__name__}: {exc}",
        )


def compare_history_docs_quality_reports(
    *,
    case_id: str,
    baseline_report: HistoryDocsQualityReport,
    candidate_report: HistoryDocsQualityReport,
) -> HistoryDocsVariantComparison:
    """Compare two quality reports deterministically without another LLM call."""

    baseline_scores = {
        score.dimension: score.score for score in baseline_report.rubric_scores
    }
    candidate_scores = {
        score.dimension: score.score for score in candidate_report.rubric_scores
    }
    deltas = [
        HistoryDocsRubricDelta(
            dimension=dimension,
            baseline_score=baseline_scores[dimension],
            candidate_score=candidate_scores[dimension],
            delta=candidate_scores[dimension] - baseline_scores[dimension],
        )
        for dimension in _RUBRIC_DIMENSIONS
    ]
    baseline_failed = baseline_report.evaluation_status != "scored"
    candidate_failed = candidate_report.evaluation_status != "scored"
    comparison_notes: list[str] = []
    if baseline_failed:
        comparison_notes.append(
            "Baseline evaluation failed; overall score defaults to 0.0."
        )
    if candidate_failed:
        comparison_notes.append(
            "Candidate evaluation failed; overall score defaults to 0.0."
        )

    preferred_variant_id: str
    if candidate_report.overall_score > baseline_report.overall_score:
        preferred_variant_id = candidate_report.variant_id
    elif candidate_report.overall_score < baseline_report.overall_score:
        preferred_variant_id = baseline_report.variant_id
    else:
        baseline_risks = len(baseline_report.unsupported_claim_risks)
        candidate_risks = len(candidate_report.unsupported_claim_risks)
        if candidate_risks < baseline_risks:
            preferred_variant_id = candidate_report.variant_id
        elif candidate_risks > baseline_risks:
            preferred_variant_id = baseline_report.variant_id
        elif (not candidate_report.tbd_overuse) and baseline_report.tbd_overuse:
            preferred_variant_id = candidate_report.variant_id
        elif candidate_report.tbd_overuse and (not baseline_report.tbd_overuse):
            preferred_variant_id = baseline_report.variant_id
        else:
            preferred_variant_id = sorted(
                [baseline_report.variant_id, candidate_report.variant_id]
            )[0]
            comparison_notes.append(
                "Tie resolved by stable variant-id ordering after score and risk tie-breaks."
            )

    return HistoryDocsVariantComparison(
        case_id=case_id,
        baseline_variant_id=baseline_report.variant_id,
        candidate_variant_id=candidate_report.variant_id,
        per_dimension_deltas=deltas,
        overall_delta=round(
            candidate_report.overall_score - baseline_report.overall_score,
            3,
        ),
        preferred_variant_id=preferred_variant_id,
        comparison_notes=comparison_notes,
        baseline_failed=baseline_failed,
        candidate_failed=candidate_failed,
    )


def run_history_docs_benchmark_suite(
    *,
    suite_id: str,
    output_root: Path,
    cases: list[PreparedHistoryDocsBenchmarkCase],
    variant_runners: list[HistoryDocsBenchmarkVariant] | None = None,
    llm_client_factory: Callable[[LLMConfig], LLMClient] = create_llm_client,
    progress_callback: ProgressCallback | None = None,
) -> HistoryDocsBenchmarkSuiteReport:
    """Run the H10 benchmark suite and persist suite, case, and quality artifacts."""

    coverage_tags = validate_benchmark_focus_coverage(cases)
    variants = variant_runners or [baseline_history_docs_benchmark_variant()]
    suite_case_reports: list[HistoryDocsBenchmarkCaseReportRef] = []
    quality_reports_by_variant: dict[str, list[HistoryDocsQualityReport]] = {
        variant.variant_id: [] for variant in variants
    }
    failed_evaluation_count = 0

    for case in cases:
        _progress(
            progress_callback,
            f"history-docs benchmark: preparing case {case.manifest.case_id}",
        )
        case_manifest_path = benchmark_case_manifest_path(
            output_root,
            suite_id,
            case.manifest.case_id,
        )
        write_json_model(case_manifest_path, case.manifest)

        quality_report_paths: dict[str, Path] = {}
        case_quality_reports: list[HistoryDocsQualityReport] = []
        llm_client = llm_client_factory(case.manifest.project_config.llm)

        for variant in variants:
            _progress(
                progress_callback,
                "history-docs benchmark: running "
                f"{variant.variant_id} for {case.manifest.case_id}",
            )
            workspace_id = (
                f"{benchmark_suite_workspace_id(suite_id)}"
                f"__{case.manifest.case_id}__{variant.variant_id}"
            )
            try:
                build_result = variant.runner(case, workspace_id)
                quality_report = evaluate_history_docs_quality(
                    case=case,
                    variant_id=variant.variant_id,
                    build_result=build_result,
                    llm_client=llm_client,
                )
            except Exception as exc:
                quality_report = _failed_quality_report(
                    case=case,
                    variant_id=variant.variant_id,
                    checkpoint_id="build-failed",
                    build_failed=True,
                    failure_note=f"Build failure: {type(exc).__name__}: {exc}",
                )
            if quality_report.evaluation_status != "scored":
                failed_evaluation_count += 1
            case_quality_reports.append(quality_report)
            quality_reports_by_variant[variant.variant_id].append(quality_report)
            quality_path = benchmark_quality_report_path(
                output_root,
                suite_id,
                case.manifest.case_id,
                variant.variant_id,
            )
            write_json_model(quality_path, quality_report)
            quality_report_paths[variant.variant_id] = quality_path

        comparison_report = HistoryDocsBenchmarkCaseComparisonReport(
            case_id=case.manifest.case_id,
            baseline_variant_id=variants[0].variant_id,
            quality_report_paths=quality_report_paths,
            comparisons=[
                compare_history_docs_quality_reports(
                    case_id=case.manifest.case_id,
                    baseline_report=left_report,
                    candidate_report=right_report,
                )
                for left_index, left_report in enumerate(case_quality_reports)
                for right_report in case_quality_reports[left_index + 1 :]
            ],
        )
        comparison_path = benchmark_comparison_report_path(
            output_root,
            suite_id,
            case.manifest.case_id,
        )
        write_json_model(comparison_path, comparison_report)
        suite_case_reports.append(
            HistoryDocsBenchmarkCaseReportRef(
                case_id=case.manifest.case_id,
                case_manifest_path=case_manifest_path,
                comparison_report_path=comparison_path,
                quality_report_paths=quality_report_paths,
            )
        )

    average_score_by_variant = {
        variant.variant_id: (
            round(
                fmean(
                    report.overall_score
                    for report in quality_reports_by_variant[variant.variant_id]
                ),
                3,
            )
            if quality_reports_by_variant[variant.variant_id]
            else 0.0
        )
        for variant in variants
    }
    suite_report = HistoryDocsBenchmarkSuiteReport(
        suite_id=suite_id,
        case_ids=[case.manifest.case_id for case in cases],
        variant_ids=[variant.variant_id for variant in variants],
        case_reports=suite_case_reports,
        average_score_by_variant=average_score_by_variant,
        failed_evaluation_count=failed_evaluation_count,
        coverage_tags=coverage_tags,
    )
    write_json_model(benchmark_suite_manifest_path(output_root, suite_id), suite_report)
    return suite_report


__all__ = [
    "PreparedHistoryDocsBenchmarkCase",
    "HistoryDocsBenchmarkVariant",
    "baseline_history_docs_benchmark_variant",
    "benchmark_case_manifest_path",
    "benchmark_case_root",
    "benchmark_comparison_report_path",
    "benchmark_quality_report_path",
    "benchmark_suite_manifest_path",
    "benchmark_suite_root",
    "benchmark_suite_workspace_id",
    "build_default_history_docs_benchmark_cases",
    "compare_history_docs_quality_reports",
    "evaluate_history_docs_quality",
    "run_history_docs_benchmark_suite",
    "semantic_structure_context_benchmark_variant",
    "semantic_history_docs_benchmark_variant",
    "validate_benchmark_focus_coverage",
]
