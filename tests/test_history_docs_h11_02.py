"""Tests for history-docs H11-02 semantic subsystem clustering."""

from __future__ import annotations

from pathlib import Path

from engllm.domain.models import CodeUnitSummary
from engllm.llm.mock import MockLLMClient
from engllm.tools.history_docs.benchmark import (
    baseline_history_docs_benchmark_variant,
    benchmark_comparison_report_path,
    benchmark_quality_report_path,
    build_default_history_docs_benchmark_cases,
    run_history_docs_benchmark_suite,
    semantic_history_docs_benchmark_variant,
)
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryDocsQualityJudgmentEnvelope,
    HistorySemanticStructureJudgment,
    HistorySemanticStructureMap,
    HistorySnapshotStructuralModel,
    HistorySubsystemCandidate,
)
from engllm.tools.history_docs.semantic_structure import (
    build_semantic_structure_map,
    build_subsystem_grouping_view,
)
from engllm.tools.history_docs.semantic_structure import (
    semantic_structure_map_path as build_semantic_structure_map_path,
)
from tests.history_docs_helpers import (
    checkpoint_markdown_path,
    checkpoint_model_path,
    commit_file,
    init_repo,
    semantic_structure_map_path,
)


def _semantic_structure_payload() -> dict[str, object]:
    return {
        "semantic_subsystems": [
            {
                "semantic_subsystem_id": "semantic-subsystem::api",
                "title": "API Layer",
                "summary": "Handles request routing and boundary validation.",
                "module_ids": ["module::src/app/api/router.py"],
                "capability_ids": ["capability::request-handling"],
            },
            {
                "semantic_subsystem_id": "semantic-subsystem::engine",
                "title": "Planning Engine",
                "summary": "Builds execution plans from request context.",
                "module_ids": ["module::src/app/engine/planner.py"],
                "capability_ids": ["capability::planning"],
            },
            {
                "semantic_subsystem_id": "semantic-subsystem::storage",
                "title": "State Storage",
                "summary": "Stores and retrieves checkpoint state.",
                "module_ids": ["module::src/app/storage/repository.py"],
                "capability_ids": ["capability::persistence"],
            },
        ],
        "capabilities": [
            {
                "capability_id": "capability::request-handling",
                "title": "Request Handling",
                "summary": "Covers routing and validation responsibilities.",
                "module_ids": ["module::src/app/api/router.py"],
                "semantic_subsystem_ids": ["semantic-subsystem::api"],
            },
            {
                "capability_id": "capability::planning",
                "title": "Planning",
                "summary": "Covers plan construction responsibilities.",
                "module_ids": ["module::src/app/engine/planner.py"],
                "semantic_subsystem_ids": ["semantic-subsystem::engine"],
            },
            {
                "capability_id": "capability::persistence",
                "title": "Persistence",
                "summary": "Covers state storage responsibilities.",
                "module_ids": ["module::src/app/storage/repository.py"],
                "semantic_subsystem_ids": ["semantic-subsystem::storage"],
            },
        ],
    }


def _benchmark_semantic_payload(case_id: str) -> dict[str, object]:
    if case_id == "medium_mixed":
        return {
            "semantic_subsystems": [
                {
                    "semantic_subsystem_id": "semantic-subsystem::application-services",
                    "title": "Application Services",
                    "summary": "Handles core processing and support utilities.",
                    "module_ids": [
                        "module::src/core/service.py",
                        "module::src/support/cache.py",
                    ],
                    "capability_ids": ["capability::service-flow"],
                },
                {
                    "semantic_subsystem_id": "semantic-subsystem::web-experience",
                    "title": "Web Experience",
                    "summary": "Renders browser-facing UI behavior.",
                    "module_ids": ["module::web/ui.js"],
                    "capability_ids": ["capability::ui-rendering"],
                },
            ],
            "capabilities": [
                {
                    "capability_id": "capability::service-flow",
                    "title": "Service Flow",
                    "summary": "Covers service assembly and cache support.",
                    "module_ids": [
                        "module::src/core/service.py",
                        "module::src/support/cache.py",
                    ],
                    "semantic_subsystem_ids": [
                        "semantic-subsystem::application-services"
                    ],
                },
                {
                    "capability_id": "capability::ui-rendering",
                    "title": "UI Rendering",
                    "summary": "Covers web rendering responsibilities.",
                    "module_ids": ["module::web/ui.js"],
                    "semantic_subsystem_ids": ["semantic-subsystem::web-experience"],
                },
            ],
        }
    return {
        "semantic_subsystems": [],
        "capabilities": [],
    }


def _quality_payload() -> dict[str, object]:
    dimensions = (
        "coverage",
        "coherence",
        "specificity",
        "algorithm_understanding",
        "dependency_understanding",
        "rationale_capture",
        "present_state_tone",
    )
    return {
        "rubric_scores": [
            {
                "dimension": dimension,
                "score": 4,
                "rationale": f"{dimension} is acceptable in the fixture.",
                "matched_expectation_ids": ["medium-architecture"],
                "cited_section_ids": ["architectural_overview"],
            }
            for dimension in dimensions
        ],
        "strengths": ["Fixture benchmark completed."],
        "weaknesses": [],
        "unsupported_claim_risks": [],
        "tbd_overuse": False,
        "evaluator_notes": ["Mock benchmark review."],
        "uncertainty": [],
    }


def _snapshot_model() -> HistorySnapshotStructuralModel:
    return HistorySnapshotStructuralModel(
        checkpoint_id="2024-05-01-demo001",
        target_commit="a" * 40,
        analyzed_source_roots=[Path("src")],
        files=[
            Path("src/app/api/router.py"),
            Path("src/app/engine/planner.py"),
            Path("src/app/storage/repository.py"),
        ],
        code_summaries=[
            CodeUnitSummary(
                path=Path("src/app/api/router.py"),
                language="python",
                functions=["route_request", "validate_request"],
                imports=["from app.engine.planner import build_plan"],
            ),
            CodeUnitSummary(
                path=Path("src/app/engine/planner.py"),
                language="python",
                functions=["build_plan"],
                imports=["from app.storage.repository import StateRepository"],
            ),
            CodeUnitSummary(
                path=Path("src/app/storage/repository.py"),
                language="python",
                classes=["StateRepository"],
            ),
        ],
        subsystem_candidates=[
            HistorySubsystemCandidate(
                candidate_id="subsystem::src::app",
                source_root=Path("src"),
                group_path=Path("src/app"),
                file_count=3,
                symbol_count=4,
                language_counts={"python": 3},
                representative_files=[
                    Path("src/app/api/router.py"),
                    Path("src/app/engine/planner.py"),
                    Path("src/app/storage/repository.py"),
                ],
            )
        ],
    )


def _create_semantic_structure_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "pyproject.toml",
        '[project]\nname = "semantic-structure"\ndependencies = ["requests>=2"]\n',
        message="add manifest",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/api/router.py",
        '"""API routing boundary."""\n\n\ndef route_request(path: str) -> str:\n    return path\n\n\ndef validate_request(path: str) -> bool:\n    return bool(path)\n',
        message="add api router",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/engine/planner.py",
        '"""Plan construction flow."""\n\n\ndef build_plan() -> list[str]:\n    return ["plan"]\n',
        message="add planning engine",
        timestamp="2024-03-01T10:00:00+00:00",
    )
    target = commit_file(
        repo_root,
        "src/app/storage/repository.py",
        '"""Checkpoint state storage."""\n\nclass StateRepository:\n    pass\n',
        message="add state storage",
        timestamp="2024-04-01T10:00:00+00:00",
    )
    return repo_root, target


def test_semantic_structure_map_path_and_invalid_clustering_fallback_are_deterministic(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot_model()
    invalid_client = MockLLMClient(
        canned={
            HistorySemanticStructureJudgment.__name__: {
                "semantic_subsystems": [
                    {
                        "semantic_subsystem_id": "semantic-subsystem::broken",
                        "title": "Broken",
                        "summary": "Invalid cluster.",
                        "module_ids": [
                            "module::src/app/api/router.py",
                            "module::missing.py",
                        ],
                        "capability_ids": [],
                    }
                ],
                "capabilities": [],
            }
        }
    )

    assert build_semantic_structure_map_path(
        tmp_path / "artifacts" / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-05-01-demo001",
    ) == semantic_structure_map_path(
        tmp_path / "artifacts",
        "repo",
        "2024-05-01-demo001",
    )

    semantic_map = build_semantic_structure_map(
        checkpoint_id=snapshot.checkpoint_id,
        target_commit=snapshot.target_commit,
        previous_checkpoint_commit=None,
        snapshot=snapshot,
        llm_client=invalid_client,
        model_name="mock-engllm",
        temperature=0.2,
    )
    grouping_view = build_subsystem_grouping_view(
        snapshot=snapshot,
        mode="semantic",
        semantic_map=semantic_map,
    )

    assert semantic_map.evaluation_status == "llm_failed"
    assert len(semantic_map.semantic_subsystems) == 1
    assert semantic_map.semantic_subsystems[0].module_ids == [
        "module::src/app/api/router.py",
        "module::src/app/engine/planner.py",
        "module::src/app/storage/repository.py",
    ]
    assert grouping_view.module_subsystem_ids == {
        Path("src/app/api/router.py"): "semantic-subsystem::src::app",
        Path("src/app/engine/planner.py"): "semantic-subsystem::src::app",
        Path("src/app/storage/repository.py"): "semantic-subsystem::src::app",
    }


def test_history_docs_build_writes_semantic_structure_map_but_keeps_path_grouping_by_default(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, target = _create_semantic_structure_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [Path("src")]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=target,
    )

    structure_path = semantic_structure_map_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    checkpoint_path = checkpoint_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    markdown_path = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    semantic_map = HistorySemanticStructureMap.model_validate_json(
        structure_path.read_text(encoding="utf-8")
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert result.semantic_structure_map_path == structure_path
    assert result.semantic_structure_status == semantic_map.evaluation_status
    assert semantic_map.evaluation_status == "llm_failed"
    assert [subsystem.concept_id for subsystem in checkpoint_model.subsystems] == [
        "subsystem::src::app"
    ]
    assert checkpoint_model.subsystems[0].display_name == "App"
    assert checkpoint_model.subsystems[0].capability_labels == []
    assert "### API Layer" not in markdown


def test_semantic_grouping_mode_rewires_checkpoint_model_and_rendering(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, target = _create_semantic_structure_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [Path("src")]
    semantic_client = MockLLMClient(
        canned={
            HistorySemanticStructureJudgment.__name__: _semantic_structure_payload()
        }
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=target,
        subsystem_grouping_mode="semantic",
        narrative_render_mode="baseline",
        llm_client_override=semantic_client,
    )

    structure_path = semantic_structure_map_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    checkpoint_path = checkpoint_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    markdown_path = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    semantic_map = HistorySemanticStructureMap.model_validate_json(
        structure_path.read_text(encoding="utf-8")
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert semantic_map.evaluation_status == "scored"
    assert result.semantic_subsystem_count == 3
    assert result.semantic_capability_count == 3
    assert [subsystem.concept_id for subsystem in checkpoint_model.subsystems] == [
        "semantic-subsystem::api",
        "semantic-subsystem::engine",
        "semantic-subsystem::storage",
    ]
    assert checkpoint_model.subsystems[0].display_name == "API Layer"
    assert checkpoint_model.subsystems[0].capability_labels == ["Request Handling"]
    assert "### API Layer" in markdown
    assert "Capability labels: `Request Handling`." in markdown
    assert "### Planning Engine" in markdown
    assert "### State Storage" in markdown


def test_h10_suite_can_compare_baseline_and_semantic_clustering_variants(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    cases = build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=output_root,
    )
    suite_report = run_history_docs_benchmark_suite(
        suite_id="semantic-compare",
        output_root=output_root,
        cases=cases,
        variant_runners=[
            baseline_history_docs_benchmark_variant(),
            semantic_history_docs_benchmark_variant(
                llm_client_builder=lambda case: MockLLMClient(
                    canned={
                        HistorySemanticStructureJudgment.__name__: _benchmark_semantic_payload(
                            case.manifest.case_id
                        )
                    }
                )
            ),
        ],
        llm_client_factory=lambda config: MockLLMClient(
            canned={HistoryDocsQualityJudgmentEnvelope.__name__: _quality_payload()}
        ),
    )

    assert suite_report.variant_ids == ["baseline", "semantic-clustering"]
    assert benchmark_quality_report_path(
        output_root,
        "semantic-compare",
        "medium_mixed",
        "baseline",
    ).exists()
    assert benchmark_quality_report_path(
        output_root,
        "semantic-compare",
        "medium_mixed",
        "semantic-clustering",
    ).exists()
    assert benchmark_comparison_report_path(
        output_root,
        "semantic-compare",
        "medium_mixed",
    ).exists()
