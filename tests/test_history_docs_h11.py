"""Tests for history-docs H11 semantic checkpoint planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.repo.history import (
    iter_first_parent_commits,
    list_reachable_tags_by_commit,
)
from engllm.domain.errors import LLMError
from engllm.llm.base import StructuredGenerationRequest
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import (
    build_semantic_checkpoint_planner_prompt,
)
from engllm.tools.history_docs.benchmark import (
    build_default_history_docs_benchmark_cases,
)
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistorySemanticCheckpointJudgmentBatch,
    HistorySemanticCheckpointPlan,
)
from engllm.tools.history_docs.semantic_planner import (
    build_semantic_checkpoint_plan,
)
from engllm.tools.history_docs.semantic_planner import (
    semantic_checkpoint_plan_path as build_semantic_checkpoint_plan_path,
)
from tests.history_docs_helpers import (
    commit_file,
    git,
    init_repo,
    semantic_checkpoint_plan_path,
    write_project_config,
)


class _FailingSemanticPlannerClient:
    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> object:
        raise LLMError("semantic checkpoint planner unavailable")


class _InjectingPlannerClient(MockLLMClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(
            canned={HistorySemanticCheckpointJudgmentBatch.__name__: payload}
        )


def _create_semantic_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "pyproject.toml",
        '[project]\nname = "semantic-demo"\ndependencies = ["requests>=2"]\n',
        message="bootstrap project manifest",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    git(repo_root, "tag", "v0.1.0", base)
    api_commit = commit_file(
        repo_root,
        "src/core/api.py",
        "def fetch_state(value: str) -> str:\n    return value\n",
        message="add core api",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    manifest_commit = commit_file(
        repo_root,
        "package.json",
        '{"devDependencies":{"vitest":"^1.0.0"}}\n',
        message="add frontend tooling",
        timestamp="2024-03-01T10:00:00+00:00",
    )
    current_branch = git(repo_root, "branch", "--show-current")
    git(repo_root, "checkout", "-b", "feature", manifest_commit)
    commit_file(
        repo_root,
        "src/engine/planner.py",
        'def build_plan() -> list[str]:\n    return ["plan"]\n',
        message="add engine planner",
        timestamp="2024-04-01T10:00:00+00:00",
    )
    git(repo_root, "checkout", current_branch)
    commit_file(
        repo_root,
        "src/core/api.py",
        "def fetch_state(value: str, strict: bool = True) -> str:\n    return value\n",
        message="tighten api contract",
        timestamp="2024-04-02T10:00:00+00:00",
    )
    git(
        repo_root,
        "merge",
        "--no-ff",
        "feature",
        "-m",
        "merge feature planner",
        env={
            "GIT_AUTHOR_DATE": "2024-04-03T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-04-03T10:00:00+00:00",
        },
    )
    target = git(repo_root, "rev-parse", "HEAD")
    return repo_root, {
        "base": base,
        "api": api_commit,
        "manifest": manifest_commit,
        "target": target,
    }


def test_semantic_checkpoint_plan_path_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"

    assert build_semantic_checkpoint_plan_path(
        output_root / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-04-03-deadbee",
    ) == semantic_checkpoint_plan_path(
        output_root,
        "repo",
        "2024-04-03-deadbee",
    )


def test_first_parent_tags_and_candidate_detection_are_deterministic(
    tmp_path: Path,
) -> None:
    repo_root, commits = _create_semantic_repo(tmp_path)
    ancestry = iter_first_parent_commits(repo_root, target_commit=commits["target"])
    tags_by_commit = list_reachable_tags_by_commit(
        repo_root,
        target_commit=commits["target"],
        commit_shas=[commit.sha for commit in ancestry],
    )
    plan = build_semantic_checkpoint_plan(
        repo_root=repo_root,
        checkpoint_id="2024-04-03-merge01",
        target_commit=commits["target"],
        previous_checkpoint_commit=commits["manifest"],
        configured_source_roots=[Path("src")],
        checkpoints=[],
        llm_client=MockLLMClient(),
        model_name="mock-engllm",
        temperature=0.2,
    )
    candidates_by_commit = {
        candidate.commit.sha: candidate for candidate in plan.candidates
    }

    assert [commit.subject for commit in ancestry] == [
        "bootstrap project manifest",
        "add core api",
        "add frontend tooling",
        "tighten api contract",
        "merge feature planner",
    ]
    assert tags_by_commit[commits["base"]] == ["v0.1.0"]
    assert plan.evaluation_status == "heuristic_only"
    assert set(candidates_by_commit[commits["base"]].signal_kinds) == {
        "build_shift",
        "tag_anchor",
    }
    assert set(candidates_by_commit[commits["api"]].signal_kinds) == {
        "interface_shift",
        "new_top_level_area",
    }
    assert "build_shift" in candidates_by_commit[commits["manifest"]].signal_kinds
    assert "merge_anchor" in candidates_by_commit[commits["target"]].signal_kinds
    assert "new_top_level_area" in candidates_by_commit[commits["target"]].signal_kinds


def test_semantic_planner_prompt_includes_compact_candidate_evidence() -> None:
    system_prompt, user_prompt = build_semantic_checkpoint_planner_prompt(
        checkpoint_id="2024-04-03-merge01",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        built_checkpoints=[
            {
                "checkpoint_id": "2024-01-01-aaaaaaa",
                "target_commit": "b" * 40,
                "target_commit_timestamp": "2024-01-01T10:00:00+00:00",
                "target_commit_subject": "bootstrap project manifest",
            }
        ],
        candidates=[
            {
                "candidate_commit_id": "a" * 40,
                "short_sha": "aaaaaaa",
                "timestamp": "2024-04-03T10:00:00+00:00",
                "subject": "merge feature planner",
                "window_start_commit": "b" * 40,
                "window_commit_count": 2,
                "tag_names": [],
                "top_level_areas": ["src/engine"],
                "change_kinds": ["interface_change"],
                "signal_kinds": ["merge_anchor", "new_top_level_area"],
                "heuristic_score": 2,
            }
        ],
    )

    assert "Do not invent new commits" in system_prompt
    assert "merge feature planner" in user_prompt
    assert "window_commit_count" in user_prompt
    assert "2024-01-01-aaaaaaa" in user_prompt


def test_llm_cannot_introduce_commits_outside_deterministic_candidate_set(
    tmp_path: Path,
) -> None:
    repo_root, commits = _create_semantic_repo(tmp_path)
    payload = {
        "judgments": [
            {
                "candidate_commit_id": commits["api"],
                "recommendation": "primary",
                "semantic_title": "API boundary milestone",
                "rationale": "Introduces the first explicit API surface.",
                "uncertainty": "",
            },
            {
                "candidate_commit_id": "f" * 40,
                "recommendation": "primary",
                "semantic_title": "Invented",
                "rationale": "Should be ignored.",
                "uncertainty": "",
            },
        ]
    }

    plan = build_semantic_checkpoint_plan(
        repo_root=repo_root,
        checkpoint_id="2024-04-03-merge01",
        target_commit=commits["target"],
        previous_checkpoint_commit=commits["manifest"],
        configured_source_roots=[Path("src")],
        checkpoints=[],
        llm_client=_InjectingPlannerClient(payload),
        model_name="mock-engllm",
        temperature=0.2,
    )
    candidates_by_commit = {
        candidate.commit.sha: candidate for candidate in plan.candidates
    }

    assert plan.evaluation_status == "scored"
    assert candidates_by_commit[commits["api"]].recommendation == "primary"
    assert (
        candidates_by_commit[commits["api"]].semantic_title == "API boundary milestone"
    )
    assert all(candidate.commit.sha != "f" * 40 for candidate in plan.candidates)


def test_build_history_docs_checkpoint_persists_llm_failed_semantic_plan(
    tmp_path: Path,
    sample_project_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, commits = _create_semantic_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [Path("src")]

    monkeypatch.setattr(
        "engllm.tools.history_docs.build.create_llm_client",
        lambda config: _FailingSemanticPlannerClient(),
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["target"],
        previous_checkpoint_commit=commits["manifest"],
    )
    plan_path = semantic_checkpoint_plan_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    plan = HistorySemanticCheckpointPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    assert result.semantic_checkpoint_plan_path == plan_path
    assert result.semantic_planner_status == "llm_failed"
    assert plan.evaluation_status == "llm_failed"
    assert all(
        candidate.recommendation == "supporting" for candidate in plan.candidates
    )
    assert all(candidate.semantic_title == "TBD" for candidate in plan.candidates)


def test_history_docs_build_writes_semantic_plan_and_cli_prints_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root, commits = _create_semantic_repo(tmp_path)
    config_path = tmp_path / "project.yaml"
    write_project_config(
        config_path,
        output_root=tmp_path / "artifacts",
        source_roots=[repo_root / "src"],
    )

    exit_code = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            commits["target"],
            "--previous-checkpoint-commit",
            commits["manifest"],
        ]
    )
    stdout = capsys.readouterr().out
    checkpoints_root = (
        tmp_path
        / "artifacts"
        / "workspaces"
        / repo_root.name
        / "tools"
        / "history_docs"
        / "checkpoints"
    )
    plan_paths = sorted(checkpoints_root.glob("*/semantic_checkpoint_plan.json"))
    assert len(plan_paths) == 1
    plan_path = plan_paths[0]
    plan = HistorySemanticCheckpointPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert f"Semantic checkpoint plan: {plan_path}" in stdout
    assert plan_path.exists()
    assert plan.target_commit == commits["target"]


def test_benchmark_algorithm_and_architecture_cases_emit_semantic_candidates(
    tmp_path: Path,
) -> None:
    cases = build_default_history_docs_benchmark_cases(
        base_root=tmp_path / "repos",
        output_root=tmp_path / "artifacts",
    )
    selected = {
        case.manifest.case_id: case
        for case in cases
        if case.manifest.case_id in {"algorithm_heavy_variants", "architecture_heavy"}
    }

    for case_id, prepared_case in selected.items():
        plan = build_semantic_checkpoint_plan(
            repo_root=prepared_case.repo_root,
            checkpoint_id=f"{case_id}-checkpoint",
            target_commit=prepared_case.manifest.target_commit,
            previous_checkpoint_commit=prepared_case.manifest.previous_checkpoint_commit,
            configured_source_roots=prepared_case.manifest.project_config.sources.roots,
            checkpoints=[],
            llm_client=MockLLMClient(),
            model_name=prepared_case.manifest.project_config.llm.model_name,
            temperature=prepared_case.manifest.project_config.llm.temperature,
        )

        assert plan.candidates
        assert any(candidate.signal_kinds for candidate in plan.candidates)
        assert any(
            signal in {"interface_shift", "new_top_level_area", "broad_change"}
            for candidate in plan.candidates
            for signal in candidate.signal_kinds
        )
