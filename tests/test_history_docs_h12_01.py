"""Tests for history-docs H12-01 interval interpretation."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.config.loader import load_project_config
from engllm.domain.errors import LLMError
from engllm.llm.base import StructuredGenerationRequest
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import build_interval_interpretation_prompt
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.interval_interpretation import (
    build_interval_interpretation,
)
from engllm.tools.history_docs.interval_interpretation import (
    interval_interpretation_path as build_interval_interpretation_path,
)
from engllm.tools.history_docs.models import (
    HistoryIntervalDeltaModel,
    HistoryIntervalInterpretation,
    HistoryIntervalInterpretationJudgment,
    HistorySnapshotStructuralModel,
)
from tests.history_docs_helpers import (
    commit_file,
    init_repo,
    interval_delta_model_path,
    interval_interpretation_path,
    snapshot_structural_model_path,
    write_project_config,
)


class _FailingIntervalClient:
    def generate_structured(
        self,
        request: StructuredGenerationRequest,
    ) -> object:
        raise LLMError("interval interpretation unavailable")


class _InjectingIntervalClient(MockLLMClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(
            canned={HistoryIntervalInterpretationJudgment.__name__: payload}
        )


def _create_interval_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "src/core/api.py",
        (
            '"""API service expects strict request validation and fallback handling."""\n'
            "def fetch_state(value: str) -> str:\n"
            "    return value\n"
        ),
        message="bootstrap api service",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    base = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
name = "interval-demo"
dependencies = ["requests>=2"]
""",
        message="add python dependency manifest",
        timestamp="2024-01-01T12:00:00+00:00",
    )
    interface_commit = commit_file(
        repo_root,
        "src/core/api.py",
        (
            '"""API service must enforce strict request validation with fallback mode."""\n'
            "def fetch_state(value: str, strict: bool = True) -> str:\n"
            "    return value\n"
        ),
        message="tighten api contract to support strict fallback mode",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "package.json",
        '{"devDependencies":{"vitest":"^1.0.0"}}\n',
        message="add frontend build tooling",
        timestamp="2024-02-01T12:00:00+00:00",
    )
    return repo_root, {"base": base, "interface": interface_commit, "head": head}


def _load_snapshot_and_delta(
    output_root: Path,
    workspace_id: str,
    checkpoint_id: str,
) -> tuple[HistorySnapshotStructuralModel, HistoryIntervalDeltaModel]:
    snapshot = HistorySnapshotStructuralModel.model_validate_json(
        snapshot_structural_model_path(
            output_root,
            workspace_id,
            checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    delta_model = HistoryIntervalDeltaModel.model_validate_json(
        interval_delta_model_path(
            output_root,
            workspace_id,
            checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    return snapshot, delta_model


def test_interval_interpretation_path_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"

    assert build_interval_interpretation_path(
        output_root / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-02-01-abcd123",
    ) == interval_interpretation_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )


def test_interval_interpretation_prompt_includes_compact_structured_evidence() -> None:
    system_prompt, user_prompt = build_interval_interpretation_prompt(
        checkpoint_id="2024-02-01-abcd123",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        commit_deltas=[
            {
                "commit_id": "a" * 40,
                "short_sha": "aaaaaaa",
                "timestamp": "2024-02-01T10:00:00+00:00",
                "subject": "tighten api contract",
                "signal_kinds": ["interface", "dependency"],
                "changed_symbol_names": ["fetch_state"],
                "affected_subsystem_ids": ["subsystem::src::core"],
                "touched_build_sources": ["pyproject.toml"],
                "impact_change_kinds": ["interface_change"],
                "impact_summary": "signature changed",
            }
        ],
        candidates={
            "subsystem_changes": [
                {
                    "change_id": "subsystem::src::core",
                    "status": "modified",
                    "group_path": "src/core",
                }
            ],
            "interface_changes": [],
            "dependency_changes": [],
            "algorithm_candidates": [],
        },
        modules=[
            {
                "module_id": "module::src/core/api.py",
                "path": "src/core/api.py",
                "language": "python",
                "functions": ["fetch_state"],
                "classes": [],
                "imports": ["requests"],
                "docstring_excerpts": ["Service expects strict fallback handling."],
                "semantic_labels": ["Core API"],
            }
        ],
        semantic_labels={
            "semantic_subsystems": [
                {
                    "semantic_subsystem_id": "semantic-subsystem::core-api",
                    "title": "Core API",
                    "summary": "Handles request boundaries.",
                    "capability_ids": ["capability::request-flow"],
                }
            ],
            "semantic_capabilities": [],
            "semantic_context_nodes": [],
        },
    )

    assert "do not invent commit ids" in system_prompt.lower()
    assert "tighten api contract" in user_prompt
    assert "semantic_subsystem_id" in user_prompt
    assert "diff --git" not in user_prompt
    assert "return value" not in user_prompt


def test_h3_change_ids_are_stable_and_interval_llm_cannot_invent_references(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_interval_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    baseline = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
    )
    snapshot, delta_model = _load_snapshot_and_delta(
        sample_project_config.workspace.output_root,
        repo_root.name,
        baseline.checkpoint_id,
    )

    assert delta_model.subsystem_changes[0].change_id == delta_model.subsystem_changes[0].candidate_id
    if delta_model.interface_changes:
        assert (
            delta_model.interface_changes[0].change_id
            == delta_model.interface_changes[0].candidate_id
        )
    if delta_model.dependency_changes:
        assert (
            delta_model.dependency_changes[0].change_id
            == delta_model.dependency_changes[0].candidate_id
        )

    interpretation = build_interval_interpretation(
        checkpoint_id=baseline.checkpoint_id,
        target_commit=baseline.target_commit,
        previous_checkpoint_commit=baseline.previous_checkpoint_commit,
        snapshot=snapshot,
        delta_model=delta_model,
        llm_client=_InjectingIntervalClient(
            {
                "insights": [
                    {
                        "insight_id": "interval-insight::invented",
                        "kind": "interface_change",
                        "title": "Invented",
                        "summary": "Should be rejected.",
                        "significance": "high",
                        "related_commit_ids": ["f" * 40],
                        "related_change_ids": ["invented-change-id"],
                        "related_subsystem_ids": ["invented-subsystem"],
                        "evidence_links": [
                            {
                                "kind": "commit",
                                "reference": "f" * 40,
                            }
                        ],
                    }
                ],
                "rationale_clues": [],
                "significant_windows": [],
            }
        ),
        model_name="mock-engllm",
        temperature=0.2,
    )

    assert interpretation.evaluation_status == "llm_failed"
    assert interpretation.insights


def test_build_history_docs_checkpoint_h12_writes_interval_interpretation_and_survives_llm_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, commits = _create_interval_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    monkeypatch.setattr(
        "engllm.tools.history_docs.build.create_llm_client",
        lambda config: _FailingIntervalClient(),
    )

    result = build_history_docs_checkpoint(
        project_config=load_project_config(config_path),
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
    )
    interpretation = HistoryIntervalInterpretation.model_validate_json(
        result.interval_interpretation_path.read_text(encoding="utf-8")
    )

    assert result.interval_interpretation_status == "llm_failed"
    assert result.interval_interpretation_path == interval_interpretation_path(
        output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert interpretation.evaluation_status == "llm_failed"
    assert interpretation.insights
    assert result.checkpoint_markdown_path is not None


def test_build_history_docs_checkpoint_h12_supports_scored_interpretation_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, commits = _create_interval_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    monkeypatch.setattr(
        "engllm.tools.history_docs.build.create_llm_client",
        lambda config: _InjectingIntervalClient(
            {
                "insights": [
                    {
                        "insight_id": "interval-insight::interface-core-api",
                        "kind": "interface_change",
                        "title": "Core API contract tightened",
                        "summary": "The fetch_state interface now carries a stricter request contract.",
                        "significance": "high",
                            "related_commit_ids": [commits["interface"]],
                        "related_change_ids": ["interface::src/core/api.py::fetch_state"],
                        "related_subsystem_ids": [],
                        "evidence_links": [
                            {
                                "kind": "commit",
                                "reference": commits["interface"],
                            },
                            {
                                "kind": "file",
                                "reference": "src/core/api.py",
                            },
                        ],
                    }
                ],
                "rationale_clues": [
                    {
                        "clue_id": "rationale-clue::signature::fetch-state",
                        "text": "strict: bool = True",
                        "confidence": 0.8,
                        "related_commit_ids": [commits["interface"]],
                        "related_change_ids": [
                            "interface::src/core/api.py::fetch_state"
                        ],
                        "source_kind": "signature_change",
                        "evidence_links": [
                            {
                                "kind": "file",
                                "reference": "src/core/api.py",
                            }
                        ],
                    }
                ],
                "significant_windows": [
                    {
                        "window_id": "change-window::01",
                        "start_commit": commits["interface"],
                        "end_commit": commits["head"],
                        "commit_ids": [commits["interface"], commits["head"]],
                        "title": "Strict API boundary update",
                        "summary": "This window concentrates interface and tooling changes.",
                        "significance": "high",
                        "related_insight_ids": [
                            "interval-insight::interface-core-api"
                        ],
                        "evidence_links": [
                            {
                                "kind": "commit",
                                "reference": commits["interface"],
                            }
                        ],
                    }
                ],
            }
        ),
    )

    project = load_project_config(config_path)
    first = build_history_docs_checkpoint(
        project_config=project,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
    )
    second = build_history_docs_checkpoint(
        project_config=project,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
    )

    first_text = first.interval_interpretation_path.read_text(encoding="utf-8")
    second_text = second.interval_interpretation_path.read_text(encoding="utf-8")
    interpretation = HistoryIntervalInterpretation.model_validate_json(first_text)

    assert first.interval_interpretation_status == "scored"
    assert first.interval_insight_count == 1
    assert first.interval_significant_window_count == 1
    assert interpretation.evaluation_status == "scored"
    assert interpretation.rationale_clues[0].source_kind == "signature_change"
    assert first_text == second_text


def test_history_docs_build_cli_prints_interval_interpretation_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root, commits = _create_interval_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    exit_code = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            commits["head"],
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Interval interpretation:" in captured.out
