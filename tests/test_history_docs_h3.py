"""Tests for history-docs H3 interval-delta analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.repo.history import describe_commit_diff
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import HistoryIntervalDeltaModel
from tests.history_docs_helpers import (
    commit_file,
    create_merge_repo,
    git,
    init_repo,
    interval_delta_model_path,
    snapshot_structural_model_path,
    write_project_config,
)


def test_describe_commit_diff_uses_root_and_first_parent(tmp_path: Path) -> None:
    repo_root = init_repo(tmp_path)
    root_commit = commit_file(
        repo_root,
        "src/app.py",
        "def root() -> int:\n    return 1\n",
        message="root commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    root_spec = describe_commit_diff(repo_root, root_commit)

    merge_repo_root, merge_commit = create_merge_repo(tmp_path / "mergecase")
    merge_spec = describe_commit_diff(merge_repo_root, merge_commit)

    assert root_spec.diff_basis == "root"
    assert root_spec.parent_commit is None
    assert root_spec.base_rev == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    assert merge_spec.diff_basis == "first_parent"
    assert merge_spec.parent_commit is not None
    assert merge_spec.base_rev == merge_spec.parent_commit


def test_build_history_docs_checkpoint_initial_run_writes_h3_interval_delta_model(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/app.py",
        "def first() -> int:\n    return 1\n",
        message="first commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    second_commit = commit_file(
        repo_root,
        "src/app.py",
        "def second(x: int) -> int:\n    return x\n",
        message="second commit",
        timestamp="2024-01-02T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    delta_path = interval_delta_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    delta_model = HistoryIntervalDeltaModel.model_validate_json(
        delta_path.read_text(encoding="utf-8")
    )

    assert result.interval_delta_model_path == delta_path
    assert result.previous_checkpoint_commit is None
    assert delta_model.previous_snapshot_available is False
    assert [item.commit.sha for item in delta_model.commit_deltas] == [
        first_commit,
        second_commit,
    ]
    assert delta_model.commit_deltas[0].diff_basis == "root"
    assert delta_model.commit_deltas[1].diff_basis == "first_parent"


def test_build_history_docs_checkpoint_emits_rich_interval_change_candidates(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_one = commit_file(
        repo_root,
        "src/core/engine.py",
        "import math\n\n\ndef run(x: int) -> int:\n    return x\n",
        message="add engine",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = []\n",
        encoding="utf-8",
    )
    git(repo_root, "add", "pyproject.toml")
    git(
        repo_root,
        "commit",
        "-m",
        "add manifest",
        env={
            "GIT_AUTHOR_DATE": "2024-01-01T12:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-01T12:00:00+00:00",
        },
    )
    commit_one = git(repo_root, "rev-parse", "HEAD")

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_one,
    )

    (repo_root / "src" / "core" / "engine.py").write_text(
        "import os\n\n\ndef run(x: int, y: int) -> int:\n    return x + y\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "service").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "service" / "alpha_strategy.py").write_text(
        (
            "class AlphaStrategy:\n"
            "    def score(self, value: int) -> int:\n"
            "        return value\n"
        ),
        encoding="utf-8",
    )
    (repo_root / "src" / "service" / "beta_strategy.py").write_text(
        (
            "class BetaStrategy:\n"
            "    def score(self, value: int) -> int:\n"
            "        return value * 2\n"
        ),
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = ['requests']\n",
        encoding="utf-8",
    )
    git(
        repo_root,
        "add",
        "src/core/engine.py",
        "src/service/alpha_strategy.py",
        "src/service/beta_strategy.py",
        "pyproject.toml",
    )
    git(
        repo_root,
        "commit",
        "-m",
        "introduce strategies",
        env={
            "GIT_AUTHOR_DATE": "2024-02-01T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-02-01T10:00:00+00:00",
        },
    )
    commit_two = git(repo_root, "rev-parse", "HEAD")

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_two,
    )
    delta_model = HistoryIntervalDeltaModel.model_validate_json(
        interval_delta_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert delta_model.previous_snapshot_available is True
    assert len(delta_model.commit_deltas) == 1
    commit_delta = delta_model.commit_deltas[0]
    assert commit_delta.signal_kinds == [
        "algorithm_candidate",
        "architectural",
        "dependency",
        "infrastructure",
        "interface",
    ]
    assert commit_delta.affected_subsystem_ids == [
        "subsystem::src::core",
        "subsystem::src::service",
    ]
    assert commit_delta.touched_build_sources == [Path("pyproject.toml")]

    subsystem_by_id = {
        item.candidate_id: item for item in delta_model.subsystem_changes
    }
    assert subsystem_by_id["subsystem::src::service"].status == "introduced"
    assert subsystem_by_id["subsystem::src::core"].status == "modified"

    interface_by_id = {
        item.candidate_id: item for item in delta_model.interface_changes
    }
    assert interface_by_id["interface::src/core/engine.py::run"].status == "modified"

    dependency_by_id = {
        item.candidate_id: item for item in delta_model.dependency_changes
    }
    assert dependency_by_id["dependency::pyproject.toml"].status == "modified"
    assert (
        dependency_by_id["dependency-signal::subsystem::src::core"].dependency_kind
        == "code_import_signal"
    )

    algorithm_by_id = {
        item.candidate_id: item for item in delta_model.algorithm_candidates
    }
    assert "algorithm::subsystem::subsystem::src::service" in algorithm_by_id
    assert (
        "variant_family"
        in algorithm_by_id["algorithm::subsystem::subsystem::src::service"].signal_kinds
    )


def test_build_history_docs_checkpoint_uses_diff_only_fallback_when_previous_snapshot_missing(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_one = commit_file(
        repo_root,
        "src/service/api.py",
        "class ApiService:\n    pass\n",
        message="add api",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    first_result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_one,
    )
    snapshot_structural_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        first_result.checkpoint_id,
    ).unlink()

    commit_two = commit_file(
        repo_root,
        "src/service/api.py",
        "class ApiService:\n    def run(self, value: int) -> int:\n        return value\n",
        message="expand api",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    second_result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_two,
    )
    delta_model = HistoryIntervalDeltaModel.model_validate_json(
        interval_delta_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            second_result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert delta_model.previous_snapshot_available is False
    assert all(item.status == "observed" for item in delta_model.subsystem_changes)
    assert all(item.status == "observed" for item in delta_model.interface_changes)
    assert all(item.status != "retired" for item in delta_model.subsystem_changes)


def test_build_history_docs_checkpoint_detects_retired_subsystems(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_one = commit_file(
        repo_root,
        "src/service/alpha_strategy.py",
        "class AlphaStrategy:\n    pass\n",
        message="add service strategy",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_one,
    )

    git(repo_root, "rm", "src/service/alpha_strategy.py")
    git(
        repo_root,
        "commit",
        "-m",
        "remove service strategy",
        env={
            "GIT_AUTHOR_DATE": "2024-02-01T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-02-01T10:00:00+00:00",
        },
    )
    commit_two = git(repo_root, "rev-parse", "HEAD")

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_two,
    )
    delta_model = HistoryIntervalDeltaModel.model_validate_json(
        interval_delta_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    subsystem_by_id = {
        item.candidate_id: item for item in delta_model.subsystem_changes
    }

    assert subsystem_by_id["subsystem::src::service"].status == "retired"


def test_history_docs_cli_build_prints_h3_interval_delta_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = init_repo(tmp_path)
    commit = commit_file(
        repo_root,
        "src/app.py",
        "def run() -> None:\n    pass\n",
        message="initial commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    config_path = tmp_path / "project.yaml"
    write_project_config(
        config_path,
        tmp_path / "artifacts",
        source_roots=["repo/src"],
    )

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            commit,
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "Interval delta model:" in output
