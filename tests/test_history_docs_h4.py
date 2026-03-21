"""Tests for history-docs H4 checkpoint documentation models."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import HistoryCheckpointModel
from tests.history_docs_helpers import (
    checkpoint_model_path,
    commit_file,
    git,
    init_repo,
    write_project_config,
)


def test_build_history_docs_checkpoint_initial_run_writes_h4_checkpoint_model(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit = commit_file(
        repo_root,
        "src/app.py",
        "def run() -> int:\n    return 1\n",
        message="initial app",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )
    git(repo_root, "add", "pyproject.toml")
    git(
        repo_root,
        "commit",
        "-m",
        "add manifest",
        env={
            "GIT_AUTHOR_DATE": "2024-01-02T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-02T10:00:00+00:00",
        },
    )
    commit = git(repo_root, "rev-parse", "HEAD")

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )
    model_path = checkpoint_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        model_path.read_text(encoding="utf-8")
    )

    assert result.checkpoint_model_path == model_path
    assert checkpoint_model.previous_checkpoint_model_available is False
    assert [section.section_id for section in checkpoint_model.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "dependencies",
        "build_development_infrastructure",
    ]
    assert all(
        concept.lifecycle_status == "active"
        for concept in [
            *checkpoint_model.subsystems,
            *checkpoint_model.modules,
            *checkpoint_model.dependencies,
        ]
    )
    assert checkpoint_model.sections[0].concept_ids == []
    assert checkpoint_model.modules[0].concept_id == "module::src/app.py"
    assert (
        checkpoint_model.dependencies[0].concept_id
        == "dependency-source::pyproject.toml"
    )


def test_build_history_docs_checkpoint_merges_previous_model_and_tracks_changes(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "src/core/engine.py",
        "def run(value: int) -> int:\n    return value\n",
        message="add core",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/cli/view.py",
        "def render() -> str:\n    return 'ok'\n",
        message="add cli",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )
    git(repo_root, "add", "pyproject.toml")
    git(
        repo_root,
        "commit",
        "-m",
        "add manifest",
        env={
            "GIT_AUTHOR_DATE": "2024-01-03T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-03T10:00:00+00:00",
        },
    )
    first_commit = git(repo_root, "rev-parse", "HEAD")

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=first_commit,
    )

    (repo_root / "src" / "core" / "engine.py").write_text(
        "def run(value: int, factor: int) -> int:\n    return value * factor\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "service").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "service" / "api.py").write_text(
        "class ApiService:\n    def execute(self) -> int:\n        return 1\n",
        encoding="utf-8",
    )
    git(repo_root, "add", "src/core/engine.py", "src/service/api.py")
    git(
        repo_root,
        "commit",
        "-m",
        "expand service",
        env={
            "GIT_AUTHOR_DATE": "2024-02-01T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-02-01T10:00:00+00:00",
        },
    )
    second_commit = git(repo_root, "rev-parse", "HEAD")

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    subsystem_by_id = {
        concept.concept_id: concept for concept in checkpoint_model.subsystems
    }
    assert subsystem_by_id["subsystem::src::cli"].change_status == "unchanged"
    assert subsystem_by_id["subsystem::src::core"].change_status == "modified"
    assert subsystem_by_id["subsystem::src::service"].change_status == "introduced"

    module_by_id = {concept.concept_id: concept for concept in checkpoint_model.modules}
    assert module_by_id["module::src/cli/view.py"].change_status == "unchanged"
    assert module_by_id["module::src/core/engine.py"].change_status == "modified"
    assert module_by_id["module::src/service/api.py"].change_status == "introduced"

    dependency_by_id = {
        concept.concept_id: concept for concept in checkpoint_model.dependencies
    }
    assert (
        dependency_by_id["dependency-source::pyproject.toml"].change_status
        == "unchanged"
    )

    section_by_id = {
        section.section_id: section for section in checkpoint_model.sections
    }
    assert section_by_id["architectural_overview"].concept_ids == [
        "subsystem::src::cli",
        "subsystem::src::core",
        "subsystem::src::service",
    ]
    assert all(
        not concept_id.startswith("retired")
        for concept_id in section_by_id["subsystems_modules"].concept_ids
    )


def test_build_history_docs_checkpoint_uses_previous_model_fallback_when_missing(
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
    checkpoint_model_path(
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
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            second_result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert checkpoint_model.previous_checkpoint_model_available is False
    assert all(
        concept.lifecycle_status == "active"
        for concept in [
            *checkpoint_model.subsystems,
            *checkpoint_model.modules,
            *checkpoint_model.dependencies,
        ]
    )
    assert all(
        concept.change_status == "observed" for concept in checkpoint_model.modules
    )


def test_build_history_docs_checkpoint_retains_retired_concepts(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_one = commit_file(
        repo_root,
        "src/service/api.py",
        "class ApiService:\n    pass\n",
        message="add service",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit_one,
    )

    git(repo_root, "rm", "src/service/api.py")
    git(
        repo_root,
        "commit",
        "-m",
        "remove service",
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
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    subsystem_by_id = {
        concept.concept_id: concept for concept in checkpoint_model.subsystems
    }
    module_by_id = {concept.concept_id: concept for concept in checkpoint_model.modules}
    assert subsystem_by_id["subsystem::src::service"].lifecycle_status == "retired"
    assert module_by_id["module::src/service/api.py"].lifecycle_status == "retired"

    section_by_id = {
        section.section_id: section for section in checkpoint_model.sections
    }
    assert (
        "subsystem::src::service"
        not in section_by_id["architectural_overview"].concept_ids
    )
    assert (
        "module::src/service/api.py"
        not in section_by_id["subsystems_modules"].concept_ids
    )


def test_history_docs_cli_build_prints_h4_checkpoint_model_path(
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
    assert "Checkpoint model:" in output
