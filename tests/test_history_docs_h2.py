"""Tests for history-docs H2 snapshot analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.analysis.history import load_snapshot_manifest
from engllm.domain.errors import RepositoryError
from engllm.domain.models import ProjectConfig
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import HistorySnapshotStructuralModel
from tests.history_docs_helpers import (
    commit_file,
    git,
    init_repo,
    snapshot_manifest_path,
    snapshot_structural_model_path,
    write_project_config,
)


def test_build_history_docs_checkpoint_writes_h2_snapshot_artifacts_and_uses_commit_state(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root = init_repo(tmp_path)
    commit = commit_file(
        repo_root,
        "app/src/main.py",
        "def committed() -> int:\n    return 1\n",
        message="add main module",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "app/src/service/api.py",
        "class ApiService:\n    pass\n",
        message="add service module",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'example'\n", encoding="utf-8"
    )
    git(repo_root, "add", "pyproject.toml")
    git(
        repo_root,
        "commit",
        "-m",
        "add project manifest",
        env={
            "GIT_AUTHOR_DATE": "2024-01-03T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-03T10:00:00+00:00",
        },
    )
    commit = git(repo_root, "rev-parse", "HEAD")

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "app" / "src"]

    # Dirty working tree changes must not leak into snapshot analysis.
    (repo_root / "app" / "src" / "main.py").write_text(
        "def uncommitted() -> int:\n    return 2\n", encoding="utf-8"
    )
    (repo_root / "app" / "src" / "dirty.py").write_text(
        "DIRTY = True\n", encoding="utf-8"
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )

    manifest_path = snapshot_manifest_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    structural_path = snapshot_structural_model_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    snapshot_manifest = load_snapshot_manifest(manifest_path)
    structural_model = HistorySnapshotStructuralModel.model_validate_json(
        structural_path.read_text(encoding="utf-8")
    )

    assert result.snapshot_manifest_path == manifest_path
    assert result.snapshot_structural_model_path == structural_path
    assert snapshot_manifest is not None
    assert snapshot_manifest.export_strategy == "git_archive_temp"
    assert snapshot_manifest.persisted_snapshot is False
    assert snapshot_manifest.analyzed_source_roots == [Path("app/src")]
    assert snapshot_manifest.skipped_source_roots == []
    assert snapshot_manifest.manifest_search_directories == [
        Path("."),
        Path("app"),
        Path("app/src"),
    ]
    assert [item.path for item in structural_model.build_sources] == [
        Path("pyproject.toml")
    ]
    symbol_names = {item.name for item in structural_model.symbol_summaries}
    assert "committed" in symbol_names
    assert "uncommitted" not in symbol_names
    assert Path("app/src/dirty.py") not in structural_model.files
    assert [item.candidate_id for item in structural_model.subsystem_candidates] == [
        "subsystem::app/src::.",
        "subsystem::app/src::service",
    ]


def test_build_history_docs_checkpoint_skips_missing_roots_and_limits_manifest_scope(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root = init_repo(tmp_path)
    commit = commit_file(
        repo_root,
        "src/app.py",
        "def run() -> None:\n    pass\n",
        message="add source root",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    (repo_root / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (repo_root / "other").mkdir()
    (repo_root / "other" / "package.json").write_text(
        '{"name": "other"}\n', encoding="utf-8"
    )
    git(repo_root, "add", "go.mod", "other/package.json")
    git(
        repo_root,
        "commit",
        "-m",
        "add build sources",
        env={
            "GIT_AUTHOR_DATE": "2024-01-02T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-02T10:00:00+00:00",
        },
    )
    commit = git(repo_root, "rev-parse", "HEAD")

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src", repo_root / "legacy"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )

    snapshot_manifest = load_snapshot_manifest(
        snapshot_manifest_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        )
    )
    structural_model = HistorySnapshotStructuralModel.model_validate_json(
        snapshot_structural_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert snapshot_manifest is not None
    assert snapshot_manifest.analyzed_source_roots == [Path("src")]
    assert snapshot_manifest.skipped_source_roots == [Path("legacy")]
    assert snapshot_manifest.manifest_search_directories == [Path("."), Path("src")]
    assert [item.path for item in structural_model.build_sources] == [Path("go.mod")]
    assert Path("other/package.json") not in {
        item.path for item in structural_model.build_sources
    }


def test_build_history_docs_checkpoint_rejects_source_roots_outside_repo(
    tmp_path: Path,
    sample_project_config: ProjectConfig,
) -> None:
    repo_root = init_repo(tmp_path)
    commit = commit_file(
        repo_root,
        "src/app.py",
        "print('ok')\n",
        message="initial commit",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [tmp_path / "outside"]

    with pytest.raises(RepositoryError, match="must live inside repo root"):
        build_history_docs_checkpoint(
            project_config=sample_project_config,
            repo_root=repo_root,
            checkpoint_commit=commit,
        )


def test_history_docs_cli_build_prints_h2_artifact_paths(
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
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'example'\n", encoding="utf-8"
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
    assert "Snapshot manifest:" in output
    assert "Snapshot structural model:" in output
