"""Tests for history-docs H5 section outline planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.analysis.history import HistoryCommitSummary
from engllm.domain.models import CommitImpact
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCommitDelta,
    HistoryDependencyConcept,
    HistoryEvidenceLink,
    HistoryIntervalDeltaModel,
    HistoryModuleConcept,
    HistorySectionOutline,
    HistorySubsystemConcept,
)
from engllm.tools.history_docs.section_outline import (
    build_section_outline,
)
from engllm.tools.history_docs.section_outline import (
    section_outline_path as build_section_outline_path,
)
from tests.history_docs_helpers import (
    checkpoint_model_path,
    commit_file,
    git,
    init_repo,
    section_outline_path,
    write_project_config,
)


def _impact(commit_range: str) -> CommitImpact:
    return CommitImpact(
        commit_range=commit_range,
        changed_files=[],
        change_kinds=[],
        impacted_sections=[],
        summary="test impact",
    )


def test_section_outline_path_is_deterministic() -> None:
    tool_root = Path("artifacts/workspaces/demo/tools/history_docs")

    assert build_section_outline_path(tool_root, "2024-01-01-abc1234") == (
        tool_root / "checkpoints" / "2024-01-01-abc1234" / "section_outline.json"
    )


def test_build_section_outline_keeps_fixed_order_and_depth_thresholds() -> None:
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234",
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        subsystems=[
            HistorySubsystemConcept(
                concept_id="subsystem::src::core",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                source_root=Path("src"),
                group_path=Path("src/core"),
                module_ids=["module::src/core/a.py", "module::src/core/b.py"],
                file_count=2,
                symbol_count=2,
                language_counts={"python": 2},
                representative_files=[Path("src/core/a.py"), Path("src/core/b.py")],
                evidence_links=[
                    HistoryEvidenceLink(
                        kind="subsystem", reference="subsystem::src::core"
                    )
                ],
            ),
            HistorySubsystemConcept(
                concept_id="subsystem::src::service",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                source_root=Path("src"),
                group_path=Path("src/service"),
                module_ids=["module::src/service/c.py", "module::src/service/d.py"],
                file_count=2,
                symbol_count=2,
                language_counts={"python": 2},
                representative_files=[
                    Path("src/service/c.py"),
                    Path("src/service/d.py"),
                ],
                evidence_links=[
                    HistoryEvidenceLink(
                        kind="subsystem", reference="subsystem::src::service"
                    )
                ],
            ),
        ],
        modules=[
            HistoryModuleConcept(
                concept_id="module::src/core/a.py",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("src/core/a.py"),
                subsystem_id="subsystem::src::core",
                language="python",
                functions=["auth_token"],
                classes=[],
                imports=[],
                docstrings=[],
                symbol_names=["auth_token"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/core/a.py")
                ],
            ),
            HistoryModuleConcept(
                concept_id="module::src/core/b.py",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("src/core/b.py"),
                subsystem_id="subsystem::src::core",
                language="python",
                functions=["jwt_secret"],
                classes=[],
                imports=[],
                docstrings=[],
                symbol_names=["jwt_secret"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/core/b.py")
                ],
            ),
            HistoryModuleConcept(
                concept_id="module::src/service/c.py",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("src/service/c.py"),
                subsystem_id="subsystem::src::service",
                language="python",
                functions=["permission_check"],
                classes=[],
                imports=[],
                docstrings=[],
                symbol_names=["permission_check"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/service/c.py")
                ],
            ),
            HistoryModuleConcept(
                concept_id="module::src/service/d.py",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("src/service/d.py"),
                subsystem_id="subsystem::src::service",
                language="python",
                functions=["run"],
                classes=[],
                imports=[],
                docstrings=[],
                symbol_names=["run"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/service/d.py")
                ],
            ),
        ],
        dependencies=[
            HistoryDependencyConcept(
                concept_id="dependency-source::pyproject.toml",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("pyproject.toml"),
                ecosystem="python",
                category="dependency_manifest",
                related_subsystem_ids=[],
                evidence_links=[
                    HistoryEvidenceLink(kind="build_source", reference="pyproject.toml")
                ],
            ),
            HistoryDependencyConcept(
                concept_id="dependency-source::requirements.txt",
                lifecycle_status="active",
                change_status="observed",
                first_seen_checkpoint="2024-01-01-abc1234",
                last_updated_checkpoint="2024-01-01-abc1234",
                path=Path("requirements.txt"),
                ecosystem="python",
                category="dependency_manifest",
                related_subsystem_ids=[],
                evidence_links=[
                    HistoryEvidenceLink(
                        kind="build_source", reference="requirements.txt"
                    )
                ],
            ),
        ],
        sections=[],
    )
    interval_delta_model = HistoryIntervalDeltaModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234",
        previous_checkpoint_commit=None,
        previous_snapshot_available=False,
        commit_deltas=[
            HistoryCommitDelta(
                commit=HistoryCommitSummary(
                    sha="a" * 40,
                    short_sha="aaaaaaa",
                    timestamp="2024-01-01T10:00:00+00:00",
                    subject="architectural commit",
                ),
                parent_commit=None,
                diff_basis="root",
                impact=_impact("root..a"),
                signal_kinds=["architectural"],
                evidence_links=[
                    HistoryEvidenceLink(kind="commit", reference="a" * 40),
                    HistoryEvidenceLink(kind="file", reference="src/core/a.py"),
                ],
            ),
            HistoryCommitDelta(
                commit=HistoryCommitSummary(
                    sha="b" * 40,
                    short_sha="bbbbbbb",
                    timestamp="2024-01-02T10:00:00+00:00",
                    subject="interface commit",
                ),
                parent_commit="a" * 40,
                diff_basis="first_parent",
                impact=_impact("a..b"),
                signal_kinds=["interface"],
                evidence_links=[
                    HistoryEvidenceLink(kind="commit", reference="b" * 40),
                    HistoryEvidenceLink(kind="file", reference="src/service/c.py"),
                ],
            ),
        ],
    )

    outline = build_section_outline(checkpoint_model, interval_delta_model)
    plans = {section.section_id: section for section in outline.sections}

    assert [section.section_id for section in outline.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "dependencies",
        "build_development_infrastructure",
        "strategy_variants_design_alternatives",
        "data_state_management",
        "error_handling_robustness",
        "performance_considerations",
        "security_considerations",
        "design_notes_rationale",
        "limitations_constraints",
    ]
    assert plans["introduction"].status == "included"
    assert plans["introduction"].depth == "brief"
    assert plans["architectural_overview"].evidence_score == 8
    assert plans["architectural_overview"].confidence_score == 80
    assert plans["architectural_overview"].depth == "deep"
    assert plans["subsystems_modules"].evidence_score == 9
    assert plans["subsystems_modules"].depth == "deep"
    assert plans["security_considerations"].status == "included"
    assert plans["security_considerations"].depth == "brief"
    assert plans["strategy_variants_design_alternatives"].status == "omitted"
    assert plans["strategy_variants_design_alternatives"].depth is None


def test_build_history_docs_checkpoint_writes_h5_section_outline_minimal_evidence(
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
    outline_path = section_outline_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    outline = HistorySectionOutline.model_validate_json(
        outline_path.read_text(encoding="utf-8")
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.section_outline_path == outline_path
    assert result.included_section_count == 5
    assert result.omitted_section_count == 8
    assert len(outline.sections) == 13
    assert sum(section.status == "included" for section in outline.sections) == 5
    assert {
        section.section_id
        for section in outline.sections
        if section.status == "omitted"
    } == {
        "algorithms_core_logic",
        "strategy_variants_design_alternatives",
        "data_state_management",
        "error_handling_robustness",
        "performance_considerations",
        "security_considerations",
        "design_notes_rationale",
        "limitations_constraints",
    }
    assert [section.section_id for section in checkpoint_model.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "algorithms_core_logic",
        "dependencies",
        "build_development_infrastructure",
    ]


def test_build_history_docs_checkpoint_h5_includes_strategy_variants_section(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/core/engine.py",
        "def run() -> int:\n    return 1\n",
        message="initial engine",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=first_commit,
    )

    commit_file(
        repo_root,
        "src/strategies/http_adapter.py",
        "class HttpAdapter:\n    def run(self) -> int:\n        return 1\n",
        message="add http adapter",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    second_commit = commit_file(
        repo_root,
        "src/strategies/grpc_adapter.py",
        "class GrpcAdapter:\n    def run(self) -> int:\n        return 2\n",
        message="add grpc adapter",
        timestamp="2024-02-02T10:00:00+00:00",
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    outline = HistorySectionOutline.model_validate_json(
        section_outline_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    plan = {section.section_id: section for section in outline.sections}[
        "strategy_variants_design_alternatives"
    ]

    assert plan.status == "included"
    assert plan.evidence_score >= 6
    assert plan.trigger_signals == ["algorithm_candidate", "variant_family"]
    assert "subsystem::src::strategies" in plan.concept_ids


def test_build_history_docs_checkpoint_h5_includes_security_section(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "src/security/auth_service.py",
        "def auth_service() -> str:\n    return 'ok'\n",
        message="add auth service",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/security/token_store.py",
        "def token_store() -> str:\n    return 'token'\n",
        message="add token store",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    commit = commit_file(
        repo_root,
        "src/security/secret_guard.py",
        "def secret_guard() -> str:\n    return 'secret'\n",
        message="add secret guard",
        timestamp="2024-01-03T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )
    outline = HistorySectionOutline.model_validate_json(
        section_outline_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    plan = {section.section_id: section for section in outline.sections}[
        "security_considerations"
    ]

    assert plan.status == "included"
    assert plan.depth == "brief"
    assert plan.trigger_signals == ["active_modules", "security_tokens"]


def test_build_history_docs_checkpoint_h5_includes_design_notes_rationale(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "src/core/engine.py",
        "def run(value: int) -> int:\n    return value\n",
        message="add engine",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/core/engine.py",
        "def run(value: int, factor: int) -> int:\n    return value * factor\n",
        message="change engine signature",
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
    (repo_root / "src" / "core" / "engine.py").write_text(
        "def run(value: int, factor: int) -> int:\n    return value * factor\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "cli").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "cli" / "view.py").write_text(
        "def render() -> str:\n    return 'ok'\n",
        encoding="utf-8",
    )
    git(repo_root, "add", "src/core/engine.py", "src/cli/view.py")
    git(
        repo_root,
        "commit",
        "-m",
        "architectural expansion",
        env={
            "GIT_AUTHOR_DATE": "2024-01-04T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-04T10:00:00+00:00",
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
    outline_path = section_outline_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    outline = HistorySectionOutline.model_validate_json(
        outline_path.read_text(encoding="utf-8")
    )
    before_checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    outline_json_before = json.loads(outline_path.read_text(encoding="utf-8"))

    second_result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )
    outline_json_after = json.loads(
        section_outline_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            second_result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    plan = {section.section_id: section for section in outline.sections}[
        "design_notes_rationale"
    ]

    assert plan.status == "included"
    assert plan.evidence_score >= 6
    assert "rationale_change" in plan.trigger_signals
    assert [section.section_id for section in before_checkpoint_model.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "algorithms_core_logic",
        "dependencies",
        "build_development_infrastructure",
    ]
    assert outline_json_before == outline_json_after


def test_history_docs_cli_build_prints_h5_section_outline_path(
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
    assert "Section outline:" in output
