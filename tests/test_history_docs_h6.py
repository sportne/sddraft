"""Tests for history-docs H6 algorithm capsules and linking."""

from __future__ import annotations

from pathlib import Path

import pytest

import engllm.tools.history_docs.algorithm_capsules as algorithm_capsules
from engllm.cli.main import main
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCandidate,
    HistoryCheckpointModel,
    HistoryEvidenceLink,
    HistoryInterfaceChangeCandidate,
    HistoryIntervalDeltaModel,
    HistoryModuleConcept,
    HistorySectionOutline,
    HistorySubsystemConcept,
)
from tests.history_docs_helpers import (
    algorithm_capsule_index_path,
    checkpoint_model_path,
    commit_file,
    init_repo,
    section_outline_path,
    write_project_config,
)


def _module(
    path: str,
    *,
    subsystem_id: str,
    functions: list[str],
    classes: list[str],
    docstrings: list[str] | None = None,
    symbols: list[str] | None = None,
) -> HistoryModuleConcept:
    module_path = Path(path)
    return HistoryModuleConcept(
        concept_id=f"module::{path}",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        path=module_path,
        subsystem_id=subsystem_id,
        language="python",
        functions=functions,
        classes=classes,
        imports=[],
        docstrings=docstrings or [],
        symbol_names=symbols or [*functions, *classes],
        evidence_links=[HistoryEvidenceLink(kind="file", reference=path)],
    )


def _subsystem(subsystem_id: str, module_ids: list[str]) -> HistorySubsystemConcept:
    group_path = Path(subsystem_id.split("::", 2)[-1])
    return HistorySubsystemConcept(
        concept_id=subsystem_id,
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        source_root=Path("src"),
        group_path=group_path,
        module_ids=module_ids,
        file_count=len(module_ids),
        symbol_count=len(module_ids) * 2,
        language_counts={"python": len(module_ids)},
        representative_files=[
            Path(module_id.removeprefix("module::")) for module_id in module_ids[:2]
        ],
        evidence_links=[HistoryEvidenceLink(kind="subsystem", reference=subsystem_id)],
    )


def test_algorithm_capsule_paths_and_filename_are_deterministic() -> None:
    tool_root = Path("artifacts/workspaces/demo/tools/history_docs")

    assert algorithm_capsules.algorithm_capsule_index_path(
        tool_root, "2024-01-01-abc1234"
    ) == (
        tool_root
        / "checkpoints"
        / "2024-01-01-abc1234"
        / "algorithm_capsules"
        / "index.json"
    )
    assert (
        algorithm_capsules.algorithm_capsule_filename(
            "algorithm-capsule::subsystem::subsystem::src::strategies"
        )
        == "algorithm-capsule__subsystem__subsystem__src__strategies.json"
    )


def test_build_algorithm_capsules_groups_candidates_and_derives_structure() -> None:
    subsystem_id = "subsystem::src::strategies"
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-02-01-abc1234",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        previous_checkpoint_model_available=True,
        subsystems=[
            _subsystem(
                subsystem_id,
                [
                    "module::src/strategies/http_adapter.py",
                    "module::src/strategies/grpc_adapter.py",
                ],
            )
        ],
        modules=[
            _module(
                "src/strategies/http_adapter.py",
                subsystem_id=subsystem_id,
                functions=["build_plan", "execute_strategy"],
                classes=["RequestContext"],
                docstrings=["HTTP adapter must stay deterministic."],
                symbols=["build_plan", "execute_strategy", "RequestContext"],
            ),
            _module(
                "src/strategies/grpc_adapter.py",
                subsystem_id=subsystem_id,
                functions=["build_plan", "execute_strategy"],
                classes=["RequestContext"],
                docstrings=["gRPC adapter only supports strict mode."],
                symbols=["build_plan", "execute_strategy", "RequestContext"],
            ),
        ],
        dependencies=[],
        sections=[],
    )
    interval_delta_model = HistoryIntervalDeltaModel(
        checkpoint_id="2024-02-01-abc1234",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        previous_snapshot_available=True,
        interface_changes=[
            HistoryInterfaceChangeCandidate(
                candidate_id="interface::src/strategies/http_adapter.py::build_plan",
                status="modified",
                scope_kind="symbol",
                source_path=Path("src/strategies/http_adapter.py"),
                symbol_name="build_plan",
                qualified_name="build_plan",
                commit_ids=["c" * 40],
                signature_changes=[
                    "def build_plan(config: RequestContext, strict: bool) -> RequestContext:"
                ],
                evidence_links=[
                    HistoryEvidenceLink(
                        kind="file", reference="src/strategies/http_adapter.py"
                    )
                ],
            )
        ],
        algorithm_candidates=[
            HistoryAlgorithmCandidate(
                candidate_id=f"algorithm::subsystem::{subsystem_id}",
                scope_kind="subsystem",
                scope_path=Path("src/strategies"),
                subsystem_id=subsystem_id,
                commit_ids=["c" * 40],
                changed_symbol_names=["build_plan", "execute_strategy"],
                variant_names=["grpc_adapter", "http_adapter"],
                signal_kinds=["introduced_module", "variant_family"],
                evidence_links=[
                    HistoryEvidenceLink(kind="subsystem", reference=subsystem_id),
                    HistoryEvidenceLink(
                        kind="file", reference="src/strategies/http_adapter.py"
                    ),
                    HistoryEvidenceLink(
                        kind="file", reference="src/strategies/grpc_adapter.py"
                    ),
                ],
            ),
            HistoryAlgorithmCandidate(
                candidate_id="algorithm::file::src/strategies/http_adapter.py",
                scope_kind="file",
                scope_path=Path("src/strategies/http_adapter.py"),
                subsystem_id=subsystem_id,
                commit_ids=["c" * 40],
                changed_symbol_names=[
                    "build_plan",
                    "execute_strategy",
                    "RequestContext",
                ],
                signal_kinds=["multi_symbol"],
                evidence_links=[
                    HistoryEvidenceLink(
                        kind="file", reference="src/strategies/http_adapter.py"
                    )
                ],
            ),
            HistoryAlgorithmCandidate(
                candidate_id="algorithm::file::src/misc/helper.py",
                scope_kind="file",
                scope_path=Path("src/misc/helper.py"),
                subsystem_id=None,
                commit_ids=["d" * 40],
                changed_symbol_names=["helper"],
                signal_kinds=[],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/misc/helper.py")
                ],
            ),
        ],
    )

    index, capsules = algorithm_capsules.build_algorithm_capsules(
        checkpoint_model, interval_delta_model
    )

    assert len(index.capsules) == 1
    assert len(capsules) == 1
    capsule = capsules[0]
    assert capsule.capsule_id == f"algorithm-capsule::subsystem::{subsystem_id}"
    assert capsule.related_subsystem_ids == [subsystem_id]
    assert capsule.related_module_ids == [
        "module::src/strategies/grpc_adapter.py",
        "module::src/strategies/http_adapter.py",
    ]
    assert capsule.variant_names == ["grpc_adapter", "http_adapter"]
    assert [item.name for item in capsule.shared_abstractions] == [
        "build_plan",
        "execute_strategy",
        "RequestContext",
    ]
    assert [item.name for item in capsule.data_structures] == ["RequestContext"]
    assert [item.phase_key for item in capsule.phases] == [
        "plan",
        "build",
        "execute",
    ]
    assert sorted(item.source_kind for item in capsule.assumptions) == [
        "docstring",
        "docstring",
        "signature_change",
    ]
    assert index.capsules[0].artifact_path == Path(
        "algorithm_capsules/algorithm-capsule__subsystem__subsystem__src__strategies.json"
    )


def test_build_algorithm_capsules_emits_standalone_file_capsule_only_when_qualified() -> (
    None
):
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-02-01-abc1234",
        target_commit="a" * 40,
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        subsystems=[_subsystem("subsystem::src::core", ["module::src/core/engine.py"])],
        modules=[
            _module(
                "src/core/engine.py",
                subsystem_id="subsystem::src::core",
                functions=["parse_plan", "build_index", "run"],
                classes=[],
                symbols=["parse_plan", "build_index", "run"],
            )
        ],
        dependencies=[],
        sections=[],
    )
    interval_delta_model = HistoryIntervalDeltaModel(
        checkpoint_id="2024-02-01-abc1234",
        target_commit="a" * 40,
        previous_checkpoint_commit=None,
        previous_snapshot_available=False,
        algorithm_candidates=[
            HistoryAlgorithmCandidate(
                candidate_id="algorithm::file::src/core/engine.py",
                scope_kind="file",
                scope_path=Path("src/core/engine.py"),
                subsystem_id="subsystem::src::core",
                commit_ids=["a" * 40],
                changed_symbol_names=["parse_plan", "build_index", "run"],
                signal_kinds=["multi_symbol"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/core/engine.py")
                ],
            ),
            HistoryAlgorithmCandidate(
                candidate_id="algorithm::file::src/core/helper.py",
                scope_kind="file",
                scope_path=Path("src/core/helper.py"),
                subsystem_id="subsystem::src::core",
                commit_ids=["b" * 40],
                changed_symbol_names=["helper"],
                signal_kinds=["multi_symbol"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/core/helper.py")
                ],
            ),
        ],
    )

    _, capsules = algorithm_capsules.build_algorithm_capsules(
        checkpoint_model, interval_delta_model
    )

    assert [capsule.capsule_id for capsule in capsules] == [
        "algorithm-capsule::file::src/core/engine.py"
    ]
    assert capsules[0].status == "observed"


def test_build_history_docs_checkpoint_h6_writes_empty_capsule_index_when_no_candidates(
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

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commit,
    )
    index_path = algorithm_capsule_index_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    outline = HistorySectionOutline.model_validate_json(
        section_outline_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    index_json = index_path.read_text(encoding="utf-8")

    assert result.algorithm_capsule_index_path == index_path
    assert result.algorithm_capsule_count == 0
    assert '"capsules": []' in index_json
    outline_by_id = {section.section_id: section for section in outline.sections}
    assert outline_by_id["algorithms_core_logic"].status == "omitted"
    assert [section.section_id for section in checkpoint_model.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "algorithms_core_logic",
        "dependencies",
        "build_development_infrastructure",
    ]
    assert checkpoint_model.algorithm_capsule_ids == []
    assert checkpoint_model.sections[3].algorithm_capsule_ids == []


def test_build_history_docs_checkpoint_h6_links_variant_capsule_into_models(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/core/base.py",
        "def base() -> int:\n    return 1\n",
        message="base",
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
        '"""HTTP adapter must remain deterministic."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add http adapter",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    second_commit = commit_file(
        repo_root,
        "src/strategies/grpc_adapter.py",
        '"""gRPC adapter only supports strict mode."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add grpc adapter",
        timestamp="2024-02-02T10:00:00+00:00",
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    index_path = algorithm_capsule_index_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    index_json = index_path.read_text(encoding="utf-8")
    outline = HistorySectionOutline.model_validate_json(
        section_outline_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.algorithm_capsule_count == 1
    assert "algorithm-capsule::subsystem::subsystem::src::strategies" in index_json
    assert checkpoint_model.algorithm_capsule_ids == [
        "algorithm-capsule::subsystem::subsystem::src::strategies"
    ]
    subsystem_by_id = {
        concept.concept_id: concept for concept in checkpoint_model.subsystems
    }
    assert subsystem_by_id["subsystem::src::strategies"].algorithm_capsule_ids == [
        "algorithm-capsule::subsystem::subsystem::src::strategies"
    ]
    assert [section.section_id for section in checkpoint_model.sections] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
        "algorithms_core_logic",
        "dependencies",
        "build_development_infrastructure",
    ]
    checkpoint_section_by_id = {
        section.section_id: section for section in checkpoint_model.sections
    }
    assert checkpoint_section_by_id["algorithms_core_logic"].algorithm_capsule_ids == [
        "algorithm-capsule::subsystem::subsystem::src::strategies"
    ]

    outline_by_id = {section.section_id: section for section in outline.sections}
    assert outline_by_id["algorithms_core_logic"].status == "included"
    assert outline_by_id["algorithms_core_logic"].algorithm_capsule_ids == [
        "algorithm-capsule::subsystem::subsystem::src::strategies"
    ]
    assert outline_by_id[
        "strategy_variants_design_alternatives"
    ].algorithm_capsule_ids == [
        "algorithm-capsule::subsystem::subsystem::src::strategies"
    ]


def test_history_docs_cli_build_prints_h6_capsule_index_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
    assert "Algorithm capsules:" in output
