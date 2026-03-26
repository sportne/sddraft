"""Tests for history-docs H11-03 semantic context and interface extraction."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.domain.models import CodeUnitSummary, SymbolSummary
from engllm.llm.mock import MockLLMClient
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryRenderManifest,
    HistorySectionOutline,
    HistorySemanticContextJudgment,
    HistorySemanticContextMap,
    HistorySemanticStructureJudgment,
    HistorySemanticStructureMap,
    HistorySemanticSubsystemCluster,
    HistorySnapshotStructuralModel,
)
from engllm.tools.history_docs.semantic_context import (
    build_semantic_context_map,
)
from engllm.tools.history_docs.semantic_context import (
    semantic_context_map_path as build_semantic_context_map_path,
)
from tests.history_docs_helpers import (
    checkpoint_markdown_path,
    commit_file,
    init_repo,
    section_outline_path,
    semantic_context_map_path,
    write_project_config,
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


def _semantic_context_payload() -> dict[str, object]:
    return {
        "context_nodes": [
            {
                "node_id": "context-node::system",
                "title": "Semantic Context Repo System",
                "kind": "system",
                "summary": "Represents the current application boundary.",
                "related_subsystem_ids": [
                    "semantic-subsystem::api",
                    "semantic-subsystem::engine",
                    "semantic-subsystem::storage",
                ],
                "related_module_ids": [
                    "module::src/app/api/router.py",
                    "module::src/app/engine/planner.py",
                    "module::src/app/storage/repository.py",
                ],
            },
            {
                "node_id": "context-node::data-store",
                "title": "State Store",
                "kind": "data_store",
                "summary": "Covers the repository-backed state boundary.",
                "related_subsystem_ids": ["semantic-subsystem::storage"],
                "related_module_ids": ["module::src/app/storage/repository.py"],
            },
        ],
        "interfaces": [
            {
                "interface_id": "interface::http_api::router",
                "title": "HTTP API: Router",
                "kind": "http_api",
                "summary": "Routes inbound requests into the planning flow.",
                "provider_subsystem_ids": ["semantic-subsystem::api"],
                "consumer_context_node_ids": ["context-node::system"],
                "related_module_ids": ["module::src/app/api/router.py"],
            }
        ],
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
                docstrings=["API routing boundary."],
                imports=["from app.engine.planner import build_plan"],
            ),
            CodeUnitSummary(
                path=Path("src/app/engine/planner.py"),
                language="python",
                functions=["build_plan"],
                docstrings=["Plan construction flow."],
                imports=["from app.storage.repository import StateRepository"],
            ),
            CodeUnitSummary(
                path=Path("src/app/storage/repository.py"),
                language="python",
                classes=["StateRepository"],
                docstrings=["Checkpoint state storage."],
            ),
        ],
        symbol_summaries=[
            SymbolSummary(
                name="route_request",
                qualified_name="route_request",
                kind="function",
                language="python",
                source_path=Path("src/app/api/router.py"),
            ),
            SymbolSummary(
                name="build_plan",
                qualified_name="build_plan",
                kind="function",
                language="python",
                source_path=Path("src/app/engine/planner.py"),
            ),
            SymbolSummary(
                name="StateRepository",
                qualified_name="StateRepository",
                kind="class",
                language="python",
                source_path=Path("src/app/storage/repository.py"),
            ),
        ],
    )


def _semantic_structure_map() -> HistorySemanticStructureMap:
    return HistorySemanticStructureMap(
        checkpoint_id="2024-05-01-demo001",
        target_commit="a" * 40,
        evaluation_status="scored",
        semantic_subsystems=[
            HistorySemanticSubsystemCluster(
                semantic_subsystem_id="semantic-subsystem::api",
                title="API Layer",
                summary="Handles request routing and boundary validation.",
                module_ids=["module::src/app/api/router.py"],
                baseline_subsystem_candidate_ids=["subsystem::src::app"],
            ),
            HistorySemanticSubsystemCluster(
                semantic_subsystem_id="semantic-subsystem::engine",
                title="Planning Engine",
                summary="Builds execution plans from request context.",
                module_ids=["module::src/app/engine/planner.py"],
                baseline_subsystem_candidate_ids=["subsystem::src::app"],
            ),
            HistorySemanticSubsystemCluster(
                semantic_subsystem_id="semantic-subsystem::storage",
                title="State Storage",
                summary="Stores and retrieves checkpoint state.",
                module_ids=["module::src/app/storage/repository.py"],
                baseline_subsystem_candidate_ids=["subsystem::src::app"],
            ),
        ],
    )


def _create_semantic_context_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "pyproject.toml",
        '[project]\nname = "semantic-context"\ndependencies = ["requests>=2"]\n',
        message="add manifest",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/api/router.py",
        '"""API routing boundary."""\n\n\ndef route_request(path: str) -> str:\n    return path\n',
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


def test_semantic_context_map_path_and_invalid_references_fall_back(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot_model()
    invalid_client = MockLLMClient(
        canned={
            HistorySemanticContextJudgment.__name__: {
                "context_nodes": [
                    {
                        "node_id": "context-node::system",
                        "title": "Broken System",
                        "kind": "system",
                        "summary": "Invalid node references unknown module.",
                        "related_subsystem_ids": ["semantic-subsystem::api"],
                        "related_module_ids": ["module::missing.py"],
                    }
                ],
                "interfaces": [],
            }
        }
    )

    assert build_semantic_context_map_path(
        tmp_path / "artifacts" / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-05-01-demo001",
    ) == semantic_context_map_path(
        tmp_path / "artifacts",
        "repo",
        "2024-05-01-demo001",
    )

    context_map = build_semantic_context_map(
        workspace_id="repo",
        checkpoint_id=snapshot.checkpoint_id,
        target_commit=snapshot.target_commit,
        previous_checkpoint_commit=None,
        snapshot=snapshot,
        semantic_map=_semantic_structure_map(),
        llm_client=invalid_client,
        model_name="mock-engllm",
        temperature=0.2,
    )

    assert context_map.evaluation_status == "llm_failed"
    assert sum(node.kind == "system" for node in context_map.context_nodes) == 1
    assert context_map.context_nodes[0].node_id == "context-node::system"


def test_history_docs_build_writes_semantic_context_map_but_keeps_sections_shadowed_by_default(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, target = _create_semantic_context_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [Path("src")]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=target,
    )

    context_path = semantic_context_map_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    outline_path = section_outline_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    markdown_path = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    context_map = HistorySemanticContextMap.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    outline = HistorySectionOutline.model_validate_json(
        outline_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert result.semantic_context_map_path == context_path
    assert result.semantic_context_status == context_map.evaluation_status
    assert sum(node.kind == "system" for node in context_map.context_nodes) == 1
    assert [
        section.section_id
        for section in outline.sections
        if section.section_id in {"system_context", "interfaces"}
    ] == []
    assert "## System Context" not in markdown
    assert "## Interfaces" not in markdown


def test_semantic_context_experimental_variant_renders_system_context_and_interfaces(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, target = _create_semantic_context_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [Path("src")]
    semantic_client = MockLLMClient(
        canned={
            HistorySemanticStructureJudgment.__name__: _semantic_structure_payload(),
            HistorySemanticContextJudgment.__name__: _semantic_context_payload(),
        }
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=target,
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=semantic_client,
    )

    context_path = semantic_context_map_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    outline_path = section_outline_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    markdown_path = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    render_manifest_path = (
        sample_project_config.workspace.output_root
        / "workspaces"
        / repo_root.name
        / "tools"
        / "history_docs"
        / "checkpoints"
        / result.checkpoint_id
        / "render_manifest.json"
    )
    context_map = HistorySemanticContextMap.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    outline = HistorySectionOutline.model_validate_json(
        outline_path.read_text(encoding="utf-8")
    )
    render_manifest = HistoryRenderManifest.model_validate_json(
        render_manifest_path.read_text(encoding="utf-8")
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert context_map.evaluation_status == "scored"
    assert result.context_node_count == 2
    assert result.interface_candidate_count == 1
    assert [
        section.section_id
        for section in outline.sections
        if section.section_id in {"system_context", "interfaces"}
        and section.status == "included"
    ] == ["system_context", "interfaces"]
    assert "## System Context" in markdown
    assert "## Interfaces" in markdown
    assert "### HTTP API: Router" in markdown
    assert "Routes inbound requests into the planning flow." in markdown
    context_sections = {
        section.section_id: section for section in render_manifest.sections
    }
    assert (
        Path("semantic_context_map.json")
        in context_sections["system_context"].source_artifact_paths
    )
    assert (
        Path("semantic_context_map.json")
        in context_sections["interfaces"].source_artifact_paths
    )


def test_history_docs_cli_build_promotes_semantic_context_rendering(
    tmp_path: Path,
    capsys,
) -> None:
    repo_root, target = _create_semantic_context_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            target,
        ]
    )
    stdout = capsys.readouterr().out
    checkpoint_root = (
        output_root
        / "workspaces"
        / repo_root.name
        / "tools"
        / "history_docs"
        / "checkpoints"
    )
    checkpoint_ids = sorted(
        path.name for path in checkpoint_root.iterdir() if path.is_dir()
    )

    markdown = checkpoint_markdown_path(
        output_root,
        repo_root.name,
        checkpoint_ids[0],
    ).read_text(encoding="utf-8")
    outline = HistorySectionOutline.model_validate_json(
        section_outline_path(
            output_root,
            repo_root.name,
            checkpoint_ids[0],
        ).read_text(encoding="utf-8")
    )

    assert rc == 0
    assert checkpoint_ids
    assert "Checkpoint markdown:" in stdout
    assert "## System Context" in markdown
    assert "## Interfaces" in markdown
    assert {
        section.section_id
        for section in outline.sections
        if section.status == "included"
    } >= {"system_context", "interfaces"}
