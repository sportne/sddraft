"""Tests for history-docs H13 shadow artifacts and internal modes."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import (
    build_algorithm_capsule_enrichment_prompt,
    build_dependency_landscape_prompt,
    build_interface_inventory_prompt,
)
from engllm.tools.history_docs.algorithm_capsule_enrichment import (
    algorithm_capsule_enrichment_index_path as build_algorithm_capsule_enrichment_index_path,
)
from engllm.tools.history_docs.benchmark import (
    semantic_structure_context_dependency_landscape_benchmark_variant,
    semantic_structure_context_enriched_algorithms_benchmark_variant,
    semantic_structure_context_h13_full_benchmark_variant,
    semantic_structure_context_interface_inventory_benchmark_variant,
)
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.checkpoint_model_enrichment import (
    checkpoint_model_enrichment_path as build_checkpoint_model_enrichment_path,
)
from engllm.tools.history_docs.dependency_landscape import (
    dependency_landscape_path as build_dependency_landscape_path,
)
from engllm.tools.history_docs.h13_evidence import (
    active_modules,
    active_subsystems,
    artifact_path,
    build_h13_evidence_pack,
    compact_checkpoint_summary,
    module_path_label,
)
from engllm.tools.history_docs.interface_inventory import (
    build_interface_inventory,
)
from engllm.tools.history_docs.interface_inventory import (
    interface_inventory_path as build_interface_inventory_path,
)
from engllm.tools.history_docs.interval_interpretation import (
    interval_interpretation_path as build_interval_interpretation_path,
)
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleEnrichmentIndex,
    HistoryAlgorithmCapsuleEnrichmentJudgment,
    HistoryAlgorithmCapsuleIndex,
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryDependencyInventory,
    HistoryDependencyLandscape,
    HistoryDependencyLandscapeJudgment,
    HistoryInterfaceInventory,
    HistoryInterfaceInventoryJudgment,
    HistoryIntervalInterpretation,
    HistorySemanticContextMap,
)
from tests.history_docs_helpers import (
    algorithm_capsule_enrichment_index_path,
    algorithm_capsule_index_path,
    checkpoint_markdown_path,
    checkpoint_model_path,
    commit_file,
    dependencies_artifact_path,
    dependency_landscape_path,
    init_repo,
    interface_inventory_path,
    semantic_context_map_path,
    write_project_config,
)


class _InvalidH13Client(MockLLMClient):
    def __init__(self) -> None:
        super().__init__(
            canned={
                HistoryAlgorithmCapsuleEnrichmentJudgment.__name__: {
                    "enrichments": [
                        {
                            "capsule_id": "invented-capsule",
                            "purpose": "Should be rejected.",
                            "phase_flow_summary": "Invented flow.",
                            "invariants": [],
                            "tradeoffs": [],
                            "variant_relationships": [],
                            "related_subsystem_ids": [],
                            "related_module_ids": [],
                            "source_insight_ids": [],
                            "source_rationale_clue_ids": [],
                            "evidence_links": [],
                        }
                    ]
                },
                HistoryInterfaceInventoryJudgment.__name__: {
                    "interfaces": [
                        {
                            "interface_id": "interface::invented",
                            "title": "Invented Interface",
                            "kind": "http_api",
                            "summary": "Should be rejected.",
                            "provider_subsystem_ids": ["invented-subsystem"],
                            "consumer_context_node_ids": [],
                            "related_module_ids": [],
                            "responsibilities": [],
                            "cross_module_contracts": [],
                            "collaboration_notes": [],
                            "source_insight_ids": [],
                            "source_rationale_clue_ids": [],
                            "evidence_links": [],
                        }
                    ]
                },
                HistoryDependencyLandscapeJudgment.__name__: {
                    "project_roles": [
                        {
                            "role_id": "role::invented",
                            "title": "Invented Role",
                            "summary": "Should be rejected.",
                            "dependency_ids": ["invented-dependency"],
                            "related_subsystem_ids": [],
                            "evidence_links": [],
                        }
                    ],
                    "clusters": [],
                    "usage_patterns": [],
                },
            }
        )


class _ValidH13Client(MockLLMClient):
    def __init__(self, payload: dict[str, dict[str, object]]) -> None:
        super().__init__(canned=payload)


def _create_h13_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "pyproject.toml",
        (
            "[project]\n"
            'name = "history-h13"\n'
            'dependencies = ["requests>=2", "click>=8"]\n'
            "[project.optional-dependencies]\n"
            'dev = ["pytest>=8"]\n'
        ),
        message="add project metadata",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/api.py",
        (
            '"""HTTP API boundary for request and response validation."""\n'
            "class RequestModel:\n"
            "    pass\n\n"
            "def fetch_state(request_id: str) -> str:\n"
            "    return request_id\n"
        ),
        message="add api boundary",
        timestamp="2024-01-10T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "src/core/engine.py",
        (
            '"""Execution engine with strict fallback handling and fast variant support."""\n'
            "def validate_request(value: str) -> bool:\n"
            "    return bool(value)\n\n"
            "def run_fast_path(value: str) -> str:\n"
            "    return value.upper()\n\n"
            "def run_safe_path(value: str) -> str:\n"
            "    return value.lower()\n\n"
            "def execute_pipeline(value: str) -> str:\n"
            "    if not validate_request(value):\n"
            '        return "fallback"\n'
            "    return run_fast_path(value)\n"
        ),
        message="add execution engine variants and fallback handling",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    return repo_root, {"base": base, "head": head}


def test_h13_artifact_paths_are_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"
    tool_root = output_root / "workspaces" / "repo" / "tools" / "history_docs"

    assert build_algorithm_capsule_enrichment_index_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == algorithm_capsule_enrichment_index_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )
    assert build_interface_inventory_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == interface_inventory_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )
    assert build_dependency_landscape_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == dependency_landscape_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )


def test_h13_prompts_are_compact() -> None:
    algorithm_system, algorithm_user = build_algorithm_capsule_enrichment_prompt(
        checkpoint_summary={"checkpoint_id": "cp-1", "algorithm_capsule_count": 1},
        capsules=[
            {
                "capsule_id": "capsule::engine",
                "title": "Execution Engine",
                "phase_keys": ["validate", "execute"],
                "related_insights": [
                    {"insight_id": "i1", "title": "Engine Tightening"}
                ],
            }
        ],
        design_note_anchors=[
            {"note_id": "n1", "title": "Strict Boundary", "summary": "Short note."}
        ],
    )
    interface_system, interface_user = build_interface_inventory_prompt(
        checkpoint_summary={"checkpoint_id": "cp-1", "module_count": 3},
        interface_context={
            "interfaces": [{"interface_id": "interface::api", "title": "HTTP API"}],
            "modules": [
                {"concept_id": "module::src/app/api.py", "path": "src/app/api.py"}
            ],
        },
        design_note_anchors=[
            {"note_id": "n1", "title": "Boundary", "summary": "Short note."}
        ],
    )
    dependency_system, dependency_user = build_dependency_landscape_prompt(
        checkpoint_summary={"checkpoint_id": "cp-1", "dependency_entry_count": 2},
        dependency_context={
            "entries": [{"dependency_id": "dep::requests", "display_name": "requests"}]
        },
        design_note_anchors=[
            {"note_id": "n1", "title": "Boundary", "summary": "Short note."}
        ],
    )

    assert "do not invent capsule ids" in algorithm_system.lower()
    assert "Execution Engine" in algorithm_user
    assert "diff --git" not in algorithm_user
    assert "return value" not in algorithm_user

    assert "do not invent interface ids" in interface_system.lower()
    assert "HTTP API" in interface_user
    assert "diff --git" not in interface_user

    assert "do not invent dependency ids" in dependency_system.lower()
    assert "requests" in dependency_user
    assert "diff --git" not in dependency_user


def test_invalid_h13_payloads_fall_back_to_persisted_shadow_artifacts(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=_InvalidH13Client(),
    )

    algorithm_index = HistoryAlgorithmCapsuleEnrichmentIndex.model_validate_json(
        algorithm_capsule_enrichment_index_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    inventory = HistoryInterfaceInventory.model_validate_json(
        interface_inventory_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    landscape = HistoryDependencyLandscape.model_validate_json(
        dependency_landscape_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.algorithm_capsule_enrichment_status == "llm_failed"
    assert result.interface_inventory_status == "llm_failed"
    assert result.dependency_landscape_status == "llm_failed"
    assert algorithm_index.evaluation_status == "llm_failed"
    assert inventory.evaluation_status == "llm_failed"
    assert landscape.evaluation_status == "llm_failed"


def test_h13_evidence_helpers_resolve_compact_labels_and_active_sets(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=_InvalidH13Client(),
    )

    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    interval_interpretation = HistoryIntervalInterpretation.model_validate_json(
        build_interval_interpretation_path(
            sample_project_config.workspace.output_root
            / "workspaces"
            / repo_root.name
            / "tools"
            / "history_docs",
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    checkpoint_enrichment = HistoryCheckpointModelEnrichment.model_validate_json(
        build_checkpoint_model_enrichment_path(
            sample_project_config.workspace.output_root
            / "workspaces"
            / repo_root.name
            / "tools"
            / "history_docs",
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    semantic_context = HistorySemanticContextMap.model_validate_json(
        semantic_context_map_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    dependency_inventory = HistoryDependencyInventory.model_validate_json(
        dependencies_artifact_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    capsule_index = HistoryAlgorithmCapsuleIndex.model_validate_json(
        algorithm_capsule_index_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    capsules = [
        HistoryAlgorithmCapsule.model_validate_json(
            (
                algorithm_capsule_index_path(
                    sample_project_config.workspace.output_root,
                    repo_root.name,
                    result.checkpoint_id,
                ).parent.parent
                / entry.artifact_path
            ).read_text(encoding="utf-8")
        )
        for entry in capsule_index.capsules
    ]

    pack = build_h13_evidence_pack(
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_enrichment,
        semantic_context_map=semantic_context,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules,
    )
    summary = compact_checkpoint_summary(pack)
    active_module_list = active_modules(pack)
    active_subsystem_list = active_subsystems(pack)

    assert summary["algorithm_capsule_count"] == len(capsules)
    assert summary["dependency_entry_count"] == len(dependency_inventory.entries)
    assert active_module_list
    assert active_subsystem_list
    assert (
        module_path_label(pack.modules_by_id, active_module_list[0].concept_id)
        == active_module_list[0].path.as_posix()
    )
    assert module_path_label(pack.modules_by_id, "module::missing") == "module::missing"
    assert (
        artifact_path(
            sample_project_config.workspace.output_root
            / "workspaces"
            / repo_root.name
            / "tools"
            / "history_docs",
            result.checkpoint_id,
            "interface_inventory.json",
        ).name
        == "interface_inventory.json"
    )


def test_h13_internal_modes_change_markdown(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]
    client = _InvalidH13Client()

    baseline = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        narrative_render_mode="baseline",
        llm_client_override=client,
    )
    enriched_algorithms = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-algorithms",
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        algorithm_capsule_mode="enriched",
        narrative_render_mode="baseline",
        llm_client_override=client,
    )
    interface_variant = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-interfaces",
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        interface_render_mode="inventory",
        narrative_render_mode="baseline",
        llm_client_override=client,
    )
    dependency_variant = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-dependencies",
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        dependency_render_mode="landscape",
        narrative_render_mode="baseline",
        llm_client_override=client,
    )

    baseline_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        baseline.checkpoint_id,
    ).read_text(encoding="utf-8")
    algorithm_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-algorithms",
        enriched_algorithms.checkpoint_id,
    ).read_text(encoding="utf-8")
    interface_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-interfaces",
        interface_variant.checkpoint_id,
    ).read_text(encoding="utf-8")
    dependency_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-dependencies",
        dependency_variant.checkpoint_id,
    ).read_text(encoding="utf-8")

    assert (
        "Current flow stages are reflected by the capsule phases"
        not in baseline_markdown
    )
    assert (
        "Current flow stages are reflected by the capsule phases" in algorithm_markdown
    )
    assert "Responsibilities:" not in baseline_markdown
    assert "Responsibilities:" in interface_markdown
    assert "Dependency Cluster" not in baseline_markdown
    assert "Dependency Cluster" in dependency_markdown


def test_interface_inventory_heuristic_fallback_uses_semantic_context_details(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    seed = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=_InvalidH13Client(),
    )

    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    interval_interpretation = HistoryIntervalInterpretation.model_validate_json(
        build_interval_interpretation_path(
            sample_project_config.workspace.output_root
            / "workspaces"
            / repo_root.name
            / "tools"
            / "history_docs",
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    checkpoint_enrichment = HistoryCheckpointModelEnrichment.model_validate_json(
        build_checkpoint_model_enrichment_path(
            sample_project_config.workspace.output_root
            / "workspaces"
            / repo_root.name
            / "tools"
            / "history_docs",
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    semantic_context = HistorySemanticContextMap.model_validate_json(
        semantic_context_map_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    dependency_inventory = HistoryDependencyInventory.model_validate_json(
        dependencies_artifact_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    capsule_index = HistoryAlgorithmCapsuleIndex.model_validate_json(
        algorithm_capsule_index_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    capsules = [
        HistoryAlgorithmCapsule.model_validate_json(
            (
                algorithm_capsule_index_path(
                    sample_project_config.workspace.output_root,
                    repo_root.name,
                    seed.checkpoint_id,
                ).parent.parent
                / entry.artifact_path
            ).read_text(encoding="utf-8")
        )
        for entry in capsule_index.capsules
    ]

    interface_candidate = semantic_context.interfaces[0]
    module_ids = sorted(module.concept_id for module in checkpoint_model.modules)
    widened_interface = interface_candidate.model_copy(
        update={
            "related_module_ids": module_ids[:2],
            "evidence_links": list(interface_candidate.evidence_links),
        }
    )
    widened_context = semantic_context.model_copy(
        update={
            "interfaces": [widened_interface, *semantic_context.interfaces[1:]],
        }
    )

    inventory = build_interface_inventory(
        checkpoint_id=seed.checkpoint_id,
        target_commit=seed.target_commit,
        previous_checkpoint_commit=seed.previous_checkpoint_commit,
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_enrichment,
        dependency_inventory=dependency_inventory,
        semantic_context_map=widened_context,
        capsule_index=capsule_index,
        capsules=capsules,
        llm_client=None,
        model_name="mock-model",
        temperature=0.0,
    )

    assert inventory.evaluation_status == "heuristic_only"
    assert inventory.interfaces
    assert inventory.interfaces[0].responsibilities
    assert (
        inventory.interfaces[0].responsibilities[0].title == "Boundary Responsibility"
    )
    assert inventory.interfaces[0].cross_module_contracts
    assert inventory.interfaces[0].cross_module_contracts[0].provider_module_ids
    assert inventory.interfaces[0].cross_module_contracts[0].consumer_module_ids
    assert inventory.interfaces[0].collaboration_notes


def test_valid_h13_payloads_score_and_render(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    seed = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=_InvalidH13Client(),
    )

    semantic_context = HistorySemanticContextMap.model_validate_json(
        semantic_context_map_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    dependency_inventory = HistoryDependencyInventory.model_validate_json(
        dependencies_artifact_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    capsule_index = HistoryAlgorithmCapsuleIndex.model_validate_json(
        algorithm_capsule_index_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            seed.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    assert capsule_index.capsules
    capsule = HistoryAlgorithmCapsule.model_validate_json(
        (
            algorithm_capsule_index_path(
                sample_project_config.workspace.output_root,
                repo_root.name,
                seed.checkpoint_id,
            ).parent.parent
            / capsule_index.capsules[0].artifact_path
        ).read_text(encoding="utf-8")
    )
    assert semantic_context.interfaces
    interface_candidate = semantic_context.interfaces[0]
    dependency_entry = dependency_inventory.entries[0]

    valid_payload = {
        HistoryAlgorithmCapsuleEnrichmentJudgment.__name__: {
            "enrichments": [
                {
                    "capsule_id": capsule.capsule_id,
                    "purpose": "Deterministic request execution pipeline with explicit fast and safe paths.",
                    "phase_flow_summary": "Flow summary: validate input, choose a path, and emit the current result.",
                    "invariants": [
                        {
                            "text": "Validation happens before path selection.",
                            "evidence_links": capsule.evidence_links,
                        }
                    ],
                    "tradeoffs": [
                        {
                            "title": "Fast vs Safe Path",
                            "summary": "The current implementation keeps both a fast path and a conservative fallback path available.",
                            "evidence_links": capsule.evidence_links,
                        }
                    ],
                    "variant_relationships": [
                        {
                            "variant_name": (
                                capsule.variant_names[0]
                                if capsule.variant_names
                                else "run_fast_path"
                            ),
                            "relationship": "performance_variant",
                            "summary": "The fast path is the performance-oriented variant in the current checkpoint.",
                            "evidence_links": capsule.evidence_links,
                        }
                    ],
                    "related_subsystem_ids": capsule.related_subsystem_ids,
                    "related_module_ids": capsule.related_module_ids,
                    "source_insight_ids": [],
                    "source_rationale_clue_ids": [],
                    "evidence_links": capsule.evidence_links,
                }
            ]
        },
        HistoryInterfaceInventoryJudgment.__name__: {
            "interfaces": [
                {
                    "interface_id": interface_candidate.interface_id,
                    "title": interface_candidate.title,
                    "kind": interface_candidate.kind,
                    "summary": "This interface coordinates the request-facing boundary for the current checkpoint.",
                    "provider_subsystem_ids": interface_candidate.provider_subsystem_ids,
                    "consumer_context_node_ids": interface_candidate.consumer_context_node_ids,
                    "related_module_ids": interface_candidate.related_module_ids,
                    "responsibilities": [
                        {
                            "title": "Request Boundary",
                            "summary": "Validate request input and expose the active response contract.",
                            "related_module_ids": interface_candidate.related_module_ids,
                            "evidence_links": interface_candidate.evidence_links,
                        }
                    ],
                    "cross_module_contracts": [],
                    "collaboration_notes": [
                        "The current interface keeps the request boundary explicit across linked modules."
                    ],
                    "source_insight_ids": [],
                    "source_rationale_clue_ids": [],
                    "evidence_links": interface_candidate.evidence_links,
                }
            ]
        },
        HistoryDependencyLandscapeJudgment.__name__: {
            "project_roles": [
                {
                    "role_id": "role::runtime",
                    "title": "Runtime Role",
                    "summary": "These dependencies support the active runtime-facing behavior in the current checkpoint.",
                    "dependency_ids": [dependency_entry.dependency_id],
                    "related_subsystem_ids": dependency_entry.related_subsystem_ids,
                    "evidence_links": [],
                }
            ],
            "clusters": [
                {
                    "cluster_id": f"cluster::{dependency_entry.ecosystem}",
                    "title": "Runtime Dependency Cluster",
                    "summary": "This cluster groups the current runtime dependencies by ecosystem and usage intent.",
                    "dependency_ids": [dependency_entry.dependency_id],
                    "ecosystems": [dependency_entry.ecosystem],
                    "scope_roles": dependency_entry.scope_roles,
                    "related_subsystem_ids": dependency_entry.related_subsystem_ids,
                    "evidence_links": [],
                }
            ],
            "usage_patterns": [
                {
                    "pattern_id": "pattern::runtime-boundary",
                    "title": "Runtime Boundary Usage",
                    "summary": "The current runtime dependency usage stays concentrated around the boundary-facing subsystem.",
                    "dependency_ids": [dependency_entry.dependency_id],
                    "related_module_ids": dependency_entry.related_module_ids,
                    "related_subsystem_ids": dependency_entry.related_subsystem_ids,
                    "source_insight_ids": [],
                    "evidence_links": [],
                }
            ],
        },
    }

    scored = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-h13-scored",
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        algorithm_capsule_mode="enriched",
        interface_render_mode="inventory",
        dependency_render_mode="landscape",
        narrative_render_mode="baseline",
        llm_client_override=_ValidH13Client(valid_payload),
    )

    markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-h13-scored",
        scored.checkpoint_id,
    ).read_text(encoding="utf-8")

    assert scored.algorithm_capsule_enrichment_status == "scored"
    assert scored.interface_inventory_status == "scored"
    assert scored.dependency_landscape_status == "scored"
    assert "Deterministic request execution pipeline" in markdown
    assert "Validate request input and expose the active response contract." in markdown
    assert "Runtime Dependency Cluster" in markdown


def test_history_docs_cli_prints_h13_artifact_paths(
    tmp_path: Path,
    capsys,
) -> None:
    repo_root, commits = _create_h13_repo(tmp_path)
    config_path = tmp_path / "engllm.yaml"
    write_project_config(
        config_path,
        tmp_path / "artifacts",
        source_roots=["repo/src"],
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
            commits["head"],
            "--previous-checkpoint-commit",
            commits["base"],
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Algorithm capsule enrichments:" in captured.out
    assert "Interface inventory:" in captured.out
    assert "Dependency landscape:" in captured.out


def test_h13_benchmark_variant_ids_are_exposed() -> None:
    assert (
        semantic_structure_context_enriched_algorithms_benchmark_variant().variant_id
        == "semantic-structure-context-enriched-algorithms"
    )
    assert (
        semantic_structure_context_interface_inventory_benchmark_variant().variant_id
        == "semantic-structure-context-interface-inventory"
    )
    assert (
        semantic_structure_context_dependency_landscape_benchmark_variant().variant_id
        == "semantic-structure-context-dependency-landscape"
    )
    assert (
        semantic_structure_context_h13_full_benchmark_variant().variant_id
        == "semantic-structure-context-h13-full"
    )
