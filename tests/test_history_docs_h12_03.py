"""Tests for history-docs H12-03 shadow LLM section planning."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import build_section_planning_llm_prompt
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryCheckpointModelEnrichmentJudgment,
    HistoryIntervalInterpretationJudgment,
    HistoryLLMSectionOutline,
    HistoryLLMSectionOutlineJudgment,
)
from engllm.tools.history_docs.section_planning_llm import (
    section_outline_llm_path as build_section_outline_llm_path,
)
from tests.history_docs_helpers import (
    checkpoint_markdown_path,
    commit_file,
    init_repo,
    section_outline_llm_path,
    section_outline_path,
    write_project_config,
)

_ALL_SECTION_IDS = (
    "introduction",
    "architectural_overview",
    "system_context",
    "subsystems_modules",
    "interfaces",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
)


class _InjectingSectionPlanningClient(MockLLMClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(
            canned={
                HistoryIntervalInterpretationJudgment.__name__: {
                    "insights": [
                        {
                            "insight_id": "interval-insight::interface-src-app-api-py-fetch-state",
                            "kind": "interface_change",
                            "title": "Interface Tightening",
                            "summary": "The API boundary now exposes strict validation.",
                            "significance": "medium",
                            "related_commit_ids": [],
                            "related_change_ids": [
                                "interface::src/app/api.py::fetch_state"
                            ],
                            "related_subsystem_ids": [],
                            "evidence_links": [
                                {"kind": "file", "reference": "src/app/api.py"},
                                {
                                    "kind": "symbol",
                                    "reference": "src/app/api.py::fetch_state",
                                },
                            ],
                        }
                    ],
                    "rationale_clues": [],
                    "significant_windows": [],
                },
                HistoryCheckpointModelEnrichmentJudgment.__name__: {
                    "subsystem_enrichments": [],
                    "module_enrichments": [],
                    "capability_proposals": [
                        {
                            "capability_id": "capability::request-validation",
                            "title": "Request Validation",
                            "summary": "Captures strict request validation at the API boundary.",
                            "related_subsystem_ids": ["subsystem::src::app"],
                            "related_module_ids": ["module::src/app/api.py"],
                            "source_insight_ids": [
                                "interval-insight::interface-src-app-api-py-fetch-state"
                            ],
                            "evidence_links": [
                                {"kind": "file", "reference": "src/app/api.py"}
                            ],
                        }
                    ],
                    "design_note_anchors": [
                        {
                            "note_id": "design-note::strict-boundary",
                            "title": "Strict Boundary",
                            "summary": "The current API boundary keeps strict validation explicit.",
                            "related_concept_ids": [
                                "subsystem::src::app",
                                "module::src/app/api.py",
                            ],
                            "source_insight_ids": [
                                "interval-insight::interface-src-app-api-py-fetch-state"
                            ],
                            "source_rationale_clue_ids": [],
                            "evidence_links": [
                                {"kind": "file", "reference": "src/app/api.py"}
                            ],
                        }
                    ],
                },
                HistoryLLMSectionOutlineJudgment.__name__: payload,
            }
        )


def _create_section_planning_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "src/app/api.py",
        (
            '"""HTTP API boundary for request and response validation."""\n'
            "def fetch_state(request_id: str) -> str:\n"
            "    return request_id\n"
        ),
        message="bootstrap api boundary",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/cli.py",
        (
            '"""CLI entrypoint for request orchestration."""\n'
            "def main() -> int:\n"
            "    return 0\n"
        ),
        message="add cli surface",
        timestamp="2024-01-10T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "src/app/api.py",
        (
            '"""HTTP API boundary for request and response validation."""\n'
            "def fetch_state(request_id: str, strict: bool = True) -> str:\n"
            "    return request_id\n"
        ),
        message="tighten request interface and strict validation",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    return repo_root, {"base": base, "head": head}


def _section_decisions() -> dict[str, object]:
    decisions: list[dict[str, object]] = []
    for section_id in _ALL_SECTION_IDS:
        status = (
            "included"
            if section_id
            in {
                "introduction",
                "architectural_overview",
                "system_context",
                "subsystems_modules",
                "dependencies",
                "build_development_infrastructure",
                "design_notes_rationale",
            }
            else "omitted"
        )
        depth = (
            "deep"
            if section_id == "system_context"
            else "standard" if status == "included" else None
        )
        decisions.append(
            {
                "section_id": section_id,
                "status": status,
                "depth": depth,
                "confidence_score": 84 if status == "included" else 35,
                "planning_rationale": (
                    "Semantic context and interval evidence justify this section."
                    if status == "included"
                    else "Planner omitted this section because evidence stayed weak."
                ),
                "source_insight_ids": (
                    ["interval-insight::interface-src-app-api-py-fetch-state"]
                    if section_id in {"system_context", "design_notes_rationale"}
                    else []
                ),
                "source_capability_ids": (
                    ["capability::request-validation"]
                    if section_id == "system_context"
                    else []
                ),
                "source_design_note_ids": (
                    ["design-note::strict-boundary"]
                    if section_id == "design_notes_rationale"
                    else []
                ),
            }
        )
    return {"sections": decisions}


def test_section_outline_llm_path_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"

    assert build_section_outline_llm_path(
        output_root / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-02-01-abcd123",
    ) == section_outline_llm_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )


def test_section_planning_prompt_is_compact() -> None:
    system_prompt, user_prompt = build_section_planning_llm_prompt(
        checkpoint_id="2024-02-01-abcd123",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        section_scaffold=[
            {
                "section_id": "system_context",
                "title": "System Context",
                "kind": "optional",
                "status": "included",
                "confidence_score": 80,
                "evidence_score": 8,
                "depth": "standard",
                "concept_ids": ["context-node::system"],
                "algorithm_capsule_ids": [],
                "trigger_signals": ["active_subsystems", "interface_change"],
                "evidence_links": [{"kind": "file", "reference": "src/app/api.py"}],
            }
        ],
        checkpoint_summary={
            "checkpoint_id": "2024-02-01-abcd123",
            "active_subsystem_count": 2,
        },
        interval_interpretation={
            "insights": [
                {
                    "insight_id": "interval-insight::interface-change",
                    "title": "Interface Change",
                    "summary": "The request boundary exposes strict validation.",
                }
            ],
            "rationale_clues": [],
            "significant_windows": [],
        },
        checkpoint_model_enrichment={
            "capability_proposals": [
                {
                    "capability_id": "capability::request-validation",
                    "title": "Request Validation",
                }
            ],
            "design_note_anchors": [
                {
                    "note_id": "design-note::strict-boundary",
                    "title": "Strict Boundary",
                }
            ],
        },
        semantic_context_summary={
            "context_node_count": 1,
            "interface_count": 1,
            "context_nodes": [{"node_id": "context-node::system", "title": "System"}],
            "interfaces": [{"interface_id": "interface::http", "title": "HTTP API"}],
        },
    )

    assert "do not invent new section ids" in system_prompt.lower()
    assert "System Context" in user_prompt
    assert "Request Validation" in user_prompt
    assert "Strict Boundary" in user_prompt
    assert "diff --git" not in user_prompt
    assert "return request_id" not in user_prompt


def test_invalid_section_planning_payload_falls_back_to_shadow_artifact(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_section_planning_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        llm_client_override=_InjectingSectionPlanningClient(
            {
                "sections": [
                    {
                        "section_id": "invented-section",
                        "status": "included",
                        "depth": "standard",
                        "confidence_score": 90,
                        "planning_rationale": "Invalid invented section.",
                        "source_insight_ids": [],
                        "source_capability_ids": [],
                        "source_design_note_ids": [],
                    }
                ]
            }
        ),
    )

    artifact = HistoryLLMSectionOutline.model_validate_json(
        section_outline_llm_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.section_planning_status == "llm_failed"
    assert artifact.evaluation_status == "llm_failed"
    assert artifact.sections
    assert all(section.planning_rationale is not None for section in artifact.sections)


def test_internal_llm_section_planning_changes_rendered_sections(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_section_planning_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    baseline = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        experimental_section_mode="semantic_context",
        llm_client_override=_InjectingSectionPlanningClient(_section_decisions()),
    )
    llm_planned = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-llm-planned",
        experimental_section_mode="semantic_context",
        section_planning_mode="llm",
        llm_client_override=_InjectingSectionPlanningClient(_section_decisions()),
    )

    baseline_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        baseline.checkpoint_id,
    ).read_text(encoding="utf-8")
    llm_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-llm-planned",
        llm_planned.checkpoint_id,
    ).read_text(encoding="utf-8")
    llm_outline = HistoryLLMSectionOutline.model_validate_json(
        section_outline_llm_path(
            sample_project_config.workspace.output_root,
            "repo-llm-planned",
            llm_planned.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    baseline_outline_json = section_outline_path(
        sample_project_config.workspace.output_root,
        "repo-llm-planned",
        llm_planned.checkpoint_id,
    ).read_text(encoding="utf-8")

    assert "## Interfaces" in baseline_markdown
    assert "## Interfaces" not in llm_markdown
    assert "## System Context" in llm_markdown
    assert llm_planned.section_planning_status == "scored"
    assert any(
        section.section_id == "system_context"
        and section.depth == "deep"
        and section.source_insight_ids
        == ["interval-insight::interface-src-app-api-py-fetch-state"]
        for section in llm_outline.sections
    )
    assert "Planner omitted this section because evidence stayed weak." in (
        section_outline_llm_path(
            sample_project_config.workspace.output_root,
            "repo-llm-planned",
            llm_planned.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    assert '"planning_rationale": null' in baseline_outline_json


def test_cli_prints_llm_section_outline_path(tmp_path: Path, capsys) -> None:
    repo_root, commits = _create_section_planning_repo(tmp_path)
    config_path = tmp_path / "project.yaml"
    write_project_config(
        config_path,
        output_root=tmp_path / "artifacts",
        source_roots=[(repo_root / "src").as_posix()],
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

    assert exit_code == 0
    assert "LLM section outline:" in capsys.readouterr().out
