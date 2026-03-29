"""Tests for history-docs H14 shadow drafting, review, and repair."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import (
    build_draft_review_prompt,
    build_section_drafting_prompt,
    build_section_repair_prompt,
)
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.draft_review import (
    draft_review_path as build_draft_review_path,
)
from engllm.tools.history_docs.models import (
    HistoryDraftReview,
    HistoryDraftReviewJudgment,
    HistorySectionDraftArtifact,
    HistorySectionDraftJudgment,
    HistorySectionRepairJudgment,
)
from engllm.tools.history_docs.section_drafting_llm import (
    checkpoint_draft_markdown_path as build_checkpoint_draft_markdown_path,
)
from engllm.tools.history_docs.section_drafting_llm import (
    render_manifest_draft_path as build_render_manifest_draft_path,
)
from engllm.tools.history_docs.section_drafting_llm import (
    section_drafts_path as build_section_drafts_path,
)
from engllm.tools.history_docs.section_drafting_llm import (
    validation_report_draft_path as build_validation_report_draft_path,
)
from engllm.tools.history_docs.section_repair_llm import (
    checkpoint_repaired_markdown_path as build_checkpoint_repaired_markdown_path,
)
from engllm.tools.history_docs.section_repair_llm import (
    render_manifest_repaired_path as build_render_manifest_repaired_path,
)
from engllm.tools.history_docs.section_repair_llm import (
    section_repairs_path as build_section_repairs_path,
)
from engllm.tools.history_docs.section_repair_llm import (
    validation_report_repaired_path as build_validation_report_repaired_path,
)
from tests.history_docs_helpers import (
    checkpoint_draft_markdown_path,
    checkpoint_markdown_path,
    checkpoint_repaired_markdown_path,
    commit_file,
    draft_review_path,
    init_repo,
    render_manifest_draft_path,
    render_manifest_repaired_path,
    section_drafts_path,
    section_repairs_path,
    validation_report_draft_path,
    validation_report_repaired_path,
    write_project_config,
)


class _InvalidDraftClient(MockLLMClient):
    def __init__(self) -> None:
        super().__init__(
            canned={
                HistorySectionDraftJudgment.__name__: {
                    "markdown_body": "   ",
                    "supporting_concept_ids": [],
                    "supporting_algorithm_capsule_ids": [],
                    "source_insight_ids": [],
                    "source_capability_ids": [],
                    "source_design_note_ids": [],
                    "evidence_links": [],
                }
            }
        )


class _RepairingH14Client(MockLLMClient):
    def __init__(self) -> None:
        super().__init__(
            canned={
                HistorySectionDraftJudgment.__name__: {
                    "markdown_body": "This drafted section body is grounded in the supplied checkpoint evidence.",
                    "supporting_concept_ids": [],
                    "supporting_algorithm_capsule_ids": [],
                    "source_insight_ids": [],
                    "source_capability_ids": [],
                    "source_design_note_ids": [],
                    "evidence_links": [],
                },
                HistoryDraftReviewJudgment.__name__: {
                    "strengths": ["The draft keeps a steady present-state tone."],
                    "findings": [
                        {
                            "finding_id": "finding::architectural-overview",
                            "kind": "weak_prose",
                            "severity": "medium",
                            "section_id": "architectural_overview",
                            "summary": "The architectural overview stays too generic.",
                            "revision_goal": "State the current architecture more concretely.",
                        }
                    ],
                    "recommended_repair_section_ids": ["architectural_overview"],
                },
                HistorySectionRepairJudgment.__name__: {
                    "revised_markdown_body": (
                        "This repaired section body states the current architecture "
                        "more concretely while staying evidence-backed."
                    ),
                    "addressed_finding_ids": ["finding::architectural-overview"],
                    "evidence_links": [],
                },
            }
        )


def _create_h14_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "pyproject.toml",
        (
            "[project]\n"
            'name = "history-h14"\n'
            'dependencies = ["requests>=2", "click>=8"]\n'
        ),
        message="add project metadata",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/app/api.py",
        (
            '"""HTTP API boundary for request validation."""\n'
            "def fetch_state(request_id: str) -> str:\n"
            "    return request_id\n"
        ),
        message="add api surface",
        timestamp="2024-01-10T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "src/app/engine.py",
        (
            '"""Execution engine with strict validation and fallback behavior."""\n'
            "def validate_request(value: str) -> bool:\n"
            "    return bool(value)\n\n"
            "def run_engine(value: str) -> str:\n"
            "    if not validate_request(value):\n"
            '        return "fallback"\n'
            "    return value.upper()\n"
        ),
        message="add engine validation and fallback behavior",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    return repo_root, {"base": base, "head": head}


def test_h14_artifact_paths_are_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"
    tool_root = output_root / "workspaces" / "repo" / "tools" / "history_docs"

    assert build_section_drafts_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == section_drafts_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_checkpoint_draft_markdown_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == checkpoint_draft_markdown_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_render_manifest_draft_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == render_manifest_draft_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_validation_report_draft_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == validation_report_draft_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_draft_review_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == draft_review_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_section_repairs_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == section_repairs_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_checkpoint_repaired_markdown_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == checkpoint_repaired_markdown_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )
    assert build_render_manifest_repaired_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == render_manifest_repaired_path(output_root, "repo", "2024-02-01-abcd123")
    assert build_validation_report_repaired_path(
        tool_root,
        "2024-02-01-abcd123",
    ) == validation_report_repaired_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )


def test_h14_prompts_are_compact() -> None:
    draft_system, draft_user = build_section_drafting_prompt(
        checkpoint_context={"checkpoint_id": "cp-1"},
        section_metadata={
            "section_id": "architectural_overview",
            "title": "Architectural Overview",
        },
        supporting_evidence={
            "concepts": [
                {"concept_id": "subsystem::src::app", "summary": "App subsystem."}
            ],
            "baseline_section_body": "Current deterministic body.",
        },
    )
    review_system, review_user = build_draft_review_prompt(
        checkpoint_context={"checkpoint_id": "cp-1"},
        draft_summary={"section_ids": ["architectural_overview"]},
        validation_summary={"error_count": 0, "warning_count": 0},
        evidence_summary={"design_note_titles": ["Strict Boundary"]},
        markdown="# doc\n",
    )
    repair_system, repair_user = build_section_repair_prompt(
        checkpoint_context={"checkpoint_id": "cp-1"},
        section_metadata={
            "section_id": "architectural_overview",
            "title": "Architectural Overview",
        },
        findings=[
            {
                "finding_id": "finding::1",
                "kind": "weak_prose",
                "severity": "medium",
                "summary": "Too generic.",
                "revision_goal": "Make it concrete.",
            }
        ],
        supporting_evidence={
            "design_notes": [{"note_id": "note::1", "title": "Strict Boundary"}]
        },
        original_markdown_body="Original body.",
    )

    assert "do not invent section ids" in draft_system.lower()
    assert "do not rewrite markdown directly" in review_system.lower()
    assert "address only the supplied findings" in repair_system.lower()
    assert "Architectural Overview" in draft_user
    assert "Strict Boundary" in review_user
    assert "Original body." in repair_user
    assert "diff --git" not in draft_user
    assert "def fetch_state" not in draft_user


def test_h14_shadow_artifacts_are_written_and_baseline_remains_authoritative(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h14_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=MockLLMClient(),
    )

    assert result.checkpoint_markdown_path == checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert result.section_drafts_path == section_drafts_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert result.draft_review_path == draft_review_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert result.section_repairs_path == section_repairs_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert result.section_drafts_path.exists()
    assert result.checkpoint_draft_markdown_path.exists()
    assert result.render_manifest_draft_path.exists()
    assert result.validation_report_draft_path.exists()
    assert result.draft_review_path.exists()
    assert result.section_repairs_path.exists()
    assert result.checkpoint_repaired_markdown_path.exists()
    assert result.render_manifest_repaired_path.exists()
    assert result.validation_report_repaired_path.exists()


def test_invalid_draft_payload_falls_back_to_exact_baseline_copy(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h14_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        llm_client_override=_InvalidDraftClient(),
    )

    baseline_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    ).read_text(encoding="utf-8")
    draft_markdown = checkpoint_draft_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    ).read_text(encoding="utf-8")
    draft_artifact = HistorySectionDraftArtifact.model_validate_json(
        section_drafts_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.draft_status == "llm_failed"
    assert draft_artifact.evaluation_status == "llm_failed"
    assert draft_markdown == baseline_markdown


def test_internal_llm_repair_mode_switches_authoritative_outputs(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_h14_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-h14-repair",
        subsystem_grouping_mode="semantic",
        experimental_section_mode="semantic_context",
        narrative_render_mode="llm_repair",
        llm_client_override=_RepairingH14Client(),
    )

    repaired_path = checkpoint_repaired_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-h14-repair",
        result.checkpoint_id,
    )
    repaired_markdown = repaired_path.read_text(encoding="utf-8")
    review = HistoryDraftReview.model_validate_json(
        draft_review_path(
            sample_project_config.workspace.output_root,
            "repo-h14-repair",
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.checkpoint_markdown_path == repaired_path
    assert result.render_manifest_path == render_manifest_repaired_path(
        sample_project_config.workspace.output_root,
        "repo-h14-repair",
        result.checkpoint_id,
    )
    assert result.validation_report_path == validation_report_repaired_path(
        sample_project_config.workspace.output_root,
        "repo-h14-repair",
        result.checkpoint_id,
    )
    assert result.draft_status == "scored"
    assert result.draft_review_status == "scored"
    assert result.repair_status == "scored"
    assert review.recommended_repair_section_ids == ["architectural_overview"]
    assert (
        "This repaired section body states the current architecture"
        in repaired_markdown
    )


def test_cli_prints_h14_artifact_paths(tmp_path: Path, capsys) -> None:
    repo_root, commits = _create_h14_repo(tmp_path)
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

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Section drafts:" in output
    assert "Draft review:" in output
    assert "Section repairs:" in output
    assert "Repaired markdown:" in output
