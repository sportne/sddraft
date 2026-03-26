"""Tests for history-docs H12-02 checkpoint-model enrichment."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.llm.mock import MockLLMClient
from engllm.prompts.history_docs.builders import (
    build_checkpoint_model_enrichment_prompt,
)
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.checkpoint_model_enrichment import (
    checkpoint_model_enrichment_path as build_checkpoint_model_enrichment_path,
)
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryCheckpointModelEnrichment,
    HistoryCheckpointModelEnrichmentJudgment,
)
from tests.history_docs_helpers import (
    checkpoint_markdown_path,
    checkpoint_model_enrichment_path,
    checkpoint_model_path,
    commit_file,
    init_repo,
    write_project_config,
)


class _InjectingEnrichmentClient(MockLLMClient):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(
            canned={HistoryCheckpointModelEnrichmentJudgment.__name__: payload}
        )


def _create_enrichment_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo_root = init_repo(tmp_path)
    base = commit_file(
        repo_root,
        "src/core/api.py",
        (
            '"""API boundary for strict validation and fallback handling."""\n'
            "def fetch_state(value: str) -> str:\n"
            "    return value\n"
        ),
        message="bootstrap api module",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "src/core/api.py",
        (
            '"""API boundary for strict validation and fallback handling."""\n'
            "def fetch_state(value: str, strict: bool = True) -> str:\n"
            "    return value\n"
        ),
        message="tighten validation interface for strict mode",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    return repo_root, {"base": base, "head": head}


def test_checkpoint_model_enrichment_path_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"

    assert build_checkpoint_model_enrichment_path(
        output_root / "workspaces" / "repo" / "tools" / "history_docs",
        "2024-02-01-abcd123",
    ) == checkpoint_model_enrichment_path(
        output_root,
        "repo",
        "2024-02-01-abcd123",
    )


def test_checkpoint_model_enrichment_prompt_is_compact() -> None:
    system_prompt, user_prompt = build_checkpoint_model_enrichment_prompt(
        checkpoint_id="2024-02-01-abcd123",
        target_commit="a" * 40,
        previous_checkpoint_commit="b" * 40,
        subsystems=[
            {
                "concept_id": "subsystem::src::core",
                "display_name": "Core",
                "summary": None,
                "module_ids": ["module::src/core/api.py"],
            }
        ],
        modules=[
            {
                "concept_id": "module::src/core/api.py",
                "path": "src/core/api.py",
                "language": "python",
                "functions": ["fetch_state"],
                "classes": [],
                "imports": ["requests"],
                "docstring_excerpts": ["Strict validation and fallback handling."],
                "symbol_names": ["fetch_state"],
            }
        ],
        interval_interpretation={
            "insights": [
                {
                    "insight_id": "interval-insight::api-contract",
                    "kind": "interface_change",
                    "title": "API Contract Tightening",
                    "summary": "The request boundary now exposes a strict mode.",
                }
            ],
            "rationale_clues": [],
            "significant_windows": [],
        },
        semantic_labels={
            "semantic_subsystems": [
                {
                    "semantic_subsystem_id": "semantic-subsystem::core-api",
                    "title": "Core API",
                    "summary": "Owns request validation.",
                }
            ],
            "capabilities": [],
            "context_nodes": [],
            "interfaces": [],
        },
    )

    assert "do not invent concept ids" in system_prompt.lower()
    assert "API Contract Tightening" in user_prompt
    assert "Strict validation and fallback handling." in user_prompt
    assert "diff --git" not in user_prompt
    assert "return value" not in user_prompt


def test_invalid_enrichment_payload_falls_back_to_persisted_shadow_artifact(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_enrichment_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        llm_client_override=_InjectingEnrichmentClient(
            {
                "subsystem_enrichments": [
                    {
                        "concept_id": "invented-subsystem",
                        "display_name": "Invented",
                        "summary": "Should be rejected.",
                        "capability_labels": [],
                        "source_insight_ids": [],
                        "source_rationale_clue_ids": [],
                        "evidence_links": [
                            {"kind": "file", "reference": "src/core/api.py"}
                        ],
                    }
                ],
                "module_enrichments": [],
                "capability_proposals": [],
                "design_note_anchors": [],
            }
        ),
    )

    artifact = HistoryCheckpointModelEnrichment.model_validate_json(
        checkpoint_model_enrichment_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )

    assert result.checkpoint_model_enrichment_status == "llm_failed"
    assert artifact.evaluation_status == "llm_failed"
    assert artifact.subsystem_enrichments
    assert artifact.module_enrichments


def test_internal_enriched_mode_updates_checkpoint_model_and_markdown(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root, commits = _create_enrichment_repo(tmp_path)
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]
    payload = {
        "subsystem_enrichments": [
            {
                "concept_id": "subsystem::src::core",
                "display_name": "Core API",
                "summary": "Owns request validation and strict boundary handling.",
                "capability_labels": ["Request Validation"],
                "source_insight_ids": [],
                "source_rationale_clue_ids": [],
                "evidence_links": [{"kind": "file", "reference": "src/core/api.py"}],
            }
        ],
        "module_enrichments": [
            {
                "concept_id": "module::src/core/api.py",
                "summary": "Handles the request-facing strict validation contract.",
                "responsibility_labels": ["Validation"],
                "source_insight_ids": [],
                "source_rationale_clue_ids": [],
                "evidence_links": [{"kind": "file", "reference": "src/core/api.py"}],
            }
        ],
        "capability_proposals": [
            {
                "capability_id": "capability::strict-validation",
                "title": "Strict Validation",
                "summary": "Covers strict request validation behavior.",
                "related_subsystem_ids": ["subsystem::src::core"],
                "related_module_ids": ["module::src/core/api.py"],
                "source_insight_ids": [],
                "evidence_links": [{"kind": "file", "reference": "src/core/api.py"}],
            }
        ],
        "design_note_anchors": [
            {
                "note_id": "design-note::strict-boundary",
                "title": "Strict Boundary Handling",
                "summary": "The current request boundary preserves an explicit strict mode.",
                "related_concept_ids": [
                    "subsystem::src::core",
                    "module::src/core/api.py",
                ],
                "source_insight_ids": [],
                "source_rationale_clue_ids": [],
                "evidence_links": [{"kind": "file", "reference": "src/core/api.py"}],
            }
        ],
    }

    baseline = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        llm_client_override=_InjectingEnrichmentClient(payload),
    )
    enriched = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=commits["head"],
        previous_checkpoint_commit=commits["base"],
        workspace_id="repo-enriched",
        checkpoint_model_enrichment_mode="enriched",
        llm_client_override=_InjectingEnrichmentClient(payload),
    )

    baseline_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            repo_root.name,
            baseline.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    enriched_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            sample_project_config.workspace.output_root,
            "repo-enriched",
            enriched.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    enriched_markdown = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        "repo-enriched",
        enriched.checkpoint_id,
    ).read_text(encoding="utf-8")

    assert baseline_model.modules[0].summary is None
    assert enriched_model.subsystems[0].display_name == "Core API"
    assert enriched_model.modules[0].summary == (
        "Handles the request-facing strict validation contract."
    )
    assert enriched_model.modules[0].responsibility_labels == ["Validation"]
    assert (
        "summary: Handles the request-facing strict validation contract."
        in enriched_markdown
    )
    assert "responsibility labels: `Validation`" in enriched_markdown


def test_cli_prints_checkpoint_model_enrichment_path(
    tmp_path: Path,
    capsys,
) -> None:
    repo_root, commits = _create_enrichment_repo(tmp_path)
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
    assert "Checkpoint model enrichment:" in capsys.readouterr().out
