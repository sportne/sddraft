"""Additional CLI tests for branch coverage."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from sddraft.analysis.retrieval import LexicalIndexer, save_retrieval_index
from sddraft.cli.main import main
from sddraft.domain.models import (
    AskResult,
    Citation,
    CommitImpact,
    ConfigBundle,
    CSCDescriptor,
    GenerationOptions,
    InspectDiffResult,
    KnowledgeChunk,
    LLMConfig,
    ProjectConfig,
    ProposeUpdatesResult,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
    RetrievalManifest,
    SDDSectionSpec,
    SDDTemplate,
    SourcesConfig,
    UpdateProposalReport,
)


def _fake_bundle(tmp_path: Path) -> ConfigBundle:
    project = ProjectConfig(
        project_name="Example",
        sources=SourcesConfig(roots=[tmp_path], include=["**/*.py"], exclude=[]),
        sdd_template=tmp_path / "t.yaml",
        llm=LLMConfig(provider="mock", model_name="mock-sddraft", temperature=0.2),
        generation=GenerationOptions(
            max_files=10, code_chunk_lines=10, retrieval_top_k=3
        ),
        output_dir=tmp_path / "artifacts",
    )
    csc = CSCDescriptor(csc_id="X", title="X", purpose="P")
    template = SDDTemplate(
        sections=[SDDSectionSpec(id="1", title="Scope", instruction="i")]
    )
    return ConfigBundle(project=project, csc_descriptors=[csc], template=template)


def _fake_manifest() -> RetrievalManifest:
    return RetrievalManifest(
        shard_size=1000,
        total_chunks=0,
        average_doc_length=0.0,
        docstats_path=Path("docstats.jsonl"),
    )


def test_cli_propose_updates_and_inspect_diff_paths(
    tmp_path: Path, monkeypatch
) -> None:
    cli_module = importlib.import_module("sddraft.cli.main")
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(cli_module, "load_config_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(
        cli_module, "create_llm_client", lambda *args, **kwargs: object()
    )

    fake_impact = CommitImpact(
        commit_range="HEAD~1..HEAD", changed_files=[], summary="s"
    )

    monkeypatch.setattr(
        cli_module,
        "propose_updates",
        lambda **kwargs: ProposeUpdatesResult(
            impact=fake_impact,
            report=UpdateProposalReport(
                commit_range="HEAD~1..HEAD", impacted_sections=[], proposals=[]
            ),
            retrieval_manifest=_fake_manifest(),
            report_markdown_path=tmp_path / "r.md",
            report_json_path=tmp_path / "r.json",
            retrieval_index_path=tmp_path / "retrieval",
            run_metrics_path=tmp_path / "run_metrics.json",
        ),
    )

    rc = main(
        [
            "propose-updates",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--existing-sdd",
            str(tmp_path / "sdd.md"),
            "--commit-range",
            "HEAD~1..HEAD",
        ]
    )
    assert rc == 0

    monkeypatch.setattr(
        cli_module,
        "inspect_diff",
        lambda **kwargs: InspectDiffResult(impact=fake_impact, raw_diff=""),
    )
    rc2 = main(["inspect-diff", "--commit-range", "HEAD~1..HEAD"])
    assert rc2 == 0


def test_cli_ask_interactive_and_error_path(tmp_path: Path, monkeypatch) -> None:
    cli_module = importlib.import_module("sddraft.cli.main")
    monkeypatch.setattr(
        cli_module, "create_llm_client", lambda *args, **kwargs: object()
    )

    answer = QueryAnswer(
        answer="A",
        citations=[
            Citation(
                chunk_id="c",
                source_path=Path("src/a.py"),
                line_start=1,
                line_end=1,
                quote="q",
            )
        ],
        confidence=0.6,
    )

    monkeypatch.setattr(
        cli_module,
        "answer_question",
        lambda **kwargs: AskResult(
            answer=answer,
            evidence_pack=QueryEvidencePack(
                request=QueryRequest(question="Q"),
                chunks=[],
                citations=answer.citations,
            ),
        ),
    )

    rendered = []

    def _render_query(_answer: QueryAnswer) -> str:
        rendered.append(_answer.answer)
        return "ok"

    monkeypatch.setattr(cli_module, "render_query_answer_text", _render_query)

    inputs = iter(["What is this?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    rc = main(
        [
            "ask",
            "--index-path",
            str(tmp_path / "i.json"),
            "--interactive",
        ]
    )
    assert rc == 0
    assert rendered == ["A"]

    rc2 = main(["ask", "--index-path", str(tmp_path / "i.json")])
    assert rc2 == 2


def test_cli_generate_and_propose_apply_runtime_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    cli_module = importlib.import_module("sddraft.cli.main")
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(cli_module, "load_config_bundle", lambda **kwargs: bundle)

    created_clients: list[tuple[str | None, str | None]] = []

    def _fake_create_llm_client(config, provider=None, model_name=None):
        created_clients.append((provider, model_name))
        return object()

    monkeypatch.setattr(cli_module, "create_llm_client", _fake_create_llm_client)

    generate_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli_module,
        "generate_sdd",
        lambda **kwargs: (
            generate_calls.append(kwargs)
            or SimpleNamespace(
                markdown_path=tmp_path / "sdd.md",
                review_json_path=tmp_path / "review.json",
                retrieval_index_path=tmp_path / "index.json",
            )
        ),
    )

    rc_generate = main(
        [
            "generate",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--provider",
            "gemini",
            "--model",
            "override-model",
            "--temperature",
            "0.73",
        ]
    )
    assert rc_generate == 0
    assert generate_calls
    assert generate_calls[0]["model_name"] == "override-model"
    assert generate_calls[0]["temperature"] == 0.73
    assert generate_calls[0]["hierarchy_docs_enabled"] is True
    assert generate_calls[0]["graph_enabled"] is True

    propose_calls: list[dict[str, object]] = []
    fake_impact = CommitImpact(
        commit_range="HEAD~1..HEAD", changed_files=[], summary="s"
    )
    monkeypatch.setattr(
        cli_module,
        "propose_updates",
        lambda **kwargs: (
            propose_calls.append(kwargs)
            or ProposeUpdatesResult(
                impact=fake_impact,
                report=UpdateProposalReport(
                    commit_range="HEAD~1..HEAD", impacted_sections=[], proposals=[]
                ),
                retrieval_manifest=_fake_manifest(),
                report_markdown_path=tmp_path / "r.md",
                report_json_path=tmp_path / "r.json",
                retrieval_index_path=tmp_path / "retrieval",
                run_metrics_path=tmp_path / "run_metrics.json",
            )
        ),
    )

    rc_propose = main(
        [
            "propose-updates",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--existing-sdd",
            str(tmp_path / "sdd.md"),
            "--commit-range",
            "HEAD~1..HEAD",
            "--provider",
            "gemini",
            "--model",
            "override-model",
            "--temperature",
            "0.73",
        ]
    )
    assert rc_propose == 0
    assert propose_calls
    assert propose_calls[0]["model_name"] == "override-model"
    assert propose_calls[0]["temperature"] == 0.73
    assert propose_calls[0]["hierarchy_docs_enabled"] is True
    assert propose_calls[0]["graph_enabled"] is True
    assert created_clients == [
        ("gemini", "override-model"),
        ("gemini", "override-model"),
    ]

    generate_calls.clear()
    rc_generate_no_hierarchy = main(
        [
            "generate",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--provider",
            "gemini",
            "--model",
            "override-model",
            "--temperature",
            "0.73",
            "--no-hierarchy-docs",
        ]
    )
    assert rc_generate_no_hierarchy == 0
    assert generate_calls[0]["hierarchy_docs_enabled"] is False


def test_cli_error_messages_for_runtime_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    temp_rc = main(
        [
            "ask",
            "--index-path",
            str(tmp_path / "missing_index.json"),
            "--question",
            "What changed?",
            "--temperature",
            "1.5",
        ]
    )
    assert temp_rc == 2
    assert "--temperature must be between 0.0 and 1.0" in capsys.readouterr().out

    ask_rc = main(
        [
            "ask",
            "--index-path",
            str(tmp_path / "missing_index.json"),
            "--question",
            "What changed?",
        ]
    )
    assert ask_rc == 2
    assert "Retrieval store not found" in capsys.readouterr().out

    cli_module = importlib.import_module("sddraft.cli.main")
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(cli_module, "load_config_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(
        cli_module, "create_llm_client", lambda *args, **kwargs: object()
    )

    propose_rc = main(
        [
            "propose-updates",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--existing-sdd",
            str(tmp_path / "missing_sdd.md"),
            "--commit-range",
            "HEAD~1..HEAD",
        ]
    )
    assert propose_rc == 2
    assert "Existing SDD file not found" in capsys.readouterr().out

    diff_rc = main(
        [
            "inspect-diff",
            "--commit-range",
            "HEAD~1..HEAD",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert diff_rc == 2
    assert "Failed to run git diff" in capsys.readouterr().out


def test_cli_no_graph_and_ask_graph_flags_propagate(
    tmp_path: Path, monkeypatch
) -> None:
    cli_module = importlib.import_module("sddraft.cli.main")
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(cli_module, "load_config_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(
        cli_module, "create_llm_client", lambda *args, **kwargs: object()
    )

    generate_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli_module,
        "generate_sdd",
        lambda **kwargs: (
            generate_calls.append(kwargs)
            or SimpleNamespace(
                markdown_path=tmp_path / "sdd.md",
                review_json_path=tmp_path / "review.json",
                retrieval_index_path=tmp_path / "retrieval",
            )
        ),
    )
    rc_generate = main(
        [
            "generate",
            "--project-config",
            str(tmp_path / "p.yaml"),
            "--csc",
            str(tmp_path / "c.yaml"),
            "--no-graph",
        ]
    )
    assert rc_generate == 0
    assert generate_calls[0]["graph_enabled"] is False

    captured_ask: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli_module,
        "answer_question",
        lambda **kwargs: (
            captured_ask.append(kwargs)
            or AskResult(
                answer=QueryAnswer(answer="A", citations=[], confidence=0.5),
                evidence_pack=QueryEvidencePack(
                    request=QueryRequest(question="Q"),
                    chunks=[],
                    citations=[],
                ),
            )
        ),
    )
    rc_ask = main(
        [
            "ask",
            "--index-path",
            str(tmp_path / "retrieval"),
            "--question",
            "what changed?",
            "--graph-depth",
            "2",
            "--graph-top-k",
            "15",
            "--no-graph",
        ]
    )
    assert rc_ask == 0
    assert captured_ask[0]["graph_enabled"] is False
    assert captured_ask[0]["graph_depth"] == 2
    assert captured_ask[0]["graph_top_k"] == 15


def test_cli_migrate_index_command(tmp_path: Path) -> None:
    legacy_path = tmp_path / "retrieval_index.json"
    legacy_index = LexicalIndexer().build(
        document_chunks=[
            KnowledgeChunk(
                chunk_id="a",
                source_type="sdd_section",
                source_path=Path("sdd.md"),
                text="scope details",
            )
        ],
        code_chunks=[],
    )
    save_retrieval_index(legacy_index, legacy_path)

    rc = main(
        [
            "migrate-index",
            "--index-path",
            str(legacy_path),
            "--shard-size",
            "4",
            "--write-batch-size",
            "2",
            "--max-in-memory-records",
            "16",
        ]
    )
    assert rc == 0
    assert (tmp_path / "retrieval" / "manifest.json").exists()
