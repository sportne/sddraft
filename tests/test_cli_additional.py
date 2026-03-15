"""Additional CLI tests for branch coverage."""

from __future__ import annotations

import importlib
from pathlib import Path

from sddraft.cli.main import main
from sddraft.domain.models import (
    AskResult,
    Citation,
    CommitImpact,
    ConfigBundle,
    CSCDescriptor,
    GenerationOptions,
    InspectDiffResult,
    LLMConfig,
    ProjectConfig,
    ProposeUpdatesResult,
    QueryAnswer,
    QueryEvidencePack,
    QueryRequest,
    RetrievalIndex,
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
        generation=GenerationOptions(max_files=10, code_chunk_lines=10, retrieval_top_k=3),
        output_dir=tmp_path / "artifacts",
    )
    csc = CSCDescriptor(csc_id="X", title="X", purpose="P")
    template = SDDTemplate(
        sections=[SDDSectionSpec(id="1", title="Scope", instruction="i")]
    )
    return ConfigBundle(project=project, csc_descriptors=[csc], template=template)


def test_cli_propose_updates_and_inspect_diff_paths(tmp_path: Path, monkeypatch) -> None:
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
            retrieval_index=RetrievalIndex(chunks=[]),
            report_markdown_path=tmp_path / "r.md",
            report_json_path=tmp_path / "r.json",
            retrieval_index_path=tmp_path / "i.json",
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
