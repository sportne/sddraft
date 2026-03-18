"""Workflow test for multi-language generation and ask."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.models import QueryRequest
from sddraft.llm.mock import MockLLMClient
from sddraft.workflows.ask import answer_question
from sddraft.workflows.generate import generate_sdd


def test_generate_and_ask_with_multilanguage_sources(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "module.py").write_text(
        "import math\n\nclass PySystem:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    (src_dir / "Main.java").write_text(
        "package demo;\nimport java.util.Map;\npublic class Main {\n  public int run(int x) { return x; }\n}\n",
        encoding="utf-8",
    )
    (src_dir / "core.cpp").write_text(
        "#include <vector>\nclass Core {};\nint run(int x) { return x; }\n",
        encoding="utf-8",
    )

    llm = MockLLMClient()
    generate_result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
    )

    assert generate_result.document.sections
    assert generate_result.retrieval_manifest.total_chunks > 0

    ask_result = answer_question(
        request=QueryRequest(question="What interfaces exist?", top_k=6),
        index_path=generate_result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert ask_result.answer.citations
    sources = {citation.source_path.suffix for citation in ask_result.answer.citations}
    assert ".md" in sources or ".json" in sources or ".py" in sources


def test_ask_dependency_question_uses_graph_import_edges_for_javascript(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "module.js").write_text(
        "export function dependency() { return 1; }\n",
        encoding="utf-8",
    )
    (src_dir / "caller.js").write_text(
        "import { dependency } from './module.js';\nexport function caller() { return dependency(); }\n",
        encoding="utf-8",
    )

    llm = MockLLMClient()

    no_graph = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=False,
    )
    no_graph_ask = answer_question(
        request=QueryRequest(question="What does caller.js depend on?", top_k=1),
        index_path=no_graph.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
        graph_enabled=True,
        graph_depth=1,
        graph_top_k=6,
    )
    assert any(
        "Engineering graph store unavailable" in item
        for item in no_graph_ask.answer.uncertainty
    )

    graph = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
        graph_enabled=True,
    )
    graph_ask = answer_question(
        request=QueryRequest(question="What does caller.js depend on?", top_k=1),
        index_path=graph.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
        graph_enabled=True,
        graph_depth=1,
        graph_top_k=6,
    )

    assert any(
        chunk.source_path == Path("src/module.js")
        for chunk in graph_ask.evidence_pack.chunks
    )
    assert any(
        reason.source == "graph" for reason in graph_ask.evidence_pack.inclusion_reasons
    )
