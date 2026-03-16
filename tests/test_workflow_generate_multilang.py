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
