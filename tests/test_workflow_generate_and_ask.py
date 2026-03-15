"""Workflow tests for generation and grounded Q&A."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.models import QueryRequest
from sddraft.llm.mock import MockLLMClient
from sddraft.workflows.ask import answer_question
from sddraft.workflows.generate import generate_sdd


def test_generate_and_ask_flow(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        """
import math


def compute_distance(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)
""".strip(),
        encoding="utf-8",
    )

    llm = MockLLMClient()
    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
    )

    assert result.markdown_path.exists()
    assert result.review_json_path.exists()
    assert result.retrieval_index_path.exists()
    assert result.document.sections

    ask_result = answer_question(
        request=QueryRequest(question="What does navigation control do?", top_k=4),
        index_path=result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert ask_result.answer.answer
    assert ask_result.answer.citations
