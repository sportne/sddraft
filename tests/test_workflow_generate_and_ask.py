"""Workflow tests for generation and grounded Q&A."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.models import QueryRequest
from sddraft.llm.base import StructuredGenerationRequest
from sddraft.llm.mock import MockLLMClient
from sddraft.workflows.ask import answer_question
from sddraft.workflows.generate import generate_sdd


class RecordingMockLLMClient(MockLLMClient):
    """Mock client that records structured generation requests."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[StructuredGenerationRequest] = []

    def generate_structured(self, request: StructuredGenerationRequest):
        self.requests.append(request)
        return super().generate_structured(request)


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
    assert result.hierarchy_json_path is not None
    assert result.hierarchy_index_path is not None
    assert result.hierarchy_json_path.exists()
    assert result.hierarchy_index_path.exists()
    assert result.document.sections

    ask_result = answer_question(
        request=QueryRequest(question="What does navigation control do?", top_k=4),
        index_path=result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert ask_result.answer.answer
    assert ask_result.answer.citations


def test_ask_uses_hierarchy_expansion_when_available(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: float, y: float) -> float:\n    return x + y\n",
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

    ask_result = answer_question(
        request=QueryRequest(question="compute_distance", top_k=1),
        index_path=result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert any(
        chunk.source_type == "directory_summary"
        for chunk in ask_result.evidence_pack.chunks
    )


def test_ask_fallback_when_hierarchy_index_missing(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: float, y: float) -> float:\n    return x + y\n",
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
    assert result.hierarchy_index_path is not None
    result.hierarchy_index_path.unlink()

    ask_result = answer_question(
        request=QueryRequest(question="compute_distance", top_k=1),
        index_path=result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert any(
        "Hierarchy index unavailable" in item for item in ask_result.answer.uncertainty
    )


def test_generate_flow_honors_runtime_model_and_temperature(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: float, y: float) -> float:\n    return x + y\n",
        encoding="utf-8",
    )

    llm = RecordingMockLLMClient()
    override_model = "mock-override"
    override_temperature = 0.61

    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        model_name=override_model,
        temperature=override_temperature,
    )

    assert result.document.sections
    assert llm.requests
    assert all(request.model_name == override_model for request in llm.requests)
    assert all(request.temperature == override_temperature for request in llm.requests)


def test_generate_reuses_existing_hierarchy_summaries(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: float, y: float) -> float:\n    return x + y\n",
        encoding="utf-8",
    )

    llm = RecordingMockLLMClient()
    generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
    )
    assert any(
        request.response_model.__name__ == "_FileSummaryDraft"
        for request in llm.requests
    )
    llm.requests.clear()

    generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
    )

    assert all(
        request.response_model.__name__
        not in {"_FileSummaryDraft", "_DirectorySummaryDraft"}
        for request in llm.requests
    )


def test_generate_progress_callback_reports_stages(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "module.py").write_text(
        "def compute_distance(x: float, y: float) -> float:\n    return x + y\n",
        encoding="utf-8",
    )

    messages: list[str] = []
    llm = MockLLMClient()
    generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        progress_callback=messages.append,
    )

    joined = "\n".join(messages)
    assert "Scanning repository" in joined
    assert "Generating section" in joined
    assert "Hierarchy planning" in joined
