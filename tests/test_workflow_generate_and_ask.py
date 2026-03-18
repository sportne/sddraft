"""Workflow tests for generation and grounded Q&A."""

from __future__ import annotations

from pathlib import Path

from sddraft.domain.models import (
    DirectorySummaryRecord,
    HierarchyManifest,
    QueryRequest,
)
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
    assert result.hierarchy_manifest_path is not None
    assert result.hierarchy_store_path is not None
    assert result.graph_manifest_path is not None
    assert result.graph_store_path is not None
    assert result.hierarchy_manifest_path.exists()
    assert result.hierarchy_store_path.exists()
    assert result.graph_manifest_path.exists()
    assert result.graph_store_path.exists()
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


def test_ask_fallback_when_hierarchy_store_missing(
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
    assert result.hierarchy_manifest_path is not None
    result.hierarchy_manifest_path.unlink()

    ask_result = answer_question(
        request=QueryRequest(question="compute_distance", top_k=1),
        index_path=result.retrieval_index_path,
        llm_client=llm,
        model_name="mock-sddraft",
    )

    assert any(
        "Hierarchy store unavailable" in item for item in ask_result.answer.uncertainty
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


def test_generate_enforces_template_section_identity(
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

    llm = MockLLMClient(
        canned={
            "SectionDraft": {
                "section_id": "OFF_TEMPLATE",
                "title": "Off Template Title",
                "content": "TBD: intentionally mismatched section identity.",
                "evidence_refs": [],
                "assumptions": [],
                "missing_information": ["TBD"],
                "confidence": 0.5,
            }
        }
    )

    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
        hierarchy_docs_enabled=False,
    )

    assert all(
        section.section_id != "OFF_TEMPLATE" for section in result.document.sections
    )
    assert all(
        section.title != "Off Template Title" for section in result.document.sections
    )
    assert [section.section_id for section in result.document.sections] == [
        item.id for item in sample_template.sections
    ]
    assert [section.title for section in result.document.sections] == [
        item.title for item in sample_template.sections
    ]

    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "OFF_TEMPLATE" not in markdown
    assert "Off Template Title" not in markdown


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


def test_generate_hierarchy_directory_rollups_are_recursive(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    nested_dir = src_dir / "sub"
    nested_dir.mkdir(parents=True)

    (src_dir / "module.py").write_text(
        "def a() -> int:\n    return 1\n", encoding="utf-8"
    )
    (nested_dir / "worker.py").write_text(
        "def b() -> int:\n    return 2\n", encoding="utf-8"
    )

    result = generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=MockLLMClient(),
        repo_root=tmp_path,
    )
    assert result.hierarchy_manifest_path is not None

    manifest = HierarchyManifest.model_validate_json(
        result.hierarchy_manifest_path.read_text(encoding="utf-8")
    )
    directory_records = [
        DirectorySummaryRecord.model_validate_json(line)
        for line in (
            result.hierarchy_manifest_path.parent / manifest.directory_summaries_path
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_path = {item.path: item for item in directory_records}

    root_rollup = by_path[Path(".")].subtree_rollup
    src_rollup = by_path[Path("src")].subtree_rollup
    sub_rollup = by_path[Path("src/sub")].subtree_rollup

    assert root_rollup.descendant_file_count == 2
    assert root_rollup.descendant_directory_count == 2
    assert root_rollup.language_counts.get("python") == 2
    assert Path("src/module.py") in root_rollup.representative_files
    assert Path("src/sub/worker.py") in root_rollup.representative_files

    assert src_rollup.descendant_file_count == 2
    assert src_rollup.descendant_directory_count == 1
    assert src_rollup.language_counts.get("python") == 2

    assert sub_rollup.descendant_file_count == 1
    assert sub_rollup.descendant_directory_count == 0
    assert sub_rollup.language_counts.get("python") == 1


def test_generate_root_directory_prompt_uses_project_level_subtree_context(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    nested_dir = src_dir / "sub"
    nested_dir.mkdir(parents=True)

    (src_dir / "module.py").write_text(
        "def a() -> int:\n    return 1\n", encoding="utf-8"
    )
    (nested_dir / "worker.py").write_text(
        "def b() -> int:\n    return 2\n", encoding="utf-8"
    )

    llm = RecordingMockLLMClient()
    generate_sdd(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        repo_root=tmp_path,
    )

    root_prompt = next(
        request.user_prompt
        for request in llm.requests
        if request.response_model.__name__ == "_DirectorySummaryDraft"
        and "Directory Path:\n." in request.user_prompt
    )
    assert "Directory Role:\nproject root" in root_prompt
    assert "Subtree Rollup" in root_prompt
    assert "project-level overview" in root_prompt
    assert "src/sub/worker.py" in root_prompt


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
