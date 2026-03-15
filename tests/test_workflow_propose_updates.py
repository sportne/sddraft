"""Workflow tests for commit-based update proposals."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sddraft.analysis.hierarchy import directory_node_id, file_node_id
from sddraft.domain.errors import ConfigError
from sddraft.domain.models import (
    DirectorySummaryDoc,
    FileSummaryDoc,
    HierarchyDocArtifact,
)
from sddraft.llm.base import StructuredGenerationRequest
from sddraft.llm.mock import MockLLMClient
from sddraft.workflows.propose_updates import propose_updates


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


class RecordingMockLLMClient(MockLLMClient):
    """Mock client that records structured generation requests."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[StructuredGenerationRequest] = []

    def generate_structured(self, request: StructuredGenerationRequest):
        self.requests.append(request)
        return super().generate_structured(request)


def test_propose_updates_flow(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    file_path = src_dir / "module.py"
    file_path.write_text(
        "import os\n\n\ndef do_work(x):\n    return x\n", encoding="utf-8"
    )

    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "initial"], cwd=tmp_path)

    file_path.write_text(
        "import sys\n\n\ndef do_work(x, y):\n    return x + y\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "change signature"], cwd=tmp_path)

    existing_sdd = tmp_path / "existing_sdd.md"
    existing_sdd.write_text(
        "# NAV_CTRL Software Design Description\n\n## 3 Interface Design\nOld text\n",
        encoding="utf-8",
    )

    llm = RecordingMockLLMClient()
    override_model = "mock-override"
    override_temperature = 0.44
    result = propose_updates(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        existing_sdd_path=existing_sdd,
        commit_range="HEAD~1..HEAD",
        repo_root=tmp_path,
        model_name=override_model,
        temperature=override_temperature,
    )

    assert result.report_markdown_path.exists()
    assert result.report_json_path.exists()
    assert result.retrieval_index_path.exists()
    assert result.hierarchy_json_path is not None
    assert result.hierarchy_index_path is not None
    assert result.hierarchy_json_path.exists()
    assert result.hierarchy_index_path.exists()
    assert result.impact.changed_files
    assert llm.requests
    assert all(request.model_name == override_model for request in llm.requests)
    assert all(request.temperature == override_temperature for request in llm.requests)


def test_propose_updates_requires_existing_sdd_file(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    missing_sdd = tmp_path / "missing.md"
    llm = MockLLMClient()
    with pytest.raises(ConfigError, match="Existing SDD file not found"):
        propose_updates(
            project_config=sample_project_config,
            csc=sample_csc,
            template=sample_template,
            llm_client=llm,
            existing_sdd_path=missing_sdd,
            commit_range="HEAD~1..HEAD",
            repo_root=tmp_path,
        )


def test_propose_updates_hierarchy_refresh_preserves_unaffected_file_summary(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
    sample_template,
) -> None:
    src_dir = tmp_path / "src"
    nested_dir = src_dir / "sub"
    nested_dir.mkdir(parents=True)

    changed_file = src_dir / "module.py"
    unchanged_file = nested_dir / "worker.py"
    changed_file.write_text("def a(x):\n    return x\n", encoding="utf-8")
    unchanged_file.write_text("def b(y):\n    return y\n", encoding="utf-8")

    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "initial"], cwd=tmp_path)

    output_root = sample_project_config.output_dir / sample_csc.csc_id
    hierarchy_root = output_root / "hierarchy"
    hierarchy_root.mkdir(parents=True, exist_ok=True)

    existing_hierarchy = HierarchyDocArtifact(
        csc_id=sample_csc.csc_id,
        root=Path("."),
        file_summaries=[
            FileSummaryDoc(
                node_id=file_node_id(Path("src/module.py")),
                path=Path("src/module.py"),
                language="python",
                summary="OLD-MODULE-SUMMARY",
            ),
            FileSummaryDoc(
                node_id=file_node_id(Path("src/sub/worker.py")),
                path=Path("src/sub/worker.py"),
                language="python",
                summary="PRESERVE-WORKER-SUMMARY",
            ),
        ],
        directory_summaries=[
            DirectorySummaryDoc(
                node_id=directory_node_id(Path(".")),
                path=Path("."),
                summary="ROOT",
                child_directories=[Path("src")],
            ),
            DirectorySummaryDoc(
                node_id=directory_node_id(Path("src")),
                path=Path("src"),
                summary="SRC",
                local_files=[Path("src/module.py")],
                child_directories=[Path("src/sub")],
            ),
            DirectorySummaryDoc(
                node_id=directory_node_id(Path("src/sub")),
                path=Path("src/sub"),
                summary="SUB",
                local_files=[Path("src/sub/worker.py")],
            ),
        ],
    )
    hierarchy_json_path = hierarchy_root / "hierarchy_artifact.json"
    hierarchy_json_path.write_text(
        existing_hierarchy.model_dump_json(indent=2), encoding="utf-8"
    )

    changed_file.write_text("def a(x, y):\n    return x + y\n", encoding="utf-8")
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "update module"], cwd=tmp_path)

    existing_sdd = tmp_path / "existing_sdd.md"
    existing_sdd.write_text(
        "# NAV_CTRL Software Design Description\n\n## 3 Interface Design\nOld text\n",
        encoding="utf-8",
    )

    llm = RecordingMockLLMClient()
    result = propose_updates(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        existing_sdd_path=existing_sdd,
        commit_range="HEAD~1..HEAD",
        repo_root=tmp_path,
    )

    assert result.hierarchy_json_path is not None
    refreshed = HierarchyDocArtifact.model_validate_json(
        result.hierarchy_json_path.read_text(encoding="utf-8")
    )
    worker_summary = next(
        item.summary
        for item in refreshed.file_summaries
        if item.path == Path("src/sub/worker.py")
    )
    assert worker_summary == "PRESERVE-WORKER-SUMMARY"
