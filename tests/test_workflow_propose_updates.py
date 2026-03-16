"""Workflow tests for commit-based update proposals."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sddraft.analysis.hierarchy import directory_node_id, file_node_id
from sddraft.domain.errors import ConfigError
from sddraft.domain.models import (
    DirectorySummaryRecord,
    FileSummaryRecord,
    HierarchyEdgeRecord,
    HierarchyManifest,
    HierarchyNodeRecord,
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
    assert result.hierarchy_manifest_path is not None
    assert result.hierarchy_store_path is not None
    assert result.hierarchy_manifest_path.exists()
    assert result.hierarchy_store_path.exists()
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

    manifest = HierarchyManifest(
        csc_id=sample_csc.csc_id,
        root=Path("."),
        file_summaries_path=Path("file_summaries.jsonl"),
        directory_summaries_path=Path("directory_summaries.jsonl"),
        nodes_path=Path("nodes.jsonl"),
        edges_path=Path("edges.jsonl"),
    )
    (hierarchy_root / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    (hierarchy_root / "file_summaries.jsonl").write_text(
        "\n".join(
            [
                FileSummaryRecord(
                    node_id=file_node_id(Path("src/module.py")),
                    path=Path("src/module.py"),
                    language="python",
                    summary="OLD-MODULE-SUMMARY",
                ).model_dump_json(),
                FileSummaryRecord(
                    node_id=file_node_id(Path("src/sub/worker.py")),
                    path=Path("src/sub/worker.py"),
                    language="python",
                    summary="PRESERVE-WORKER-SUMMARY",
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (hierarchy_root / "directory_summaries.jsonl").write_text(
        "\n".join(
            [
                DirectorySummaryRecord(
                    node_id=directory_node_id(Path(".")),
                    path=Path("."),
                    summary="ROOT",
                    child_directories=[Path("src")],
                ).model_dump_json(),
                DirectorySummaryRecord(
                    node_id=directory_node_id(Path("src")),
                    path=Path("src"),
                    summary="SRC",
                    local_files=[Path("src/module.py")],
                    child_directories=[Path("src/sub")],
                ).model_dump_json(),
                DirectorySummaryRecord(
                    node_id=directory_node_id(Path("src/sub")),
                    path=Path("src/sub"),
                    summary="SUB",
                    local_files=[Path("src/sub/worker.py")],
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (hierarchy_root / "nodes.jsonl").write_text(
        "\n".join(
            [
                HierarchyNodeRecord(
                    node_id=directory_node_id(Path(".")),
                    kind="directory",
                    path=Path("."),
                    doc_path=Path("_directory.md"),
                    abstract="ROOT",
                ).model_dump_json(),
                HierarchyNodeRecord(
                    node_id=directory_node_id(Path("src")),
                    kind="directory",
                    path=Path("src"),
                    parent_id=directory_node_id(Path(".")),
                    doc_path=Path("src/_directory.md"),
                    abstract="SRC",
                ).model_dump_json(),
                HierarchyNodeRecord(
                    node_id=directory_node_id(Path("src/sub")),
                    kind="directory",
                    path=Path("src/sub"),
                    parent_id=directory_node_id(Path("src")),
                    doc_path=Path("src/sub/_directory.md"),
                    abstract="SUB",
                ).model_dump_json(),
                HierarchyNodeRecord(
                    node_id=file_node_id(Path("src/module.py")),
                    kind="file",
                    path=Path("src/module.py"),
                    parent_id=directory_node_id(Path("src")),
                    doc_path=Path("src/module.py.md"),
                    abstract="OLD-MODULE-SUMMARY",
                ).model_dump_json(),
                HierarchyNodeRecord(
                    node_id=file_node_id(Path("src/sub/worker.py")),
                    kind="file",
                    path=Path("src/sub/worker.py"),
                    parent_id=directory_node_id(Path("src/sub")),
                    doc_path=Path("src/sub/worker.py.md"),
                    abstract="PRESERVE-WORKER-SUMMARY",
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (hierarchy_root / "edges.jsonl").write_text(
        "\n".join(
            [
                HierarchyEdgeRecord(
                    parent_id=directory_node_id(Path(".")),
                    child_id=directory_node_id(Path("src")),
                ).model_dump_json(),
                HierarchyEdgeRecord(
                    parent_id=directory_node_id(Path("src")),
                    child_id=directory_node_id(Path("src/sub")),
                ).model_dump_json(),
                HierarchyEdgeRecord(
                    parent_id=directory_node_id(Path("src")),
                    child_id=file_node_id(Path("src/module.py")),
                ).model_dump_json(),
                HierarchyEdgeRecord(
                    parent_id=directory_node_id(Path("src/sub")),
                    child_id=file_node_id(Path("src/sub/worker.py")),
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
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

    assert result.hierarchy_manifest_path is not None
    refreshed_files = [
        FileSummaryRecord.model_validate_json(line)
        for line in (result.hierarchy_manifest_path.parent / "file_summaries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    worker_summary = next(
        item.summary
        for item in refreshed_files
        if item.path == Path("src/sub/worker.py")
    )
    assert worker_summary == "PRESERVE-WORKER-SUMMARY"


def test_propose_updates_progress_callback_reports_stages(
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

    progress_messages: list[str] = []
    llm = MockLLMClient()
    propose_updates(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        existing_sdd_path=existing_sdd,
        commit_range="HEAD~1..HEAD",
        repo_root=tmp_path,
        progress_callback=progress_messages.append,
    )

    joined = "\n".join(progress_messages)
    assert "Scanning repository" in joined
    assert "Parsing commit range" in joined
    assert "Refreshing hierarchy documentation" in joined
