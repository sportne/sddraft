"""Workflow tests for commit-based update proposals."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sddraft.llm.mock import MockLLMClient
from sddraft.workflows.propose_updates import propose_updates


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


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

    llm = MockLLMClient()
    result = propose_updates(
        project_config=sample_project_config,
        csc=sample_csc,
        template=sample_template,
        llm_client=llm,
        existing_sdd_path=existing_sdd,
        commit_range="HEAD~1..HEAD",
        repo_root=tmp_path,
    )

    assert result.report_markdown_path.exists()
    assert result.report_json_path.exists()
    assert result.retrieval_index_path.exists()
    assert result.impact.changed_files
