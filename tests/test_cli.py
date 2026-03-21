"""CLI behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

from engllm.cli.main import main
from engllm.core.analysis.retrieval import load_retrieval_manifest


def _write_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    src_dir = tmp_path / "src"
    templates_dir = tmp_path / "templates"
    examples_dir = tmp_path / "examples"
    src_dir.mkdir()
    templates_dir.mkdir()
    examples_dir.mkdir()

    (src_dir / "module.py").write_text(
        "def fn() -> str:\n    return 'ok'\n", encoding="utf-8"
    )

    template_path = templates_dir / "template.yaml"
    template_path.write_text(
        """
document_type: sdd
sections:
  - id: "1"
    title: "Scope"
    instruction: "Describe"
""".strip(),
        encoding="utf-8",
    )

    project_path = examples_dir / "project.yaml"
    project_path.write_text(
        """
project_name: Example
workspace:
  output_root: ../artifacts
sources:
  roots: [../src]
  include: ["**/*.py"]
  exclude: []
llm:
  provider: mock
  model_name: mock-engllm
  temperature: 0.2
tools:
  sdd:
    template: ../templates/template.yaml
""".strip(),
        encoding="utf-8",
    )

    csc_path = examples_dir / "csc.yaml"
    csc_path.write_text(
        """
csc_id: NAV_CTRL
title: Navigation Control
purpose: Controls navigation.
source_roots: [../src]
""".strip(),
        encoding="utf-8",
    )

    return project_path, csc_path, template_path


def test_cli_validate_generate_and_ask(tmp_path: Path, monkeypatch) -> None:
    project_path, csc_path, _ = _write_files(tmp_path)

    monkeypatch.chdir(tmp_path)

    validate_exit = main(
        [
            "sdd",
            "validate-config",
            "--config",
            str(project_path),
            "--target",
            str(csc_path),
        ]
    )
    assert validate_exit == 0

    generate_exit = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_path),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "mock",
            "--model",
            "mock-engllm",
        ]
    )
    assert generate_exit == 0

    index_path = (
        tmp_path / "artifacts" / "workspaces" / "NAV_CTRL" / "shared" / "retrieval"
    )
    ask_exit = main(
        [
            "ask",
            "answer",
            "--index-path",
            str(index_path),
            "--question",
            "What does this component do?",
            "--provider",
            "mock",
            "--model",
            "mock-engllm",
        ]
    )
    assert ask_exit == 0


def test_cli_generate_no_hierarchy_docs_flag(tmp_path: Path, monkeypatch) -> None:
    project_path, csc_path, _ = _write_files(tmp_path)
    monkeypatch.chdir(tmp_path)

    generate_exit = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_path),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "mock",
            "--model",
            "mock-engllm",
            "--no-hierarchy-docs",
        ]
    )
    assert generate_exit == 0

    hierarchy_dir = (
        tmp_path / "artifacts" / "workspaces" / "NAV_CTRL" / "shared" / "hierarchy"
    )
    assert not hierarchy_dir.exists()

    index_path = (
        tmp_path / "artifacts" / "workspaces" / "NAV_CTRL" / "shared" / "retrieval"
    )
    manifest, store_root = load_retrieval_manifest(index_path)
    source_types: set[str] = set()
    for shard in manifest.chunk_shards:
        rows = (
            json.loads(line)
            for line in (store_root / shard.path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        source_types.update(str(row["source_type"]) for row in rows)
    assert "file_summary" not in source_types
    assert "directory_summary" not in source_types
