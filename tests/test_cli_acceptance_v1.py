"""V1 acceptance tests for required CLI commands and offline UX behavior."""

from __future__ import annotations

import subprocess
import urllib.error
from pathlib import Path

from sddraft.cli.main import main
from sddraft.llm import ollama as ollama_module


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _write_project_files(
    tmp_path: Path, *, include_second_csc: bool
) -> tuple[Path, list[Path], Path]:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    module_path = src_dir / "module.py"
    module_path.write_text(
        "def do_work(x: int) -> int:\n    return x + 1\n", encoding="utf-8"
    )

    template_path = _write_file(
        tmp_path / "templates" / "template.yaml",
        """
document_type: sdd
sections:
  - id: "1"
    title: "Scope"
    instruction: "Describe scope."
  - id: "2"
    title: "Design Overview"
    instruction: "Describe overview."
""",
    )

    project_path = _write_file(
        tmp_path / "examples" / "project.yaml",
        """
project_name: ExampleProject
sources:
  roots: [../src]
  include: ["**/*.py"]
  exclude: []
sdd_template: ../templates/template.yaml
llm:
  provider: mock
  model_name: mock-sddraft
  temperature: 0.2
output_dir: ../artifacts
""",
    )

    csc_paths = [
        _write_file(
            tmp_path / "examples" / "csc_nav.yaml",
            """
csc_id: NAV_CTRL
title: Navigation Control
purpose: Controls navigation.
source_roots: [../src]
""",
        )
    ]

    if include_second_csc:
        csc_paths.append(
            _write_file(
                tmp_path / "examples" / "csc_power.yaml",
                """
csc_id: POWER_CTRL
title: Power Control
purpose: Controls power.
source_roots: [../src]
""",
            )
        )

    return project_path, csc_paths, template_path


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def test_cli_acceptance_required_commands_and_batch_generation(
    tmp_path: Path, monkeypatch
) -> None:
    project_path, csc_paths, _ = _write_project_files(tmp_path, include_second_csc=True)
    monkeypatch.chdir(tmp_path)

    validate_rc = main(
        [
            "validate-config",
            "--project-config",
            str(project_path),
            "--csc",
            str(csc_paths[0]),
            str(csc_paths[1]),
        ]
    )
    assert validate_rc == 0

    generate_rc = main(
        [
            "generate",
            "--project-config",
            str(project_path),
            "--csc",
            str(csc_paths[0]),
            str(csc_paths[1]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "mock",
            "--model",
            "mock-sddraft",
            "--temperature",
            "0.2",
        ]
    )
    assert generate_rc == 0

    nav_root = tmp_path / "artifacts" / "NAV_CTRL"
    power_root = tmp_path / "artifacts" / "POWER_CTRL"
    for root in (nav_root, power_root):
        assert (root / "sdd.md").exists()
        assert (root / "review_artifact.json").exists()
        assert (root / "retrieval" / "manifest.json").exists()


def test_cli_acceptance_propose_updates_and_inspect_diff(
    tmp_path: Path, monkeypatch
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)

    src_file = tmp_path / "src" / "module.py"
    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "initial"], cwd=tmp_path)

    src_file.write_text(
        "import math\n\n\ndef do_work(x: int, y: int) -> int:\n    return x + y\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "signature update"], cwd=tmp_path)

    existing_sdd = _write_file(
        tmp_path / "existing_sdd.md",
        """
# NAV_CTRL Software Design Description

## 3 Interface Design
Old text
""",
    )

    propose_rc = main(
        [
            "propose-updates",
            "--project-config",
            str(project_path),
            "--csc",
            str(csc_paths[0]),
            "--existing-sdd",
            str(existing_sdd),
            "--commit-range",
            "HEAD~1..HEAD",
            "--repo-root",
            str(tmp_path),
            "--provider",
            "mock",
            "--model",
            "mock-sddraft",
            "--temperature",
            "0.2",
        ]
    )
    assert propose_rc == 0

    out_root = tmp_path / "artifacts" / "NAV_CTRL"
    assert (out_root / "update_report.md").exists()
    assert (out_root / "update_proposals.json").exists()
    assert (out_root / "retrieval" / "manifest.json").exists()

    inspect_rc = main(
        [
            "inspect-diff",
            "--commit-range",
            "HEAD~1..HEAD",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert inspect_rc == 0


def test_cli_acceptance_gemini_provider_missing_config_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    rc = main(
        [
            "generate",
            "--project-config",
            str(project_path),
            "--csc",
            str(csc_paths[0]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "gemini",
            "--model",
            "gemini-2.5-flash",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert (
        "GEMINI_API_KEY is not configured" in output
        or "Gemini provider dependencies are unavailable" in output
    )


def test_cli_acceptance_ollama_provider_unreachable_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)

    def _raise_connection_error(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ollama_module.request_lib, "urlopen", _raise_connection_error)

    rc = main(
        [
            "generate",
            "--project-config",
            str(project_path),
            "--csc",
            str(csc_paths[0]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "ollama",
            "--model",
            "qwen2.5:14b-instruct-q4_K_M",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert "Cannot connect to Ollama" in output
