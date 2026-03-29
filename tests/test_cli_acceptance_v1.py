"""V1 acceptance tests for required CLI commands and offline UX behavior."""

from __future__ import annotations

import subprocess
import urllib.error
from pathlib import Path

from engllm.cli.main import main
from engllm.llm import anthropic as anthropic_module
from engllm.llm import grok as grok_module
from engllm.llm import ollama as ollama_module
from engllm.llm import openai as openai_module


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
            "sdd",
            "validate-config",
            "--config",
            str(project_path),
            "--target",
            str(csc_paths[0]),
            str(csc_paths[1]),
        ]
    )
    assert validate_rc == 0

    generate_rc = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_paths[0]),
            str(csc_paths[1]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "mock",
            "--model",
            "mock-engllm",
            "--temperature",
            "0.2",
        ]
    )
    assert generate_rc == 0

    nav_root = tmp_path / "artifacts" / "workspaces" / "NAV_CTRL"
    power_root = tmp_path / "artifacts" / "workspaces" / "POWER_CTRL"
    for root in (nav_root, power_root):
        assert (root / "tools" / "sdd" / "sdd.md").exists()
        assert (root / "tools" / "sdd" / "review_artifact.json").exists()
        assert (root / "shared" / "retrieval" / "manifest.json").exists()


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
            "sdd",
            "propose-updates",
            "--config",
            str(project_path),
            "--target",
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
            "mock-engllm",
            "--temperature",
            "0.2",
        ]
    )
    assert propose_rc == 0

    out_root = tmp_path / "artifacts" / "workspaces" / "NAV_CTRL"
    assert (out_root / "tools" / "sdd" / "update_report.md").exists()
    assert (out_root / "tools" / "sdd" / "update_proposals.json").exists()
    assert (out_root / "shared" / "retrieval" / "manifest.json").exists()

    inspect_rc = main(
        [
            "repo",
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
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
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
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
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


def test_cli_acceptance_openai_provider_missing_config_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _FakeOpenAIResponses:
        def parse(self, **_kwargs):
            raise AssertionError("parse should not run when api key is missing")

    class _FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = _FakeOpenAIResponses()

    monkeypatch.setattr(openai_module, "OpenAI", _FakeOpenAI)

    rc = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_paths[0]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "openai",
            "--model",
            "gpt-4.1-mini",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert "OPENAI_API_KEY is not configured" in output


def test_cli_acceptance_anthropic_provider_missing_config_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class _FakeAnthropicMessages:
        def create(self, **_kwargs):
            raise AssertionError("create should not run when api key is missing")

    class _FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = _FakeAnthropicMessages()

    monkeypatch.setattr(anthropic_module, "Anthropic", _FakeAnthropic)

    rc = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_paths[0]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-latest",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY is not configured" in output


def test_cli_acceptance_grok_provider_missing_config_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project_path, csc_paths, _ = _write_project_files(
        tmp_path, include_second_csc=False
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    class _FakeGrokCompletions:
        def parse(self, **_kwargs):
            raise AssertionError("parse should not run when api key is missing")

    class _FakeGrokBeta:
        def __init__(self) -> None:
            self.chat = type(
                "_FakeChat",
                (),
                {"completions": _FakeGrokCompletions()},
            )()

    class _FakeGrokOpenAI:
        def __init__(self, **_kwargs):
            self.beta = _FakeGrokBeta()

    monkeypatch.setattr(grok_module, "OpenAI", _FakeGrokOpenAI)

    rc = main(
        [
            "sdd",
            "generate",
            "--config",
            str(project_path),
            "--target",
            str(csc_paths[0]),
            "--repo-root",
            str(tmp_path),
            "--provider",
            "grok",
            "--model",
            "grok-4-fast",
        ]
    )
    assert rc == 2
    output = capsys.readouterr().out
    assert "XAI_API_KEY is not configured" in output
