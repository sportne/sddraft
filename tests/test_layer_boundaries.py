"""Architectural boundary tests for the EngLLM package layout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "engllm"


def _python_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*.py") if item.is_file())


def test_domain_does_not_import_project_modules() -> None:
    for file_path in _python_files(ROOT / "domain"):
        text = file_path.read_text(encoding="utf-8")
        assert "from engllm." not in text.replace("from engllm.domain", "")
        assert "import engllm." not in text


def test_core_repo_does_not_import_llm_module() -> None:
    for file_path in _python_files(ROOT / "core" / "repo"):
        text = file_path.read_text(encoding="utf-8")
        assert "engllm.llm" not in text


def test_core_analysis_does_not_import_provider_sdks() -> None:
    for file_path in _python_files(ROOT / "core" / "analysis"):
        text = file_path.read_text(encoding="utf-8")
        assert "google.genai" not in text
        assert "from google" not in text


def test_core_render_does_not_import_repo_module() -> None:
    for file_path in _python_files(ROOT / "core" / "render"):
        text = file_path.read_text(encoding="utf-8")
        assert "engllm.core.repo" not in text


def test_integrations_do_not_import_tool_modules() -> None:
    for file_path in _python_files(ROOT / "integrations"):
        text = file_path.read_text(encoding="utf-8")
        assert "engllm.tools" not in text
