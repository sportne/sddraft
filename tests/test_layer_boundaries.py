"""Architectural boundary tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "sddraft"


def _python_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*.py") if item.is_file())


def test_domain_does_not_import_project_modules() -> None:
    for file_path in _python_files(ROOT / "domain"):
        text = file_path.read_text(encoding="utf-8")
        assert "from sddraft." not in text.replace("from sddraft.domain", "")
        assert "import sddraft." not in text


def test_repo_does_not_import_llm_module() -> None:
    for file_path in _python_files(ROOT / "repo"):
        text = file_path.read_text(encoding="utf-8")
        assert "sddraft.llm" not in text


def test_analysis_does_not_import_provider_sdks() -> None:
    for file_path in _python_files(ROOT / "analysis"):
        text = file_path.read_text(encoding="utf-8")
        assert "google.genai" not in text
        assert "from google" not in text


def test_render_does_not_import_repo_module() -> None:
    for file_path in _python_files(ROOT / "render"):
        text = file_path.read_text(encoding="utf-8")
        assert "sddraft.repo" not in text
