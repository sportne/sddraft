"""Tests for repository scanning."""

from __future__ import annotations

from pathlib import Path

from engllm.core.repo.scanner import scan_repository


def test_scan_repository_extracts_summaries_and_chunks(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "tests").mkdir()
    (tmp_path / ".gitignore").write_text("src/ignored.md\n", encoding="utf-8")

    module_path = src_dir / "module.py"
    module_path.write_text(
        '"""Module docs."""\n\nimport os\n\n\nclass NavService:\n    def run(self) -> None:\n        """Run loop"""\n        pass\n\n\ndef plan_route() -> str:\n    return "ok"\n',
        encoding="utf-8",
    )
    makefile_path = src_dir / "Makefile"
    makefile_path.write_text("include common.mk\nall:\n\t@echo ok\n", encoding="utf-8")
    ignored_doc = src_dir / "ignored.md"
    ignored_doc.write_text("# ignored", encoding="utf-8")
    (src_dir / "tests" / "ignored.py").write_text("x=1", encoding="utf-8")

    result = scan_repository(sample_project_config, sample_csc, repo_root=tmp_path)

    assert Path("src/module.py") in result.files
    assert Path("src/Makefile") in result.files
    assert Path("src/ignored.md") not in result.files
    assert all("tests" not in path.as_posix() for path in result.files)

    assert result.code_summaries
    python_summary = next(
        item for item in result.code_summaries if item.path == Path("src/module.py")
    )
    assert "plan_route" in python_summary.functions
    assert "NavService" in python_summary.classes
    assert any("import os" in dep for dep in python_summary.imports)
    unknown_summary = next(
        item for item in result.code_summaries if item.path == Path("src/Makefile")
    )
    assert unknown_summary.language == "unknown"
    assert "include common.mk" in unknown_summary.imports

    symbol_names = {item.name for item in result.symbol_summaries}
    assert "NavService" in symbol_names
    assert "plan_route" in symbol_names

    assert result.code_chunks
    chunk_languages = {item.metadata.get("language") for item in result.code_chunks}
    assert "python" in chunk_languages
    assert "unknown" in chunk_languages
