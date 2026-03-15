"""Multi-language repository scanning tests."""

from __future__ import annotations

from pathlib import Path

from sddraft.repo.scanner import scan_repository


def test_scan_repository_multilanguage(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "module.py").write_text(
        "import os\n\nclass PyService:\n    def run(self):\n        pass\n",
        encoding="utf-8",
    )
    (src_dir / "Main.java").write_text(
        "package demo;\nimport java.util.List;\npublic class Main {\n  public int go(int x) { return x; }\n}\n",
        encoding="utf-8",
    )
    (src_dir / "core.cpp").write_text(
        "#include <vector>\nclass Core {};\nint compute(int x) { return x; }\n",
        encoding="utf-8",
    )

    result = scan_repository(sample_project_config, sample_csc, repo_root=tmp_path)

    languages = {summary.language for summary in result.code_summaries}
    assert {"python", "java", "cpp"}.issubset(languages)

    interface_names = {item.name for item in result.interface_summaries}
    assert "PyService" in interface_names
    assert "Main" in interface_names
    assert "Core" in interface_names

    chunk_languages = {chunk.metadata.get("language") for chunk in result.code_chunks}
    assert "python" in chunk_languages
    assert "java" in chunk_languages
    assert "cpp" in chunk_languages
