"""Tests for repository scanning."""

from __future__ import annotations

from pathlib import Path

from sddraft.repo.scanner import scan_repository


def test_scan_repository_extracts_summaries_and_chunks(
    tmp_path: Path,
    sample_project_config,
    sample_csc,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "tests").mkdir()

    module_path = src_dir / "module.py"
    module_path.write_text(
        '"""Module docs."""\n\nimport os\n\n\nclass NavService:\n    def run(self) -> None:\n        """Run loop"""\n        pass\n\n\ndef plan_route() -> str:\n    return "ok"\n',
        encoding="utf-8",
    )
    (src_dir / "tests" / "ignored.py").write_text("x=1", encoding="utf-8")

    result = scan_repository(sample_project_config, sample_csc, repo_root=tmp_path)

    assert module_path.resolve() in result.files
    assert all("tests" not in path.as_posix() for path in result.files)

    assert result.code_summaries
    summary = result.code_summaries[0]
    assert "plan_route" in summary.functions
    assert "NavService" in summary.classes
    assert any("import os" in dep for dep in summary.imports)

    interface_names = {item.name for item in result.interface_summaries}
    assert "NavService" in interface_names
    assert "plan_route" in interface_names

    assert result.code_chunks
    first_chunk = result.code_chunks[0]
    assert first_chunk.source_type == "code"
    assert first_chunk.line_start == 1
    assert first_chunk.line_end is not None
    assert first_chunk.metadata.get("language") == "python"
