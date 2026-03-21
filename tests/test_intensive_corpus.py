"""Tests for deterministic intensive ask corpus building."""

from __future__ import annotations

from pathlib import Path

from engllm.core.analysis.intensive_corpus import (
    default_intensive_corpus_root,
    iter_intensive_corpus_chunks,
    prepare_intensive_corpus,
)


def test_prepare_intensive_corpus_packs_whole_files_across_boundaries(
    tmp_path: Path,
    sample_project_config,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.py").write_text("alpha beta\ngamma delta\n", encoding="utf-8")
    (src_dir / "b.py").write_text("epsilon zeta\neta theta\n", encoding="utf-8")

    output_root = tmp_path / "artifacts" / "workspaces" / "NAV_CTRL" / "tools" / "ask"
    manifest, reused = prepare_intensive_corpus(
        project_config=sample_project_config,
        repo_root=tmp_path,
        output_root=output_root,
        csc_id="NAV_CTRL",
        chunk_tokens=16,
    )

    assert reused is False
    assert manifest.file_count == 2
    chunks = list(
        iter_intensive_corpus_chunks(
            default_intensive_corpus_root(output_root) / "manifest.json"
        )
    )
    assert len(chunks) == 1
    assert [segment.source_path.as_posix() for segment in chunks[0].segments] == [
        "src/a.py",
        "src/b.py",
    ]


def test_prepare_intensive_corpus_splits_only_oversized_files_and_reuses_store(
    tmp_path: Path,
    sample_project_config,
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "big.py").write_text(
        "one two three\nfour five six\nseven eight nine\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "artifacts" / "workspaces" / "NAV_CTRL" / "tools" / "ask"
    first_manifest, first_reused = prepare_intensive_corpus(
        project_config=sample_project_config,
        repo_root=tmp_path,
        output_root=output_root,
        csc_id="NAV_CTRL",
        chunk_tokens=5,
    )
    second_manifest, second_reused = prepare_intensive_corpus(
        project_config=sample_project_config,
        repo_root=tmp_path,
        output_root=output_root,
        csc_id="NAV_CTRL",
        chunk_tokens=5,
    )

    assert first_reused is False
    assert second_reused is True
    assert first_manifest.corpus_fingerprint == second_manifest.corpus_fingerprint

    chunks = list(
        iter_intensive_corpus_chunks(
            default_intensive_corpus_root(output_root) / "manifest.json"
        )
    )
    assert len(chunks) == 3
    assert [
        (segment.line_start, segment.line_end)
        for chunk in chunks
        for segment in chunk.segments
    ] == [(1, 1), (2, 2), (3, 3)]
