"""Focused tests for incremental graph planner and fragment helper branches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sddraft.analysis.graph_build import (
    _build_fragment,
    _cleanup_stale_fragments,
    _commit_fragment_id,
    _fragment_path,
    _fragment_status,
    _GraphFragment,
    _load_prior_manifest,
    _plan_build,
    _PreparedInputs,
    _read_fragment,
    _write_fragment,
    _write_manifest,
)
from sddraft.analysis.graph_models import (
    GraphBuildPlan,
    GraphInputFingerprint,
    GraphManifest,
    GraphNodeRecord,
)


def _fingerprint(digest: str) -> GraphInputFingerprint:
    return GraphInputFingerprint(
        digest=digest,
        scan_digest=f"scan-{digest}",
        retrieval_digest=f"retrieval-{digest}",
    )


def _prior_manifest() -> GraphManifest:
    return GraphManifest(
        csc_id="NAV_CTRL",
        source_retrieval_manifest=Path("retrieval/manifest.json"),
        nodes_path=Path("nodes.jsonl"),
        edges_path=Path("edges.jsonl"),
        symbol_index_path=Path("symbol_index.json"),
        adjacency_path=Path("adjacency.json"),
        node_counts={
            "directory": 0,
            "file": 0,
            "symbol": 0,
            "chunk": 0,
            "sdd_section": 0,
            "commit": 0,
        },
        edge_counts={
            "contains": 0,
            "defines": 0,
            "references": 0,
            "documents": 0,
            "parent_of": 0,
            "imports": 0,
            "changed_in": 0,
            "impacts_section": 0,
        },
        build_version="v3-incremental-fragment",
        input_fingerprint=_fingerprint("old"),
    )


def test_plan_build_requires_scope_for_partial(tmp_path: Path) -> None:
    graph_root = tmp_path / "graph"
    fragments_root = graph_root / "fragments"
    fragments_root.mkdir(parents=True)

    expected = {"structure": "new-fingerprint"}
    (_fragment_path(fragments_root, "structure")).write_text(
        json.dumps(
            {
                "row_type": "meta",
                "fragment_id": "structure",
                "fingerprint": "old-fingerprint",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan = _plan_build(
        prior_manifest=_prior_manifest(),
        graph_root=graph_root,
        fragments_root=fragments_root,
        input_fingerprint=_fingerprint("new"),
        expected_fingerprints=expected,
        changed_files=set(),
        impacted_sections=set(),
        commit_impact=None,
    )

    assert isinstance(plan, GraphBuildPlan)
    assert plan.decision == "full"
    assert "change scope" in plan.reason


def test_fragment_status_and_cleanup_cover_corrupt_and_stale_files(
    tmp_path: Path,
) -> None:
    fragments_root = tmp_path / "fragments"
    fragments_root.mkdir(parents=True)

    expected_id = "structure"
    expected_path = _fragment_path(fragments_root, expected_id)
    expected_path.write_text(
        json.dumps(
            {
                "row_type": "meta",
                "fragment_id": expected_id,
                "fingerprint": "fp-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        _fragment_status(
            fragments_root=fragments_root,
            fragment_id=expected_id,
            expected_fingerprint="fp-a",
        )
        == "match"
    )

    corrupt_path = fragments_root / "corrupt.jsonl"
    corrupt_path.write_text("not-json\n", encoding="utf-8")

    unexpected_id = "section::3"
    unexpected_path = _fragment_path(fragments_root, unexpected_id)
    unexpected_path.write_text(
        json.dumps(
            {
                "row_type": "meta",
                "fragment_id": unexpected_id,
                "fingerprint": "fp-b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _cleanup_stale_fragments(fragments_root, {expected_id})

    assert expected_path.exists()
    assert not corrupt_path.exists()
    assert not unexpected_path.exists()


def test_read_fragment_error_branches(tmp_path: Path) -> None:
    fragments_root = tmp_path / "fragments"
    fragments_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="Missing graph fragment"):
        _read_fragment(fragments_root, "file::src/missing.py")

    empty_id = "file::src/empty.py"
    _fragment_path(fragments_root, empty_id).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty graph fragment"):
        _read_fragment(fragments_root, empty_id)

    invalid_id = "file::src/invalid.py"
    _fragment_path(fragments_root, invalid_id).write_text(
        json.dumps({"row_type": "meta", "fragment_id": invalid_id}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fragment fingerprint"):
        _read_fragment(fragments_root, invalid_id)

    unknown_row_id = "file::src/unknown.py"
    _fragment_path(fragments_root, unknown_row_id).write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_type": "meta",
                        "fragment_id": unknown_row_id,
                        "fingerprint": "fp",
                    }
                ),
                json.dumps({"row_type": "mystery", "payload": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown fragment row type"):
        _read_fragment(fragments_root, unknown_row_id)


def test_build_fragment_requires_commit_payload_for_commit_fragment() -> None:
    prepared = _PreparedInputs(
        files=[],
        file_summary_by_path={},
        symbol_records=[],
        symbols_by_file={},
        symbols_by_name={},
        symbol_lookup={},
        dependency_records_by_source={},
        code_chunks_by_source={},
        misc_chunks=[],
        sections=[],
        section_targets={},
        all_file_node_ids=set(),
    )

    commit_fragment_id = _commit_fragment_id("HEAD~1..HEAD")
    with pytest.raises(ValueError, match="without commit impact"):
        _build_fragment(
            fragment_id=commit_fragment_id,
            expected_fingerprints={commit_fragment_id: "fp"},
            prepared=prepared,
            commit_impact=None,
        )


def test_write_manifest_handles_non_relative_retrieval_path(tmp_path: Path) -> None:
    graph_root = tmp_path / "graph"
    graph_root.mkdir(parents=True)
    external_retrieval_root = tmp_path.parent / "outside-retrieval"

    manifest_path = _write_manifest(
        graph_root=graph_root,
        csc_id="NAV_CTRL",
        output_root=tmp_path,
        retrieval_root=external_retrieval_root,
        nodes={},
        edges={},
        input_fingerprint=_fingerprint("x"),
        planner_decision="full",
        previous_manifest_rel=None,
        fragment_stats={
            "total_fragments": 0,
            "rebuilt_fragments": 0,
            "reused_fragments": 0,
        },
    )

    manifest = _load_prior_manifest(manifest_path)
    assert manifest is not None
    assert (
        manifest.source_retrieval_manifest == external_retrieval_root / "manifest.json"
    )


def test_write_and_read_fragment_round_trip(tmp_path: Path) -> None:
    fragments_root = tmp_path / "fragments"
    fragments_root.mkdir(parents=True)

    node = GraphNodeRecord(
        node_id="file::src/a.py",
        node_type="file",
        label="a.py",
        path=Path("src/a.py"),
        language="python",
    )
    fragment = _GraphFragment(
        fragment_id="file::src/a.py",
        fingerprint="fp",
        nodes=[node],
        edges=[],
    )
    _write_fragment(fragments_root, fragment)

    loaded = _read_fragment(fragments_root, "file::src/a.py")
    assert loaded.fragment_id == fragment.fragment_id
    assert loaded.fingerprint == fragment.fingerprint
    assert loaded.nodes[0].node_id == node.node_id
