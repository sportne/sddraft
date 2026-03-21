"""Tests for history-docs H8 checkpoint markdown rendering."""

from __future__ import annotations

from pathlib import Path

from engllm.cli.main import main
from engllm.tools.history_docs.algorithm_capsules import algorithm_capsule_filename
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryAlgorithmCapsuleIndexEntry,
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryEvidenceLink,
    HistoryModuleConcept,
    HistoryRenderManifest,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionState,
    HistorySubsystemConcept,
)
from engllm.tools.history_docs.render import (
    checkpoint_markdown_path as build_checkpoint_markdown_path,
)
from engllm.tools.history_docs.render import (
    render_checkpoint_markdown,
)
from engllm.tools.history_docs.render import (
    render_manifest_path as build_render_manifest_path,
)
from tests.history_docs_helpers import (
    checkpoint_markdown_path,
    commit_file,
    dependencies_artifact_path,
    init_repo,
    render_manifest_path,
    write_project_config,
)


def _subsystem_concept() -> HistorySubsystemConcept:
    return HistorySubsystemConcept(
        concept_id="subsystem::src::core",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        source_root=Path("src"),
        group_path=Path("core"),
        module_ids=["module::src/core/app.py"],
        file_count=1,
        symbol_count=3,
        language_counts={"python": 1},
        representative_files=[Path("src/core/app.py")],
        evidence_links=[
            HistoryEvidenceLink(kind="subsystem", reference="subsystem::src::core")
        ],
    )


def _module_concept(*, imports: list[str] | None = None) -> HistoryModuleConcept:
    return HistoryModuleConcept(
        concept_id="module::src/core/app.py",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        path=Path("src/core/app.py"),
        subsystem_id="subsystem::src::core",
        language="python",
        functions=["load_state", "validate_request"],
        classes=["AppState"],
        imports=imports or ["requests"],
        docstrings=["Request processing must remain deterministic."],
        symbol_names=["load_state", "validate_request", "AppState"],
        evidence_links=[HistoryEvidenceLink(kind="file", reference="src/core/app.py")],
    )


def _dependency_concept() -> HistoryDependencyConcept:
    return HistoryDependencyConcept(
        concept_id="dependency-source::pyproject.toml",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        path=Path("pyproject.toml"),
        ecosystem="python",
        category="dependency_manifest",
        related_subsystem_ids=["subsystem::src::core"],
        documented_dependency_ids=["dependency::python::requests"],
        documented_dependency_count=1,
        evidence_links=[
            HistoryEvidenceLink(kind="build_source", reference="pyproject.toml")
        ],
    )


def _checkpoint_model() -> HistoryCheckpointModel:
    return HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        subsystems=[_subsystem_concept()],
        modules=[_module_concept()],
        dependencies=[_dependency_concept()],
        sections=[
            HistorySectionState(section_id="introduction", title="Introduction"),
            HistorySectionState(
                section_id="architectural_overview",
                title="Architectural Overview",
                concept_ids=["subsystem::src::core"],
            ),
            HistorySectionState(
                section_id="subsystems_modules",
                title="Subsystems and Modules",
                concept_ids=[
                    "subsystem::src::core",
                    "module::src/core/app.py",
                ],
            ),
            HistorySectionState(
                section_id="algorithms_core_logic",
                title="Algorithms and Core Logic",
            ),
            HistorySectionState(
                section_id="dependencies",
                title="Dependencies",
                concept_ids=["dependency-source::pyproject.toml"],
            ),
            HistorySectionState(
                section_id="build_development_infrastructure",
                title="Build and Development Infrastructure",
                concept_ids=["dependency-source::pyproject.toml"],
            ),
        ],
    )


def _dependency_inventory() -> HistoryDependencyInventory:
    return HistoryDependencyInventory(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyEntry(
                dependency_id="dependency::python::requests",
                display_name="requests",
                normalized_name="requests",
                ecosystem="python",
                source_manifest_paths=[Path("pyproject.toml")],
                source_dependency_concept_ids=["dependency-source::pyproject.toml"],
                related_subsystem_ids=["subsystem::src::core"],
                related_module_ids=["module::src/core/app.py"],
                scope_roles=["runtime"],
                section_target="dependencies",
                general_description="Requests is an HTTP client library used for network communication.",
                project_usage_description="This project uses Requests for external service calls in the application layer.",
                summary_status="documented",
            )
        ],
    )


def _section_outline(*, include_algorithms: bool = False) -> HistorySectionOutline:
    sections = [
        HistorySectionPlan(
            section_id="introduction",
            title="Introduction",
            kind="core",
            status="included",
            confidence_score=100,
            evidence_score=10,
            depth="brief",
        ),
        HistorySectionPlan(
            section_id="dependencies",
            title="Dependencies",
            kind="core",
            status="included",
            confidence_score=90,
            evidence_score=9,
            depth="deep",
            concept_ids=["dependency-source::pyproject.toml"],
        ),
        HistorySectionPlan(
            section_id="security_considerations",
            title="Security Considerations",
            kind="optional",
            status="included",
            confidence_score=60,
            evidence_score=6,
            depth="brief",
            concept_ids=["module::src/core/app.py"],
        ),
        HistorySectionPlan(
            section_id="limitations_constraints",
            title="Limitations and Constraints",
            kind="optional",
            status="omitted",
            confidence_score=20,
            evidence_score=2,
            omission_reason="insufficient_evidence",
        ),
    ]
    if include_algorithms:
        sections.insert(
            2,
            HistorySectionPlan(
                section_id="algorithms_core_logic",
                title="Algorithms and Core Logic",
                kind="optional",
                status="included",
                confidence_score=80,
                evidence_score=8,
                depth="standard",
                concept_ids=[
                    "subsystem::src::core",
                    "module::src/core/app.py",
                ],
                algorithm_capsule_ids=["algorithm-capsule::file::src/core/app.py"],
            ),
        )
    return HistorySectionOutline(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        sections=sections,
    )


def _capsules() -> tuple[HistoryAlgorithmCapsuleIndex, list[HistoryAlgorithmCapsule]]:
    capsule = HistoryAlgorithmCapsule(
        capsule_id="algorithm-capsule::file::src/core/app.py",
        title="Algorithm Module: App",
        status="introduced",
        scope_kind="file",
        scope_path=Path("src/core/app.py"),
        related_subsystem_ids=["subsystem::src::core"],
        related_module_ids=["module::src/core/app.py"],
        source_candidate_ids=["alg-1"],
        commit_ids=["abc1234deadbeef"],
        changed_symbol_names=["load_state", "validate_request"],
        variant_names=["strict", "safe"],
        signal_kinds=["introduced_module", "variant_family"],
        phases=[],
        shared_abstractions=[],
        data_structures=[],
        assumptions=[],
        evidence_links=[HistoryEvidenceLink(kind="file", reference="src/core/app.py")],
    )
    index = HistoryAlgorithmCapsuleIndex(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        capsules=[
            HistoryAlgorithmCapsuleIndexEntry(
                capsule_id=capsule.capsule_id,
                title=capsule.title,
                status=capsule.status,
                scope_kind=capsule.scope_kind,
                scope_path=capsule.scope_path,
                artifact_path=Path("algorithm_capsules")
                / algorithm_capsule_filename(capsule.capsule_id),
            )
        ],
    )
    return index, [capsule]


def test_render_paths_are_deterministic(tmp_path: Path) -> None:
    tool_root = (
        tmp_path / "artifacts" / "workspaces" / "demo" / "tools" / "history_docs"
    )

    assert build_checkpoint_markdown_path(tool_root, "2024-01-01-abc1234") == (
        tool_root / "checkpoints" / "2024-01-01-abc1234" / "checkpoint.md"
    )
    assert build_render_manifest_path(tool_root, "2024-01-01-abc1234") == (
        tool_root / "checkpoints" / "2024-01-01-abc1234" / "render_manifest.json"
    )


def test_render_checkpoint_markdown_filters_to_included_sections_and_preserves_order() -> (
    None
):
    markdown, manifest = render_checkpoint_markdown(
        workspace_id="demo",
        checkpoint_model=_checkpoint_model(),
        section_outline=_section_outline(),
        dependency_inventory=_dependency_inventory(),
        capsule_index=HistoryAlgorithmCapsuleIndex(
            checkpoint_id="2024-01-01-abc1234",
            target_commit="abc1234deadbeef",
            capsules=[],
        ),
        capsules=[],
    )

    assert markdown.index("## Introduction") < markdown.index("## Dependencies")
    assert markdown.index("## Dependencies") < markdown.index(
        "## Security Considerations"
    )
    assert "## Limitations and Constraints" not in markdown
    assert (
        "### requests\n\n"
        "Requests is an HTTP client library used for network communication.\n\n"
        "This project uses Requests for external service calls in the application layer.\n"
        in markdown
    )
    assert [section.section_id for section in manifest.sections] == [
        "introduction",
        "dependencies",
        "security_considerations",
    ]
    dependencies_section = manifest.sections[1]
    assert dependencies_section.dependency_ids == ["dependency::python::requests"]
    assert dependencies_section.source_artifact_paths == [
        Path("checkpoint_model.json"),
        Path("section_outline.json"),
        Path("dependencies.json"),
    ]
    assert manifest.sections[2].source_artifact_paths == [
        Path("checkpoint_model.json"),
        Path("section_outline.json"),
    ]


def test_render_checkpoint_markdown_renders_algorithm_section_and_manifest_paths() -> (
    None
):
    capsule_index, capsules = _capsules()

    markdown, manifest = render_checkpoint_markdown(
        workspace_id="demo",
        checkpoint_model=_checkpoint_model(),
        section_outline=_section_outline(include_algorithms=True),
        dependency_inventory=_dependency_inventory(),
        capsule_index=capsule_index,
        capsules=capsules,
    )

    assert "## Algorithms and Core Logic" in markdown
    assert "### Algorithm Module: App" in markdown
    assert "- Scope: file `src/core/app.py`." in markdown
    assert "- Variant names: `strict`, `safe`." in markdown
    algorithms_section = next(
        section
        for section in manifest.sections
        if section.section_id == "algorithms_core_logic"
    )
    assert algorithms_section.algorithm_capsule_ids == [
        "algorithm-capsule::file::src/core/app.py"
    ]
    assert algorithms_section.source_artifact_paths == [
        Path("checkpoint_model.json"),
        Path("section_outline.json"),
        Path("algorithm_capsules") / "index.json",
        Path("algorithm_capsules")
        / algorithm_capsule_filename("algorithm-capsule::file::src/core/app.py"),
    ]


def test_build_history_docs_checkpoint_h8_writes_markdown_and_render_manifest(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/core/base.py",
        "def base() -> int:\n    return 1\n",
        message="base",
        timestamp="2024-01-01T10:00:00+00:00",
    )

    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=first_commit,
    )

    commit_file(
        repo_root,
        "src/strategies/http_adapter.py",
        '"""HTTP adapter must remain deterministic."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add http adapter",
        timestamp="2024-02-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/strategies/grpc_adapter.py",
        '"""gRPC adapter only supports strict mode."""\n\nclass RequestContext:\n    pass\n\n\ndef build_plan(strict: bool) -> RequestContext:\n    return RequestContext()\n',
        message="add grpc adapter",
        timestamp="2024-02-02T10:00:00+00:00",
    )
    second_commit = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = [\"requests>=2\"]
[project.optional-dependencies]
dev = [\"pytest>=8\"]
""",
        message="add python deps",
        timestamp="2024-02-03T10:00:00+00:00",
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    markdown_path = checkpoint_markdown_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    manifest_path = render_manifest_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )

    assert result.checkpoint_markdown_path == markdown_path
    assert result.render_manifest_path == manifest_path
    assert result.rendered_section_count > 0

    markdown = markdown_path.read_text(encoding="utf-8")
    manifest = HistoryRenderManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    inventory_path = dependencies_artifact_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    inventory_json = inventory_path.read_text(encoding="utf-8")

    assert "# repo Documentation" in markdown
    assert "## Introduction" in markdown
    assert "## Dependencies" in markdown
    assert "## Algorithms and Core Logic" in markdown
    assert "## Strategy Variants and Design Alternatives" in markdown
    assert "## Limitations and Constraints" not in markdown
    assert "TBD" in inventory_json
    assert "TBD" in markdown
    assert [section.section_id for section in manifest.sections][:3] == [
        "introduction",
        "architectural_overview",
        "subsystems_modules",
    ]
    assert any(
        section.section_id == "dependencies"
        and Path("dependencies.json") in section.source_artifact_paths
        for section in manifest.sections
    )
    assert any(
        section.section_id == "algorithms_core_logic"
        and Path("algorithm_capsules") / "index.json" in section.source_artifact_paths
        for section in manifest.sections
    )

    rerun = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    assert markdown == rerun.checkpoint_markdown_path.read_text(encoding="utf-8")
    assert manifest_path.read_text(
        encoding="utf-8"
    ) == rerun.render_manifest_path.read_text(encoding="utf-8")


def test_history_docs_cli_build_prints_h8_artifact_paths(
    tmp_path: Path,
    capsys,
) -> None:
    repo_root = init_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    commit_file(
        repo_root,
        "src/app.py",
        "def run() -> int:\n    return 1\n",
        message="add app",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = [\"requests>=2\"]
""",
        message="add deps",
        timestamp="2024-01-02T10:00:00+00:00",
    )

    rc = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            head,
        ]
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert "Checkpoint markdown:" in output
    assert "Render manifest:" in output
