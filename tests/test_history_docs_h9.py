"""Tests for history-docs H9 validation reports and quality checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import engllm.tools.history_docs.build as build_module
from engllm.cli.main import main
from engllm.domain.errors import ValidationError
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCapsule,
    HistoryAlgorithmCapsuleIndex,
    HistoryAlgorithmCapsuleIndexEntry,
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryDependencyNarrativeShadow,
    HistoryDependencyNarrativeShadowEntry,
    HistoryEvidenceLink,
    HistoryModuleConcept,
    HistoryRenderedSection,
    HistoryRenderManifest,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionState,
    HistorySubsystemConcept,
    HistoryValidationReport,
)
from engllm.tools.history_docs.render import render_checkpoint_markdown
from engllm.tools.history_docs.validation import (
    validate_checkpoint_render,
)
from engllm.tools.history_docs.validation import (
    validation_report_path as build_validation_report_path,
)
from tests.history_docs_helpers import (
    commit_file,
    init_repo,
    validation_report_path,
    write_project_config,
)


def _checkpoint_model() -> HistoryCheckpointModel:
    return HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        subsystems=[
            HistorySubsystemConcept(
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
                    HistoryEvidenceLink(
                        kind="subsystem",
                        reference="subsystem::src::core",
                    )
                ],
            )
        ],
        modules=[
            HistoryModuleConcept(
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
                imports=["requests"],
                docstrings=["Requests must remain deterministic."],
                symbol_names=["load_state", "validate_request", "AppState"],
                evidence_links=[
                    HistoryEvidenceLink(kind="file", reference="src/core/app.py")
                ],
            )
        ],
        dependencies=[
            HistoryDependencyConcept(
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
                    HistoryEvidenceLink(
                        kind="build_source",
                        reference="pyproject.toml",
                    )
                ],
            )
        ],
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


def _dependency_inventory(*, tbd: bool = False) -> HistoryDependencyInventory:
    summary = "TBD" if tbd else "Requests is an HTTP client library."
    usage = (
        "TBD"
        if tbd
        else "This project uses Requests for external service calls in the application layer."
    )
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
                general_description=summary,
                project_usage_description=usage,
                summary_status="tbd" if tbd else "documented",
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
            confidence_score=80,
            evidence_score=8,
            depth="standard",
            concept_ids=["dependency-source::pyproject.toml"],
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
            1,
            HistorySectionPlan(
                section_id="algorithms_core_logic",
                title="Algorithms and Core Logic",
                kind="optional",
                status="included",
                confidence_score=70,
                evidence_score=7,
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


def _capsules(
    *, thin: bool = False
) -> tuple[HistoryAlgorithmCapsuleIndex, list[HistoryAlgorithmCapsule]]:
    capsule = HistoryAlgorithmCapsule(
        capsule_id="algorithm-capsule::file::src/core/app.py",
        title="Algorithm Module: App",
        status="observed",
        scope_kind="file",
        scope_path=Path("src/core/app.py"),
        related_subsystem_ids=[] if thin else ["subsystem::src::core"],
        related_module_ids=[] if thin else ["module::src/core/app.py"],
        source_candidate_ids=["alg-1"],
        commit_ids=["abc1234deadbeef"],
        changed_symbol_names=[] if thin else ["load_state", "validate_request"],
        variant_names=[],
        signal_kinds=["multi_symbol"],
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
                / "algorithm_capsule__file__src_core_app.py.json",
            )
        ],
    )
    return index, [capsule]


def _write_source_artifacts(checkpoint_dir: Path, relative_paths: list[Path]) -> None:
    for relative_path in relative_paths:
        artifact_path = checkpoint_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("{}\n", encoding="utf-8")


def _render_and_validate(
    tmp_path: Path,
    *,
    include_algorithms: bool = False,
    thin_capsule: bool = False,
    dependency_tbd: bool = False,
    markdown_mutator: Callable[[str], str] | None = None,
) -> HistoryValidationReport:
    checkpoint_model = _checkpoint_model()
    dependency_inventory = _dependency_inventory(tbd=dependency_tbd)
    section_outline = _section_outline(include_algorithms=include_algorithms)
    capsule_index, capsules = _capsules(thin=thin_capsule)
    markdown, render_manifest = render_checkpoint_markdown(
        workspace_id="repo",
        checkpoint_model=checkpoint_model,
        section_outline=section_outline,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        capsules=capsules if include_algorithms else [],
    )
    if markdown_mutator is not None:
        markdown = markdown_mutator(markdown)

    checkpoint_dir = tmp_path / "checkpoint"
    relative_paths: list[Path] = [
        Path("checkpoint_model.json"),
        Path("section_outline.json"),
    ]
    for section in render_manifest.sections:
        relative_paths.extend(section.source_artifact_paths)
    _write_source_artifacts(checkpoint_dir, relative_paths)

    return validate_checkpoint_render(
        checkpoint_dir=checkpoint_dir,
        checkpoint_model=checkpoint_model,
        section_outline=section_outline,
        dependency_inventory=dependency_inventory,
        capsule_index=capsule_index,
        markdown=markdown,
        render_manifest=render_manifest,
    )


def test_validation_report_path_is_deterministic(tmp_path: Path) -> None:
    tool_root = (
        tmp_path / "artifacts" / "workspaces" / "repo" / "tools" / "history_docs"
    )

    assert build_validation_report_path(tool_root, "2024-01-01-abc1234") == (
        tool_root / "checkpoints" / "2024-01-01-abc1234" / "validation_report.json"
    )


def test_validate_checkpoint_render_flags_included_and_omitted_section_mismatches(
    tmp_path: Path,
) -> None:
    report = _render_and_validate(
        tmp_path,
        markdown_mutator=lambda markdown: markdown
        + "\n## Limitations and Constraints\n\nThis should not render.\n",
    )

    check_ids = {finding.check_id for finding in report.findings}

    assert "omitted_section_rendered" in check_ids
    assert "render_manifest_mismatch" in check_ids
    assert report.error_count >= 1


def test_validate_checkpoint_render_ignores_metadata_for_release_note_phrases(
    tmp_path: Path,
) -> None:
    report = _render_and_validate(
        tmp_path,
        markdown_mutator=lambda markdown: markdown.replace(
            "# repo Documentation",
            "# repo Documentation\n\nSince last version, metadata stays outside the rendered sections.",
            1,
        ),
    )

    assert not any(
        finding.check_id == "release_note_phrase" for finding in report.findings
    )

    warned = _render_and_validate(
        tmp_path,
        markdown_mutator=lambda markdown: markdown.replace(
            "## Dependencies",
            "## Dependencies\n\nSince last release, this section regressed.",
            1,
        ),
    )
    assert any(finding.check_id == "release_note_phrase" for finding in warned.findings)


def test_validate_checkpoint_render_enforces_two_dependency_paragraphs(
    tmp_path: Path,
) -> None:
    report = _render_and_validate(
        tmp_path,
        markdown_mutator=lambda markdown: markdown.replace(
            "This project uses Requests for external service calls in the application layer.",
            "- This bullet breaks the paragraph contract.",
            1,
        ),
    )

    assert any(
        finding.check_id == "dependency_subsection_shape_invalid"
        for finding in report.findings
    )
    assert report.error_count >= 1


def test_validate_checkpoint_render_warns_on_thin_algorithm_capsules(
    tmp_path: Path,
) -> None:
    report = _render_and_validate(
        tmp_path,
        include_algorithms=True,
        thin_capsule=True,
    )

    assert any(
        finding.check_id == "algorithm_capsule_thin" for finding in report.findings
    )
    assert report.error_count == 0


def test_build_history_docs_checkpoint_h9_writes_validation_report_and_keeps_tbd_as_warning(
    tmp_path: Path,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/core/app.py",
        "def run() -> int:\n    return 1\n",
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

    second_commit = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
""",
        message="add dependencies",
        timestamp="2024-01-02T10:00:00+00:00",
    )

    result = build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=second_commit,
    )
    report_path = validation_report_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    report = HistoryValidationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )

    assert result.validation_report_path == report_path
    assert report.error_count == 0
    assert result.validation_error_count == 0
    assert result.validation_warning_count == report.warning_count
    assert any(
        finding.check_id == "dependency_summary_tbd" for finding in report.findings
    )


def test_validate_checkpoint_render_flags_internal_id_leaks_and_contradictory_tbd() -> (
    None
):
    outline = _section_outline()
    inventory = _dependency_inventory()
    manifest = HistoryRenderManifest(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        markdown_path=Path("checkpoint.md"),
        sections=[
            HistoryRenderedSection(
                section_id="dependencies",
                title="Dependencies",
                order=1,
                kind="core",
                dependency_ids=["dependency::python::requests"],
                source_artifact_paths=[Path("dependencies.json")],
                subheading_count=1,
            )
        ],
    )
    markdown = (
        "# Example Documentation\n\n"
        "## Dependencies\n\n"
        "### requests\n\n"
        "TBD - requests is an HTTP client library.\n\n"
        "Project-specific usage is not strongly evidenced.\n\n"
        "- subsystem::src::core\n"
    )
    report = validate_checkpoint_render(
        checkpoint_dir=Path("."),
        checkpoint_model=_checkpoint_model(),
        section_outline=outline,
        dependency_inventory=inventory,
        capsule_index=HistoryAlgorithmCapsuleIndex(
            checkpoint_id="2024-01-01-abc1234",
            target_commit="abc1234deadbeef",
            capsules=[],
        ),
        markdown=markdown,
        render_manifest=manifest,
    )
    assert any(f.check_id == "contradictory_tbd_phrase" for f in report.findings)
    assert any(f.check_id == "raw_internal_id_leak" for f in report.findings)


def test_validate_checkpoint_render_allows_grouped_tooling_shadow_entries() -> None:
    checkpoint_model = _checkpoint_model()
    inventory = HistoryDependencyInventory(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyEntry(
                dependency_id="dependency::python::pytest",
                display_name="pytest",
                normalized_name="pytest",
                ecosystem="python",
                source_manifest_paths=[Path("pyproject.toml")],
                source_dependency_concept_ids=["dependency-source::pyproject.toml"],
                scope_roles=["test"],
                section_target="build_development_infrastructure",
                general_description="pytest is a Python test runner.",
                project_usage_description="Project-specific usage is not strongly evidenced.",
            )
        ],
    )
    outline = HistorySectionOutline(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        sections=[
            HistorySectionPlan(
                section_id="build_development_infrastructure",
                title="Build and Development Infrastructure",
                kind="core",
                status="included",
                confidence_score=80,
                evidence_score=8,
                depth="standard",
            )
        ],
    )
    markdown = (
        "# Example Documentation\n\n"
        "## Build and Development Infrastructure\n\n"
        "### Python Test Tooling\n\n"
        "This grouped tooling summary covers low-evidence support packages.\n\n"
        "- `pytest`: pytest is a Python test runner. Project-specific usage is not strongly evidenced.\n"
    )
    manifest = HistoryRenderManifest(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        markdown_path=Path("checkpoint.md"),
        sections=[
            HistoryRenderedSection(
                section_id="build_development_infrastructure",
                title="Build and Development Infrastructure",
                order=1,
                kind="core",
                dependency_ids=["dependency::python::pytest"],
                source_artifact_paths=[Path("dependency_narratives_shadow.json")],
                subheading_count=1,
            )
        ],
    )
    shadow = HistoryDependencyNarrativeShadow(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyNarrativeShadowEntry(
                dependency_id="dependency::python::pytest",
                display_name="pytest",
                ecosystem="python",
                section_target="build_development_infrastructure",
                scope_roles=["test"],
                render_style="grouped_tooling",
                group_title="Python Test Tooling",
                general_description="pytest is a Python test runner.",
                general_description_basis="package_general_knowledge",
                project_usage_description="Project-specific usage is not strongly evidenced.",
                project_usage_basis="tbd",
            )
        ],
    )
    report = validate_checkpoint_render(
        checkpoint_dir=Path("."),
        checkpoint_model=checkpoint_model,
        section_outline=outline,
        dependency_inventory=inventory,
        capsule_index=HistoryAlgorithmCapsuleIndex(
            checkpoint_id="2024-01-01-abc1234",
            target_commit="abc1234deadbeef",
            capsules=[],
        ),
        markdown=markdown,
        render_manifest=manifest,
        dependency_narratives_shadow=shadow,
    )
    assert not any(
        finding.check_id == "dependency_subsection_shape_invalid"
        for finding in report.findings
    )


def test_render_checkpoint_markdown_groups_tooling_without_package_subsections() -> (
    None
):
    checkpoint_model = _checkpoint_model()
    inventory = HistoryDependencyInventory(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyEntry(
                dependency_id="dependency::python::pytest",
                display_name="pytest",
                normalized_name="pytest",
                ecosystem="python",
                source_manifest_paths=[Path("pyproject.toml")],
                source_dependency_concept_ids=["dependency-source::pyproject.toml"],
                scope_roles=["test"],
                section_target="build_development_infrastructure",
                general_description="TBD: pytest is a Python test runner.",
                project_usage_description="TBD - exact project usage is unclear.",
            )
        ],
    )
    outline = HistorySectionOutline(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        sections=[
            HistorySectionPlan(
                section_id="build_development_infrastructure",
                title="Build and Development Infrastructure",
                kind="core",
                status="included",
                confidence_score=80,
                evidence_score=8,
                depth="standard",
            )
        ],
    )
    shadow = HistoryDependencyNarrativeShadow(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyNarrativeShadowEntry(
                dependency_id="dependency::python::pytest",
                display_name="pytest",
                ecosystem="python",
                section_target="build_development_infrastructure",
                scope_roles=["test"],
                render_style="grouped_tooling",
                group_title="Python Test Tooling",
                general_description="pytest is a Python test runner.",
                general_description_basis="package_general_knowledge",
                project_usage_description="Project-specific usage is not strongly evidenced by the current manifests and import signals.",
                project_usage_basis="tbd",
            )
        ],
    )

    markdown, _render_manifest = render_checkpoint_markdown(
        workspace_id="repo",
        checkpoint_model=checkpoint_model,
        section_outline=outline,
        dependency_inventory=inventory,
        capsule_index=HistoryAlgorithmCapsuleIndex(
            checkpoint_id="2024-01-01-abc1234",
            target_commit="abc1234deadbeef",
            capsules=[],
        ),
        capsules=[],
        dependency_narratives_shadow=shadow,
    )

    assert "### Python Test Tooling" in markdown
    assert "`pytest`" in markdown
    assert "### pytest" not in markdown
    assert "TBD:" not in markdown
    assert "TBD -" not in markdown


def test_render_checkpoint_markdown_marks_package_knowledge_provenance_for_dependency_blurbs() -> (
    None
):
    checkpoint_model = _checkpoint_model()
    inventory = HistoryDependencyInventory(
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
                scope_roles=["runtime"],
                section_target="dependencies",
                general_description="TBD",
                project_usage_description="TBD",
            )
        ],
    )
    outline = HistorySectionOutline(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        sections=[
            HistorySectionPlan(
                section_id="dependencies",
                title="Dependencies",
                kind="core",
                status="included",
                confidence_score=80,
                evidence_score=8,
                depth="standard",
            )
        ],
    )
    shadow = HistoryDependencyNarrativeShadow(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="abc1234deadbeef",
        entries=[
            HistoryDependencyNarrativeShadowEntry(
                dependency_id="dependency::python::requests",
                display_name="requests",
                ecosystem="python",
                section_target="dependencies",
                scope_roles=["runtime"],
                render_style="standard",
                general_description="requests is a Python HTTP client library commonly used for outbound web requests.",
                general_description_basis="package_general_knowledge",
                project_usage_description="Project-specific usage is not strongly evidenced by the current manifests and import signals.",
                project_usage_basis="tbd",
            )
        ],
    )

    markdown, _render_manifest = render_checkpoint_markdown(
        workspace_id="repo",
        checkpoint_model=checkpoint_model,
        section_outline=outline,
        dependency_inventory=inventory,
        capsule_index=HistoryAlgorithmCapsuleIndex(
            checkpoint_id="2024-01-01-abc1234",
            target_commit="abc1234deadbeef",
            capsules=[],
        ),
        capsules=[],
        dependency_narratives_shadow=shadow,
    )

    assert "### requests" in markdown
    assert (
        "This general description is based on package-level knowledge rather than repository-specific evidence."
        in markdown
    )
    assert (
        "Project-specific usage is not strongly evidenced by the current manifests and import signals."
        in markdown
    )


def test_build_history_docs_checkpoint_h9_raises_validation_error_with_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_project_config,
) -> None:
    repo_root = init_repo(tmp_path)
    first_commit = commit_file(
        repo_root,
        "src/app.py",
        "def run() -> int:\n    return 1\n",
        message="base",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    second_commit = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
""",
        message="deps",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    sample_project_config.workspace.output_root = tmp_path / "artifacts"
    sample_project_config.sources.roots = [repo_root / "src"]

    build_history_docs_checkpoint(
        project_config=sample_project_config,
        repo_root=repo_root,
        checkpoint_commit=first_commit,
    )

    original_render = build_module.render_checkpoint_markdown

    def _bad_render(**kwargs):
        markdown, manifest = original_render(**kwargs)
        return (
            markdown.replace("## Dependencies", "## Dependency Inventory", 1),
            manifest,
        )

    monkeypatch.setattr(build_module, "render_checkpoint_markdown", _bad_render)

    with pytest.raises(ValidationError) as exc_info:
        build_history_docs_checkpoint(
            project_config=sample_project_config,
            repo_root=repo_root,
            checkpoint_commit=second_commit,
        )

    checkpoint_id = f"2024-01-02-{second_commit[:7]}"
    report_path = validation_report_path(
        sample_project_config.workspace.output_root,
        repo_root.name,
        checkpoint_id,
    )
    report = HistoryValidationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )

    assert "validation_report.json" in str(exc_info.value)
    assert report.error_count >= 1
    assert any(
        finding.check_id == "included_section_missing" for finding in report.findings
    )


def test_history_docs_cli_build_prints_validation_report_path(
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
        message="base",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
""",
        message="deps",
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
    assert "Validation report:" in output
