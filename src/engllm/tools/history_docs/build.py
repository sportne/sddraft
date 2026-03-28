"""History-docs workflow: checkpoint manifests, snapshot analysis, and deltas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from engllm.core.analysis.history import (
    HistoryBuildSource,
    HistoryCheckpoint,
    HistoryCheckpointPlan,
    HistoryCommitSummary,
    HistoryInterval,
    HistorySnapshotManifest,
    HistorySourceRootMapping,
    PreviousCheckpointSource,
    checkpoint_id_for,
    default_checkpoint_plan_path,
    default_intervals_path,
    default_snapshot_manifest_path,
    load_checkpoint_plan,
    load_intervals,
    normalize_checkpoint_plan,
    save_checkpoint_plan,
    save_intervals,
    save_snapshot_manifest,
)
from engllm.core.render.json_artifacts import write_json_model
from engllm.core.repo.history import (
    export_commit_snapshot,
    get_commit_metadata,
    is_strict_ancestor,
    iter_interval_commits,
    read_file_at_commit,
    resolve_commit,
)
from engllm.core.repo.scanner import scan_paths
from engllm.core.workspaces import build_workspace_context, tool_artifact_root
from engllm.domain.errors import (
    AnalysisError,
    GitError,
    RepositoryError,
    ValidationError,
)
from engllm.domain.models import ProjectConfig
from engllm.llm.base import LLMClient
from engllm.llm.factory import create_llm_client
from engllm.tools.history_docs.algorithm_capsule_enrichment import (
    algorithm_capsule_enrichment_dir,
    algorithm_capsule_enrichment_filename,
    algorithm_capsule_enrichment_index_path,
    build_algorithm_capsule_enrichments,
)
from engllm.tools.history_docs.algorithm_capsules import (
    algorithm_capsule_index_path,
    build_algorithm_capsules,
    link_algorithm_capsules_to_checkpoint_model,
    link_algorithm_capsules_to_section_outline,
)
from engllm.tools.history_docs.checkpoint_model import (
    build_checkpoint_model,
    checkpoint_model_path,
    load_checkpoint_model,
)
from engllm.tools.history_docs.checkpoint_model_enrichment import (
    apply_checkpoint_model_enrichment,
    build_checkpoint_model_enrichment,
    checkpoint_model_enrichment_path,
)
from engllm.tools.history_docs.delta import (
    build_interval_delta_model,
    interval_delta_model_path,
)
from engllm.tools.history_docs.dependencies import (
    build_dependency_inventory,
    dependency_inventory_path,
    link_dependency_inventory_to_checkpoint_model,
)
from engllm.tools.history_docs.dependency_landscape import (
    build_dependency_landscape,
    dependency_landscape_path,
)
from engllm.tools.history_docs.interface_inventory import (
    build_interface_inventory,
    interface_inventory_path,
)
from engllm.tools.history_docs.interval_interpretation import (
    build_interval_interpretation,
    interval_interpretation_path,
)
from engllm.tools.history_docs.models import (
    HistoryBuildResult,
    HistoryCheckpointModel,
    HistoryDependencyInventory,
    HistorySectionOutline,
    HistorySnapshotStructuralModel,
)
from engllm.tools.history_docs.render import (
    checkpoint_markdown_path,
    render_checkpoint_markdown,
    render_manifest_path,
    write_checkpoint_markdown,
)
from engllm.tools.history_docs.section_outline import (
    build_section_outline,
    section_outline_path,
)
from engllm.tools.history_docs.section_planning_llm import (
    build_llm_section_outline,
    build_section_planning_scaffold,
    link_section_planning_outline_to_scaffold,
    section_outline_llm_path,
)
from engllm.tools.history_docs.semantic_context import (
    build_semantic_context_map,
    semantic_context_map_path,
)
from engllm.tools.history_docs.semantic_planner import (
    build_semantic_checkpoint_plan,
    semantic_checkpoint_plan_path,
)
from engllm.tools.history_docs.semantic_structure import (
    build_semantic_structure_map,
    build_subsystem_grouping_view,
    load_semantic_structure_map,
    semantic_structure_map_path,
)
from engllm.tools.history_docs.structure import (
    build_subsystem_candidates,
    normalize_relative_path,
)
from engllm.tools.history_docs.validation import (
    validate_checkpoint_render,
    validation_report_path,
)

ProgressCallback = Callable[[str], None]
BuildSourceCategory = Literal[
    "dependency_manifest",
    "dependency_lockfile",
    "build_config",
]

_MAX_REPRESENTATIVE_FILES = 12
_EXACT_BUILD_SOURCE_RULES: tuple[tuple[str, str, BuildSourceCategory], ...] = (
    ("pyproject.toml", "python", "dependency_manifest"),
    ("requirements.txt", "python", "dependency_manifest"),
    ("setup.py", "python", "build_config"),
    ("setup.cfg", "python", "build_config"),
    ("Pipfile", "python", "dependency_manifest"),
    ("Pipfile.lock", "python", "dependency_lockfile"),
    ("poetry.lock", "python", "dependency_lockfile"),
    ("package.json", "javascript", "dependency_manifest"),
    ("package-lock.json", "javascript", "dependency_lockfile"),
    ("yarn.lock", "javascript", "dependency_lockfile"),
    ("pnpm-lock.yaml", "javascript", "dependency_lockfile"),
    ("Cargo.toml", "rust", "dependency_manifest"),
    ("Cargo.lock", "rust", "dependency_lockfile"),
    ("go.mod", "go", "dependency_manifest"),
    ("go.sum", "go", "dependency_lockfile"),
    ("pom.xml", "jvm", "dependency_manifest"),
    ("build.gradle", "jvm", "build_config"),
    ("build.gradle.kts", "jvm", "build_config"),
    ("settings.gradle", "jvm", "build_config"),
    ("settings.gradle.kts", "jvm", "build_config"),
    ("gradle.properties", "jvm", "build_config"),
    ("packages.lock.json", "dotnet", "dependency_lockfile"),
    ("Directory.Build.props", "dotnet", "build_config"),
    ("global.json", "dotnet", "build_config"),
    ("CMakeLists.txt", "cpp", "build_config"),
    ("vcpkg.json", "cpp", "dependency_manifest"),
    ("conanfile.py", "cpp", "dependency_manifest"),
    ("conanfile.txt", "cpp", "dependency_manifest"),
    ("meson.build", "cpp", "build_config"),
    ("BUILD", "generic", "build_config"),
    ("WORKSPACE", "generic", "build_config"),
    ("Makefile", "generic", "build_config"),
)
_GLOB_BUILD_SOURCE_RULES: tuple[tuple[str, str, BuildSourceCategory], ...] = (
    ("requirements-*.txt", "python", "dependency_manifest"),
    ("*.csproj", "dotnet", "dependency_manifest"),
)


def _progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _resolve_workspace_id(repo_root: Path, workspace_id: str | None) -> str:
    return workspace_id or repo_root.resolve().name


def _select_previous_from_artifacts(
    checkpoints: list[HistoryCheckpoint],
    *,
    repo_root: Path,
    target_commit: str,
) -> str | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.target_commit == target_commit:
            continue
        if is_strict_ancestor(repo_root, checkpoint.target_commit, target_commit):
            return checkpoint.target_commit
    return None


def _validate_previous_boundary(
    repo_root: Path,
    *,
    target_commit: str,
    previous_commit: str,
) -> str:
    resolved_previous = resolve_commit(repo_root, previous_commit)
    if resolved_previous == target_commit:
        raise GitError(
            "Previous checkpoint commit must be a strict ancestor of the target commit"
        )
    if not is_strict_ancestor(repo_root, resolved_previous, target_commit):
        raise GitError(
            f"Previous checkpoint commit {resolved_previous} is not an ancestor of {target_commit}"
        )
    return resolved_previous


def _resolve_source_root_mappings(
    *,
    configured_roots: list[Path],
    repo_root: Path,
    snapshot_root: Path,
) -> list[HistorySourceRootMapping]:
    mappings: list[HistorySourceRootMapping] = []
    for configured_root in configured_roots:
        resolved_root = (
            configured_root
            if configured_root.is_absolute()
            else (repo_root / configured_root)
        ).resolve()
        try:
            repo_relative_root = normalize_relative_path(
                resolved_root.relative_to(repo_root.resolve())
            )
        except ValueError as exc:
            raise RepositoryError(
                f"Configured source root must live inside repo root for history-docs: {configured_root}"
            ) from exc

        snapshot_path = (
            snapshot_root
            if repo_relative_root == Path(".")
            else snapshot_root / repo_relative_root
        )
        if snapshot_path.exists():
            mappings.append(
                HistorySourceRootMapping(
                    requested_root=repo_relative_root,
                    snapshot_relative_root=repo_relative_root,
                    status="analyzed",
                )
            )
        else:
            mappings.append(
                HistorySourceRootMapping(
                    requested_root=repo_relative_root,
                    snapshot_relative_root=repo_relative_root,
                    status="missing_at_checkpoint",
                    reason="root_missing_at_checkpoint",
                )
            )
    return mappings


def _manifest_search_directories(analyzed_roots: list[Path]) -> list[Path]:
    directories: set[Path] = set()
    for root in analyzed_roots:
        cursor = normalize_relative_path(root)
        while True:
            directories.add(cursor)
            if cursor == Path("."):
                break
            parent = cursor.parent if cursor.parent != Path("") else Path(".")
            cursor = normalize_relative_path(parent)
    return sorted(directories, key=lambda item: item.as_posix())


def _discover_build_sources(
    *,
    snapshot_root: Path,
    manifest_search_directories: list[Path],
) -> list[HistoryBuildSource]:
    discovered: dict[Path, HistoryBuildSource] = {}
    for directory in manifest_search_directories:
        search_root = (
            snapshot_root if directory == Path(".") else snapshot_root / directory
        )
        if not search_root.exists() or not search_root.is_dir():
            continue

        for name, ecosystem, category in _EXACT_BUILD_SOURCE_RULES:
            candidate = search_root / name
            if candidate.exists() and candidate.is_file():
                relative_path = (
                    Path(name) if directory == Path(".") else directory / name
                )
                discovered[relative_path] = HistoryBuildSource(
                    path=relative_path,
                    ecosystem=ecosystem,
                    category=category,
                )

        for pattern, ecosystem, category in _GLOB_BUILD_SOURCE_RULES:
            for candidate in sorted(search_root.glob(pattern)):
                if not candidate.is_file():
                    continue
                relative_path = (
                    Path(candidate.name)
                    if directory == Path(".")
                    else directory / candidate.name
                )
                discovered[relative_path] = HistoryBuildSource(
                    path=relative_path,
                    ecosystem=ecosystem,
                    category=category,
                )

    return [
        discovered[key] for key in sorted(discovered, key=lambda item: item.as_posix())
    ]


def _snapshot_structural_model_path(tool_root: Path, checkpoint_id: str) -> Path:
    return tool_root / "checkpoints" / checkpoint_id / "snapshot_structural_model.json"


def _load_snapshot_structural_model(
    path: Path,
) -> HistorySnapshotStructuralModel | None:
    if not path.exists():
        return None
    return HistorySnapshotStructuralModel.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _load_previous_checkpoint_model(
    tool_root: Path,
    checkpoint_id: str,
) -> HistoryCheckpointModel | None:
    return load_checkpoint_model(checkpoint_model_path(tool_root, checkpoint_id))


def _count_section_status(
    section_outline: HistorySectionOutline,
    status: str,
) -> int:
    return sum(section.status == status for section in section_outline.sections)


def _count_dependency_summary_failures(
    dependency_inventory: HistoryDependencyInventory,
) -> int:
    return sum(
        entry.summary_status == "llm_failed" for entry in dependency_inventory.entries
    )


def build_history_docs_checkpoint(
    *,
    project_config: ProjectConfig,
    repo_root: Path,
    checkpoint_commit: str,
    previous_checkpoint_commit: str | None = None,
    workspace_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    subsystem_grouping_mode: Literal["path", "semantic"] = "path",
    experimental_section_mode: Literal["default", "semantic_context"] = "default",
    checkpoint_model_enrichment_mode: Literal["baseline", "enriched"] = "baseline",
    section_planning_mode: Literal["baseline", "llm"] = "baseline",
    algorithm_capsule_mode: Literal["baseline", "enriched"] = "baseline",
    interface_render_mode: Literal["baseline", "inventory"] = "baseline",
    dependency_render_mode: Literal["baseline", "landscape"] = "baseline",
    llm_client_override: LLMClient | None = None,
) -> HistoryBuildResult:
    """Persist checkpoint manifests plus H2-H9 history-docs artifacts."""

    resolved_repo_root = repo_root.resolve()
    target_metadata = get_commit_metadata(resolved_repo_root, checkpoint_commit)
    resolved_workspace_id = _resolve_workspace_id(resolved_repo_root, workspace_id)
    workspace = build_workspace_context(
        output_root=project_config.workspace.output_root,
        workspace_id=resolved_workspace_id,
        kind="repo",
        repo_root=resolved_repo_root,
    )
    history_tool_root = tool_artifact_root(workspace, "history_docs")
    checkpoint_plan_path = default_checkpoint_plan_path(workspace.shared_root)
    intervals_path = default_intervals_path(workspace.shared_root)

    existing_plan = load_checkpoint_plan(checkpoint_plan_path)
    if (
        existing_plan is not None
        and existing_plan.repo_root.resolve() != resolved_repo_root
    ):
        raise AnalysisError(
            "Existing history checkpoint plan repo root does not match this run"
        )

    if previous_checkpoint_commit is not None:
        resolved_previous = _validate_previous_boundary(
            resolved_repo_root,
            target_commit=target_metadata.sha,
            previous_commit=previous_checkpoint_commit,
        )
        previous_source: PreviousCheckpointSource = "explicit_override"
    else:
        resolved_previous = None
        if existing_plan is not None:
            resolved_previous = _select_previous_from_artifacts(
                existing_plan.checkpoints,
                repo_root=resolved_repo_root,
                target_commit=target_metadata.sha,
            )
        previous_source = "artifact" if resolved_previous is not None else "initial"

    interval_commits = iter_interval_commits(
        resolved_repo_root,
        target_commit=target_metadata.sha,
        previous_commit=resolved_previous,
    )
    checkpoint_id = checkpoint_id_for(
        target_metadata.timestamp, target_metadata.short_sha
    )
    checkpoint = HistoryCheckpoint(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        target_commit_short=target_metadata.short_sha,
        target_commit_timestamp=target_metadata.timestamp,
        target_commit_subject=target_metadata.subject,
        tree_sha=target_metadata.tree_sha,
        previous_checkpoint_commit=resolved_previous,
        previous_checkpoint_source=previous_source,
    )
    interval = HistoryInterval(
        checkpoint_id=checkpoint_id,
        start_commit=resolved_previous,
        end_commit=target_metadata.sha,
        commit_count=len(interval_commits),
        commits=[
            HistoryCommitSummary(
                sha=commit.sha,
                short_sha=commit.short_sha,
                timestamp=commit.timestamp,
                subject=commit.subject,
            )
            for commit in interval_commits
        ],
    )

    prior_checkpoints = (
        existing_plan.checkpoints[:] if existing_plan is not None else []
    )
    checkpoint_by_commit = {
        item.target_commit: item
        for item in prior_checkpoints
        if item.target_commit != checkpoint.target_commit
    }
    checkpoint_by_commit[checkpoint.target_commit] = checkpoint
    normalized_plan = normalize_checkpoint_plan(
        HistoryCheckpointPlan(
            workspace_id=resolved_workspace_id,
            repo_root=resolved_repo_root,
            checkpoints=list(checkpoint_by_commit.values()),
        )
    )

    prior_intervals = load_intervals(intervals_path)
    interval_by_checkpoint = {
        item.checkpoint_id: item
        for item in prior_intervals
        if item.checkpoint_id != interval.checkpoint_id
    }
    interval_by_checkpoint[interval.checkpoint_id] = interval
    ordered_intervals = [
        interval_by_checkpoint[item.checkpoint_id]
        for item in normalized_plan.checkpoints
        if item.checkpoint_id in interval_by_checkpoint
    ]

    save_checkpoint_plan(normalized_plan, checkpoint_plan_path)
    save_intervals(ordered_intervals, intervals_path)

    llm_client = llm_client_override or create_llm_client(project_config.llm)

    _progress(
        progress_callback,
        "history-docs: planning semantic checkpoint candidates",
    )
    semantic_plan = build_semantic_checkpoint_plan(
        repo_root=resolved_repo_root,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        configured_source_roots=project_config.sources.roots,
        checkpoints=normalized_plan.checkpoints,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    semantic_plan_artifact_path = semantic_checkpoint_plan_path(
        history_tool_root,
        checkpoint_id,
    )
    write_json_model(semantic_plan_artifact_path, semantic_plan)

    _progress(
        progress_callback,
        "history-docs: exporting checkpoint snapshot and analyzing structure",
    )
    with TemporaryDirectory(prefix="engllm-history-snapshot-") as temp_dir:
        temp_root = Path(temp_dir)
        snapshot_root = temp_root / "snapshot"
        export_commit_snapshot(
            resolved_repo_root,
            target_commit=target_metadata.sha,
            destination_root=snapshot_root,
        )

        source_root_mappings = _resolve_source_root_mappings(
            configured_roots=project_config.sources.roots,
            repo_root=resolved_repo_root,
            snapshot_root=snapshot_root,
        )
        requested_source_roots = [
            mapping.requested_root for mapping in source_root_mappings
        ]
        analyzed_source_roots = [
            mapping.requested_root
            for mapping in source_root_mappings
            if mapping.status == "analyzed"
        ]
        skipped_source_roots = [
            mapping.requested_root
            for mapping in source_root_mappings
            if mapping.status != "analyzed"
        ]
        snapshot_scan_roots = [
            snapshot_root if root == Path(".") else snapshot_root / root
            for root in analyzed_source_roots
        ]
        scan_result = scan_paths(
            roots=snapshot_scan_roots,
            include=project_config.sources.include,
            exclude=project_config.sources.exclude,
            repo_root=snapshot_root,
            max_files=project_config.generation.max_files,
            chunk_lines=project_config.generation.code_chunk_lines,
            include_code_chunks=False,
        )
        manifest_search_directories = _manifest_search_directories(
            analyzed_source_roots
        )
        build_sources = _discover_build_sources(
            snapshot_root=snapshot_root,
            manifest_search_directories=manifest_search_directories,
        )
        symbol_counts_by_path: dict[Path, int] = Counter(
            symbol.source_path for symbol in scan_result.symbol_summaries
        )
        language_by_path = {
            summary.path: summary.language for summary in scan_result.code_summaries
        }
        subsystem_candidates = build_subsystem_candidates(
            files=scan_result.files,
            symbol_counts_by_path=symbol_counts_by_path,
            language_by_path=language_by_path,
            analyzed_roots=analyzed_source_roots,
        )
        snapshot_manifest = HistorySnapshotManifest(
            checkpoint_id=checkpoint_id,
            target_commit=target_metadata.sha,
            tree_sha=target_metadata.tree_sha,
            source_root_mappings=source_root_mappings,
            requested_source_roots=requested_source_roots,
            analyzed_source_roots=analyzed_source_roots,
            skipped_source_roots=skipped_source_roots,
            manifest_search_directories=manifest_search_directories,
            file_count=len(scan_result.files),
            symbol_count=len(scan_result.symbol_summaries),
            subsystem_count=len(subsystem_candidates),
            build_source_count=len(build_sources),
        )
        structural_model = HistorySnapshotStructuralModel(
            checkpoint_id=checkpoint_id,
            target_commit=target_metadata.sha,
            requested_source_roots=requested_source_roots,
            analyzed_source_roots=analyzed_source_roots,
            skipped_source_roots=skipped_source_roots,
            files=scan_result.files,
            code_summaries=scan_result.code_summaries,
            symbol_summaries=scan_result.symbol_summaries,
            subsystem_candidates=subsystem_candidates,
            build_sources=build_sources,
        )

    snapshot_manifest_path = default_snapshot_manifest_path(
        workspace.shared_root, checkpoint_id
    )
    snapshot_structural_model_path = _snapshot_structural_model_path(
        history_tool_root, checkpoint_id
    )
    save_snapshot_manifest(snapshot_manifest, snapshot_manifest_path)
    write_json_model(snapshot_structural_model_path, structural_model)

    _progress(
        progress_callback,
        "history-docs: building semantic subsystem and capability map",
    )
    semantic_structure_map = build_semantic_structure_map(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        snapshot=structural_model,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    semantic_structure_artifact_path = semantic_structure_map_path(
        history_tool_root,
        checkpoint_id,
    )
    write_json_model(semantic_structure_artifact_path, semantic_structure_map)

    _progress(
        progress_callback,
        "history-docs: building semantic context and interface map",
    )
    semantic_context_map = build_semantic_context_map(
        workspace_id=resolved_workspace_id,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        snapshot=structural_model,
        semantic_map=semantic_structure_map,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    semantic_context_artifact_path = semantic_context_map_path(
        history_tool_root,
        checkpoint_id,
    )
    write_json_model(semantic_context_artifact_path, semantic_context_map)

    previous_snapshot = None
    previous_checkpoint_model = None
    previous_checkpoint = None
    previous_semantic_structure_map = None
    if resolved_previous is not None:
        previous_checkpoint = next(
            (
                item
                for item in normalized_plan.checkpoints
                if item.target_commit == resolved_previous
            ),
            None,
        )
        if previous_checkpoint is not None:
            previous_snapshot = _load_snapshot_structural_model(
                _snapshot_structural_model_path(
                    history_tool_root,
                    previous_checkpoint.checkpoint_id,
                )
            )
            previous_checkpoint_model = _load_previous_checkpoint_model(
                history_tool_root,
                previous_checkpoint.checkpoint_id,
            )
            previous_semantic_structure_map = load_semantic_structure_map(
                semantic_structure_map_path(
                    history_tool_root,
                    previous_checkpoint.checkpoint_id,
                )
            )

    current_grouping = build_subsystem_grouping_view(
        snapshot=structural_model,
        mode=subsystem_grouping_mode,
        semantic_map=semantic_structure_map,
    )
    previous_grouping = (
        None
        if previous_snapshot is None
        else build_subsystem_grouping_view(
            snapshot=previous_snapshot,
            mode=subsystem_grouping_mode,
            semantic_map=previous_semantic_structure_map,
        )
    )

    _progress(
        progress_callback,
        "history-docs: analyzing interval deltas",
    )
    interval_delta_model = build_interval_delta_model(
        repo_root=resolved_repo_root,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        interval_commits=interval.commits,
        current_snapshot=structural_model,
        previous_snapshot=previous_snapshot,
        current_grouping=current_grouping,
        previous_grouping=previous_grouping,
    )
    interval_delta_path = interval_delta_model_path(history_tool_root, checkpoint_id)
    write_json_model(interval_delta_path, interval_delta_model)

    _progress(
        progress_callback,
        "history-docs: interpreting interval change significance",
    )
    interval_interpretation = build_interval_interpretation(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        snapshot=structural_model,
        delta_model=interval_delta_model,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
        semantic_structure_map=semantic_structure_map,
        semantic_context_map=semantic_context_map,
    )
    interval_interpretation_artifact_path = interval_interpretation_path(
        history_tool_root,
        checkpoint_id,
    )
    write_json_model(
        interval_interpretation_artifact_path,
        interval_interpretation,
    )

    _progress(
        progress_callback,
        "history-docs: building checkpoint documentation model",
    )
    checkpoint_model = build_checkpoint_model(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        current_snapshot=structural_model,
        current_delta=interval_delta_model,
        previous_model=previous_checkpoint_model,
        current_grouping=current_grouping,
    )
    checkpoint_model_artifact_path = checkpoint_model_path(
        history_tool_root, checkpoint_id
    )
    checkpoint_model_enrichment_artifact_path = checkpoint_model_enrichment_path(
        history_tool_root,
        checkpoint_id,
    )

    _progress(
        progress_callback,
        "history-docs: enriching checkpoint model in shadow mode",
    )
    checkpoint_model_enrichment = build_checkpoint_model_enrichment(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        checkpoint_model=checkpoint_model,
        snapshot=structural_model,
        interval_interpretation=interval_interpretation,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
        semantic_structure_map=semantic_structure_map,
        semantic_context_map=semantic_context_map,
    )
    write_json_model(
        checkpoint_model_enrichment_artifact_path,
        checkpoint_model_enrichment,
    )
    if checkpoint_model_enrichment_mode == "enriched":
        checkpoint_model = apply_checkpoint_model_enrichment(
            checkpoint_model,
            checkpoint_model_enrichment,
        )

    _progress(
        progress_callback,
        "history-docs: planning evidence-scored section outline",
    )
    section_outline = build_section_outline(
        checkpoint_model=checkpoint_model,
        interval_delta_model=interval_delta_model,
    )
    section_outline_artifact_path = section_outline_path(
        history_tool_root, checkpoint_id
    )
    section_outline_llm_artifact_path = section_outline_llm_path(
        history_tool_root,
        checkpoint_id,
    )
    scaffold_outline = build_section_planning_scaffold(
        checkpoint_model=checkpoint_model,
        section_outline=section_outline,
        semantic_context_map=semantic_context_map,
        include_semantic_context=experimental_section_mode == "semantic_context",
    )
    algorithm_capsule_index_artifact_path = algorithm_capsule_index_path(
        history_tool_root, checkpoint_id
    )

    _progress(
        progress_callback,
        "history-docs: building deterministic algorithm capsules",
    )
    algorithm_capsule_index, algorithm_capsules = build_algorithm_capsules(
        checkpoint_model=checkpoint_model,
        interval_delta_model=interval_delta_model,
    )
    checkpoint_model = link_algorithm_capsules_to_checkpoint_model(
        checkpoint_model,
        algorithm_capsules,
    )
    scaffold_outline = link_algorithm_capsules_to_section_outline(
        checkpoint_model,
        scaffold_outline,
        interval_delta_model,
        algorithm_capsules,
    )
    _progress(
        progress_callback,
        "history-docs: building shadow LLM section outline",
    )
    section_outline_llm = build_llm_section_outline(
        checkpoint_model=checkpoint_model,
        section_scaffold=scaffold_outline,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        semantic_context_map=(
            semantic_context_map
            if experimental_section_mode == "semantic_context"
            else None
        ),
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    section_outline_llm = link_section_planning_outline_to_scaffold(
        section_scaffold=scaffold_outline,
        llm_section_outline=section_outline_llm,
    )
    authoritative_section_outline = (
        HistorySectionOutline(
            checkpoint_id=section_outline_llm.checkpoint_id,
            target_commit=section_outline_llm.target_commit,
            previous_checkpoint_commit=section_outline_llm.previous_checkpoint_commit,
            sections=section_outline_llm.sections,
        )
        if section_planning_mode == "llm"
        else scaffold_outline
    )

    checkpoint_root = history_tool_root / "checkpoints" / checkpoint_id
    for index_entry, capsule in zip(
        algorithm_capsule_index.capsules,
        algorithm_capsules,
        strict=True,
    ):
        write_json_model(checkpoint_root / index_entry.artifact_path, capsule)
    write_json_model(algorithm_capsule_index_artifact_path, algorithm_capsule_index)
    write_json_model(section_outline_artifact_path, scaffold_outline)
    write_json_model(section_outline_llm_artifact_path, section_outline_llm)

    _progress(
        progress_callback,
        "history-docs: documenting direct dependencies",
    )
    dependency_inventory = build_dependency_inventory(
        repo_root=resolved_repo_root,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        checkpoint_model=checkpoint_model,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
        read_file_at_commit=read_file_at_commit,
    )
    checkpoint_model = link_dependency_inventory_to_checkpoint_model(
        checkpoint_model,
        dependency_inventory,
    )
    dependencies_artifact_path = dependency_inventory_path(
        history_tool_root,
        checkpoint_id,
    )
    algorithm_capsule_enrichment_index_artifact_path = (
        algorithm_capsule_enrichment_index_path(history_tool_root, checkpoint_id)
    )
    interface_inventory_artifact_path = interface_inventory_path(
        history_tool_root,
        checkpoint_id,
    )
    dependency_landscape_artifact_path = dependency_landscape_path(
        history_tool_root,
        checkpoint_id,
    )
    write_json_model(dependencies_artifact_path, dependency_inventory)
    write_json_model(checkpoint_model_artifact_path, checkpoint_model)

    _progress(
        progress_callback,
        "history-docs: enriching algorithm capsules in shadow mode",
    )
    algorithm_capsule_enrichment_index, algorithm_capsule_enrichments = (
        build_algorithm_capsule_enrichments(
            checkpoint_id=checkpoint_id,
            target_commit=target_metadata.sha,
            previous_checkpoint_commit=resolved_previous,
            checkpoint_model=checkpoint_model,
            interval_interpretation=interval_interpretation,
            checkpoint_model_enrichment=checkpoint_model_enrichment,
            dependency_inventory=dependency_inventory,
            capsule_index=algorithm_capsule_index,
            capsules=algorithm_capsules,
            semantic_context_map=semantic_context_map,
            llm_client=llm_client,
            model_name=project_config.llm.model_name,
            temperature=project_config.llm.temperature,
        )
    )
    enriched_capsule_root = algorithm_capsule_enrichment_dir(
        history_tool_root,
        checkpoint_id,
    )
    for enrichment in algorithm_capsule_enrichments:
        write_json_model(
            enriched_capsule_root
            / algorithm_capsule_enrichment_filename(enrichment.capsule_id),
            enrichment,
        )
    write_json_model(
        algorithm_capsule_enrichment_index_artifact_path,
        algorithm_capsule_enrichment_index,
    )

    _progress(
        progress_callback,
        "history-docs: building interface inventory in shadow mode",
    )
    interface_inventory = build_interface_inventory(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        dependency_inventory=dependency_inventory,
        semantic_context_map=semantic_context_map,
        capsule_index=algorithm_capsule_index,
        capsules=algorithm_capsules,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    write_json_model(interface_inventory_artifact_path, interface_inventory)

    _progress(
        progress_callback,
        "history-docs: building dependency landscape in shadow mode",
    )
    dependency_landscape = build_dependency_landscape(
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        checkpoint_model=checkpoint_model,
        interval_interpretation=interval_interpretation,
        checkpoint_model_enrichment=checkpoint_model_enrichment,
        dependency_inventory=dependency_inventory,
        semantic_context_map=semantic_context_map,
        capsule_index=algorithm_capsule_index,
        capsules=algorithm_capsules,
        llm_client=llm_client,
        model_name=project_config.llm.model_name,
        temperature=project_config.llm.temperature,
    )
    write_json_model(dependency_landscape_artifact_path, dependency_landscape)

    _progress(
        progress_callback,
        "history-docs: rendering checkpoint markdown",
    )
    checkpoint_markdown_artifact_path = checkpoint_markdown_path(
        history_tool_root,
        checkpoint_id,
    )
    render_manifest_artifact_path = render_manifest_path(
        history_tool_root,
        checkpoint_id,
    )
    checkpoint_markdown, render_manifest = render_checkpoint_markdown(
        workspace_id=resolved_workspace_id,
        checkpoint_model=checkpoint_model,
        section_outline=authoritative_section_outline,
        dependency_inventory=dependency_inventory,
        capsule_index=algorithm_capsule_index,
        capsules=algorithm_capsules,
        algorithm_capsule_enrichment_index=(
            algorithm_capsule_enrichment_index
            if algorithm_capsule_mode == "enriched"
            else None
        ),
        algorithm_capsule_enrichments=(
            algorithm_capsule_enrichments
            if algorithm_capsule_mode == "enriched"
            else None
        ),
        semantic_context_map=(
            semantic_context_map
            if experimental_section_mode == "semantic_context"
            else None
        ),
        interface_inventory=(
            interface_inventory if interface_render_mode == "inventory" else None
        ),
        dependency_landscape=(
            dependency_landscape if dependency_render_mode == "landscape" else None
        ),
    )
    write_checkpoint_markdown(
        checkpoint_markdown_artifact_path,
        checkpoint_markdown,
    )
    write_json_model(render_manifest_artifact_path, render_manifest)

    _progress(
        progress_callback,
        "history-docs: validating rendered checkpoint artifacts",
    )
    validation_report_artifact_path = validation_report_path(
        history_tool_root,
        checkpoint_id,
    )
    validation_report = validate_checkpoint_render(
        checkpoint_dir=checkpoint_root,
        checkpoint_model=checkpoint_model,
        section_outline=authoritative_section_outline,
        dependency_inventory=dependency_inventory,
        capsule_index=algorithm_capsule_index,
        markdown=checkpoint_markdown,
        render_manifest=render_manifest,
    )
    write_json_model(validation_report_artifact_path, validation_report)

    if validation_report.error_count > 0:
        raise ValidationError(
            "History-docs validation failed with "
            f"{validation_report.error_count} error(s); see {validation_report_artifact_path}"
        )

    retired_concept_count = (
        sum(
            concept.lifecycle_status == "retired"
            for concept in checkpoint_model.subsystems
        )
        + sum(
            concept.lifecycle_status == "retired"
            for concept in checkpoint_model.modules
        )
        + sum(
            concept.lifecycle_status == "retired"
            for concept in checkpoint_model.dependencies
        )
    )

    return HistoryBuildResult(
        workspace_id=resolved_workspace_id,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        previous_checkpoint_source=previous_source,
        commit_count=len(interval_commits),
        checkpoint_plan_path=checkpoint_plan_path,
        intervals_path=intervals_path,
        semantic_checkpoint_plan_path=semantic_plan_artifact_path,
        semantic_structure_map_path=semantic_structure_artifact_path,
        semantic_context_map_path=semantic_context_artifact_path,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_structural_model_path=snapshot_structural_model_path,
        interval_delta_model_path=interval_delta_path,
        interval_interpretation_path=interval_interpretation_artifact_path,
        checkpoint_model_path=checkpoint_model_artifact_path,
        checkpoint_model_enrichment_path=checkpoint_model_enrichment_artifact_path,
        algorithm_capsule_enrichment_index_path=(
            algorithm_capsule_enrichment_index_artifact_path
        ),
        interface_inventory_path=interface_inventory_artifact_path,
        dependency_landscape_path=dependency_landscape_artifact_path,
        section_outline_path=section_outline_artifact_path,
        section_outline_llm_path=section_outline_llm_artifact_path,
        algorithm_capsule_index_path=algorithm_capsule_index_artifact_path,
        dependencies_artifact_path=dependencies_artifact_path,
        checkpoint_markdown_path=checkpoint_markdown_artifact_path,
        render_manifest_path=render_manifest_artifact_path,
        validation_report_path=validation_report_artifact_path,
        file_count=len(scan_result.files),
        symbol_count=len(scan_result.symbol_summaries),
        subsystem_count=len(subsystem_candidates),
        build_source_count=len(build_sources),
        semantic_candidate_count=len(semantic_plan.candidates),
        semantic_primary_candidate_count=sum(
            candidate.recommendation == "primary"
            for candidate in semantic_plan.candidates
        ),
        semantic_planner_status=semantic_plan.evaluation_status,
        semantic_subsystem_count=len(semantic_structure_map.semantic_subsystems),
        semantic_capability_count=len(semantic_structure_map.capabilities),
        semantic_structure_status=semantic_structure_map.evaluation_status,
        semantic_context_status=semantic_context_map.evaluation_status,
        interval_interpretation_status=interval_interpretation.evaluation_status,
        checkpoint_model_enrichment_status=(
            checkpoint_model_enrichment.evaluation_status
        ),
        algorithm_capsule_enrichment_status=(
            algorithm_capsule_enrichment_index.evaluation_status
        ),
        interface_inventory_status=interface_inventory.evaluation_status,
        dependency_landscape_status=dependency_landscape.evaluation_status,
        section_planning_status=section_outline_llm.evaluation_status,
        context_node_count=len(semantic_context_map.context_nodes),
        interface_candidate_count=len(semantic_context_map.interfaces),
        subsystem_change_count=len(interval_delta_model.subsystem_changes),
        interface_change_count=len(interval_delta_model.interface_changes),
        dependency_change_count=len(interval_delta_model.dependency_changes),
        algorithm_candidate_count=len(interval_delta_model.algorithm_candidates),
        interval_insight_count=len(interval_interpretation.insights),
        interval_significant_window_count=len(
            interval_interpretation.significant_windows
        ),
        enriched_subsystem_count=len(checkpoint_model_enrichment.subsystem_enrichments),
        enriched_module_count=len(checkpoint_model_enrichment.module_enrichments),
        capability_proposal_count=len(checkpoint_model_enrichment.capability_proposals),
        design_note_anchor_count=len(checkpoint_model_enrichment.design_note_anchors),
        enriched_algorithm_capsule_count=len(algorithm_capsule_enrichments),
        interface_inventory_count=len(interface_inventory.interfaces),
        dependency_cluster_count=len(dependency_landscape.clusters),
        dependency_usage_pattern_count=len(dependency_landscape.usage_patterns),
        subsystem_concept_count=len(checkpoint_model.subsystems),
        module_concept_count=len(checkpoint_model.modules),
        dependency_concept_count=len(checkpoint_model.dependencies),
        retired_concept_count=retired_concept_count,
        included_section_count=_count_section_status(scaffold_outline, "included"),
        omitted_section_count=_count_section_status(scaffold_outline, "omitted"),
        algorithm_capsule_count=len(algorithm_capsules),
        llm_included_section_count=_count_section_status(
            HistorySectionOutline(
                checkpoint_id=section_outline_llm.checkpoint_id,
                target_commit=section_outline_llm.target_commit,
                previous_checkpoint_commit=section_outline_llm.previous_checkpoint_commit,
                sections=section_outline_llm.sections,
            ),
            "included",
        ),
        documented_dependency_count=len(dependency_inventory.entries),
        dependency_warning_count=len(dependency_inventory.warnings),
        dependency_summary_failure_count=_count_dependency_summary_failures(
            dependency_inventory
        ),
        rendered_section_count=len(render_manifest.sections),
        validation_error_count=validation_report.error_count,
        validation_warning_count=validation_report.warning_count,
    )
