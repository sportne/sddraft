"""History-docs workflow: checkpoint manifests plus snapshot structural analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
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
    resolve_commit,
)
from engllm.core.repo.scanner import scan_paths
from engllm.core.workspaces import build_workspace_context, tool_artifact_root
from engllm.domain.errors import AnalysisError, GitError, RepositoryError
from engllm.domain.models import ProjectConfig
from engllm.tools.history_docs.models import (
    HistoryBuildResult,
    HistorySnapshotStructuralModel,
    HistorySubsystemCandidate,
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


def _normalize_relative_path(path: Path) -> Path:
    return Path(".") if path in {Path(""), Path(".")} else path


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
            repo_relative_root = _normalize_relative_path(
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
        cursor = _normalize_relative_path(root)
        while True:
            directories.add(cursor)
            if cursor == Path("."):
                break
            parent = cursor.parent if cursor.parent != Path("") else Path(".")
            cursor = _normalize_relative_path(parent)
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


def _owning_source_root(file_path: Path, analyzed_roots: list[Path]) -> Path:
    ordered_roots = sorted(
        analyzed_roots,
        key=lambda item: (-len(item.parts), item.as_posix()),
    )
    for root in ordered_roots:
        if root == Path("."):
            return root
        try:
            file_path.relative_to(root)
            return root
        except ValueError:
            continue
    return Path(".")


def _build_subsystem_candidates(
    *,
    files: list[Path],
    symbol_counts_by_path: Mapping[Path, int],
    language_by_path: Mapping[Path, str],
    analyzed_roots: list[Path],
) -> list[HistorySubsystemCandidate]:
    grouped_files: dict[tuple[Path, Path], list[Path]] = defaultdict(list)
    for file_path in sorted(files, key=lambda item: item.as_posix()):
        source_root = _owning_source_root(file_path, analyzed_roots)
        relative_under_root = (
            file_path
            if source_root == Path(".")
            else file_path.relative_to(source_root)
        )
        if len(relative_under_root.parts) <= 1:
            group_path = source_root
        else:
            first_segment = Path(relative_under_root.parts[0])
            group_path = (
                first_segment
                if source_root == Path(".")
                else source_root / first_segment
            )
        grouped_files[(source_root, group_path)].append(file_path)

    candidates: list[HistorySubsystemCandidate] = []
    for (source_root, group_path), grouped in sorted(
        grouped_files.items(),
        key=lambda item: (item[0][0].as_posix(), item[0][1].as_posix()),
    ):
        language_counts: Counter[str] = Counter()
        symbol_count = 0
        for file_path in grouped:
            language_counts[language_by_path[file_path]] += 1
            symbol_count += symbol_counts_by_path.get(file_path, 0)
        if group_path == source_root:
            group_relative = Path(".")
        elif source_root == Path("."):
            group_relative = group_path
        else:
            group_relative = group_path.relative_to(source_root)
        candidates.append(
            HistorySubsystemCandidate(
                candidate_id=(
                    f"subsystem::{source_root.as_posix()}::{group_relative.as_posix()}"
                ),
                source_root=source_root,
                group_path=group_path,
                file_count=len(grouped),
                symbol_count=symbol_count,
                language_counts={
                    key: language_counts[key] for key in sorted(language_counts)
                },
                representative_files=grouped[:_MAX_REPRESENTATIVE_FILES],
            )
        )
    return candidates


def _snapshot_structural_model_path(tool_root: Path, checkpoint_id: str) -> Path:
    return tool_root / "checkpoints" / checkpoint_id / "snapshot_structural_model.json"


def build_history_docs_checkpoint(
    *,
    project_config: ProjectConfig,
    repo_root: Path,
    checkpoint_commit: str,
    previous_checkpoint_commit: str | None = None,
    workspace_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HistoryBuildResult:
    """Persist checkpoint manifests plus H2 snapshot structural analysis."""

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
        subsystem_candidates = _build_subsystem_candidates(
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

    return HistoryBuildResult(
        workspace_id=resolved_workspace_id,
        checkpoint_id=checkpoint_id,
        target_commit=target_metadata.sha,
        previous_checkpoint_commit=resolved_previous,
        previous_checkpoint_source=previous_source,
        commit_count=len(interval_commits),
        checkpoint_plan_path=checkpoint_plan_path,
        intervals_path=intervals_path,
        snapshot_manifest_path=snapshot_manifest_path,
        snapshot_structural_model_path=snapshot_structural_model_path,
        file_count=len(scan_result.files),
        symbol_count=len(scan_result.symbol_summaries),
        subsystem_count=len(subsystem_candidates),
        build_source_count=len(build_sources),
    )
