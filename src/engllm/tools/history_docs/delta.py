"""History-docs interval-delta analysis helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from engllm.core.analysis.commit_impact import build_commit_impact
from engllm.core.analysis.history import HistoryBuildSource, HistoryCommitSummary
from engllm.core.repo.diff_parser import (
    extract_changed_symbol_names,
    get_git_diff_between,
    parse_diff,
)
from engllm.core.repo.history import describe_commit_diff
from engllm.domain.models import SymbolSummary
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCandidate,
    HistoryAlgorithmScopeKind,
    HistoryAlgorithmSignalKind,
    HistoryCommitDelta,
    HistoryCommitSignalKind,
    HistoryDeltaEvidenceKind,
    HistoryDeltaEvidenceLink,
    HistoryDeltaStatus,
    HistoryDependencyChangeCandidate,
    HistoryDependencyKind,
    HistoryInterfaceChangeCandidate,
    HistoryInterfaceScopeKind,
    HistoryIntervalDeltaModel,
    HistorySnapshotStructuralModel,
    HistorySubsystemCandidate,
    HistorySubsystemChangeCandidate,
)
from engllm.tools.history_docs.structure import (
    subsystem_candidate_id_for_file,
    subsystem_is_root_scope,
)

_VARIANT_TOKENS = ("strategy", "variant", "policy", "mode", "backend", "adapter")
_EvidenceLinkTuple = tuple[HistoryDeltaEvidenceKind, str, str | None]


@dataclass(slots=True)
class _SubsystemAccumulator:
    candidate: HistorySubsystemCandidate
    statuses: set[HistoryDeltaStatus] = field(default_factory=set)
    commit_ids: set[str] = field(default_factory=set)
    file_paths: set[Path] = field(default_factory=set)
    changed_symbol_names: set[str] = field(default_factory=set)
    evidence_links: set[_EvidenceLinkTuple] = field(default_factory=set)


@dataclass(slots=True)
class _InterfaceAccumulator:
    candidate_id: str
    source_path: Path
    scope_kind: HistoryInterfaceScopeKind
    symbol_name: str | None = None
    qualified_name: str | None = None
    statuses: set[HistoryDeltaStatus] = field(default_factory=set)
    commit_ids: set[str] = field(default_factory=set)
    signature_changes: set[str] = field(default_factory=set)
    evidence_links: set[_EvidenceLinkTuple] = field(default_factory=set)


@dataclass(slots=True)
class _DependencyAccumulator:
    candidate_id: str
    dependency_kind: HistoryDependencyKind
    path: Path | None = None
    subsystem_id: str | None = None
    ecosystem: str | None = None
    category: str | None = None
    statuses: set[HistoryDeltaStatus] = field(default_factory=set)
    commit_ids: set[str] = field(default_factory=set)
    file_paths: set[Path] = field(default_factory=set)
    dependency_change_lines: set[str] = field(default_factory=set)
    evidence_links: set[_EvidenceLinkTuple] = field(default_factory=set)


@dataclass(slots=True)
class _AlgorithmAccumulator:
    candidate_id: str
    scope_kind: HistoryAlgorithmScopeKind
    scope_path: Path
    subsystem_id: str | None = None
    commit_ids: set[str] = field(default_factory=set)
    changed_symbol_names: set[str] = field(default_factory=set)
    variant_names: set[str] = field(default_factory=set)
    signal_kinds: set[HistoryAlgorithmSignalKind] = field(default_factory=set)
    evidence_links: set[_EvidenceLinkTuple] = field(default_factory=set)


def interval_delta_model_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the tool-scoped interval-delta artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "interval_delta_model.json"


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _evidence_link(
    kind: HistoryDeltaEvidenceKind,
    reference: str,
    detail: str | None = None,
) -> tuple[HistoryDeltaEvidenceKind, str, str | None]:
    return (kind, reference, detail)


def _finalize_evidence_links(
    links: set[_EvidenceLinkTuple],
) -> list[HistoryDeltaEvidenceLink]:
    return [
        HistoryDeltaEvidenceLink(kind=kind, reference=reference, detail=detail)
        for kind, reference, detail in sorted(
            links,
            key=lambda item: (item[0], item[1], item[2] or ""),
        )
    ]


def _pick_status(statuses: set[HistoryDeltaStatus]) -> HistoryDeltaStatus:
    for status in ("introduced", "retired", "modified", "observed"):
        if status in statuses:
            return status
    return "observed"


def _index_symbols(
    snapshot: HistorySnapshotStructuralModel | None,
) -> dict[Path, list[SymbolSummary]]:
    indexed: dict[Path, list[SymbolSummary]] = defaultdict(list)
    if snapshot is None:
        return indexed
    for symbol in snapshot.symbol_summaries:
        indexed[symbol.source_path].append(symbol)
    for path in indexed:
        indexed[path] = sorted(
            indexed[path],
            key=lambda item: (
                item.qualified_name or "",
                item.name,
                item.kind,
                item.line_start or -1,
                item.line_end or -1,
            ),
        )
    return indexed


def _resolve_symbol(
    indexed: dict[Path, list[SymbolSummary]],
    *,
    source_path: Path,
    name: str,
) -> SymbolSummary | None:
    for symbol in indexed.get(source_path, []):
        if symbol.name == name or (symbol.qualified_name or "") == name:
            return symbol
    return None


def _index_subsystems(
    snapshot: HistorySnapshotStructuralModel | None,
) -> dict[str, HistorySubsystemCandidate]:
    if snapshot is None:
        return {}
    return {
        candidate.candidate_id: candidate for candidate in snapshot.subsystem_candidates
    }


def _index_build_sources(
    snapshot: HistorySnapshotStructuralModel | None,
) -> dict[Path, HistoryBuildSource]:
    if snapshot is None:
        return {}
    return {source.path: source for source in snapshot.build_sources}


def _file_set(snapshot: HistorySnapshotStructuralModel | None) -> set[Path]:
    if snapshot is None:
        return set()
    return set(snapshot.files)


def _symbol_counts(snapshot: HistorySnapshotStructuralModel | None) -> Counter[Path]:
    if snapshot is None:
        return Counter()
    return Counter(symbol.source_path for symbol in snapshot.symbol_summaries)


def _path_is_within_roots(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        if root == Path("."):
            return True
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_subsystem_id(
    snapshot: HistorySnapshotStructuralModel | None,
    indexed: dict[str, HistorySubsystemCandidate],
    file_path: Path,
) -> str | None:
    if snapshot is None or not snapshot.analyzed_source_roots:
        return None
    if not _path_is_within_roots(file_path, snapshot.analyzed_source_roots):
        return None
    candidate_id = subsystem_candidate_id_for_file(
        file_path=file_path,
        analyzed_roots=snapshot.analyzed_source_roots,
    )
    return candidate_id if candidate_id in indexed else None


def _primary_subsystem_id(
    current_id: str | None, previous_id: str | None
) -> str | None:
    return current_id or previous_id


def _variant_name(path: Path) -> str | None:
    stem = path.stem.lower()
    if any(token in stem for token in _VARIANT_TOKENS):
        return stem
    return None


def _record_subsystem_change(
    accumulators: dict[str, _SubsystemAccumulator],
    *,
    current_index: dict[str, HistorySubsystemCandidate],
    previous_index: dict[str, HistorySubsystemCandidate],
    current_id: str | None,
    previous_id: str | None,
    previous_snapshot_available: bool,
    commit_id: str,
    file_path: Path,
    changed_symbol_names: set[str],
) -> set[HistoryDeltaStatus]:
    touched_statuses: set[HistoryDeltaStatus] = set()

    def record(
        candidate_id: str,
        status: HistoryDeltaStatus,
        candidate: HistorySubsystemCandidate,
    ) -> None:
        accumulator = accumulators.setdefault(
            candidate_id,
            _SubsystemAccumulator(candidate=candidate),
        )
        accumulator.statuses.add(status)
        accumulator.commit_ids.add(commit_id)
        accumulator.file_paths.add(file_path)
        accumulator.changed_symbol_names.update(changed_symbol_names)
        accumulator.evidence_links.add(_evidence_link("subsystem", candidate_id))
        accumulator.evidence_links.add(_evidence_link("file", file_path.as_posix()))
        for name in sorted(changed_symbol_names):
            accumulator.evidence_links.add(
                _evidence_link(
                    "symbol",
                    f"{file_path.as_posix()}::{name}",
                )
            )
        touched_statuses.add(status)

    if not previous_snapshot_available:
        candidate_id = current_id or previous_id
        if candidate_id is None:
            return touched_statuses
        candidate = current_index.get(candidate_id) or previous_index.get(candidate_id)
        if candidate is None:
            return touched_statuses
        record(candidate_id, "observed", candidate)
        return touched_statuses

    if current_id is not None and previous_id is not None:
        if current_id == previous_id:
            candidate = current_index.get(current_id) or previous_index[current_id]
            record(current_id, "modified", candidate)
        else:
            if current_id in current_index:
                record(current_id, "introduced", current_index[current_id])
            if previous_id in previous_index:
                record(previous_id, "retired", previous_index[previous_id])
        return touched_statuses

    if current_id is not None and current_id in current_index:
        record(current_id, "introduced", current_index[current_id])
    if previous_id is not None and previous_id in previous_index:
        record(previous_id, "retired", previous_index[previous_id])
    return touched_statuses


def _record_interface_change(
    accumulators: dict[str, _InterfaceAccumulator],
    *,
    current_symbols: dict[Path, list[SymbolSummary]],
    previous_symbols: dict[Path, list[SymbolSummary]],
    previous_snapshot_available: bool,
    commit_id: str,
    file_path: Path,
    name: str,
    signature_changes: list[str],
) -> None:
    current_symbol = _resolve_symbol(current_symbols, source_path=file_path, name=name)
    previous_symbol = _resolve_symbol(
        previous_symbols, source_path=file_path, name=name
    )

    if current_symbol is not None:
        candidate_id = (
            f"interface::{file_path.as_posix()}::"
            f"{current_symbol.qualified_name or current_symbol.name}"
        )
        scope_kind: HistoryInterfaceScopeKind = "symbol"
        symbol_name = current_symbol.name
        qualified_name = current_symbol.qualified_name
    elif previous_symbol is not None:
        candidate_id = (
            f"interface::{file_path.as_posix()}::"
            f"{previous_symbol.qualified_name or previous_symbol.name}"
        )
        scope_kind = "symbol"
        symbol_name = previous_symbol.name
        qualified_name = previous_symbol.qualified_name
    else:
        candidate_id = f"interface::{file_path.as_posix()}::."
        scope_kind = "file"
        symbol_name = None
        qualified_name = None

    if not previous_snapshot_available:
        status: HistoryDeltaStatus = "observed"
    elif current_symbol is not None and previous_symbol is None:
        status = "introduced"
    elif current_symbol is None and previous_symbol is not None:
        status = "retired"
    else:
        status = "modified"

    accumulator = accumulators.setdefault(
        candidate_id,
        _InterfaceAccumulator(
            candidate_id=candidate_id,
            scope_kind=scope_kind,
            source_path=file_path,
            symbol_name=symbol_name,
            qualified_name=qualified_name,
        ),
    )
    accumulator.statuses.add(status)
    accumulator.commit_ids.add(commit_id)
    accumulator.signature_changes.update(
        line for line in signature_changes if name in line
    )
    if not accumulator.signature_changes:
        accumulator.signature_changes.update(signature_changes)
    accumulator.evidence_links.add(_evidence_link("file", file_path.as_posix()))
    if symbol_name is not None:
        accumulator.evidence_links.add(
            _evidence_link(
                "symbol",
                f"{file_path.as_posix()}::{qualified_name or symbol_name}",
            )
        )


def _record_build_source_dependency(
    accumulators: dict[str, _DependencyAccumulator],
    *,
    current_build_sources: dict[Path, HistoryBuildSource],
    previous_build_sources: dict[Path, HistoryBuildSource],
    previous_snapshot_available: bool,
    commit_id: str,
    file_path: Path,
) -> None:
    current_source = current_build_sources.get(file_path)
    previous_source = previous_build_sources.get(file_path)
    candidate_id = f"dependency::{file_path.as_posix()}"
    if not previous_snapshot_available:
        status: HistoryDeltaStatus = "observed"
        source = current_source or previous_source
    elif current_source is not None and previous_source is None:
        status = "introduced"
        source = current_source
    elif current_source is None and previous_source is not None:
        status = "retired"
        source = previous_source
    else:
        status = "modified"
        source = current_source or previous_source
    if source is None:
        return
    accumulator = accumulators.setdefault(
        candidate_id,
        _DependencyAccumulator(
            candidate_id=candidate_id,
            dependency_kind="build_source",
            path=file_path,
            ecosystem=source.ecosystem,
            category=source.category,
        ),
    )
    accumulator.statuses.add(status)
    accumulator.commit_ids.add(commit_id)
    accumulator.file_paths.add(file_path)
    accumulator.evidence_links.add(_evidence_link("build_source", file_path.as_posix()))
    accumulator.evidence_links.add(_evidence_link("file", file_path.as_posix()))


def _record_code_import_signal(
    accumulators: dict[str, _DependencyAccumulator],
    *,
    previous_snapshot_available: bool,
    commit_id: str,
    file_path: Path,
    dependency_changes: list[str],
    subsystem_id: str | None,
) -> None:
    if subsystem_id is not None:
        candidate_id = f"dependency-signal::{subsystem_id}"
    else:
        candidate_id = f"dependency-signal::{file_path.as_posix()}"
    accumulator = accumulators.setdefault(
        candidate_id,
        _DependencyAccumulator(
            candidate_id=candidate_id,
            dependency_kind="code_import_signal",
            path=None if subsystem_id is not None else file_path,
            subsystem_id=subsystem_id,
        ),
    )
    accumulator.statuses.add("modified" if previous_snapshot_available else "observed")
    accumulator.commit_ids.add(commit_id)
    accumulator.file_paths.add(file_path)
    accumulator.dependency_change_lines.update(dependency_changes)
    if subsystem_id is not None:
        accumulator.evidence_links.add(_evidence_link("subsystem", subsystem_id))
    accumulator.evidence_links.add(_evidence_link("file", file_path.as_posix()))


def _finalize_subsystem_changes(
    accumulators: dict[str, _SubsystemAccumulator],
) -> list[HistorySubsystemChangeCandidate]:
    return [
        HistorySubsystemChangeCandidate(
            candidate_id=candidate_id,
            status=_pick_status(accumulator.statuses),
            source_root=accumulator.candidate.source_root,
            group_path=accumulator.candidate.group_path,
            commit_ids=sorted(accumulator.commit_ids),
            file_paths=sorted(accumulator.file_paths, key=_path_sort_key),
            changed_symbol_names=sorted(accumulator.changed_symbol_names),
            evidence_links=_finalize_evidence_links(accumulator.evidence_links),
        )
        for candidate_id, accumulator in sorted(accumulators.items())
    ]


def _finalize_interface_changes(
    accumulators: dict[str, _InterfaceAccumulator],
) -> list[HistoryInterfaceChangeCandidate]:
    return [
        HistoryInterfaceChangeCandidate(
            candidate_id=candidate_id,
            status=_pick_status(accumulator.statuses),
            scope_kind=accumulator.scope_kind,
            source_path=accumulator.source_path,
            symbol_name=accumulator.symbol_name,
            qualified_name=accumulator.qualified_name,
            commit_ids=sorted(accumulator.commit_ids),
            signature_changes=sorted(accumulator.signature_changes),
            evidence_links=_finalize_evidence_links(accumulator.evidence_links),
        )
        for candidate_id, accumulator in sorted(accumulators.items())
    ]


def _finalize_dependency_changes(
    accumulators: dict[str, _DependencyAccumulator],
) -> list[HistoryDependencyChangeCandidate]:
    return [
        HistoryDependencyChangeCandidate(
            candidate_id=candidate_id,
            status=_pick_status(accumulator.statuses),
            dependency_kind=accumulator.dependency_kind,
            path=accumulator.path,
            subsystem_id=accumulator.subsystem_id,
            ecosystem=accumulator.ecosystem,
            category=accumulator.category,
            commit_ids=sorted(accumulator.commit_ids),
            file_paths=sorted(accumulator.file_paths, key=_path_sort_key),
            dependency_change_lines=sorted(accumulator.dependency_change_lines),
            evidence_links=_finalize_evidence_links(accumulator.evidence_links),
        )
        for candidate_id, accumulator in sorted(accumulators.items())
    ]


def _finalize_algorithm_candidates(
    accumulators: dict[str, _AlgorithmAccumulator],
) -> list[HistoryAlgorithmCandidate]:
    return [
        HistoryAlgorithmCandidate(
            candidate_id=candidate_id,
            scope_kind=accumulator.scope_kind,
            scope_path=accumulator.scope_path,
            subsystem_id=accumulator.subsystem_id,
            commit_ids=sorted(accumulator.commit_ids),
            changed_symbol_names=sorted(accumulator.changed_symbol_names),
            variant_names=sorted(accumulator.variant_names),
            signal_kinds=sorted(accumulator.signal_kinds),
            evidence_links=_finalize_evidence_links(accumulator.evidence_links),
        )
        for candidate_id, accumulator in sorted(accumulators.items())
    ]


def build_interval_delta_model(
    *,
    repo_root: Path,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    interval_commits: list[HistoryCommitSummary],
    current_snapshot: HistorySnapshotStructuralModel,
    previous_snapshot: HistorySnapshotStructuralModel | None,
) -> HistoryIntervalDeltaModel:
    """Build a tool-scoped interval-delta model for one checkpoint."""

    previous_snapshot_available = previous_snapshot is not None
    current_subsystems = _index_subsystems(current_snapshot)
    previous_subsystems = _index_subsystems(previous_snapshot)
    current_build_sources = _index_build_sources(current_snapshot)
    previous_build_sources = _index_build_sources(previous_snapshot)
    current_symbols = _index_symbols(current_snapshot)
    previous_symbols = _index_symbols(previous_snapshot)
    current_files = _file_set(current_snapshot)
    previous_files = _file_set(previous_snapshot)
    current_symbol_counts = _symbol_counts(current_snapshot)

    commit_deltas: list[HistoryCommitDelta] = []
    subsystem_accumulators: dict[str, _SubsystemAccumulator] = {}
    interface_accumulators: dict[str, _InterfaceAccumulator] = {}
    dependency_accumulators: dict[str, _DependencyAccumulator] = {}
    algorithm_accumulators: dict[str, _AlgorithmAccumulator] = {}
    file_algorithm_names: dict[Path, set[str]] = defaultdict(set)
    file_algorithm_commits: dict[Path, set[str]] = defaultdict(set)
    file_algorithm_subsystems: dict[Path, set[str]] = defaultdict(set)
    file_algorithm_evidence: dict[Path, set[_EvidenceLinkTuple]] = defaultdict(set)
    subsystem_variant_names: dict[str, set[str]] = defaultdict(set)
    subsystem_variant_commits: dict[str, set[str]] = defaultdict(set)
    subsystem_variant_evidence: dict[str, set[_EvidenceLinkTuple]] = defaultdict(set)

    for commit in interval_commits:
        diff_spec = describe_commit_diff(repo_root, commit.sha)
        diff_text = get_git_diff_between(
            diff_spec.base_rev,
            diff_spec.commit_sha,
            repo_root,
        )
        file_diffs = parse_diff(diff_text)
        impact = build_commit_impact(
            f"{diff_spec.base_rev}..{diff_spec.commit_sha}",
            file_diffs,
        )

        commit_signal_kinds: set[HistoryCommitSignalKind] = set()
        changed_symbol_names: set[str] = set()
        affected_subsystem_ids: set[str] = set()
        touched_build_sources: set[Path] = set()
        evidence_links: set[_EvidenceLinkTuple] = {
            _evidence_link("commit", commit.sha, commit.subject)
        }
        touched_subsystem_statuses: set[HistoryDeltaStatus] = set()

        if file_diffs and all(file_diff.comment_only for file_diff in file_diffs):
            commit_signal_kinds.add("documentation_only")

        for file_diff in file_diffs:
            file_path = file_diff.path
            file_names = extract_changed_symbol_names(file_diff.signature_changes)
            changed_symbol_names.update(file_names)
            evidence_links.add(_evidence_link("file", file_path.as_posix()))
            for name in sorted(file_names):
                evidence_links.add(
                    _evidence_link("symbol", f"{file_path.as_posix()}::{name}")
                )

            current_subsystem_id = _resolve_subsystem_id(
                current_snapshot,
                current_subsystems,
                file_path,
            )
            previous_subsystem_id = _resolve_subsystem_id(
                previous_snapshot,
                previous_subsystems,
                file_path,
            )
            if current_subsystem_id is not None:
                affected_subsystem_ids.add(current_subsystem_id)
                evidence_links.add(_evidence_link("subsystem", current_subsystem_id))
            if previous_subsystem_id is not None:
                affected_subsystem_ids.add(previous_subsystem_id)
                evidence_links.add(_evidence_link("subsystem", previous_subsystem_id))
            touched_subsystem_statuses.update(
                _record_subsystem_change(
                    subsystem_accumulators,
                    current_index=current_subsystems,
                    previous_index=previous_subsystems,
                    current_id=current_subsystem_id,
                    previous_id=previous_subsystem_id,
                    previous_snapshot_available=previous_snapshot_available,
                    commit_id=commit.sha,
                    file_path=file_path,
                    changed_symbol_names=file_names,
                )
            )

            if file_diff.signature_changes:
                commit_signal_kinds.add("interface")
                interface_names = sorted(file_names) or [file_path.as_posix()]
                for name in interface_names:
                    if name == file_path.as_posix():
                        _record_interface_change(
                            interface_accumulators,
                            current_symbols=current_symbols,
                            previous_symbols=previous_symbols,
                            previous_snapshot_available=previous_snapshot_available,
                            commit_id=commit.sha,
                            file_path=file_path,
                            name="",
                            signature_changes=file_diff.signature_changes,
                        )
                    else:
                        _record_interface_change(
                            interface_accumulators,
                            current_symbols=current_symbols,
                            previous_symbols=previous_symbols,
                            previous_snapshot_available=previous_snapshot_available,
                            commit_id=commit.sha,
                            file_path=file_path,
                            name=name,
                            signature_changes=file_diff.signature_changes,
                        )

            if (
                file_path in current_build_sources
                or file_path in previous_build_sources
            ):
                touched_build_sources.add(file_path)
                commit_signal_kinds.update({"dependency", "infrastructure"})
                evidence_links.add(_evidence_link("build_source", file_path.as_posix()))
                _record_build_source_dependency(
                    dependency_accumulators,
                    current_build_sources=current_build_sources,
                    previous_build_sources=previous_build_sources,
                    previous_snapshot_available=previous_snapshot_available,
                    commit_id=commit.sha,
                    file_path=file_path,
                )

            if file_diff.dependency_changes:
                commit_signal_kinds.add("dependency")
                primary_subsystem_id = _primary_subsystem_id(
                    current_subsystem_id,
                    previous_subsystem_id,
                )
                _record_code_import_signal(
                    dependency_accumulators,
                    previous_snapshot_available=previous_snapshot_available,
                    commit_id=commit.sha,
                    file_path=file_path,
                    dependency_changes=file_diff.dependency_changes,
                    subsystem_id=primary_subsystem_id,
                )

            if not file_diff.comment_only and (
                file_diff.added_lines or file_diff.removed_lines
            ):
                file_algorithm_names[file_path].update(file_names)
                file_algorithm_commits[file_path].add(commit.sha)
                file_algorithm_evidence[file_path].add(
                    _evidence_link("file", file_path.as_posix())
                )
                primary_subsystem_id = _primary_subsystem_id(
                    current_subsystem_id,
                    previous_subsystem_id,
                )
                if primary_subsystem_id is not None:
                    file_algorithm_subsystems[file_path].add(primary_subsystem_id)
                    file_algorithm_evidence[file_path].add(
                        _evidence_link("subsystem", primary_subsystem_id)
                    )
                    variant_name = _variant_name(file_path)
                    if variant_name is not None:
                        subsystem_variant_names[primary_subsystem_id].add(variant_name)
                        subsystem_variant_commits[primary_subsystem_id].add(commit.sha)
                        subsystem_variant_evidence[primary_subsystem_id].add(
                            _evidence_link("file", file_path.as_posix())
                        )

        if len(affected_subsystem_ids) > 1:
            commit_signal_kinds.add("architectural")
        if {"introduced", "retired"} & touched_subsystem_statuses:
            commit_signal_kinds.add("architectural")
        if (
            affected_subsystem_ids
            and any(
                subsystem_is_root_scope(candidate_id)
                for candidate_id in affected_subsystem_ids
            )
            and any(
                not subsystem_is_root_scope(candidate_id)
                for candidate_id in affected_subsystem_ids
            )
        ):
            commit_signal_kinds.add("architectural")
        if (
            "documentation_only" not in commit_signal_kinds
            and "architectural" not in commit_signal_kinds
            and "interface" not in commit_signal_kinds
            and "dependency" not in commit_signal_kinds
            and "infrastructure" not in commit_signal_kinds
            and any(
                (file_diff.added_lines or file_diff.removed_lines)
                and not file_diff.comment_only
                for file_diff in file_diffs
            )
        ):
            commit_signal_kinds.add("logic_only")

        commit_deltas.append(
            HistoryCommitDelta(
                commit=commit,
                parent_commit=diff_spec.parent_commit,
                diff_basis=diff_spec.diff_basis,
                impact=impact,
                signal_kinds=sorted(commit_signal_kinds),
                changed_symbol_names=sorted(changed_symbol_names),
                affected_subsystem_ids=sorted(affected_subsystem_ids),
                touched_build_sources=sorted(touched_build_sources, key=_path_sort_key),
                evidence_links=_finalize_evidence_links(evidence_links),
            )
        )

    subsystem_changes = _finalize_subsystem_changes(subsystem_accumulators)
    interface_changes = _finalize_interface_changes(interface_accumulators)
    dependency_changes = _finalize_dependency_changes(dependency_accumulators)

    for candidate in subsystem_changes:
        current_candidate = current_subsystems.get(candidate.candidate_id)
        if (
            candidate.status == "introduced"
            and current_candidate is not None
            and current_candidate.symbol_count >= 2
        ):
            accumulator = algorithm_accumulators.setdefault(
                f"algorithm::subsystem::{candidate.candidate_id}",
                _AlgorithmAccumulator(
                    candidate_id=f"algorithm::subsystem::{candidate.candidate_id}",
                    scope_kind="subsystem",
                    scope_path=current_candidate.group_path,
                    subsystem_id=candidate.candidate_id,
                ),
            )
            accumulator.signal_kinds.add("introduced_module")
            accumulator.commit_ids.update(candidate.commit_ids)
            accumulator.changed_symbol_names.update(candidate.changed_symbol_names)
            accumulator.evidence_links.update(
                _evidence_link(link.kind, link.reference, link.detail)
                for link in candidate.evidence_links
            )

    for file_path, names in file_algorithm_names.items():
        if len(names) >= 2:
            accumulator = algorithm_accumulators.setdefault(
                f"algorithm::file::{file_path.as_posix()}",
                _AlgorithmAccumulator(
                    candidate_id=f"algorithm::file::{file_path.as_posix()}",
                    scope_kind="file",
                    scope_path=file_path,
                    subsystem_id=(
                        sorted(file_algorithm_subsystems[file_path])[0]
                        if file_algorithm_subsystems[file_path]
                        else None
                    ),
                ),
            )
            accumulator.signal_kinds.add("multi_symbol")
            accumulator.commit_ids.update(file_algorithm_commits[file_path])
            accumulator.changed_symbol_names.update(names)
            accumulator.evidence_links.update(file_algorithm_evidence[file_path])

        if (
            previous_snapshot_available
            and file_path in current_files
            and file_path not in previous_files
            and current_symbol_counts[file_path] >= 2
        ):
            accumulator = algorithm_accumulators.setdefault(
                f"algorithm::file::{file_path.as_posix()}",
                _AlgorithmAccumulator(
                    candidate_id=f"algorithm::file::{file_path.as_posix()}",
                    scope_kind="file",
                    scope_path=file_path,
                    subsystem_id=(
                        sorted(file_algorithm_subsystems[file_path])[0]
                        if file_algorithm_subsystems[file_path]
                        else None
                    ),
                ),
            )
            accumulator.signal_kinds.add("introduced_module")
            accumulator.commit_ids.update(file_algorithm_commits[file_path])
            accumulator.changed_symbol_names.update(names)
            accumulator.evidence_links.update(file_algorithm_evidence[file_path])

    for subsystem_id, variant_names in subsystem_variant_names.items():
        if len(variant_names) >= 2 and subsystem_id in current_subsystems:
            accumulator = algorithm_accumulators.setdefault(
                f"algorithm::subsystem::{subsystem_id}",
                _AlgorithmAccumulator(
                    candidate_id=f"algorithm::subsystem::{subsystem_id}",
                    scope_kind="subsystem",
                    scope_path=current_subsystems[subsystem_id].group_path,
                    subsystem_id=subsystem_id,
                ),
            )
            accumulator.signal_kinds.add("variant_family")
            accumulator.commit_ids.update(subsystem_variant_commits[subsystem_id])
            accumulator.variant_names.update(variant_names)
            accumulator.evidence_links.update(subsystem_variant_evidence[subsystem_id])

    algorithm_candidates = _finalize_algorithm_candidates(algorithm_accumulators)
    algorithm_candidate_commit_ids = {
        commit_id
        for candidate in algorithm_candidates
        for commit_id in candidate.commit_ids
    }
    for commit_delta in commit_deltas:
        if commit_delta.commit.sha in algorithm_candidate_commit_ids:
            commit_delta.signal_kinds = sorted(
                {*commit_delta.signal_kinds, "algorithm_candidate"}
            )

    return HistoryIntervalDeltaModel(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        previous_snapshot_available=previous_snapshot_available,
        commit_deltas=commit_deltas,
        subsystem_changes=subsystem_changes,
        interface_changes=interface_changes,
        dependency_changes=dependency_changes,
        algorithm_candidates=algorithm_candidates,
    )


__all__ = [
    "build_interval_delta_model",
    "interval_delta_model_path",
]
