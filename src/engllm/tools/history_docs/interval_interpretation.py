"""H12-01 interval interpretation helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from engllm.domain.models import CodeUnitSummary
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_interval_interpretation_prompt
from engllm.tools.history_docs.models import (
    HistoryAlgorithmCandidate,
    HistoryCommitDelta,
    HistoryDependencyChangeCandidate,
    HistoryDesignChangeInsight,
    HistoryEvidenceLink,
    HistoryInterfaceChangeCandidate,
    HistoryIntervalDeltaModel,
    HistoryIntervalInsightKind,
    HistoryIntervalInterpretation,
    HistoryIntervalInterpretationJudgment,
    HistoryIntervalInterpretationStatus,
    HistoryIntervalSignificance,
    HistoryRationaleClue,
    HistorySemanticContextMap,
    HistorySemanticStructureMap,
    HistorySignificantChangeWindow,
    HistorySnapshotStructuralModel,
    HistorySubsystemChangeCandidate,
)

_MAX_PROMPT_COMMITS = 12
_MAX_PROMPT_SUMMARIES = 16
_MAX_FALLBACK_INSIGHTS = 12
_MAX_FALLBACK_CLUES = 10
_RATIONALE_TOKENS = (
    "because",
    "ensure",
    "support",
    "enable",
    "avoid",
    "strict",
    "fallback",
    "deterministic",
    "conservative",
    "require",
    "must",
    "expects",
    "introduce",
    "tighten",
)


def interval_interpretation_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H12 interval interpretation artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "interval_interpretation.json"


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(
    *groups: Iterable[HistoryEvidenceLink],
) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _change_id(change: object) -> str:
    if isinstance(change, HistorySubsystemChangeCandidate):
        return str(change.change_id or change.candidate_id)
    if isinstance(change, HistoryInterfaceChangeCandidate):
        return str(change.change_id or change.candidate_id)
    if isinstance(change, HistoryDependencyChangeCandidate):
        return str(change.change_id or change.candidate_id)
    if isinstance(change, HistoryAlgorithmCandidate):
        return str(change.candidate_id)
    raise TypeError("unsupported change object")


def _title_case(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("-", " ").split())


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in cleaned.split("-") if part) or "item"


def _short_docstrings(summary: CodeUnitSummary) -> list[str]:
    return [value.strip()[:160] for value in summary.docstrings if value.strip()][:2]


def _collect_known_evidence_links(
    *,
    snapshot: HistorySnapshotStructuralModel,
    delta_model: HistoryIntervalDeltaModel,
    semantic_structure_map: HistorySemanticStructureMap | None,
    semantic_context_map: HistorySemanticContextMap | None,
) -> set[tuple[str, str]]:
    links: set[tuple[str, str]] = set()
    for commit_delta in delta_model.commit_deltas:
        for link in commit_delta.evidence_links:
            links.add((link.kind, link.reference))
    for group in (
        delta_model.subsystem_changes,
        delta_model.interface_changes,
        delta_model.dependency_changes,
        delta_model.algorithm_candidates,
    ):
        for candidate in group:
            for link in candidate.evidence_links:
                links.add((link.kind, link.reference))
    for summary in snapshot.code_summaries:
        links.add(("file", summary.path.as_posix()))
    for source in snapshot.build_sources:
        links.add(("build_source", source.path.as_posix()))
    if semantic_structure_map is not None:
        for subsystem in semantic_structure_map.semantic_subsystems:
            links.add(("subsystem", subsystem.semantic_subsystem_id))
    if semantic_context_map is not None:
        for node in semantic_context_map.context_nodes:
            for subsystem_id in node.related_subsystem_ids:
                links.add(("subsystem", subsystem_id))
    return links


def _candidate_payloads(
    *,
    subsystem_changes: list[HistorySubsystemChangeCandidate],
    interface_changes: list[HistoryInterfaceChangeCandidate],
    dependency_changes: list[HistoryDependencyChangeCandidate],
    algorithm_candidates: list[HistoryAlgorithmCandidate],
) -> dict[str, list[dict[str, object]]]:
    return {
        "subsystem_changes": [
            {
                "change_id": _change_id(change),
                "status": change.status,
                "source_root": change.source_root.as_posix(),
                "group_path": change.group_path.as_posix(),
                "commit_ids": change.commit_ids,
                "file_paths": [path.as_posix() for path in change.file_paths[:8]],
                "changed_symbol_names": change.changed_symbol_names[:12],
            }
            for change in subsystem_changes
        ],
        "interface_changes": [
            {
                "change_id": _change_id(change),
                "status": change.status,
                "scope_kind": change.scope_kind,
                "source_path": change.source_path.as_posix(),
                "symbol_name": change.symbol_name,
                "qualified_name": change.qualified_name,
                "commit_ids": change.commit_ids,
                "signature_changes": change.signature_changes[:8],
            }
            for change in interface_changes
        ],
        "dependency_changes": [
            {
                "change_id": _change_id(change),
                "status": change.status,
                "dependency_kind": change.dependency_kind,
                "path": None if change.path is None else change.path.as_posix(),
                "subsystem_id": change.subsystem_id,
                "ecosystem": change.ecosystem,
                "category": change.category,
                "commit_ids": change.commit_ids,
                "dependency_change_lines": change.dependency_change_lines[:8],
            }
            for change in dependency_changes
        ],
        "algorithm_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "scope_kind": candidate.scope_kind,
                "scope_path": candidate.scope_path.as_posix(),
                "subsystem_id": candidate.subsystem_id,
                "commit_ids": candidate.commit_ids,
                "changed_symbol_names": candidate.changed_symbol_names[:12],
                "variant_names": candidate.variant_names,
                "signal_kinds": candidate.signal_kinds,
            }
            for candidate in algorithm_candidates
        ],
    }


def _commit_strength(commit_delta: object) -> int:
    signal_count = len(getattr(commit_delta, "signal_kinds", []))
    affected_subsystems = len(getattr(commit_delta, "affected_subsystem_ids", []))
    build_sources = len(getattr(commit_delta, "touched_build_sources", []))
    changed_symbols = min(len(getattr(commit_delta, "changed_symbol_names", [])), 4)
    return (
        signal_count * 3 + affected_subsystems * 2 + build_sources * 2 + changed_symbols
    )


def _sorted_prompt_commits(
    delta_model: HistoryIntervalDeltaModel,
) -> list[dict[str, object]]:
    ranked = sorted(
        delta_model.commit_deltas,
        key=lambda item: (
            -_commit_strength(item),
            item.commit.timestamp,
            item.commit.sha,
        ),
    )[:_MAX_PROMPT_COMMITS]
    return [
        {
            "commit_id": item.commit.sha,
            "short_sha": item.commit.short_sha,
            "timestamp": item.commit.timestamp,
            "subject": item.commit.subject,
            "signal_kinds": item.signal_kinds,
            "changed_symbol_names": item.changed_symbol_names[:12],
            "affected_subsystem_ids": item.affected_subsystem_ids,
            "touched_build_sources": [
                path.as_posix() for path in item.touched_build_sources
            ],
            "impact_change_kinds": item.impact.change_kinds,
            "impact_summary": item.impact.summary,
        }
        for item in ranked
    ]


def _module_payloads(
    snapshot: HistorySnapshotStructuralModel,
    semantic_structure_map: HistorySemanticStructureMap | None,
) -> list[dict[str, object]]:
    semantic_titles_by_module: dict[str, list[str]] = defaultdict(list)
    if semantic_structure_map is not None:
        for subsystem in semantic_structure_map.semantic_subsystems:
            for module_id in subsystem.module_ids:
                semantic_titles_by_module[module_id].append(subsystem.title)

    def module_id_for(summary: CodeUnitSummary) -> str:
        return f"module::{summary.path.as_posix()}"

    return [
        {
            "module_id": module_id_for(summary),
            "path": summary.path.as_posix(),
            "language": summary.language,
            "functions": summary.functions[:8],
            "classes": summary.classes[:8],
            "imports": summary.imports[:8],
            "docstring_excerpts": _short_docstrings(summary),
            "semantic_labels": semantic_titles_by_module.get(
                module_id_for(summary), []
            ),
        }
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )[:_MAX_PROMPT_SUMMARIES]
    ]


def _semantic_labels_payload(
    semantic_structure_map: HistorySemanticStructureMap | None,
    semantic_context_map: HistorySemanticContextMap | None,
) -> dict[str, list[dict[str, object]]]:
    subsystem_labels: list[dict[str, object]] = []
    capability_labels: list[dict[str, object]] = []
    context_labels: list[dict[str, object]] = []
    if semantic_structure_map is not None:
        subsystem_labels = [
            {
                "semantic_subsystem_id": subsystem.semantic_subsystem_id,
                "title": subsystem.title,
                "summary": subsystem.summary,
                "capability_ids": subsystem.capability_ids,
            }
            for subsystem in sorted(
                semantic_structure_map.semantic_subsystems,
                key=lambda item: item.semantic_subsystem_id,
            )
        ]
        capability_labels = [
            {
                "capability_id": capability.capability_id,
                "title": capability.title,
                "summary": capability.summary,
            }
            for capability in sorted(
                semantic_structure_map.capabilities,
                key=lambda item: item.capability_id,
            )
        ]
    if semantic_context_map is not None:
        context_labels = [
            {
                "node_id": node.node_id,
                "title": node.title,
                "kind": node.kind,
            }
            for node in sorted(
                semantic_context_map.context_nodes, key=lambda item: item.node_id
            )
        ]
    return {
        "semantic_subsystems": subsystem_labels,
        "semantic_capabilities": capability_labels,
        "semantic_context_nodes": context_labels,
    }


def _significance_for_score(score: int) -> HistoryIntervalSignificance:
    if score >= 10:
        return "high"
    if score >= 6:
        return "medium"
    return "low"


def _fallback_insight_from_subsystem(
    change: HistorySubsystemChangeCandidate,
) -> HistoryDesignChangeInsight:
    title = f"Subsystem {change.status}: {change.group_path.as_posix()}"
    changed_symbols = ", ".join(change.changed_symbol_names[:4])
    summary = f"The {change.group_path.as_posix()} subsystem is {change.status} in this interval."
    if changed_symbols:
        summary = f"{summary} Changed symbols include {changed_symbols}."
    return HistoryDesignChangeInsight(
        insight_id=f"interval-insight::{_slug(_change_id(change))}",
        kind="subsystem_change",
        title=title,
        summary=summary,
        significance=_significance_for_score(
            len(change.commit_ids) + len(change.file_paths)
        ),
        related_commit_ids=sorted(change.commit_ids),
        related_change_ids=[_change_id(change)],
        related_subsystem_ids=[_change_id(change)],
        evidence_links=change.evidence_links,
    )


def _fallback_insight_from_interface(
    change: HistoryInterfaceChangeCandidate,
) -> HistoryDesignChangeInsight:
    symbol_label = (
        change.qualified_name or change.symbol_name or change.source_path.as_posix()
    )
    return HistoryDesignChangeInsight(
        insight_id=f"interval-insight::{_slug(_change_id(change))}",
        kind="interface_change",
        title=f"Interface {change.status}: {symbol_label}",
        summary=(
            f"The interface surface for {symbol_label} is {change.status} in this interval."
        ),
        significance=_significance_for_score(
            len(change.commit_ids) + len(change.signature_changes)
        ),
        related_commit_ids=sorted(change.commit_ids),
        related_change_ids=[_change_id(change)],
        related_subsystem_ids=[],
        evidence_links=change.evidence_links,
    )


def _fallback_insight_from_dependency(
    change: HistoryDependencyChangeCandidate,
) -> HistoryDesignChangeInsight:
    label = (
        change.path.as_posix()
        if change.path is not None
        else (change.subsystem_id or "dependency")
    )
    kind: HistoryIntervalInsightKind = (
        "build_change"
        if change.dependency_kind == "build_source"
        else "dependency_change"
    )
    return HistoryDesignChangeInsight(
        insight_id=f"interval-insight::{_slug(_change_id(change))}",
        kind=kind,
        title=f"{_title_case(kind.replace('_', ' '))}: {label}",
        summary=(
            f"{label} contributes a {change.status} {change.dependency_kind.replace('_', ' ')} signal in this interval."
        ),
        significance=_significance_for_score(
            len(change.commit_ids)
            + len(change.file_paths)
            + len(change.dependency_change_lines)
        ),
        related_commit_ids=sorted(change.commit_ids),
        related_change_ids=[_change_id(change)],
        related_subsystem_ids=(
            [change.subsystem_id] if change.subsystem_id is not None else []
        ),
        evidence_links=change.evidence_links,
    )


def _fallback_insight_from_algorithm(
    candidate: HistoryAlgorithmCandidate,
) -> HistoryDesignChangeInsight:
    label = candidate.scope_path.as_posix()
    summary = f"{label} carries algorithm-oriented signals: {', '.join(candidate.signal_kinds) or 'TBD'}."
    if candidate.variant_names:
        summary = f"{summary} Variant families include {', '.join(candidate.variant_names[:4])}."
    return HistoryDesignChangeInsight(
        insight_id=f"interval-insight::{_slug(candidate.candidate_id)}",
        kind="algorithm_change",
        title=f"Algorithm signal: {label}",
        summary=summary,
        significance=_significance_for_score(
            len(candidate.commit_ids)
            + len(candidate.changed_symbol_names)
            + len(candidate.variant_names)
        ),
        related_commit_ids=sorted(candidate.commit_ids),
        related_change_ids=[candidate.candidate_id],
        related_subsystem_ids=(
            [candidate.subsystem_id] if candidate.subsystem_id is not None else []
        ),
        evidence_links=candidate.evidence_links,
    )


def _heuristic_insights(
    delta_model: HistoryIntervalDeltaModel,
) -> list[HistoryDesignChangeInsight]:
    insights: list[HistoryDesignChangeInsight] = []
    for subsystem_change in delta_model.subsystem_changes:
        insights.append(_fallback_insight_from_subsystem(subsystem_change))
    for interface_change in delta_model.interface_changes:
        insights.append(_fallback_insight_from_interface(interface_change))
    for dependency_change in delta_model.dependency_changes:
        insights.append(_fallback_insight_from_dependency(dependency_change))
    for candidate in delta_model.algorithm_candidates:
        insights.append(_fallback_insight_from_algorithm(candidate))
    insights.sort(key=lambda item: (item.kind, item.title, item.insight_id))
    return insights[:_MAX_FALLBACK_INSIGHTS]


def _token_match(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in _RATIONALE_TOKENS)


def _heuristic_rationale_clues(
    delta_model: HistoryIntervalDeltaModel,
    snapshot: HistorySnapshotStructuralModel,
) -> list[HistoryRationaleClue]:
    clues: list[HistoryRationaleClue] = []
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()

    for commit_delta in delta_model.commit_deltas:
        subject = commit_delta.commit.subject.strip()
        if not subject or not _token_match(subject):
            continue
        key: tuple[str, str, tuple[str, ...], tuple[str, ...]] = (
            "commit_message",
            subject,
            (commit_delta.commit.sha,),
            (),
        )
        if key in seen:
            continue
        seen.add(key)
        clues.append(
            HistoryRationaleClue(
                clue_id=f"rationale-clue::commit::{_slug(commit_delta.commit.sha)}",
                text=subject,
                confidence=0.55,
                related_commit_ids=[commit_delta.commit.sha],
                related_change_ids=[],
                source_kind="commit_message",
                evidence_links=_dedupe_evidence(
                    [
                        HistoryEvidenceLink(
                            kind="commit", reference=commit_delta.commit.sha
                        )
                    ]
                ),
            )
        )

    for change in delta_model.interface_changes:
        for line in change.signature_changes:
            text = line.strip()
            if not text or not _token_match(text):
                continue
            key = (
                "signature_change",
                text,
                tuple(change.commit_ids),
                (_change_id(change),),
            )
            if key in seen:
                continue
            seen.add(key)
            clues.append(
                HistoryRationaleClue(
                    clue_id=f"rationale-clue::signature::{_slug(_change_id(change))}",
                    text=text,
                    confidence=0.65,
                    related_commit_ids=sorted(change.commit_ids),
                    related_change_ids=[_change_id(change)],
                    source_kind="signature_change",
                    evidence_links=change.evidence_links,
                )
            )

    for summary in sorted(
        snapshot.code_summaries, key=lambda item: item.path.as_posix()
    ):
        for docstring in summary.docstrings:
            text = " ".join(docstring.split())
            if not text or not _token_match(text):
                continue
            key = ("docstring", text, (), (summary.path.as_posix(),))
            if key in seen:
                continue
            seen.add(key)
            clues.append(
                HistoryRationaleClue(
                    clue_id=f"rationale-clue::docstring::{_slug(summary.path.as_posix())}",
                    text=text[:220],
                    confidence=0.6,
                    related_commit_ids=[],
                    related_change_ids=[],
                    source_kind="docstring",
                    evidence_links=_dedupe_evidence(
                        [
                            HistoryEvidenceLink(
                                kind="file", reference=summary.path.as_posix()
                            )
                        ]
                    ),
                )
            )
            break

    clues.sort(key=lambda item: (item.source_kind, item.text, item.clue_id))
    return clues[:_MAX_FALLBACK_CLUES]


def _commit_windows(
    delta_model: HistoryIntervalDeltaModel,
) -> list[list[HistoryCommitDelta]]:
    if not delta_model.commit_deltas:
        return []
    chronological = list(delta_model.commit_deltas)
    scores = [_commit_strength(item) for item in chronological]
    threshold = max(6, sorted(scores, reverse=True)[min(len(scores) - 1, 2)])
    strong_indexes = [index for index, score in enumerate(scores) if score >= threshold]
    if not strong_indexes:
        strong_indexes = [scores.index(max(scores))]
    windows: list[list[HistoryCommitDelta]] = []
    current: list[HistoryCommitDelta] = []
    previous_index: int | None = None
    for index in strong_indexes:
        if previous_index is None or index == previous_index + 1:
            current.append(chronological[index])
        else:
            windows.append(current)
            current = [chronological[index]]
        previous_index = index
    if current:
        windows.append(current)
    return windows


def _heuristic_windows(
    delta_model: HistoryIntervalDeltaModel,
    insights: list[HistoryDesignChangeInsight],
) -> list[HistorySignificantChangeWindow]:
    insight_ids_by_commit: dict[str, list[str]] = defaultdict(list)
    for insight in insights:
        for commit_id in insight.related_commit_ids:
            insight_ids_by_commit[commit_id].append(insight.insight_id)

    windows: list[HistorySignificantChangeWindow] = []
    for index, group in enumerate(_commit_windows(delta_model), start=1):
        commit_ids = [item.commit.sha for item in group]
        titles = [item.commit.subject for item in group]
        signal_kinds = sorted(
            {signal for item in group for signal in item.signal_kinds}
        )
        evidence_links = _dedupe_evidence(*(item.evidence_links for item in group))
        max_score = max(_commit_strength(item) for item in group)
        windows.append(
            HistorySignificantChangeWindow(
                window_id=f"change-window::{index:02d}",
                start_commit=group[0].commit.sha,
                end_commit=group[-1].commit.sha,
                commit_ids=commit_ids,
                title=titles[0] if len(titles) == 1 else f"{titles[0]} to {titles[-1]}",
                summary=(
                    "This interval window concentrates "
                    f"{', '.join(signal_kinds) or 'logic'} signals across {len(commit_ids)} commit(s)."
                ),
                significance=_significance_for_score(max_score),
                related_insight_ids=sorted(
                    {
                        insight_id
                        for commit_id in commit_ids
                        for insight_id in insight_ids_by_commit.get(commit_id, [])
                    }
                ),
                evidence_links=evidence_links,
            )
        )
    return windows


def _fallback_interval_interpretation(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    delta_model: HistoryIntervalDeltaModel,
    snapshot: HistorySnapshotStructuralModel,
    status: HistoryIntervalInterpretationStatus,
) -> HistoryIntervalInterpretation:
    insights = _heuristic_insights(delta_model)
    clues = _heuristic_rationale_clues(delta_model, snapshot)
    windows = _heuristic_windows(delta_model, insights)
    return HistoryIntervalInterpretation(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        insights=insights,
        rationale_clues=clues,
        significant_windows=windows,
    )


def _known_change_ids(delta_model: HistoryIntervalDeltaModel) -> set[str]:
    return {
        *(_change_id(change) for change in delta_model.subsystem_changes),
        *(_change_id(change) for change in delta_model.interface_changes),
        *(_change_id(change) for change in delta_model.dependency_changes),
        *(candidate.candidate_id for candidate in delta_model.algorithm_candidates),
    }


def _known_subsystem_ids(
    delta_model: HistoryIntervalDeltaModel,
    snapshot: HistorySnapshotStructuralModel,
    semantic_structure_map: HistorySemanticStructureMap | None,
) -> set[str]:
    flattened: set[str] = {
        candidate.candidate_id for candidate in snapshot.subsystem_candidates
    }
    for commit_delta in delta_model.commit_deltas:
        flattened.update(commit_delta.affected_subsystem_ids)
    flattened.update(_change_id(change) for change in delta_model.subsystem_changes)
    if semantic_structure_map is not None:
        flattened.update(
            subsystem.semantic_subsystem_id
            for subsystem in semantic_structure_map.semantic_subsystems
        )
    return flattened


def _validate_interpretation(
    interpretation: HistoryIntervalInterpretationJudgment,
    *,
    known_commit_ids: set[str],
    known_change_ids: set[str],
    known_subsystem_ids: set[str],
    known_evidence_links: set[tuple[str, str]],
) -> None:
    insight_ids = {insight.insight_id for insight in interpretation.insights}
    for insight in interpretation.insights:
        if not insight.title.strip() or not insight.summary.strip():
            raise ValueError("interval insights must have non-empty title and summary")
        if not set(insight.related_commit_ids) <= known_commit_ids:
            raise ValueError("interval insight referenced unknown commit ids")
        if not set(insight.related_change_ids) <= known_change_ids:
            raise ValueError("interval insight referenced unknown change ids")
        if not set(insight.related_subsystem_ids) <= known_subsystem_ids:
            raise ValueError("interval insight referenced unknown subsystem ids")
        if (
            not {(link.kind, link.reference) for link in insight.evidence_links}
            <= known_evidence_links
        ):
            raise ValueError("interval insight referenced unknown evidence links")
    for clue in interpretation.rationale_clues:
        if not clue.text.strip():
            raise ValueError("rationale clues must have non-empty text")
        if not set(clue.related_commit_ids) <= known_commit_ids:
            raise ValueError("rationale clue referenced unknown commit ids")
        if not set(clue.related_change_ids) <= known_change_ids:
            raise ValueError("rationale clue referenced unknown change ids")
        if (
            not {(link.kind, link.reference) for link in clue.evidence_links}
            <= known_evidence_links
        ):
            raise ValueError("rationale clue referenced unknown evidence links")
    for window in interpretation.significant_windows:
        if not window.title.strip() or not window.summary.strip():
            raise ValueError(
                "significant windows must have non-empty title and summary"
            )
        if (
            window.start_commit not in known_commit_ids
            or window.end_commit not in known_commit_ids
        ):
            raise ValueError("significant window referenced unknown boundary commits")
        if not set(window.commit_ids) <= known_commit_ids:
            raise ValueError("significant window referenced unknown commit ids")
        if not set(window.related_insight_ids) <= insight_ids:
            raise ValueError("significant window referenced unknown insight ids")
        if (
            not {(link.kind, link.reference) for link in window.evidence_links}
            <= known_evidence_links
        ):
            raise ValueError("significant window referenced unknown evidence links")


def _materialize_interpretation(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    judgment: HistoryIntervalInterpretationJudgment,
    status: HistoryIntervalInterpretationStatus,
) -> HistoryIntervalInterpretation:
    insights = sorted(
        judgment.insights, key=lambda item: (item.kind, item.title, item.insight_id)
    )
    clues = sorted(
        judgment.rationale_clues,
        key=lambda item: (item.source_kind, item.text, item.clue_id),
    )
    windows = sorted(
        judgment.significant_windows,
        key=lambda item: (item.start_commit, item.end_commit, item.window_id),
    )
    return HistoryIntervalInterpretation(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        insights=insights,
        rationale_clues=clues,
        significant_windows=windows,
    )


def build_interval_interpretation(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    snapshot: HistorySnapshotStructuralModel,
    delta_model: HistoryIntervalDeltaModel,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
    semantic_structure_map: HistorySemanticStructureMap | None = None,
    semantic_context_map: HistorySemanticContextMap | None = None,
) -> HistoryIntervalInterpretation:
    """Build the H12-01 interval interpretation artifact for one checkpoint."""

    if llm_client is None:
        return _fallback_interval_interpretation(
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            delta_model=delta_model,
            snapshot=snapshot,
            status="heuristic_only",
        )

    system_prompt, user_prompt = build_interval_interpretation_prompt(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        commit_deltas=_sorted_prompt_commits(delta_model),
        candidates=_candidate_payloads(
            subsystem_changes=delta_model.subsystem_changes,
            interface_changes=delta_model.interface_changes,
            dependency_changes=delta_model.dependency_changes,
            algorithm_candidates=delta_model.algorithm_candidates,
        ),
        modules=_module_payloads(snapshot, semantic_structure_map),
        semantic_labels=_semantic_labels_payload(
            semantic_structure_map,
            semantic_context_map,
        ),
    )

    known_commit_ids = {item.commit.sha for item in delta_model.commit_deltas}
    known_change_ids = _known_change_ids(delta_model)
    known_subsystem_ids = _known_subsystem_ids(
        delta_model,
        snapshot,
        semantic_structure_map,
    )
    known_evidence_links = _collect_known_evidence_links(
        snapshot=snapshot,
        delta_model=delta_model,
        semantic_structure_map=semantic_structure_map,
        semantic_context_map=semantic_context_map,
    )

    try:
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryIntervalInterpretationJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistoryIntervalInterpretationJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        _validate_interpretation(
            judgment,
            known_commit_ids=known_commit_ids,
            known_change_ids=known_change_ids,
            known_subsystem_ids=known_subsystem_ids,
            known_evidence_links=known_evidence_links,
        )
        if not (
            judgment.insights
            or judgment.rationale_clues
            or judgment.significant_windows
        ):
            return _fallback_interval_interpretation(
                checkpoint_id=checkpoint_id,
                target_commit=target_commit,
                previous_checkpoint_commit=previous_checkpoint_commit,
                delta_model=delta_model,
                snapshot=snapshot,
                status="heuristic_only",
            )
        return _materialize_interpretation(
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            judgment=judgment,
            status="scored",
        )
    except Exception:
        return _fallback_interval_interpretation(
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            delta_model=delta_model,
            snapshot=snapshot,
            status="llm_failed",
        )


__all__ = ["build_interval_interpretation", "interval_interpretation_path"]
