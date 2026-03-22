"""H11 semantic subsystem/capability clustering helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from engllm.domain.models import CodeUnitSummary
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_semantic_structure_prompt
from engllm.tools.history_docs.models import (
    HistoryEvidenceLink,
    HistorySemanticCapabilityCluster,
    HistorySemanticStructureJudgment,
    HistorySemanticStructureMap,
    HistorySemanticStructureStatus,
    HistorySemanticSubsystemCluster,
    HistorySnapshotStructuralModel,
    HistorySubsystemCandidate,
)
from engllm.tools.history_docs.structure import (
    subsystem_candidate_id_for_file,
    subsystem_is_root_scope,
)

_MAX_REPRESENTATIVE_FILES = 12


@dataclass(frozen=True, slots=True)
class HistorySubsystemGroupingView:
    """Normalized subsystem grouping view used by downstream history-docs stages."""

    subsystem_candidates: list[HistorySubsystemCandidate]
    module_subsystem_ids: dict[Path, str]
    display_names: dict[str, str]
    summaries: dict[str, str | None]
    capability_labels: dict[str, list[str]]
    baseline_subsystem_ids: dict[str, list[str]]
    root_scope_subsystem_ids: set[str]


def semantic_structure_map_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H11 semantic-structure artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "semantic_structure_map.json"


def load_semantic_structure_map(path: Path) -> HistorySemanticStructureMap | None:
    """Load one semantic structure artifact if it exists."""

    if not path.exists():
        return None
    return HistorySemanticStructureMap.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _module_concept_id(path: Path) -> str:
    return f"module::{path.as_posix()}"


def _title_case(value: str) -> str:
    return (
        " ".join(part.capitalize() for part in value.replace("-", " ").split())
        or "Root"
    )


def _slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in value.strip()
    )
    normalized = "-".join(part for part in slug.split("-") if part)
    return normalized or "root"


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _symbol_names_by_path(
    snapshot: HistorySnapshotStructuralModel,
) -> dict[Path, list[str]]:
    by_path: dict[Path, set[str]] = defaultdict(set)
    for symbol in snapshot.symbol_summaries:
        by_path[symbol.source_path].add(symbol.qualified_name or symbol.name)
    return {
        path: sorted(names)
        for path, names in sorted(by_path.items(), key=lambda item: item[0].as_posix())
    }


def _baseline_module_membership(
    snapshot: HistorySnapshotStructuralModel,
) -> dict[str, str]:
    return {
        _module_concept_id(summary.path): subsystem_candidate_id_for_file(
            file_path=summary.path,
            analyzed_roots=snapshot.analyzed_source_roots,
        )
        for summary in snapshot.code_summaries
    }


def _heuristic_semantic_subsystem_id(candidate: HistorySubsystemCandidate) -> str:
    return candidate.candidate_id.replace("subsystem::", "semantic-subsystem::", 1)


def _baseline_display_name(candidate: HistorySubsystemCandidate) -> str:
    label = (
        candidate.source_root.as_posix()
        if candidate.group_path == candidate.source_root
        else candidate.group_path.name
    )
    return _title_case(label.replace("_", " "))


def _short_docstrings(summary: CodeUnitSummary) -> list[str]:
    return [value.strip()[:160] for value in summary.docstrings[:2] if value.strip()]


def _baseline_subsystem_payloads(
    snapshot: HistorySnapshotStructuralModel,
    module_membership: dict[str, str],
) -> list[dict[str, object]]:
    module_ids_by_subsystem: dict[str, list[str]] = defaultdict(list)
    for module_id, subsystem_id in sorted(module_membership.items()):
        module_ids_by_subsystem[subsystem_id].append(module_id)

    payloads: list[dict[str, object]] = []
    for candidate in sorted(
        snapshot.subsystem_candidates, key=lambda item: item.candidate_id
    ):
        payloads.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_root": candidate.source_root.as_posix(),
                "group_path": candidate.group_path.as_posix(),
                "title_hint": _baseline_display_name(candidate),
                "file_count": candidate.file_count,
                "symbol_count": candidate.symbol_count,
                "language_counts": candidate.language_counts,
                "representative_files": [
                    path.as_posix() for path in candidate.representative_files
                ],
                "module_ids": module_ids_by_subsystem.get(candidate.candidate_id, []),
            }
        )
    return payloads


def _module_payloads(
    snapshot: HistorySnapshotStructuralModel,
    *,
    module_membership: dict[str, str],
) -> list[dict[str, object]]:
    symbol_names = _symbol_names_by_path(snapshot)
    return [
        {
            "module_id": _module_concept_id(summary.path),
            "path": summary.path.as_posix(),
            "language": summary.language,
            "baseline_subsystem_candidate_id": module_membership[
                _module_concept_id(summary.path)
            ],
            "functions": summary.functions[:8],
            "classes": summary.classes[:8],
            "imports": summary.imports[:8],
            "symbol_names": symbol_names.get(summary.path, [])[:12],
            "docstring_excerpts": _short_docstrings(summary),
        }
        for summary in sorted(
            snapshot.code_summaries,
            key=lambda item: item.path.as_posix(),
        )
    ]


def _fallback_semantic_structure_map(
    *,
    snapshot: HistorySnapshotStructuralModel,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    status: HistorySemanticStructureStatus,
) -> HistorySemanticStructureMap:
    module_membership = _baseline_module_membership(snapshot)
    semantic_subsystems: list[HistorySemanticSubsystemCluster] = []
    for candidate in sorted(
        snapshot.subsystem_candidates, key=lambda item: item.candidate_id
    ):
        subsystem_module_ids = sorted(
            module_id
            for module_id, subsystem_id in module_membership.items()
            if subsystem_id == candidate.candidate_id
        )
        semantic_subsystems.append(
            HistorySemanticSubsystemCluster(
                semantic_subsystem_id=_heuristic_semantic_subsystem_id(candidate),
                title=_baseline_display_name(candidate),
                summary="",
                module_ids=subsystem_module_ids,
                baseline_subsystem_candidate_ids=[candidate.candidate_id],
                capability_ids=[],
                representative_files=candidate.representative_files,
                evidence_links=_dedupe_evidence(
                    [
                        HistoryEvidenceLink(
                            kind="subsystem",
                            reference=candidate.candidate_id,
                        ),
                        *[
                            HistoryEvidenceLink(kind="file", reference=path.as_posix())
                            for path in candidate.representative_files
                        ],
                    ]
                ),
            )
        )
    return HistorySemanticStructureMap(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        semantic_subsystems=semantic_subsystems,
        capabilities=[],
    )


def _validate_judgment(
    judgment: HistorySemanticStructureJudgment,
    *,
    known_module_ids: set[str],
) -> None:
    subsystem_ids = {
        subsystem.semantic_subsystem_id for subsystem in judgment.semantic_subsystems
    }
    assigned_modules: set[str] = set()
    for subsystem in judgment.semantic_subsystems:
        if not subsystem.module_ids:
            raise ValueError("semantic subsystem clusters must not be empty")
        module_ids = set(subsystem.module_ids)
        if not module_ids <= known_module_ids:
            raise ValueError("semantic subsystem cluster referenced unknown module ids")
        if assigned_modules & module_ids:
            raise ValueError(
                "semantic subsystem clusters must partition modules exclusively"
            )
        assigned_modules.update(module_ids)
    if assigned_modules != known_module_ids:
        raise ValueError("semantic subsystem clusters must cover all known module ids")

    capability_ids = {capability.capability_id for capability in judgment.capabilities}
    for subsystem in judgment.semantic_subsystems:
        if not set(subsystem.capability_ids) <= capability_ids:
            raise ValueError(
                "semantic subsystem cluster referenced unknown capability ids"
            )
    for capability in judgment.capabilities:
        if not set(capability.module_ids) <= known_module_ids:
            raise ValueError("capability cluster referenced unknown module ids")
        if not set(capability.semantic_subsystem_ids) <= subsystem_ids:
            raise ValueError(
                "capability cluster referenced unknown semantic subsystem ids"
            )


def _materialize_semantic_structure_map(
    *,
    judgment: HistorySemanticStructureJudgment,
    snapshot: HistorySnapshotStructuralModel,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    status: HistorySemanticStructureStatus,
) -> HistorySemanticStructureMap:
    module_membership = _baseline_module_membership(snapshot)
    modules_by_id = {
        _module_concept_id(summary.path): summary
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
    }
    capability_titles = {
        capability.capability_id: capability.title.strip() or capability.capability_id
        for capability in judgment.capabilities
    }

    semantic_subsystems: list[HistorySemanticSubsystemCluster] = []
    for subsystem in sorted(
        judgment.semantic_subsystems,
        key=lambda item: item.semantic_subsystem_id,
    ):
        module_ids = sorted(set(subsystem.module_ids))
        module_paths = [modules_by_id[module_id].path for module_id in module_ids]
        baseline_ids = sorted(
            {
                module_membership[module_id]
                for module_id in module_ids
                if module_id in module_membership
            }
        )
        representative_files = module_paths[:_MAX_REPRESENTATIVE_FILES]
        semantic_subsystems.append(
            HistorySemanticSubsystemCluster(
                semantic_subsystem_id=subsystem.semantic_subsystem_id,
                title=subsystem.title.strip() or subsystem.semantic_subsystem_id,
                summary=subsystem.summary.strip(),
                module_ids=module_ids,
                baseline_subsystem_candidate_ids=baseline_ids,
                capability_ids=sorted(
                    capability_id
                    for capability_id in set(subsystem.capability_ids)
                    if capability_id in capability_titles
                ),
                representative_files=representative_files,
                evidence_links=_dedupe_evidence(
                    [
                        *[
                            HistoryEvidenceLink(kind="subsystem", reference=item)
                            for item in baseline_ids
                        ],
                        *[
                            HistoryEvidenceLink(kind="file", reference=path.as_posix())
                            for path in representative_files
                        ],
                    ]
                ),
            )
        )

    capabilities: list[HistorySemanticCapabilityCluster] = []
    for capability in sorted(
        judgment.capabilities,
        key=lambda item: item.capability_id,
    ):
        module_ids = sorted(set(capability.module_ids))
        subsystem_ids = sorted(set(capability.semantic_subsystem_ids))
        capabilities.append(
            HistorySemanticCapabilityCluster(
                capability_id=capability.capability_id,
                title=capability.title.strip() or capability.capability_id,
                summary=capability.summary.strip(),
                module_ids=module_ids,
                semantic_subsystem_ids=subsystem_ids,
                evidence_links=_dedupe_evidence(
                    [
                        *[
                            HistoryEvidenceLink(
                                kind="file",
                                reference=modules_by_id[module_id].path.as_posix(),
                            )
                            for module_id in module_ids
                            if module_id in modules_by_id
                        ],
                        *[
                            HistoryEvidenceLink(
                                kind="subsystem", reference=subsystem_id
                            )
                            for subsystem_id in subsystem_ids
                        ],
                    ]
                ),
            )
        )

    return HistorySemanticStructureMap(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        semantic_subsystems=semantic_subsystems,
        capabilities=capabilities,
    )


def build_semantic_structure_map(
    *,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    snapshot: HistorySnapshotStructuralModel,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> HistorySemanticStructureMap:
    """Build the H11 semantic structure artifact for one snapshot."""

    if not snapshot.code_summaries:
        return HistorySemanticStructureMap(
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            evaluation_status="heuristic_only",
            semantic_subsystems=[],
            capabilities=[],
        )

    if llm_client is None:
        return _fallback_semantic_structure_map(
            snapshot=snapshot,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="heuristic_only",
        )

    module_membership = _baseline_module_membership(snapshot)
    system_prompt, user_prompt = build_semantic_structure_prompt(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        baseline_subsystems=_baseline_subsystem_payloads(snapshot, module_membership),
        modules=_module_payloads(snapshot, module_membership=module_membership),
    )
    known_module_ids = set(module_membership)
    try:
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistorySemanticStructureJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistorySemanticStructureJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        _validate_judgment(judgment, known_module_ids=known_module_ids)
        return _materialize_semantic_structure_map(
            judgment=judgment,
            snapshot=snapshot,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="scored",
        )
    except Exception:
        return _fallback_semantic_structure_map(
            snapshot=snapshot,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="llm_failed",
        )


def _synthetic_candidate_from_cluster(
    *,
    cluster: HistorySemanticSubsystemCluster,
    modules_by_id: dict[str, CodeUnitSummary],
    symbol_counts_by_path: Counter[Path],
    baseline_candidates: dict[str, HistorySubsystemCandidate],
) -> HistorySubsystemCandidate:
    module_ids = sorted(cluster.module_ids)
    module_paths = [modules_by_id[module_id].path for module_id in module_ids]
    language_counts: Counter[str] = Counter(
        modules_by_id[module_id].language for module_id in module_ids
    )
    source_roots = [
        baseline_candidates[item].source_root
        for item in cluster.baseline_subsystem_candidate_ids
        if item in baseline_candidates
    ]
    chosen_source_root = (
        sorted(
            source_roots,
            key=lambda item: (source_roots.count(item), item.as_posix()),
            reverse=True,
        )[0]
        if source_roots
        else Path(".")
    )
    if len(cluster.baseline_subsystem_candidate_ids) == 1:
        baseline_candidate = baseline_candidates.get(
            cluster.baseline_subsystem_candidate_ids[0]
        )
        chosen_group_path = (
            baseline_candidate.group_path
            if baseline_candidate is not None
            else Path(_slug(cluster.title))
        )
    else:
        group_leaf = Path(_slug(cluster.title))
        chosen_group_path = (
            group_leaf
            if chosen_source_root == Path(".")
            else chosen_source_root / group_leaf
        )

    return HistorySubsystemCandidate(
        candidate_id=cluster.semantic_subsystem_id,
        source_root=chosen_source_root,
        group_path=chosen_group_path,
        file_count=len(module_ids),
        symbol_count=sum(symbol_counts_by_path[path] for path in module_paths),
        language_counts={
            language: language_counts[language] for language in sorted(language_counts)
        },
        representative_files=module_paths[:_MAX_REPRESENTATIVE_FILES],
    )


def build_subsystem_grouping_view(
    *,
    snapshot: HistorySnapshotStructuralModel,
    mode: str,
    semantic_map: HistorySemanticStructureMap | None = None,
) -> HistorySubsystemGroupingView:
    """Return the normalized subsystem grouping view for one snapshot."""

    modules_by_id = {
        _module_concept_id(summary.path): summary
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
    }
    symbol_counts_by_path: Counter[Path] = Counter(
        symbol.source_path for symbol in snapshot.symbol_summaries
    )
    baseline_membership = _baseline_module_membership(snapshot)
    baseline_candidates = {
        candidate.candidate_id: candidate for candidate in snapshot.subsystem_candidates
    }

    if mode != "semantic":
        return HistorySubsystemGroupingView(
            subsystem_candidates=sorted(
                snapshot.subsystem_candidates,
                key=lambda item: item.candidate_id,
            ),
            module_subsystem_ids={
                summary.path: baseline_membership[_module_concept_id(summary.path)]
                for summary in snapshot.code_summaries
            },
            display_names={
                candidate.candidate_id: _baseline_display_name(candidate)
                for candidate in snapshot.subsystem_candidates
            },
            summaries={
                candidate.candidate_id: None
                for candidate in snapshot.subsystem_candidates
            },
            capability_labels={
                candidate.candidate_id: []
                for candidate in snapshot.subsystem_candidates
            },
            baseline_subsystem_ids={
                candidate.candidate_id: [candidate.candidate_id]
                for candidate in snapshot.subsystem_candidates
            },
            root_scope_subsystem_ids={
                candidate.candidate_id
                for candidate in snapshot.subsystem_candidates
                if subsystem_is_root_scope(candidate.candidate_id)
            },
        )

    effective_map = semantic_map or _fallback_semantic_structure_map(
        snapshot=snapshot,
        checkpoint_id=snapshot.checkpoint_id,
        target_commit=snapshot.target_commit,
        previous_checkpoint_commit=None,
        status="heuristic_only",
    )
    capability_titles = {
        capability.capability_id: capability.title
        for capability in effective_map.capabilities
    }
    synthetic_candidates: list[HistorySubsystemCandidate] = []
    for cluster in sorted(
        effective_map.semantic_subsystems,
        key=lambda item: item.semantic_subsystem_id,
    ):
        synthetic_candidates.append(
            _synthetic_candidate_from_cluster(
                cluster=cluster,
                modules_by_id=modules_by_id,
                symbol_counts_by_path=symbol_counts_by_path,
                baseline_candidates=baseline_candidates,
            )
        )
    module_subsystem_ids = {
        modules_by_id[module_id].path: cluster.semantic_subsystem_id
        for cluster in effective_map.semantic_subsystems
        for module_id in cluster.module_ids
        if module_id in modules_by_id
    }
    return HistorySubsystemGroupingView(
        subsystem_candidates=synthetic_candidates,
        module_subsystem_ids=module_subsystem_ids,
        display_names={
            cluster.semantic_subsystem_id: cluster.title
            for cluster in effective_map.semantic_subsystems
        },
        summaries={
            cluster.semantic_subsystem_id: (cluster.summary or None)
            for cluster in effective_map.semantic_subsystems
        },
        capability_labels={
            cluster.semantic_subsystem_id: sorted(
                capability_titles[capability_id]
                for capability_id in cluster.capability_ids
                if capability_id in capability_titles
            )
            for cluster in effective_map.semantic_subsystems
        },
        baseline_subsystem_ids={
            cluster.semantic_subsystem_id: sorted(
                cluster.baseline_subsystem_candidate_ids
            )
            for cluster in effective_map.semantic_subsystems
        },
        root_scope_subsystem_ids={
            cluster.semantic_subsystem_id
            for cluster in effective_map.semantic_subsystems
            if any(
                subsystem_is_root_scope(candidate_id)
                for candidate_id in cluster.baseline_subsystem_candidate_ids
            )
        },
    )


__all__ = [
    "HistorySubsystemGroupingView",
    "build_semantic_structure_map",
    "build_subsystem_grouping_view",
    "load_semantic_structure_map",
    "semantic_structure_map_path",
]
