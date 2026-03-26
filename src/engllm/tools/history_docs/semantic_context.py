"""H11-03 semantic context and interface helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from engllm.domain.models import CodeUnitSummary
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_semantic_context_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryEvidenceLink,
    HistorySectionDepth,
    HistorySectionOutline,
    HistorySectionPlan,
    HistorySectionPlanId,
    HistorySemanticContextJudgment,
    HistorySemanticContextMap,
    HistorySemanticContextStatus,
    HistorySemanticInterfaceCandidate,
    HistorySemanticInterfaceKind,
    HistorySemanticStructureMap,
    HistorySnapshotStructuralModel,
    HistorySystemContextNode,
)

_HTTP_TOKENS = (
    "api",
    "router",
    "route",
    "controller",
    "endpoint",
    "request",
    "response",
)
_CLI_TOKENS = ("cli", "command", "argparse", "click", "main")
_DATA_CONTRACT_TOKENS = ("request", "response", "model", "dto", "schema")
_LIBRARY_SURFACE_TOKENS = ("public", "exports", "export", "client", "sdk")
_DATA_STORE_TOKENS = (
    "store",
    "storage",
    "repository",
    "database",
    "db",
    "cache",
    "queue",
)
_CONTEXT_SECTION_ORDER: tuple[HistorySectionPlanId, ...] = (
    "introduction",
    "architectural_overview",
    "system_context",
    "subsystems_modules",
    "interfaces",
    "algorithms_core_logic",
    "dependencies",
    "build_development_infrastructure",
    "strategy_variants_design_alternatives",
    "data_state_management",
    "error_handling_robustness",
    "performance_considerations",
    "security_considerations",
    "design_notes_rationale",
    "limitations_constraints",
)
_SECTION_TITLES = {
    "system_context": "System Context",
    "interfaces": "Interfaces",
}


def semantic_context_map_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H11-03 semantic context artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "semantic_context_map.json"


def load_semantic_context_map(path: Path) -> HistorySemanticContextMap | None:
    """Load one semantic context artifact if it exists."""

    if not path.exists():
        return None
    return HistorySemanticContextMap.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _module_concept_id(path: Path) -> str:
    return f"module::{path.as_posix()}"


def _evidence_sort_key(link: HistoryEvidenceLink) -> tuple[str, str, str]:
    return (link.kind, link.reference, link.detail or "")


def _dedupe_evidence(*groups: list[HistoryEvidenceLink]) -> list[HistoryEvidenceLink]:
    deduped: dict[tuple[str, str, str | None], HistoryEvidenceLink] = {}
    for group in groups:
        for link in group:
            deduped[(link.kind, link.reference, link.detail)] = link
    return sorted(deduped.values(), key=_evidence_sort_key)


def _title_case(value: str) -> str:
    return (
        " ".join(part.capitalize() for part in value.replace("-", " ").split())
        or "Root"
    )


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in cleaned.split("-") if part) or "item"


def _terms(summary: CodeUnitSummary) -> list[str]:
    return [
        summary.path.as_posix().lower(),
        *(value.lower() for value in summary.functions),
        *(value.lower() for value in summary.classes),
        *(value.lower() for value in summary.imports),
        *(value.lower() for value in summary.docstrings),
    ]


def _short_docstrings(summary: CodeUnitSummary) -> list[str]:
    return [value.strip()[:160] for value in summary.docstrings[:2] if value.strip()]


def _module_to_semantic_subsystems(
    semantic_map: HistorySemanticStructureMap,
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for subsystem in semantic_map.semantic_subsystems:
        for module_id in subsystem.module_ids:
            mapping[module_id].append(subsystem.semantic_subsystem_id)
    return {key: sorted(value) for key, value in sorted(mapping.items())}


def _semantic_subsystem_payloads(
    semantic_map: HistorySemanticStructureMap,
) -> list[dict[str, object]]:
    return [
        {
            "semantic_subsystem_id": subsystem.semantic_subsystem_id,
            "title": subsystem.title,
            "summary": subsystem.summary,
            "module_ids": subsystem.module_ids,
            "capability_ids": subsystem.capability_ids,
            "baseline_subsystem_candidate_ids": subsystem.baseline_subsystem_candidate_ids,
            "representative_files": [
                path.as_posix() for path in subsystem.representative_files
            ],
        }
        for subsystem in sorted(
            semantic_map.semantic_subsystems,
            key=lambda item: item.semantic_subsystem_id,
        )
    ]


def _module_payloads(
    snapshot: HistorySnapshotStructuralModel,
    *,
    semantic_membership: dict[str, list[str]],
) -> list[dict[str, object]]:
    symbol_names_by_path: dict[Path, list[str]] = defaultdict(list)
    for symbol in snapshot.symbol_summaries:
        name = symbol.qualified_name or symbol.name
        if name not in symbol_names_by_path[symbol.source_path]:
            symbol_names_by_path[symbol.source_path].append(name)
    return [
        {
            "module_id": _module_concept_id(summary.path),
            "path": summary.path.as_posix(),
            "language": summary.language,
            "semantic_subsystem_ids": semantic_membership.get(
                _module_concept_id(summary.path), []
            ),
            "functions": summary.functions[:8],
            "classes": summary.classes[:8],
            "imports": summary.imports[:8],
            "symbol_names": symbol_names_by_path.get(summary.path, [])[:12],
            "docstring_excerpts": _short_docstrings(summary),
        }
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
    ]


def _build_source_payloads(
    snapshot: HistorySnapshotStructuralModel,
) -> list[dict[str, object]]:
    return [
        {
            "path": source.path.as_posix(),
            "ecosystem": source.ecosystem,
            "category": source.category,
        }
        for source in snapshot.build_sources
    ]


def _fallback_context_nodes(
    *,
    workspace_id: str,
    snapshot: HistorySnapshotStructuralModel,
    semantic_map: HistorySemanticStructureMap,
) -> list[HistorySystemContextNode]:
    active_subsystem_ids = sorted(
        subsystem.semantic_subsystem_id
        for subsystem in semantic_map.semantic_subsystems
    )
    module_ids = sorted(
        _module_concept_id(summary.path) for summary in snapshot.code_summaries
    )
    nodes = [
        HistorySystemContextNode(
            node_id="context-node::system",
            title=f"{workspace_id} System",
            kind="system",
            summary="Represents the current checkpointed system boundary.",
            related_subsystem_ids=active_subsystem_ids,
            related_module_ids=module_ids[:12],
            evidence_links=_dedupe_evidence(
                [
                    *[
                        HistoryEvidenceLink(kind="subsystem", reference=subsystem_id)
                        for subsystem_id in active_subsystem_ids
                    ],
                    *[
                        HistoryEvidenceLink(
                            kind="file", reference=summary.path.as_posix()
                        )
                        for summary in sorted(
                            snapshot.code_summaries,
                            key=lambda item: item.path.as_posix(),
                        )[:12]
                    ],
                ]
            ),
        )
    ]

    storage_modules = [
        summary
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
        if any(
            token in term for term in _terms(summary) for token in _DATA_STORE_TOKENS
        )
    ]
    if storage_modules:
        related_module_ids = [
            _module_concept_id(summary.path) for summary in storage_modules
        ]
        related_subsystem_ids = sorted(
            {
                subsystem_id
                for module_id in related_module_ids
                for subsystem_id in next(
                    (
                        subsystem.module_ids and [subsystem.semantic_subsystem_id]
                        for subsystem in semantic_map.semantic_subsystems
                        if module_id in subsystem.module_ids
                    ),
                    [],
                )
            }
        )
        nodes.append(
            HistorySystemContextNode(
                node_id="context-node::data-store",
                title="Data Stores",
                kind="data_store",
                summary="Covers repository, storage, cache, or queue modules with explicit data-state responsibilities.",
                related_subsystem_ids=related_subsystem_ids,
                related_module_ids=sorted(related_module_ids),
                evidence_links=_dedupe_evidence(
                    [
                        *[
                            HistoryEvidenceLink(
                                kind="subsystem", reference=subsystem_id
                            )
                            for subsystem_id in related_subsystem_ids
                        ],
                        *[
                            HistoryEvidenceLink(
                                kind="file", reference=summary.path.as_posix()
                            )
                            for summary in storage_modules
                        ],
                    ]
                ),
            )
        )

    return nodes


def _fallback_interfaces(
    *,
    snapshot: HistorySnapshotStructuralModel,
    semantic_map: HistorySemanticStructureMap,
) -> list[HistorySemanticInterfaceCandidate]:
    semantic_membership = _module_to_semantic_subsystems(semantic_map)
    interfaces: list[HistorySemanticInterfaceCandidate] = []
    seen_ids: set[str] = set()
    for summary in sorted(
        snapshot.code_summaries, key=lambda item: item.path.as_posix()
    ):
        module_id = _module_concept_id(summary.path)
        terms = _terms(summary)
        interface_kind: HistorySemanticInterfaceKind | None = None
        title = ""
        if any(token in term for term in terms for token in _HTTP_TOKENS):
            interface_kind = "http_api"
            title = f"HTTP API: {_title_case(summary.path.stem)}"
        elif any(token in term for term in terms for token in _CLI_TOKENS):
            interface_kind = "cli_surface"
            title = f"CLI Surface: {_title_case(summary.path.stem)}"
        elif any(token in term for term in terms for token in _DATA_CONTRACT_TOKENS):
            interface_kind = "data_contract"
            title = f"Data Contract: {_title_case(summary.path.stem)}"
        elif summary.path.name == "__init__.py" or any(
            token in term for term in terms for token in _LIBRARY_SURFACE_TOKENS
        ):
            interface_kind = "library_surface"
            title = f"Library Surface: {_title_case(summary.path.stem)}"

        if interface_kind is None:
            continue
        interface_id = f"interface::{interface_kind}::{_slug(summary.path.as_posix())}"
        if interface_id in seen_ids:
            continue
        seen_ids.add(interface_id)
        provider_subsystem_ids = semantic_membership.get(module_id, [])
        interfaces.append(
            HistorySemanticInterfaceCandidate(
                interface_id=interface_id,
                title=title,
                kind=interface_kind,
                summary=(
                    f"Evidence in `{summary.path.as_posix()}` suggests a current {interface_kind.replace('_', ' ')} boundary."
                ),
                provider_subsystem_ids=provider_subsystem_ids,
                consumer_context_node_ids=[],
                related_module_ids=[module_id],
                evidence_links=_dedupe_evidence(
                    [
                        *[
                            HistoryEvidenceLink(
                                kind="subsystem", reference=subsystem_id
                            )
                            for subsystem_id in provider_subsystem_ids
                        ],
                        HistoryEvidenceLink(
                            kind="file", reference=summary.path.as_posix()
                        ),
                    ]
                ),
            )
        )
    return interfaces


def _fallback_semantic_context_map(
    *,
    workspace_id: str,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    snapshot: HistorySnapshotStructuralModel,
    semantic_map: HistorySemanticStructureMap,
    status: HistorySemanticContextStatus,
) -> HistorySemanticContextMap:
    return HistorySemanticContextMap(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        context_nodes=_fallback_context_nodes(
            workspace_id=workspace_id,
            snapshot=snapshot,
            semantic_map=semantic_map,
        ),
        interfaces=_fallback_interfaces(snapshot=snapshot, semantic_map=semantic_map),
    )


def _validate_context_judgment(
    judgment: HistorySemanticContextJudgment,
    *,
    known_module_ids: set[str],
    known_subsystem_ids: set[str],
) -> None:
    if sum(node.kind == "system" for node in judgment.context_nodes) != 1:
        raise ValueError(
            "semantic context judgment must contain exactly one system node"
        )
    node_ids = {node.node_id for node in judgment.context_nodes}
    for node in judgment.context_nodes:
        if not node.title.strip() or not node.summary.strip():
            raise ValueError(
                "semantic context nodes must have non-empty title and summary"
            )
        if not set(node.related_module_ids) <= known_module_ids:
            raise ValueError("semantic context node referenced unknown module ids")
        if not set(node.related_subsystem_ids) <= known_subsystem_ids:
            raise ValueError("semantic context node referenced unknown subsystem ids")
    for interface in judgment.interfaces:
        if not interface.title.strip() or not interface.summary.strip():
            raise ValueError(
                "semantic interfaces must have non-empty title and summary"
            )
        if not set(interface.related_module_ids) <= known_module_ids:
            raise ValueError("semantic interface referenced unknown module ids")
        if not set(interface.provider_subsystem_ids) <= known_subsystem_ids:
            raise ValueError(
                "semantic interface referenced unknown provider subsystem ids"
            )
        if not set(interface.consumer_context_node_ids) <= node_ids:
            raise ValueError("semantic interface referenced unknown context node ids")


def _materialize_context_map(
    *,
    judgment: HistorySemanticContextJudgment,
    snapshot: HistorySnapshotStructuralModel,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    status: HistorySemanticContextStatus,
) -> HistorySemanticContextMap:
    modules_by_id = {
        _module_concept_id(summary.path): summary
        for summary in sorted(
            snapshot.code_summaries, key=lambda item: item.path.as_posix()
        )
    }
    context_nodes = [
        HistorySystemContextNode(
            node_id=node.node_id,
            title=node.title.strip(),
            kind=node.kind,
            summary=node.summary.strip(),
            related_subsystem_ids=sorted(set(node.related_subsystem_ids)),
            related_module_ids=sorted(set(node.related_module_ids)),
            evidence_links=_dedupe_evidence(
                [
                    *[
                        HistoryEvidenceLink(kind="subsystem", reference=subsystem_id)
                        for subsystem_id in sorted(set(node.related_subsystem_ids))
                    ],
                    *[
                        HistoryEvidenceLink(
                            kind="file",
                            reference=modules_by_id[module_id].path.as_posix(),
                        )
                        for module_id in sorted(set(node.related_module_ids))
                        if module_id in modules_by_id
                    ],
                ]
            ),
        )
        for node in sorted(judgment.context_nodes, key=lambda item: item.node_id)
    ]
    interfaces = [
        HistorySemanticInterfaceCandidate(
            interface_id=interface.interface_id,
            title=interface.title.strip(),
            kind=interface.kind,
            summary=interface.summary.strip(),
            provider_subsystem_ids=sorted(set(interface.provider_subsystem_ids)),
            consumer_context_node_ids=sorted(set(interface.consumer_context_node_ids)),
            related_module_ids=sorted(set(interface.related_module_ids)),
            evidence_links=_dedupe_evidence(
                [
                    *[
                        HistoryEvidenceLink(kind="subsystem", reference=subsystem_id)
                        for subsystem_id in sorted(
                            set(interface.provider_subsystem_ids)
                        )
                    ],
                    *[
                        HistoryEvidenceLink(
                            kind="file",
                            reference=modules_by_id[module_id].path.as_posix(),
                        )
                        for module_id in sorted(set(interface.related_module_ids))
                        if module_id in modules_by_id
                    ],
                ]
            ),
        )
        for interface in sorted(judgment.interfaces, key=lambda item: item.interface_id)
    ]
    return HistorySemanticContextMap(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=status,
        context_nodes=context_nodes,
        interfaces=interfaces,
    )


def build_semantic_context_map(
    *,
    workspace_id: str,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    snapshot: HistorySnapshotStructuralModel,
    semantic_map: HistorySemanticStructureMap,
    llm_client: LLMClient | None,
    model_name: str,
    temperature: float,
) -> HistorySemanticContextMap:
    """Build the H11-03 semantic context artifact for one snapshot."""

    if llm_client is None:
        return _fallback_semantic_context_map(
            workspace_id=workspace_id,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            snapshot=snapshot,
            semantic_map=semantic_map,
            status="heuristic_only",
        )

    semantic_membership = _module_to_semantic_subsystems(semantic_map)
    known_module_ids = set(
        _module_concept_id(summary.path) for summary in snapshot.code_summaries
    )
    known_subsystem_ids = {
        subsystem.semantic_subsystem_id
        for subsystem in semantic_map.semantic_subsystems
    }
    system_prompt, user_prompt = build_semantic_context_prompt(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        semantic_subsystems=_semantic_subsystem_payloads(semantic_map),
        modules=_module_payloads(snapshot, semantic_membership=semantic_membership),
        build_sources=_build_source_payloads(snapshot),
    )
    try:
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistorySemanticContextJudgment,
                model_name=model_name,
                temperature=temperature,
            )
        )
        judgment = HistorySemanticContextJudgment.model_validate(
            response.content.model_dump(mode="python")
        )
        _validate_context_judgment(
            judgment,
            known_module_ids=known_module_ids,
            known_subsystem_ids=known_subsystem_ids,
        )
        return _materialize_context_map(
            judgment=judgment,
            snapshot=snapshot,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            status="scored",
        )
    except Exception:
        return _fallback_semantic_context_map(
            workspace_id=workspace_id,
            checkpoint_id=checkpoint_id,
            target_commit=target_commit,
            previous_checkpoint_commit=previous_checkpoint_commit,
            snapshot=snapshot,
            semantic_map=semantic_map,
            status="llm_failed",
        )


def _concept_ids_for_modules(
    checkpoint_model: HistoryCheckpointModel,
    module_ids: set[str],
) -> list[str]:
    active_modules = {
        module.concept_id: module
        for module in checkpoint_model.modules
        if module.lifecycle_status == "active"
    }
    active_subsystem_ids = {
        subsystem.concept_id
        for subsystem in checkpoint_model.subsystems
        if subsystem.lifecycle_status == "active"
    }
    concept_ids: set[str] = set()
    for module_id in module_ids:
        module = active_modules.get(module_id)
        if module is None:
            continue
        concept_ids.add(module.concept_id)
        if module.subsystem_id in active_subsystem_ids:
            concept_ids.add(module.subsystem_id)
    return sorted(
        concept_ids,
        key=lambda item: (
            (0 if item.startswith("subsystem::") else 1),
            item,
        ),
    )


def _interface_concept_ids(
    checkpoint_model: HistoryCheckpointModel,
    semantic_context_map: HistorySemanticContextMap,
) -> list[str]:
    module_ids: set[str] = set()
    for interface in semantic_context_map.interfaces:
        module_ids.update(interface.related_module_ids)
    return _concept_ids_for_modules(checkpoint_model, module_ids)


def _depth_for_optional_score(score: int) -> HistorySectionDepth | None:
    if score < 5:
        return None
    if score <= 6:
        return "brief"
    if score <= 8:
        return "standard"
    return "deep"


def augment_section_outline_with_semantic_context(
    *,
    checkpoint_model: HistoryCheckpointModel,
    section_outline: HistorySectionOutline,
    semantic_context_map: HistorySemanticContextMap,
) -> HistorySectionOutline:
    """Insert candidate-only context/interface sections into an outline."""

    active_subsystem_count = sum(
        subsystem.lifecycle_status == "active"
        for subsystem in checkpoint_model.subsystems
    )
    has_system_node = any(
        node.kind == "system" for node in semantic_context_map.context_nodes
    )
    non_system_nodes = [
        node for node in semantic_context_map.context_nodes if node.kind != "system"
    ]

    system_context_score = 4
    if non_system_nodes:
        system_context_score += min(len(non_system_nodes), 2)
    if active_subsystem_count >= 2:
        system_context_score += 1
    if semantic_context_map.interfaces:
        system_context_score += 1
    system_context_included = has_system_node and (
        bool(non_system_nodes) or active_subsystem_count >= 2
    )
    system_context_links = _dedupe_evidence(
        *[node.evidence_links for node in semantic_context_map.context_nodes]
    )
    system_context_section = HistorySectionPlan(
        section_id="system_context",
        title=_SECTION_TITLES["system_context"],
        kind="optional",
        status="included" if system_context_included else "omitted",
        confidence_score=min(100, system_context_score * 10),
        evidence_score=system_context_score,
        depth=(
            _depth_for_optional_score(system_context_score)
            if system_context_included
            else None
        ),
        concept_ids=_concept_ids_for_modules(
            checkpoint_model,
            {
                module_id
                for node in semantic_context_map.context_nodes
                for module_id in node.related_module_ids
            },
        ),
        evidence_links=system_context_links,
        trigger_signals=(
            ["active_subsystems", "interface_change"]
            if semantic_context_map.interfaces
            else ["active_subsystems"]
        ),
        omission_reason=None if system_context_included else "insufficient_evidence",
    )

    interface_score = 4 + min(len(semantic_context_map.interfaces), 3)
    if any(
        interface.provider_subsystem_ids
        for interface in semantic_context_map.interfaces
    ):
        interface_score += 1
    interface_included = bool(semantic_context_map.interfaces)
    interface_links = _dedupe_evidence(
        *[interface.evidence_links for interface in semantic_context_map.interfaces]
    )
    interfaces_section = HistorySectionPlan(
        section_id="interfaces",
        title=_SECTION_TITLES["interfaces"],
        kind="optional",
        status="included" if interface_included else "omitted",
        confidence_score=min(100, interface_score * 10),
        evidence_score=interface_score,
        depth=(
            _depth_for_optional_score(interface_score) if interface_included else None
        ),
        concept_ids=_interface_concept_ids(checkpoint_model, semantic_context_map),
        evidence_links=interface_links,
        trigger_signals=(
            ["active_modules", "interface_change"]
            if semantic_context_map.interfaces
            else []
        ),
        omission_reason=None if interface_included else "insufficient_evidence",
    )

    sections_by_id = {
        section.section_id: section for section in section_outline.sections
    }
    sections_by_id["system_context"] = system_context_section
    sections_by_id["interfaces"] = interfaces_section
    return HistorySectionOutline(
        checkpoint_id=section_outline.checkpoint_id,
        target_commit=section_outline.target_commit,
        previous_checkpoint_commit=section_outline.previous_checkpoint_commit,
        sections=[
            sections_by_id[section_id]
            for section_id in _CONTEXT_SECTION_ORDER
            if section_id in sections_by_id
        ],
    )


__all__ = [
    "augment_section_outline_with_semantic_context",
    "build_semantic_context_map",
    "load_semantic_context_map",
    "semantic_context_map_path",
]
