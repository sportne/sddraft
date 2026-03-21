"""Section-scoped deterministic evidence builders."""

from __future__ import annotations

from engllm.domain.models import (
    CodeUnitSummary,
    CommitImpact,
    CSCDescriptor,
    EvidenceReference,
    SDDSectionSpec,
    SDDTemplate,
    SectionEvidencePack,
    SymbolSummary,
)


def _has_kind(section_kinds: list[str], *candidates: str) -> bool:
    normalized = {kind.lower() for kind in section_kinds}
    return any(candidate in normalized for candidate in candidates)


def _slice_for_section(
    *,
    section_kinds: list[str],
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
) -> tuple[list[CodeUnitSummary], list[SymbolSummary], list[str]]:
    include_code = _has_kind(section_kinds, "code_summary", "code", "implementation")
    include_symbols = _has_kind(
        section_kinds,
        "symbols",
        "symbol",
        "interfaces",
        "interface",
    )
    include_dependencies = _has_kind(section_kinds, "dependencies", "dependency")

    return (
        code_summaries if include_code else [],
        symbol_summaries if include_symbols else [],
        dependencies if include_dependencies else [],
    )


def _build_references(
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
    hierarchy_summaries: list[str],
    commit_impact: CommitImpact | None,
    existing_text: str | None,
) -> list[EvidenceReference]:
    references: list[EvidenceReference] = []

    references.extend(
        EvidenceReference(kind="code_summary", source=summary.path.as_posix())
        for summary in code_summaries
    )
    references.extend(
        EvidenceReference(
            kind="symbol",
            source=item.source_path.as_posix(),
            detail=item.qualified_name or item.name,
        )
        for item in symbol_summaries
    )
    references.extend(
        EvidenceReference(kind="dependency", source=dependency)
        for dependency in dependencies
    )
    references.extend(
        EvidenceReference(kind="hierarchy_summary", source=summary)
        for summary in hierarchy_summaries
    )

    if commit_impact is not None:
        references.append(
            EvidenceReference(
                kind="commit_impact",
                source=commit_impact.commit_range,
                detail=commit_impact.summary,
            )
        )

    if existing_text is not None:
        references.append(
            EvidenceReference(
                kind="existing_section",
                source="existing_sdd",
                detail=existing_text[:120],
            )
        )

    return references


def build_generation_evidence_pack(
    *,
    section: SDDSectionSpec,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
    hierarchy_summaries: list[str] | None = None,
) -> SectionEvidencePack:
    """Build one generation evidence pack for a single section."""

    section_code, section_symbols, section_dependencies = _slice_for_section(
        section_kinds=section.evidence_kinds,
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        dependencies=dependencies,
    )
    hierarchy_values = hierarchy_summaries or []
    refs = _build_references(
        code_summaries=section_code,
        symbol_summaries=section_symbols,
        dependencies=section_dependencies,
        hierarchy_summaries=hierarchy_values,
        commit_impact=None,
        existing_text=None,
    )
    return SectionEvidencePack(
        section=section,
        csc=csc,
        code_summaries=section_code,
        symbol_summaries=section_symbols,
        dependency_summaries=section_dependencies,
        hierarchy_summaries=hierarchy_values,
        evidence_references=refs,
    )


def build_generation_evidence_packs(
    template: SDDTemplate,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
    hierarchy_summaries: list[str] | None = None,
) -> list[SectionEvidencePack]:
    """Build evidence packs for initial SDD generation."""

    packs: list[SectionEvidencePack] = []
    for section in template.sections:
        packs.append(
            build_generation_evidence_pack(
                section=section,
                csc=csc,
                code_summaries=code_summaries,
                symbol_summaries=symbol_summaries,
                dependencies=dependencies,
                hierarchy_summaries=hierarchy_summaries,
            )
        )
    return packs


def build_update_evidence_pack(
    *,
    section: SDDSectionSpec,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
    commit_impact: CommitImpact,
    existing_sections: dict[str, str],
) -> SectionEvidencePack | None:
    """Build one update evidence pack for a single section, if impacted."""

    impacted_titles = {item.lower() for item in commit_impact.impacted_sections}
    if section.title.lower() not in impacted_titles:
        return None

    section_code, section_symbols, section_dependencies = _slice_for_section(
        section_kinds=section.evidence_kinds,
        code_summaries=code_summaries,
        symbol_summaries=symbol_summaries,
        dependencies=dependencies,
    )
    if _has_kind(section.evidence_kinds, "commit_impact", "diff", "commit"):
        section_commit_impact = commit_impact
    else:
        section_commit_impact = None
    existing_text = existing_sections.get(section.id, "TBD")
    refs = _build_references(
        code_summaries=section_code,
        symbol_summaries=section_symbols,
        dependencies=section_dependencies,
        hierarchy_summaries=[],
        commit_impact=section_commit_impact,
        existing_text=existing_text,
    )
    return SectionEvidencePack(
        section=section,
        csc=csc,
        code_summaries=section_code,
        symbol_summaries=section_symbols,
        dependency_summaries=section_dependencies,
        commit_impact=section_commit_impact,
        existing_section_text=existing_text,
        evidence_references=refs,
    )


def build_update_evidence_packs(
    template: SDDTemplate,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    symbol_summaries: list[SymbolSummary],
    dependencies: list[str],
    commit_impact: CommitImpact,
    existing_sections: dict[str, str],
) -> list[SectionEvidencePack]:
    """Build evidence packs for impacted sections during update proposals."""

    packs: list[SectionEvidencePack] = []

    for section in template.sections:
        pack = build_update_evidence_pack(
            section=section,
            csc=csc,
            code_summaries=code_summaries,
            symbol_summaries=symbol_summaries,
            dependencies=dependencies,
            commit_impact=commit_impact,
            existing_sections=existing_sections,
        )
        if pack is not None:
            packs.append(pack)

    return packs
