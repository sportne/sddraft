"""Section-scoped deterministic evidence builders."""

from __future__ import annotations

from sddraft.domain.models import (
    CodeUnitSummary,
    CommitImpact,
    CSCDescriptor,
    EvidenceReference,
    InterfaceSummary,
    SDDTemplate,
    SectionEvidencePack,
)


def _build_references(
    code_summaries: list[CodeUnitSummary],
    interface_summaries: list[InterfaceSummary],
    dependencies: list[str],
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
            kind="interface", source=item.source_path.as_posix(), detail=item.name
        )
        for item in interface_summaries
    )
    references.extend(
        EvidenceReference(kind="dependency", source=dependency)
        for dependency in dependencies
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


def build_generation_evidence_packs(
    template: SDDTemplate,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    interface_summaries: list[InterfaceSummary],
    dependencies: list[str],
) -> list[SectionEvidencePack]:
    """Build evidence packs for initial SDD generation."""

    packs: list[SectionEvidencePack] = []
    for section in template.sections:
        refs = _build_references(
            code_summaries=code_summaries,
            interface_summaries=interface_summaries,
            dependencies=dependencies,
            commit_impact=None,
            existing_text=None,
        )
        packs.append(
            SectionEvidencePack(
                section=section,
                csc=csc,
                code_summaries=code_summaries,
                interface_summaries=interface_summaries,
                dependency_summaries=dependencies,
                evidence_references=refs,
            )
        )
    return packs


def build_update_evidence_packs(
    template: SDDTemplate,
    csc: CSCDescriptor,
    code_summaries: list[CodeUnitSummary],
    interface_summaries: list[InterfaceSummary],
    dependencies: list[str],
    commit_impact: CommitImpact,
    existing_sections: dict[str, str],
) -> list[SectionEvidencePack]:
    """Build evidence packs for impacted sections during update proposals."""

    impacted_titles = {section.lower() for section in commit_impact.impacted_sections}
    packs: list[SectionEvidencePack] = []

    for section in template.sections:
        if section.title.lower() not in impacted_titles:
            continue

        existing_text = existing_sections.get(section.id, "TBD")
        refs = _build_references(
            code_summaries=code_summaries,
            interface_summaries=interface_summaries,
            dependencies=dependencies,
            commit_impact=commit_impact,
            existing_text=existing_text,
        )

        packs.append(
            SectionEvidencePack(
                section=section,
                csc=csc,
                code_summaries=code_summaries,
                interface_summaries=interface_summaries,
                dependency_summaries=dependencies,
                commit_impact=commit_impact,
                existing_section_text=existing_text,
                evidence_references=refs,
            )
        )

    return packs
