"""Deterministic prompt builders."""

from __future__ import annotations

import json

from sddraft.domain.models import (
    CodeUnitSummary,
    DirectorySummaryDoc,
    FileSummaryDoc,
    QueryEvidencePack,
    SectionEvidencePack,
    SymbolSummary,
)
from sddraft.prompts.templates import (
    DIRECTORY_SUMMARY_SYSTEM_PROMPT,
    FILE_SUMMARY_SYSTEM_PROMPT,
    QUERY_SYSTEM_PROMPT,
    SECTION_SYSTEM_PROMPT,
    UPDATE_SYSTEM_PROMPT,
)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_section_generation_prompt(pack: SectionEvidencePack) -> tuple[str, str]:
    """Return system and user prompts for section generation."""

    user_prompt = (
        "Generate an SDD section draft from this evidence.\n"
        f"Section Spec:\n{_json(pack.section.model_dump(mode='json'))}\n\n"
        f"CSC Descriptor:\n{_json(pack.csc.model_dump(mode='json'))}\n\n"
        f"Evidence Pack:\n{_json(pack.model_dump(mode='json'))}\n"
    )
    return SECTION_SYSTEM_PROMPT, user_prompt


def build_update_proposal_prompt(pack: SectionEvidencePack) -> tuple[str, str]:
    """Return system and user prompts for update proposal generation."""

    user_prompt = (
        "Propose a revision for the impacted section using commit evidence.\n"
        f"Section Spec:\n{_json(pack.section.model_dump(mode='json'))}\n\n"
        f"Existing Section Text:\n{pack.existing_section_text or 'TBD'}\n\n"
        f"Evidence Pack:\n{_json(pack.model_dump(mode='json'))}\n"
    )
    return UPDATE_SYSTEM_PROMPT, user_prompt


def build_query_prompt(pack: QueryEvidencePack) -> tuple[str, str]:
    """Return system and user prompts for grounded Q&A."""

    user_prompt = (
        f"Question:\n{pack.request.question}\n\n"
        f"Session History:\n{_json(pack.request.session_history)}\n\n"
        f"Evidence Chunks:\n{_json([chunk.model_dump(mode='json') for chunk in pack.chunks])}\n\n"
        f"Citations:\n{_json([citation.model_dump(mode='json') for citation in pack.citations])}\n"
    )
    return QUERY_SYSTEM_PROMPT, user_prompt


def build_file_summary_prompt(
    *,
    code_summary: CodeUnitSummary,
    symbols: list[SymbolSummary],
    code_excerpt: str,
) -> tuple[str, str]:
    """Return system and user prompts for file summary generation."""

    user_prompt = (
        "Generate a concise summary for one source file.\n"
        f"Code Summary:\n{_json(code_summary.model_dump(mode='json'))}\n\n"
        f"Symbols:\n{_json([item.model_dump(mode='json') for item in symbols])}\n\n"
        f"Code Excerpt:\n{code_excerpt or 'TBD'}\n"
    )
    return FILE_SUMMARY_SYSTEM_PROMPT, user_prompt


def build_directory_summary_prompt(
    *,
    directory_path: str,
    local_files: list[FileSummaryDoc],
    child_directories: list[DirectorySummaryDoc],
) -> tuple[str, str]:
    """Return system and user prompts for bottom-up directory summary generation."""

    user_prompt = (
        "Generate a directory summary from direct evidence only.\n"
        f"Directory Path:\n{directory_path}\n\n"
        f"Local File Summaries:\n{_json([item.model_dump(mode='json') for item in local_files])}\n\n"
        f"Child Directory Summaries:\n{_json([item.model_dump(mode='json') for item in child_directories])}\n"
    )
    return DIRECTORY_SUMMARY_SYSTEM_PROMPT, user_prompt
