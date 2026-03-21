"""Deterministic prompt builders."""

from __future__ import annotations

import json

from sddraft.domain.models import (
    CodeUnitSummary,
    DirectorySummaryDoc,
    FileSummaryDoc,
    IntensiveCorpusChunk,
    QueryEvidencePack,
    SectionEvidencePack,
    SubtreeRollup,
    SymbolSummary,
)
from sddraft.prompts.templates import (
    DIRECTORY_SUMMARY_SYSTEM_PROMPT,
    FILE_SUMMARY_SYSTEM_PROMPT,
    INTENSIVE_SCREENING_SYSTEM_PROMPT,
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
        f"Primary Chunks:\n{_json([chunk.model_dump(mode='json') for chunk in pack.primary_chunks])}\n\n"
        f"Selected Chunks:\n{_json([chunk.model_dump(mode='json') for chunk in pack.chunks])}\n\n"
        f"Citations:\n{_json([citation.model_dump(mode='json') for citation in pack.citations])}\n\n"
        f"Related Files:\n{_json([path.as_posix() for path in pack.related_files])}\n\n"
        f"Related Symbols:\n{_json(pack.related_symbols)}\n\n"
        f"Related Sections:\n{_json(pack.related_sections)}\n\n"
        f"Related Commits:\n{_json(pack.related_commits)}\n\n"
        f"Inclusion Reasons:\n{_json([reason.model_dump(mode='json') for reason in pack.inclusion_reasons])}\n"
    )
    return QUERY_SYSTEM_PROMPT, user_prompt


def build_intensive_screening_prompt(
    *,
    question: str,
    session_history: list[str],
    chunk: IntensiveCorpusChunk,
) -> tuple[str, str]:
    """Return prompts for one intensive corpus chunk screening pass."""

    user_prompt = (
        "Screen this structured repository chunk for relevance to the question.\n"
        f"Screening Input:\n{_json({'question': question, 'session_history': session_history, 'chunk': chunk.model_dump(mode='json')})}\n"
    )
    return INTENSIVE_SCREENING_SYSTEM_PROMPT, user_prompt


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
    subtree_rollup: SubtreeRollup,
) -> tuple[str, str]:
    """Return prompts for recursive directory subtree summary generation."""

    role = "project root" if directory_path == "." else "directory subtree"
    root_instruction = (
        "This path is the repository root. Write a project-level overview.\n\n"
        if directory_path == "."
        else ""
    )

    user_prompt = (
        "Generate a recursive subtree summary for this directory.\n"
        f"Directory Path:\n{directory_path}\n\n"
        f"Directory Role:\n{role}\n\n"
        f"{root_instruction}"
        f"Local File Summaries:\n{_json([item.model_dump(mode='json') for item in local_files])}\n\n"
        f"Child Directory Summaries:\n{_json([item.model_dump(mode='json') for item in child_directories])}\n"
        f"\nSubtree Rollup:\n{_json(subtree_rollup.model_dump(mode='json'))}\n"
    )
    return DIRECTORY_SUMMARY_SYSTEM_PROMPT, user_prompt
