"""Deterministic prompt builders."""

from __future__ import annotations

import json

from sddraft.domain.models import QueryEvidencePack, SectionEvidencePack
from sddraft.prompts.templates import (
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
