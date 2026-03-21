"""Deterministic prompt builders for the SDD tool."""

from __future__ import annotations

import json

from engllm.prompts.sdd.templates import SECTION_SYSTEM_PROMPT, UPDATE_SYSTEM_PROMPT
from engllm.tools.sdd.models import SectionEvidencePack


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
