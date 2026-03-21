"""SDD tool model namespace.

The classes are still defined in :mod:`engllm.domain.models`, but this module is
now the canonical import surface for SDD-specific models.
"""

from engllm.domain.models import (
    ConfigBundle,
    CSCDescriptor,
    GenerateResult,
    ProjectConfig,
    ProposeUpdatesResult,
    ReviewArtifact,
    SDDDocument,
    SDDSectionSpec,
    SDDTemplate,
    SectionDraft,
    SectionEvidencePack,
    SectionReviewArtifact,
    SectionUpdateProposal,
    UpdateProposalReport,
)

__all__ = [
    "CSCDescriptor",
    "ConfigBundle",
    "GenerateResult",
    "ProjectConfig",
    "ProposeUpdatesResult",
    "ReviewArtifact",
    "SDDDocument",
    "SDDSectionSpec",
    "SDDTemplate",
    "SectionDraft",
    "SectionEvidencePack",
    "SectionReviewArtifact",
    "SectionUpdateProposal",
    "UpdateProposalReport",
]
