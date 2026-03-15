"""Project error taxonomy."""


class SDDraftError(Exception):
    """Base error type for SDDraft."""


class ConfigError(SDDraftError):
    """Raised for configuration loading/normalization issues."""


class RepositoryError(SDDraftError):
    """Raised for repository scanning problems."""


class GitError(SDDraftError):
    """Raised for git command and diff issues."""


class AnalysisError(SDDraftError):
    """Raised for deterministic analysis failures."""


class LLMError(SDDraftError):
    """Raised for provider and generation failures."""


class ValidationError(SDDraftError):
    """Raised for schema validation failures."""


class RenderingError(SDDraftError):
    """Raised for document/artifact rendering failures."""
