"""External system integration contracts and adapters."""

from engllm.integrations.base import CiLogClient, IssueTrackerClient, RepoHostClient

__all__ = ["RepoHostClient", "IssueTrackerClient", "CiLogClient"]
