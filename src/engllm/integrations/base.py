"""Capability-oriented external integration protocols.

These interfaces let future tools depend on stable capabilities rather than on a
single vendor family such as Atlassian or GitLab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RepoHostClient(Protocol):
    """Capability interface for pull/merge-request style code-host operations."""

    def fetch_change_metadata(self, change_id: str) -> dict[str, object]:
        """Return normalized metadata for a code review change."""

    def fetch_diff(self, change_id: str) -> str:
        """Return the raw unified diff for a change."""

    def publish_review_comment(self, change_id: str, body: str) -> None:
        """Publish one review comment back to the hosting product."""


class IssueTrackerClient(Protocol):
    """Capability interface for issue and ticket lookups."""

    def fetch_issue(self, issue_id: str) -> dict[str, object]:
        """Return normalized issue metadata for downstream tooling."""


class CiLogClient(Protocol):
    """Capability interface for CI job log retrieval."""

    def fetch_job_log(self, job_id: str) -> str:
        """Return the raw log text for a CI job."""

    def download_artifact(self, job_id: str, artifact_name: str, target: Path) -> Path:
        """Persist one CI artifact locally and return the written path."""
