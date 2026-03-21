"""Workspace path helpers for shared and tool-scoped artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from engllm.domain.models import DomainModel

WorkspaceKind = Literal["csc", "repo", "change", "review", "release"]


class WorkspaceContext(DomainModel):
    """Resolved workspace paths for one tool run."""

    kind: WorkspaceKind
    workspace_id: str
    repo_root: Path
    workspace_root: Path
    shared_root: Path
    tools_root: Path


def build_workspace_context(
    *,
    output_root: Path,
    workspace_id: str,
    kind: WorkspaceKind,
    repo_root: Path,
) -> WorkspaceContext:
    """Return deterministic shared and tool-specific artifact roots."""

    workspace_root = output_root / "workspaces" / workspace_id
    return WorkspaceContext(
        kind=kind,
        workspace_id=workspace_id,
        repo_root=repo_root.resolve(),
        workspace_root=workspace_root,
        shared_root=workspace_root / "shared",
        tools_root=workspace_root / "tools",
    )


def tool_artifact_root(context: WorkspaceContext, tool_name: str) -> Path:
    """Return the artifact root for one tool inside a workspace."""

    return context.tools_root / tool_name


def workspace_root_from_shared_artifact(path: Path) -> Path:
    """Return the workspace root inferred from a shared artifact path."""

    shared_root = path if path.name == "shared" else path.parent
    if shared_root.name != "shared":
        raise ValueError(f"Path is not inside a shared workspace root: {path}")
    return shared_root.parent


def tool_root_from_shared_artifact(path: Path, tool_name: str) -> Path:
    """Return a tool root inferred from any path beneath a workspace shared root."""

    workspace_root = workspace_root_from_shared_artifact(path)
    return workspace_root / "tools" / tool_name
