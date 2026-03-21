"""Repository analysis API."""

from .diff_parser import get_git_diff, parse_diff
from .history import (
    get_commit_metadata,
    is_strict_ancestor,
    iter_interval_commits,
    resolve_commit,
)
from .language_analyzers import (
    detect_language,
    get_analyzer_for_language,
    get_analyzer_for_path,
    get_registered_analyzers,
)
from .scanner import discover_source_files, scan_repository

__all__ = [
    "discover_source_files",
    "scan_repository",
    "get_git_diff",
    "parse_diff",
    "resolve_commit",
    "get_commit_metadata",
    "is_strict_ancestor",
    "iter_interval_commits",
    "detect_language",
    "get_analyzer_for_path",
    "get_analyzer_for_language",
    "get_registered_analyzers",
]
