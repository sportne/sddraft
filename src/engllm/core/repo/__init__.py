"""Repository analysis API."""

from .diff_parser import (
    extract_changed_symbol_names,
    get_git_diff,
    get_git_diff_between,
    parse_diff,
)
from .history import (
    describe_commit_diff,
    export_commit_snapshot,
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
    "extract_changed_symbol_names",
    "get_git_diff",
    "get_git_diff_between",
    "parse_diff",
    "describe_commit_diff",
    "export_commit_snapshot",
    "resolve_commit",
    "get_commit_metadata",
    "is_strict_ancestor",
    "iter_interval_commits",
    "detect_language",
    "get_analyzer_for_path",
    "get_analyzer_for_language",
    "get_registered_analyzers",
]
