"""Repository analysis API."""

from .diff_parser import get_git_diff, parse_diff
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
    "detect_language",
    "get_analyzer_for_path",
    "get_analyzer_for_language",
    "get_registered_analyzers",
]
