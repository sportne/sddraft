"""Repository analysis API."""

from .diff_parser import get_git_diff, parse_diff
from .scanner import discover_source_files, scan_repository

__all__ = ["discover_source_files", "scan_repository", "get_git_diff", "parse_diff"]
