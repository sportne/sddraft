"""Tests for diff parsing and commit impact mapping."""

from __future__ import annotations

from sddraft.analysis.commit_impact import build_commit_impact
from sddraft.repo.diff_parser import parse_diff


def test_parse_diff_and_build_impact() -> None:
    diff = """
diff --git a/src/a.py b/src/a.py
index abc..def 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1,2 +1,3 @@
-import os
+import sys
-def old_fn(x):
+def new_fn(x, y):
     return x
""".strip()

    summaries = parse_diff(diff)
    assert len(summaries) == 1

    summary = summaries[0]
    assert summary.path.as_posix() == "src/a.py"
    assert summary.added_lines == 2
    assert summary.removed_lines == 2
    assert summary.signature_changes
    assert summary.dependency_changes

    impact = build_commit_impact("HEAD~1..HEAD", summaries)
    assert "interface_change" in impact.change_kinds
    assert "Interface Design" in impact.impacted_sections
