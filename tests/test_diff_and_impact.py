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
    assert summary.language == "python"
    assert summary.added_lines == 2
    assert summary.removed_lines == 2
    assert summary.signature_changes
    assert summary.dependency_changes

    impact = build_commit_impact("HEAD~1..HEAD", summaries)
    assert "interface_change" in impact.change_kinds
    assert "Interface Design" in impact.impacted_sections


def test_parse_diff_multilanguage_signature_and_dependency_detection() -> None:
    diff = """
diff --git a/src/Main.java b/src/Main.java
index abc..def 100644
--- a/src/Main.java
+++ b/src/Main.java
@@ -1,2 +1,3 @@
-import java.util.List;
+import java.util.Map;
-public void oldMethod(int x) {
+public void newMethod(int x, int y) {
diff --git a/src/core.cpp b/src/core.cpp
index abc..def 100644
--- a/src/core.cpp
+++ b/src/core.cpp
@@ -1,2 +1,3 @@
-#include <vector>
+#include <map>
-int run(int x) {
+int run(int x, int y) {
""".strip()

    summaries = parse_diff(diff)
    assert len(summaries) == 2

    java_summary = summaries[0]
    cpp_summary = summaries[1]

    assert java_summary.language == "java"
    assert java_summary.signature_changes
    assert java_summary.dependency_changes

    assert cpp_summary.language == "cpp"
    assert cpp_summary.signature_changes
    assert cpp_summary.dependency_changes
