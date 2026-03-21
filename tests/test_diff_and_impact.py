"""Tests for diff parsing and commit impact mapping."""

from __future__ import annotations

from engllm.core.analysis.commit_impact import build_commit_impact
from engllm.core.repo.diff_parser import parse_diff


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
diff --git a/src/app.ts b/src/app.ts
index abc..def 100644
--- a/src/app.ts
+++ b/src/app.ts
@@ -1,2 +1,3 @@
-import {A} from './a';
+import {B} from './b';
-export function run(x: number) {
+export function run(x: number, y: number) {
diff --git a/src/main.go b/src/main.go
index abc..def 100644
--- a/src/main.go
+++ b/src/main.go
@@ -1,2 +1,3 @@
-import "fmt"
+import "os"
-func run(x int) {
+func run(x int, y int) {
""".strip()

    summaries = parse_diff(diff)
    assert len(summaries) == 4

    by_path = {item.path.as_posix(): item for item in summaries}
    java_summary = by_path["src/Main.java"]
    cpp_summary = by_path["src/core.cpp"]
    ts_summary = by_path["src/app.ts"]
    go_summary = by_path["src/main.go"]

    assert java_summary.language == "java"
    assert java_summary.signature_changes
    assert java_summary.dependency_changes

    assert cpp_summary.language == "cpp"
    assert cpp_summary.signature_changes
    assert cpp_summary.dependency_changes

    assert ts_summary.language == "typescript"
    assert ts_summary.signature_changes
    assert ts_summary.dependency_changes

    assert go_summary.language == "go"
    assert go_summary.signature_changes
    assert go_summary.dependency_changes
