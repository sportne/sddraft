# Extending SDDraft

This guide describes how to add new capabilities while preserving SDDraft
architecture boundaries, and how to pick up the current roadmap safely.

## Design Rules To Preserve

- Keep provider-specific code in `src/sddraft/llm/` only.
- Keep repository logic out of workflows and renderers.
- Keep deterministic analysis outside the LLM boundary.
- Keep structured JSON validation mandatory for LLM outputs.
- Keep inspectable file-backed artifacts as the source of truth.

## Add an LLM Provider

1. Implement a provider client in `src/sddraft/llm/` that satisfies `LLMClient` from `llm/base.py`.
2. Ensure provider responses are parsed as structured JSON and validated against `response_model`.
3. Register the provider in `llm/factory.py` without importing provider SDKs outside `llm/`.
4. Add provider tests with mocks/fakes so tests remain network-independent.

## Add a Language Analyzer

1. Add a `LanguageAnalyzer` implementation in `src/sddraft/repo/language_analyzers.py`.
2. Map file extensions to a normalized `SourceLanguage` and register the analyzer.
3. Ensure analyzer output feeds `CodeUnitSummary` and `SymbolSummary`.
4. Add scan/diff tests to validate deterministic extraction and classification.

## Add a Renderer Output

1. Implement rendering in `src/sddraft/render/` from structured domain models only.
2. Keep repository inspection out of renderers; workflows provide input models.
3. Add tests for deterministic output formatting and error behavior.

## Roadmap Pickup Guide

Use this section when taking over a roadmap phase from `TASKS.md`.

### Phase 1 — Symbol Fidelity And Graph Correctness

Primary code areas:
- `src/sddraft/repo/language_analyzers.py`
- `src/sddraft/repo/scanner.py`
- `src/sddraft/analysis/graph_build.py`

Primary tests:
- `tests/test_language_analyzers.py`
- `tests/test_graph_build_and_retrieval.py`
- `tests/test_graph_index_and_candidate_sources.py`

### Phase 2 — Multi-Language Dependency Expansion

Primary code areas:
- `src/sddraft/analysis/dependency_resolution.py`
- `src/sddraft/repo/language_analyzers.py`
- `src/sddraft/analysis/graph_build.py`

Primary tests:
- `tests/test_graph_build_and_retrieval.py`
- `tests/test_workflow_generate_multilang.py`
- `tests/test_graph_index_and_candidate_sources.py`

### Phase 3 — Incremental Graph Build And Reuse

Primary code areas:
- `src/sddraft/analysis/graph_build.py`
- `src/sddraft/analysis/graph_models.py`
- `src/sddraft/workflows/propose_updates.py`

Primary tests:
- `tests/test_graph_build_incremental_helpers.py`
- `tests/test_workflow_propose_updates.py`
- `tests/test_graph_build_and_retrieval.py`

### Phase 4 — Ask Evidence Quality

Primary code areas:
- `src/sddraft/analysis/graph_retrieval.py`
- `src/sddraft/workflows/ask.py`
- `src/sddraft/prompts/builders.py`

Primary tests:
- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_render_and_workflow_misc.py`

### Phase 5 — Vector-Readiness Formalization

Primary code areas:
- `src/sddraft/analysis/graph_retrieval.py`
- `src/sddraft/workflows/ask.py`
- `src/sddraft/cli/main.py`

Primary tests:
- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_cli_additional.py`

### Phase 6 — Commit-Aware Q&A

Primary code areas:
- `src/sddraft/analysis/graph_retrieval.py`
- `src/sddraft/workflows/ask.py`
- `src/sddraft/workflows/propose_updates.py`

Primary tests:
- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_propose_updates.py`
- `tests/test_workflow_generate_and_ask.py`

### Phase 7 — Docs Hardening

Primary docs:
- `docs/USAGE.md`
- `ARCHITECTURE.md`
- `docs/EXTENDING.md`

Primary validation:
- `TASKS.md`
- `rg` verification commands recorded in `TASKS.md`
- normal repo quality gates (`ruff`, `mypy`, `pytest`)

## Practical Contributor Workflow

1. Read the matching phase entry in `TASKS.md`.
2. Inspect the code areas listed above before editing anything.
3. Use the listed tests first for fast iteration.
4. Run the full quality gates before marking the phase complete.
5. Update `TASKS.md` in the same change when task status changes.
