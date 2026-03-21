# Extending EngLLM

This guide is the contributor handoff document for EngLLM. It explains where new
work should live, which boundaries matter, and which modules and tests are most
important for each roadmap phase.

## Design Rules To Preserve

- Keep provider-specific code in `src/engllm/llm/` only.
- Keep deterministic repo analysis in `src/engllm/core/`.
- Keep prompt text centralized under `src/engllm/prompts/`.
- Keep tool orchestration in `src/engllm/tools/<tool_name>/`.
- Keep external-system abstractions in `src/engllm/integrations/`.
- Keep structured JSON validation mandatory for every LLM boundary.
- Keep artifacts file-backed and inspectable.

## Current Package Shape

```text
src/engllm/
  cli/
  core/
    analysis/
    config/
    render/
    repo/
  domain/
  integrations/
  llm/
  prompts/
    ask/
    core/
    history_docs/
    sdd/
  tools/
    ask/
    history_docs/
    repo/
    sdd/
```

Use this split when deciding where new code should live:

- `core/`: shared deterministic services used by multiple tools.
- `tools/`: tool-specific workflows, renderers, and tool-facing models.
- `integrations/`: reusable external capability interfaces and adapters.
- `domain/`: cross-tool models and shared errors only.

## Add an LLM Provider

1. Implement the provider in `src/engllm/llm/`.
2. Satisfy the `LLMClient` contract from `src/engllm/llm/base.py`.
3. Parse and validate structured JSON responses against `response_model`.
4. Register the provider in `src/engllm/llm/factory.py`.
5. Add isolated tests in `tests/test_llm_components.py` or a neighboring provider test.

## Add a Language Analyzer

1. Update `src/engllm/core/repo/language_analyzers.py`.
2. Register the language in `src/engllm/core/repo/scanner.py` if needed.
3. Emit `CodeUnitSummary`, `SymbolSummary`, and dependency signals conservatively.
4. Add deterministic scan and graph tests.

Primary tests:

- `tests/test_language_analyzers.py`
- `tests/test_repo_scanner.py`
- `tests/test_repo_scanner_multilang.py`
- `tests/test_graph_build_and_retrieval.py`

## Add a Shared Analysis Capability

Use `core/` when the behavior is reusable across tools.

Good fits for `core/`:

- retrieval and indexing
- hierarchy analysis
- graph build and traversal
- diff and commit impact analysis
- workspace and artifact helpers
- future history traversal or reusable repo summarization logic

Primary shared-analysis tests:

- `tests/test_retrieval.py`
- `tests/test_hierarchy_analysis.py`
- `tests/test_graph_build_and_retrieval.py`
- `tests/test_graph_build_incremental_helpers.py`
- `tests/test_diff_and_impact.py`
- `tests/test_commit_impact_extra.py`

## Add or Change a Tool

Use `tools/` when the behavior is specific to one user-facing capability.

Current tool namespaces:

- `src/engllm/tools/sdd/`
- `src/engllm/tools/ask/`
- `src/engllm/tools/repo/`
- `src/engllm/tools/history_docs/`

If you add a future tool, keep its workflow, tool-specific models, and renderers
inside `src/engllm/tools/<tool_name>/`, then register it through the CLI/tooling
layer instead of coupling it directly to existing tools.

## Add an Integration

External-system code belongs in `src/engllm/integrations/`.

Current shared interfaces:

- `RepoHostClient`
- `IssueTrackerClient`
- `CiLogClient`

Future tools such as review automation or release-note enrichment should depend
on these capability interfaces rather than on vendor-specific clients.

## Roadmap Pickup Guide

Use this section when taking over a roadmap phase from `TASKS.md`.

### Phase 1 — Symbol Fidelity And Graph Correctness

Primary code areas:

- `src/engllm/core/repo/language_analyzers.py`
- `src/engllm/core/repo/scanner.py`
- `src/engllm/core/analysis/graph_build.py`

Primary tests:

- `tests/test_language_analyzers.py`
- `tests/test_repo_scanner_multilang.py`
- `tests/test_graph_build_and_retrieval.py`

### Phase 2 — Multi-Language Dependency Expansion

Primary code areas:

- `src/engllm/core/analysis/dependency_resolution.py`
- `src/engllm/core/repo/language_analyzers.py`
- `src/engllm/core/analysis/graph_build.py`

Primary tests:

- `tests/test_graph_build_and_retrieval.py`
- `tests/test_workflow_generate_multilang.py`
- `tests/test_graph_index_and_candidate_sources.py`

### Phase 3 — Incremental Graph Build And Reuse

Primary code areas:

- `src/engllm/core/analysis/graph_build.py`
- `src/engllm/core/analysis/graph_models.py`
- `src/engllm/tools/sdd/propose_updates.py`

Primary tests:

- `tests/test_graph_build_incremental_helpers.py`
- `tests/test_workflow_propose_updates.py`
- `tests/test_graph_build_and_retrieval.py`

### Phase 4 — Ask Evidence Quality

Primary code areas:

- `src/engllm/core/analysis/graph_retrieval.py`
- `src/engllm/tools/ask/ask.py`
- `src/engllm/prompts/ask/builders.py`

Primary tests:

- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_render_and_workflow_misc.py`

### Phase 5 — Vector-Readiness Formalization

Primary code areas:

- `src/engllm/core/analysis/graph_retrieval.py`
- `src/engllm/tools/ask/ask.py`
- `src/engllm/cli/main.py`

Primary tests:

- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_cli_additional.py`

### Phase 6 — Commit-Aware Q&A

Primary code areas:

- `src/engllm/core/analysis/graph_retrieval.py`
- `src/engllm/tools/ask/ask.py`
- `src/engllm/tools/sdd/propose_updates.py`

Primary tests:

- `tests/test_graph_index_and_candidate_sources.py`
- `tests/test_workflow_propose_updates.py`
- `tests/test_workflow_generate_and_ask.py`

### Phase 7 — Docs Hardening

Primary docs:

- `README.md`
- `docs/USAGE.md`
- `docs/EXTENDING.md`
- `ARCHITECTURE.md`
- `SPEC.md`
- `TASKS.md`

Primary validation:

- `rg` verification commands recorded in `TASKS.md`
- `.venv/bin/python -m black --check src tests`
- `.venv/bin/python -m isort --check-only src tests`
- `.venv/bin/ruff check src tests`
- `.venv/bin/mypy src`
- `.venv/bin/pytest -q`

### Phase 8 — Intensive Ask Mode

Primary code areas:

- `src/engllm/core/analysis/intensive_corpus.py`
- `src/engllm/tools/ask/intensive.py`
- `src/engllm/prompts/ask/builders.py`
- `src/engllm/cli/main.py`

Primary tests:

- `tests/test_intensive_corpus.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_cli_additional.py`

### Phase 9 — EngLLM Multi-Tool Restructure

Primary code areas:

- `src/engllm/core/workspaces.py`
- `src/engllm/core/tooling.py`
- `src/engllm/cli/main.py`
- `src/engllm/tools/sdd/`
- `src/engllm/tools/ask/`
- `src/engllm/tools/repo/`
- `src/engllm/integrations/base.py`

Primary tests:

- `tests/test_cli.py`
- `tests/test_cli_additional.py`
- `tests/test_imports.py`
- `tests/test_layer_boundaries.py`
- `tests/test_workflow_generate_and_ask.py`
- `tests/test_workflow_propose_updates.py`

### History-Walk Documentation Track

Primary design docs:

- `docs/HISTORY_DOCS.md`
- `TASKS.md`
- `ARCHITECTURE.md`

Primary code areas for H1-H2:

- `src/engllm/core/repo/history.py`
- `src/engllm/core/repo/scanner.py`
- `src/engllm/core/analysis/history.py`
- `src/engllm/tools/history_docs/build.py`
- `src/engllm/tools/history_docs/models.py`
- `src/engllm/cli/main.py`

Primary tests for H1-H5:

- `tests/test_history_docs_h1.py`
- `tests/test_history_docs_h2.py`
- `tests/test_history_docs_h3.py`
- `tests/test_history_docs_h4.py`
- `tests/test_history_docs_h5.py`
- `tests/history_docs_helpers.py`
- `tests/test_imports.py`
- `tests/test_diff_and_impact.py`

Current H1-H5 behavior:

- single-checkpoint, manual-first `engllm history-docs build`
- explicit `--checkpoint-commit`
- optional `--previous-checkpoint-commit`
- previous checkpoint defaults to the latest prior ancestor checkpoint already
  recorded in shared history artifacts for the workspace
- temporary checkpoint snapshot export via `git archive`
- structural scanning runs against the exported snapshot, not the live working
  tree
- missing historical source roots are recorded and skipped
- manifest search scope is limited to analyzed source roots plus their ancestor
  chain back to repo root
- interval deltas are derived from first-parent commit diffs plus snapshot
  comparison
- interval artifacts are written to
  `tools/history_docs/checkpoints/<checkpoint_id>/interval_delta_model.json`
- missing previous snapshot artifacts trigger diff-only fallback with
  conservative `observed` statuses instead of failure
- checkpoint-state artifacts are written to
  `tools/history_docs/checkpoints/<checkpoint_id>/checkpoint_model.json`
- checkpoint models keep both active and retired concepts while section records
  reference active concepts only
- section-plan artifacts are written to
  `tools/history_docs/checkpoints/<checkpoint_id>/section_outline.json`
- H5 keeps checkpoint-model sections as H4 core stubs and writes the scored
  section outline separately
- quarterly checkpoint auto-selection is deferred to a later phase

## Future Tool Notes

These are planned directions, not implemented tools yet:

- CI log summarizer built on shared repo and CI-log integration surfaces.
- Code review automation built on `RepoHostClient` and `IssueTrackerClient`.
- Release-note generation using commit ranges plus optional issue enrichment.
- History-walk documentation generation built on reusable repo-history analysis.

## Practical Contributor Workflow

1. Read the matching phase entry in `TASKS.md`.
2. Inspect the code areas listed above before editing anything.
3. Start with the fastest targeted tests for the phase.
4. Run the full quality gates before marking work complete.
5. Update `TASKS.md` in the same change when the roadmap status changes.
