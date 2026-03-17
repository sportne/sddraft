# SDDraft Task Board

## Update Rules

- Use statuses: `[ ]` not started, `[~]` in progress, `[x]` done.
- Keep this file as the single implementation roadmap.
- Update task status in the same PR that changes implementation.
- Preserve deterministic behavior and inspectable artifacts unless a task explicitly changes that.

## 1) Current State Summary (Grounded In Repo)

- Graph-enhanced retrieval is implemented end-to-end: `generate` and `propose-updates` build `graph/manifest.json`, `nodes.jsonl`, `edges.jsonl`, `symbol_index.json`, and `adjacency.json`.
- `ask` currently runs lexical retrieval, optional hierarchy expansion, graph expansion, and deterministic reranking with fallback when hierarchy/graph artifacts are missing.
- Symbol inventory exists, but it is conservative and regex-assisted; spans and ownership quality are strongest for Python and weaker for other languages.
- `imports` graph edges are currently resolved only for Python in graph build logic.
- Graph build is currently full-rebuild per run; no manifest fingerprinting or partial graph reuse yet.
- Vector readiness is partial: placeholder vector candidate source exists, and generation config has `vector_*` fields, but no real retrieval orchestration path is wired.
- Commit-aware graph edges (`changed_in`, `impacts_section`) are generated in propose-updates, but commit-oriented Q&A traversal is not yet intentionally tuned.
- Docs and tests are good baseline quality, but graph schema/intents/incremental behavior are not yet documented as deeply as implementation now warrants.

## 2) Guiding Principles / Scope

- Keep lexical retrieval as the baseline and backward-compatible behavior.
- Keep all analysis deterministic and artifacts inspectable on disk.
- Improve symbol/dependency fidelity before introducing real vector retrieval.
- Avoid heavy external infrastructure (no hosted vector DB, no graph DB).
- Keep provider isolation and structured JSON validation requirements unchanged.
- Add vector retrieval only through clean abstractions and disabled-by-default paths first.

## 3) Sequenced Task Plan (Priority Order)

### Phase 1 — Symbol Fidelity And Graph Correctness

`Objective:` Improve symbol quality and graph correctness before expanding retrieval complexity.
`Why:` Better symbol nodes/edges improve both reranking and future vector grounding.
`Dependencies:` None (start here).
`Completion Criteria:` Symbol spans/ownership are analyzer-derived where possible, IDs remain stable, and cross-language symbol tests pass.

- [ ] **G1-01**  
  `Outcome:` Create a symbol-quality inventory by language from current analyzers and graph artifacts.  
  `Definition of Done:` Gaps are documented in comments/tests for Python, Java, C++, JS/TS, Go, Rust, and C#.  
  `Verification Command(s):` `pytest tests/test_language_analyzers.py tests/test_graph_build_and_retrieval.py`

- [ ] **G1-02**  
  `Outcome:` Extend deterministic symbol extraction to prefer analyzer-derived symbol facts (name, kind, qualified name, span, owner) over regex fallback whenever available.  
  `Definition of Done:` Symbol inventory path uses analyzer facts first; regex span matching is fallback-only.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_repo_scanner_multilang.py`

- [ ] **G1-03**  
  `Outcome:` Improve parent/child symbol ownership mapping for nested symbols and methods.  
  `Definition of Done:` Graph has reliable `parent_of`/`contains` symbol edges for nested ownership cases.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py`

- [ ] **G1-04**  
  `Outcome:` Preserve stable symbol node IDs while improving symbol metadata quality.  
  `Definition of Done:` ID derivation remains deterministic and unaffected by span-only changes.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py::test_generate_writes_deterministic_graph_artifacts`

### Phase 2 — Multi-Language Dependency / Import Edges

`Objective:` Broaden `imports` edge creation beyond Python with conservative resolution.
`Why:` Dependency-oriented questions currently underperform for non-Python projects.
`Dependencies:` Phase 1.
`Completion Criteria:` Imports edges are emitted for major supported languages when target resolution is reliable; unresolved imports are deterministic and non-fatal.

- [ ] **G2-01**  
  `Outcome:` Define normalized import/dependency resolution strategy per language (`java`, `cpp`, `javascript`, `typescript`, `go`, `rust`, `csharp`).  
  `Definition of Done:` Graph build has a language-router for import edge extraction rather than Python-only logic.  
  `Verification Command(s):` `pytest tests/test_repo_scanner_multilang.py tests/test_graph_build_and_retrieval.py`

- [ ] **G2-02**  
  `Outcome:` Implement Java + JS/TS import edge resolution to known files/modules.  
  `Definition of Done:` `imports` edges appear in graph artifacts for these languages in mixed-language fixtures.  
  `Verification Command(s):` `pytest tests/test_workflow_generate_multilang.py`

- [ ] **G2-03**  
  `Outcome:` Implement conservative C++ include and C#/Go/Rust dependency edge resolution where deterministic mapping is possible.  
  `Definition of Done:` Supported patterns emit edges; ambiguous/unresolved entries are skipped consistently with reason metadata.  
  `Verification Command(s):` `pytest tests/test_language_analyzers.py tests/test_graph_build_and_retrieval.py`

- [ ] **G2-04**  
  `Outcome:` Add regression tests for import edge creation and tie-break determinism across languages.  
  `Definition of Done:` Edge counts/types are asserted in graph artifact tests for multilingual fixtures.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`

### Phase 3 — Incremental Graph Build And Reuse

`Objective:` Move graph generation from full rebuild to deterministic incremental behavior.
`Why:` Full rewrite is expensive and limits scalability for large repos.
`Dependencies:` Phases 1-2.
`Completion Criteria:` Unchanged runs reuse graph state; changed runs update only impacted graph substructures while preserving deterministic output.

- [ ] **G3-01**  
  `Outcome:` Extend graph manifest with deterministic fingerprints (scan/retrieval/hierarchy inputs and build version).  
  `Definition of Done:` Manifest can detect no-op build conditions and reproducibility context.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py`

- [ ] **G3-02**  
  `Outcome:` Introduce incremental graph planner (full rebuild vs partial update vs no-op).  
  `Definition of Done:` Planner decisions are deterministic and test-covered.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_workflow_propose_updates.py`

- [ ] **G3-03**  
  `Outcome:` Persist/reuse per-file or per-node graph fragments to avoid rewriting unaffected regions.  
  `Definition of Done:` Propose-updates path updates impacted subtree + related section/commit edges only.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py`

- [ ] **G3-04**  
  `Outcome:` Add deterministic equivalence tests for full rebuild vs incremental rebuild outputs.  
  `Definition of Done:` Outputs match for unchanged inputs; partial changes alter only expected graph regions.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`

### Phase 4 — Ask Evidence Quality Tightening

`Objective:` Improve graph-derived evidence quality and prompt-facing metadata.
`Why:` Current `related_symbols` derivation is token-based and can include noise; prompt currently underuses graph context fields.
`Dependencies:` Phases 1-3.
`Completion Criteria:` `ask` evidence pack uses graph-grounded related entities and richer inclusion reasons, and prompt consumes them explicitly.

- [ ] **G4-01**  
  `Outcome:` Derive `related_symbols` from traversed graph symbol nodes and symbol index, not token sweeps from candidate chunk text.  
  `Definition of Done:` Related symbols are stable, relevant, and traceable to graph nodes.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py tests/test_graph_build_and_retrieval.py`

- [ ] **G4-02**  
  `Outcome:` Add graph-path evidence metadata (edge/path reason) to inclusion reasons used by prompts and audits.  
  `Definition of Done:` Inclusion reasons include deterministic graph rationale beyond `graph:<node_type>:<label>`.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`

- [ ] **G4-03**  
  `Outcome:` Update query prompt builder to include `primary_chunks`, `related_files`, `related_symbols`, `related_sections`, and inclusion score breakdowns.  
  `Definition of Done:` Prompt inputs reflect full evidence pack shape.  
  `Verification Command(s):` `pytest tests/test_render_and_workflow_misc.py tests/test_workflow_generate_and_ask.py`

- [ ] **G4-04**  
  `Outcome:` Tighten rerank signal definitions and validate deterministic tie-break behavior under mixed lexical/graph evidence.  
  `Definition of Done:` Rerank math and tie-break rules are test-covered and documented.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`

### Phase 5 — Vector-Readiness Formalization (Disabled By Default)

`Objective:` Formalize extension seams so vector retrieval can be added later without ask/workflow rework.
`Why:` Current placeholder exists but orchestration/config/CLI contracts are only partially wired.
`Dependencies:` Phase 4.
`Completion Criteria:` Candidate-source orchestration, config schema, and CLI placeholders are coherent and no-op when vector is disabled.

- [ ] **G5-01**  
  `Outcome:` Unify candidate-source abstraction so lexical, graph, and vector sources share an orchestration contract.  
  `Definition of Done:` Ask retrieval orchestration is source-pluggable without source-specific branching leaks.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`

- [ ] **G5-02**  
  `Outcome:` Thread `vector_enabled`/`vector_top_k` through ask workflow settings and CLI/config resolution paths (still disabled by default).  
  `Definition of Done:` Config and CLI parse/propagate vector placeholders without changing current default behavior.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_workflow_generate_and_ask.py`

- [ ] **G5-03**  
  `Outcome:` Document vector extension points in usage + architecture docs with explicit non-goals for current phase.  
  `Definition of Done:` Docs explain how to add vector source later without implying it is active now.  
  `Verification Command(s):` `rg -n \"vector\" docs/USAGE.md ARCHITECTURE.md`

### Phase 6 — Commit-Aware Q&A Integration

`Objective:` Make commit edges (`changed_in`, `impacts_section`) intentionally useful in `ask`.
`Why:` Commit-aware edges exist in graph artifacts but are not yet leveraged by intent routing and evidence assembly.
`Dependencies:` Phases 2-4.
`Completion Criteria:` Change-impact questions reliably retrieve commit/file/section evidence via graph traversal and reranking.

- [ ] **G6-01**  
  `Outcome:` Add deterministic change-impact intent classification for commit-oriented questions.  
  `Definition of Done:` Intent router explicitly recognizes change/impact/commit question patterns and edge preferences.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`

- [ ] **G6-02**  
  `Outcome:` Add traversal preferences for `changed_in` and `impacts_section` edges and include commit nodes in related evidence outputs.  
  `Definition of Done:` Ask evidence pack exposes commit-aware context for applicable queries.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py tests/test_workflow_generate_and_ask.py`

- [ ] **G6-03**  
  `Outcome:` Add regression tests for commit-impact Q&A using propose-updates-produced artifacts.  
  `Definition of Done:` Tests assert grounded citations and non-fabricated unknown handling (`TBD`) for change-impact questions.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py tests/test_graph_build_and_retrieval.py`

### Phase 7 — Documentation First-Class Graph Subsystem

`Objective:` Make graph subsystem docs complete and maintainable.
`Why:` Current docs describe graph at high level but not enough for contributors implementing next steps.
`Dependencies:` Phases 1-6 (can start in parallel for baseline docs).
`Completion Criteria:` Docs cover graph schema, build modes, ask flow, commit-aware behavior, and extension seams with examples.

- [ ] **G7-01**  
  `Outcome:` Expand usage docs with graph node/edge schema, stable IDs, artifact examples, and troubleshooting.  
  `Definition of Done:` `docs/USAGE.md` has a dedicated graph section with practical inspection commands.  
  `Verification Command(s):` `rg -n \"graph/manifest|nodes.jsonl|edges.jsonl|changed_in|impacts_section\" docs/USAGE.md`

- [ ] **G7-02**  
  `Outcome:` Expand architecture docs with graph build lifecycle (full vs incremental), ask evidence flow, and commit-aware routing.  
  `Definition of Done:` `ARCHITECTURE.md` reflects current implementation and next-step architecture boundaries.  
  `Verification Command(s):` `rg -n \"Engineering Graph|incremental|candidate source|commit-aware\" ARCHITECTURE.md`

- [ ] **G7-03**  
  `Outcome:` Add contributor notes linking TASKS roadmap phases to relevant modules/tests for pickup by future agents.  
  `Definition of Done:` Docs map each major phase to code areas and test files.  
  `Verification Command(s):` `rg -n \"Phase 1|Phase 2|test_graph\" docs/USAGE.md docs/EXTENDING.md TASKS.md`

## 4) Testing And Validation Requirements (For All Phases)

- Unit tests for symbol extraction, edge construction, graph loading, traversal, and rerank math.
- Integration tests for `generate`, `propose-updates`, and `ask` with and without graph artifacts.
- Deterministic artifact tests for graph IDs, counts, ordering, and full-vs-incremental equivalence.
- Regression tests for fallback behavior when hierarchy/graph artifacts are missing or corrupt.
- Commit-aware Q&A tests that assert grounded citations and conservative TBD behavior.
- Quality gates required on every phase PR:
  - `ruff check src tests`
  - `mypy src`
  - `pytest -q`
  - coverage must remain `>= 90%`

## 5) Nice-To-Have / Later (Non-Blocking)

- [ ] Real embedding generation and local vector index implementation (still file-backed, deterministic build metadata).
- [ ] Advanced cross-language semantic/call-graph inference beyond import/dependency-level relations.
- [ ] Richer section impact inference at symbol-flow level, not only evidence overlap.
- [ ] Graph visualization/export tooling for artifact inspection (CLI report or static visualization).
- [ ] Optional benchmark harness for large mixed-language repos with memory/runtime trend reporting.

## 6) Historical Completed Milestones (Preserved)

- [x] **T-001..T-006** v1 hardening baseline (task board, CLI override parity, error handling, regression coverage).
- [x] **T-007..T-010** v1 acceptance-proofing baseline (extensibility docs, Gemini optional dependency/docs, acceptance tests, quality gate pass).
- [x] Graph baseline implemented: graph artifacts + ask graph augmentation + deterministic reranking + CLI graph flags + fallback behavior.

## 7) Blocked

- [ ] None currently.
