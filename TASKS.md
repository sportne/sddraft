# EngLLM Task Board

## Update Rules

- Use statuses: `[ ]` not started, `[~]` in progress, `[x]` done.
- Keep this file as the single implementation roadmap.
- Update task status in the same PR that changes implementation.
- Preserve deterministic behavior and inspectable artifacts unless a task explicitly changes that.

## 1) Current State Summary (Grounded In Repo)

- EngLLM is now organized as a shared platform plus tool namespaces: `core/`, `tools/`, `integrations/`, `llm/`, and tool-scoped prompt namespaces under `prompts/`.
- The CLI is now tool-first and breaking by design: `engllm sdd ...`, `engllm ask ...`, and `engllm repo ...`. Legacy flat `sddraft` package and command names are gone.
- Shared analysis artifacts now live under `artifacts/workspaces/<workspace_id>/shared/`, while tool outputs live under `artifacts/workspaces/<workspace_id>/tools/<tool_name>/`.
- Graph-enhanced retrieval is implemented end-to-end: SDD workflows build shared `graph/manifest.json`, `nodes.jsonl`, `edges.jsonl`, `symbol_index.json`, and `adjacency.json`.
- `ask` currently supports both the standard lexical/hierarchy/graph pipeline and an intensive whole-repo screening mode with deterministic corpus artifacts.
- Symbol inventory is now analyzer-emitted and symbol-first (`SymbolSummary`), with deterministic owner/span metadata where available; quality remains conservative for some language-specific edge cases.
- `imports` graph edges now use a normalized multi-language resolver (`python`, `java`, `javascript`, `typescript`, `go`, `rust`, `csharp`, `cpp`) with conservative repo-local resolution and inspectable reason metadata.
- Graph build now supports deterministic planner decisions (`no_op`, `partial`, `full`) with manifest fingerprinting and fragment reuse in `graph/fragments/`.
- Vector readiness is formalized but still placeholder-only: `ask` can orchestrate lexical, hierarchy, graph, and future vector sources through one contract, while the vector backend remains disabled-by-default and intentionally empty.
- Commit-aware graph edges (`changed_in`, `impacts_section`) are generated in propose-updates and are now used intentionally in `ask` for change-impact questions.
- `ask` intensive mode screens a structured cross-file corpus chunk-by-chunk and persists corpus/run artifacts under `artifacts/workspaces/<workspace_id>/tools/ask/intensive/`.
- Shared integration capability interfaces now exist for future repo-host, issue-tracker, and CI-backed tools, but those future tools are not implemented yet.
- The planned history-walk documentation tool now has a dedicated design spec in `docs/HISTORY_DOCS.md` plus scaffolded package namespaces under `tools/history_docs/` and `prompts/history_docs/`, but no executable workflow yet.

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

- [x] **G1-01**  
  `Outcome:` Create a symbol-quality inventory by language from current analyzers and graph artifacts.  
  `Definition of Done:` Gaps are documented in comments/tests for Python, Java, C++, JS/TS, Go, Rust, and C#.  
  `Verification Command(s):` `pytest tests/test_language_analyzers.py tests/test_graph_build_and_retrieval.py`

- [x] **G1-02**  
  `Outcome:` Extend deterministic symbol extraction to prefer analyzer-derived symbol facts (name, kind, qualified name, span, owner) over regex fallback whenever available.  
  `Definition of Done:` Symbol inventory path uses analyzer facts first; regex span matching is fallback-only.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_repo_scanner_multilang.py`

- [x] **G1-03**  
  `Outcome:` Improve parent/child symbol ownership mapping for nested symbols and methods.  
  `Definition of Done:` Graph has reliable `parent_of`/`contains` symbol edges for nested ownership cases.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py`

- [x] **G1-04**  
  `Outcome:` Preserve stable symbol node IDs while improving symbol metadata quality.  
  `Definition of Done:` ID derivation remains deterministic and unaffected by span-only changes.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py::test_generate_writes_deterministic_graph_artifacts`

### Phase 2 — Multi-Language Dependency / Import Edges

`Objective:` Broaden `imports` edge creation beyond Python with conservative resolution.
`Why:` Dependency-oriented questions currently underperform for non-Python projects.
`Dependencies:` Phase 1.
`Completion Criteria:` Imports edges are emitted for major supported languages when target resolution is reliable; unresolved imports are deterministic and non-fatal.

- [x] **G2-01**  
  `Outcome:` Define normalized import/dependency resolution strategy per language (`java`, `cpp`, `javascript`, `typescript`, `go`, `rust`, `csharp`).  
  `Definition of Done:` Graph build has a language-router for import edge extraction rather than Python-only logic.  
  `Verification Command(s):` `pytest tests/test_repo_scanner_multilang.py tests/test_graph_build_and_retrieval.py`

- [x] **G2-02**  
  `Outcome:` Implement Java + JS/TS import edge resolution to known files/modules.  
  `Definition of Done:` `imports` edges appear in graph artifacts for these languages in mixed-language fixtures.  
  `Verification Command(s):` `pytest tests/test_workflow_generate_multilang.py`

- [x] **G2-03**  
  `Outcome:` Implement conservative C++ include and C#/Go/Rust dependency edge resolution where deterministic mapping is possible.  
  `Definition of Done:` Supported patterns emit edges; ambiguous/unresolved entries are skipped consistently with reason metadata.  
  `Verification Command(s):` `pytest tests/test_language_analyzers.py tests/test_graph_build_and_retrieval.py`

- [x] **G2-04**  
  `Outcome:` Add regression tests for import edge creation and tie-break determinism across languages.  
  `Definition of Done:` Edge counts/types are asserted in graph artifact tests for multilingual fixtures.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`

### Phase 3 — Incremental Graph Build And Reuse

`Objective:` Move graph generation from full rebuild to deterministic incremental behavior.
`Why:` Full rewrite is expensive and limits scalability for large repos.
`Dependencies:` Phases 1-2.
`Completion Criteria:` Unchanged runs reuse graph state; changed runs update only impacted graph substructures while preserving deterministic output.

- [x] **G3-01**  
  `Outcome:` Extend graph manifest with deterministic fingerprints (scan/retrieval/hierarchy inputs and build version).  
  `Definition of Done:` Manifest can detect no-op build conditions and reproducibility context.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py`  
  `Result:` pass

- [x] **G3-02**  
  `Outcome:` Introduce incremental graph planner (full rebuild vs partial update vs no-op).  
  `Definition of Done:` Planner decisions are deterministic and test-covered.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_workflow_propose_updates.py`  
  `Result:` pass

- [x] **G3-03**  
  `Outcome:` Persist/reuse per-file or per-node graph fragments to avoid rewriting unaffected regions.  
  `Definition of Done:` Propose-updates path updates impacted subtree + related section/commit edges only.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py`  
  `Result:` pass

- [x] **G3-04**  
  `Outcome:` Add deterministic equivalence tests for full rebuild vs incremental rebuild outputs.  
  `Definition of Done:` Outputs match for unchanged inputs; partial changes alter only expected graph regions.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`  
  `Result:` pass

### Phase 4 — Ask Evidence Quality Tightening

`Objective:` Improve graph-derived evidence quality and prompt-facing metadata.
`Why:` Current `related_symbols` derivation is token-based and can include noise; prompt currently underuses graph context fields.
`Dependencies:` Phases 1-3.
`Completion Criteria:` `ask` evidence pack uses graph-grounded related entities and richer inclusion reasons, and prompt consumes them explicitly.

- [x] **G4-01**  
  `Outcome:` Derive `related_symbols` from traversed graph symbol nodes and symbol index, not token sweeps from candidate chunk text.  
  `Definition of Done:` Related symbols are stable, relevant, and traceable to graph nodes.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py tests/test_graph_build_and_retrieval.py`  
  `Result:` pass

- [x] **G4-02**  
  `Outcome:` Add graph-path evidence metadata (edge/path reason) to inclusion reasons used by prompts and audits.  
  `Definition of Done:` Inclusion reasons include deterministic graph rationale beyond `graph:<node_type>:<label>`.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`  
  `Result:` pass

- [x] **G4-03**  
  `Outcome:` Update query prompt builder to include `primary_chunks`, `related_files`, `related_symbols`, `related_sections`, and inclusion score breakdowns.  
  `Definition of Done:` Prompt inputs reflect full evidence pack shape.  
  `Verification Command(s):` `pytest tests/test_render_and_workflow_misc.py tests/test_workflow_generate_and_ask.py`  
  `Result:` pass

- [x] **G4-04**  
  `Outcome:` Tighten rerank signal definitions and validate deterministic tie-break behavior under mixed lexical/graph evidence.  
  `Definition of Done:` Rerank math and tie-break rules are test-covered and documented.  
  `Verification Command(s):` `pytest tests/test_graph_build_and_retrieval.py tests/test_graph_index_and_candidate_sources.py`  
  `Result:` pass

### Phase 5 — Vector-Readiness Formalization (Disabled By Default)

`Objective:` Formalize extension seams so vector retrieval can be added later without ask/workflow rework.
`Why:` Current placeholder exists but orchestration/config/CLI contracts are only partially wired.
`Dependencies:` Phase 4.
`Completion Criteria:` Candidate-source orchestration, config schema, and CLI placeholders are coherent and no-op when vector is disabled.

- [x] **G5-01**  
  `Outcome:` Unify candidate-source abstraction so lexical, graph, and vector sources share an orchestration contract.  
  `Definition of Done:` Ask retrieval orchestration is source-pluggable without source-specific branching leaks.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`  
  `Result:` pass

- [x] **G5-02**  
  `Outcome:` Thread `vector_enabled`/`vector_top_k` through ask workflow settings and CLI/config resolution paths (still disabled by default).  
  `Definition of Done:` Config and CLI parse/propagate vector placeholders without changing current default behavior.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_workflow_generate_and_ask.py`  
  `Result:` pass

- [x] **G5-03**  
  `Outcome:` Document vector extension points in usage + architecture docs with explicit non-goals for current phase.  
  `Definition of Done:` Docs explain how to add vector source later without implying it is active now.  
  `Verification Command(s):` `rg -n \"vector\" docs/USAGE.md ARCHITECTURE.md`  
  `Result:` pass

### Phase 6 — Commit-Aware Q&A Integration

`Objective:` Make commit edges (`changed_in`, `impacts_section`) intentionally useful in `ask`.
`Why:` Commit-aware edges exist in graph artifacts but are not yet leveraged by intent routing and evidence assembly.
`Dependencies:` Phases 2-4.
`Completion Criteria:` Change-impact questions reliably retrieve commit/file/section evidence via graph traversal and reranking.

- [x] **G6-01**  
  `Outcome:` Add deterministic change-impact intent classification for commit-oriented questions.  
  `Definition of Done:` Intent router explicitly recognizes change/impact/commit question patterns and edge preferences.  
  `Verification Command(s):` `pytest tests/test_graph_index_and_candidate_sources.py`  
  `Result:` pass

- [x] **G6-02**  
  `Outcome:` Add traversal preferences for `changed_in` and `impacts_section` edges and include commit nodes in related evidence outputs.  
  `Definition of Done:` Ask evidence pack exposes commit-aware context for applicable queries.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py tests/test_workflow_generate_and_ask.py`  
  `Result:` pass

- [x] **G6-03**  
  `Outcome:` Add regression tests for commit-impact Q&A using propose-updates-produced artifacts.  
  `Definition of Done:` Tests assert grounded citations and non-fabricated unknown handling (`TBD`) for change-impact questions.  
  `Verification Command(s):` `pytest tests/test_workflow_propose_updates.py tests/test_graph_build_and_retrieval.py`  
  `Result:` pass

### Phase 7 — Documentation First-Class Graph Subsystem

`Objective:` Make graph subsystem docs complete and maintainable.
`Why:` Current docs describe graph at high level but not enough for contributors implementing next steps.
`Dependencies:` Phases 1-6 (can start in parallel for baseline docs).
`Completion Criteria:` Docs cover graph schema, build modes, ask flow, commit-aware behavior, and extension seams with examples.

- [x] **G7-01**  
  `Outcome:` Expand usage docs with graph node/edge schema, stable IDs, artifact examples, and troubleshooting.  
  `Definition of Done:` `docs/USAGE.md` has a dedicated graph section with practical inspection commands.  
  `Verification Command(s):` `rg -n \"graph/manifest|nodes.jsonl|edges.jsonl|changed_in|impacts_section\" docs/USAGE.md`  
  `Result:` pass

- [x] **G7-02**  
  `Outcome:` Expand architecture docs with graph build lifecycle (full vs incremental), ask evidence flow, and commit-aware routing.  
  `Definition of Done:` `ARCHITECTURE.md` reflects current implementation and next-step architecture boundaries.  
  `Verification Command(s):` `rg -n \"Engineering Graph|incremental|candidate source|commit-aware\" ARCHITECTURE.md`  
  `Result:` pass

- [x] **G7-03**  
  `Outcome:` Add contributor notes linking TASKS roadmap phases to relevant modules/tests for pickup by future agents.  
  `Definition of Done:` Docs map each major phase to code areas and test files.  
  `Verification Command(s):` `rg -n \"Phase 1|Phase 2|test_graph\" docs/USAGE.md docs/EXTENDING.md TASKS.md`  
  `Result:` pass

### Phase 8 — Intensive Ask Mode (Structured Cross-File Corpus Screening)

`Objective:` Add a high-compute `ask` mode that screens a deterministic whole-repo corpus before the final answer call.
`Why:` Some questions need broader repo context than lexical/graph retrieval can cheaply surface, especially when larger-context hosted models are available.
`Dependencies:` Phases 4-7.
`Completion Criteria:` `ask --mode intensive` builds or reuses a structured corpus, screens each chunk through the provider abstraction, persists artifacts, and returns conservative answers when no excerpts are selected.

- [x] **G8-01**  
  `Outcome:` Add file-aware intensive corpus building with cross-file chunk packing, whole-file preservation under budget, and oversized-file splitting only when necessary.  
  `Definition of Done:` Corpus artifacts are deterministic, reusable by fingerprint, and preserve exact file/line provenance via ordered segments.  
  `Verification Command(s):` `pytest tests/test_intensive_corpus.py`  
  `Result:` pass

- [x] **G8-02**  
  `Outcome:` Add structured chunk screening and final-answer orchestration for `ask --mode intensive`.  
  `Definition of Done:` Intensive mode persists corpus/run artifacts, validates excerpt ranges against chunk segments, and returns `TBD` conservatively when screening finds nothing relevant.  
  `Verification Command(s):` `pytest tests/test_workflow_generate_and_ask.py`  
  `Result:` pass

- [x] **G8-03**  
  `Outcome:` Thread intensive mode through CLI/config defaults and document the public interface.  
  `Definition of Done:` `ask` accepts `--mode`, `--repo-root`, `--intensive-chunk-tokens`, and `--intensive-max-selected-excerpts`, while `docs/USAGE.md` explains the mode and artifacts.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_render_and_workflow_misc.py && rg -n \"intensive\" docs/USAGE.md`  
  `Result:` pass

### Phase 9 — EngLLM Multi-Tool Restructure

`Objective:` Reshape the repo from an SDD-first package into a multi-tool repository-analysis toolkit.
`Why:` SDD generation, grounded Q&A, and future repo-focused tools need a shared platform instead of continuing to accrete inside one workflow-centric package.
`Dependencies:` Phases 1-8.
`Completion Criteria:` Package/CLI rename is complete, shared deterministic logic lives under `core/`, tool orchestration lives under `tools/`, future integrations have explicit capability interfaces, and artifacts use the workspace/tool layout.

- [x] **G9-01**  
  `Outcome:` Rename the package and CLI from `sddraft` to `engllm` with no compatibility shim.  
  `Definition of Done:` Imports, package metadata, docs, and CLI entrypoints use `engllm` only.  
  `Verification Command(s):` `pytest tests/test_imports.py tests/test_cli.py`  
  `Result:` pass

- [x] **G9-02**  
  `Outcome:` Split shared deterministic logic into `core/` and tool-specific orchestration into `tools/ask`, `tools/sdd`, and `tools/repo`.  
  `Definition of Done:` Ask, SDD, and repo utility commands execute from tool namespaces while shared analysis stays reusable in `core/`.  
  `Verification Command(s):` `pytest tests/test_workflow_generate_and_ask.py tests/test_workflow_propose_updates.py`  
  `Result:` pass

- [x] **G9-03**  
  `Outcome:` Introduce workspace-first artifact layout and internal tool/integration contracts.  
  `Definition of Done:` Shared artifacts write to `artifacts/workspaces/<workspace_id>/shared/`, tool outputs write to `artifacts/workspaces/<workspace_id>/tools/<tool_name>/`, and future integration seams exist under `integrations/`.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_layer_boundaries.py`  
  `Result:` pass

## 4) History-Walk Documentation Tool Roadmap

This roadmap governs the new history-walk documentation tool. It is separate
from the completed graph/ask phases above because it is a new tool family with
its own internal phases.

### History Phase 0 — Concept Capture And Scaffolding

`Objective:` Capture the product concept, define terminology, and create the minimum repo scaffolding for future implementation.
`Why:` Later implementation work will span multiple agents and phases, so the tool needs one authoritative design baseline before deep coding starts.
`Dependencies:` None.
`Completion Criteria:` The tool has a dedicated design doc, explicit phase plan, initial package skeleton, and a documented first implementation slice.

- [x] **H0-01**  
  `Outcome:` Create a dedicated design document that defines purpose, non-goals, rendering philosophy, section-inclusion rules, artifacts, and major planned models.  
  `Definition of Done:` `docs/HISTORY_DOCS.md` exists and is the canonical planning reference for the tool.  
  `Verification Command(s):` `rg -n "History-Walk Documentation Tool Design|Section Inclusion Philosophy|Phased Implementation Plan" docs/HISTORY_DOCS.md`  
  `Result:` pass

- [x] **H0-02**  
  `Outcome:` Fit the tool into the EngLLM architecture and contributor docs.  
  `Definition of Done:` `ARCHITECTURE.md`, `docs/EXTENDING.md`, and `README.md` all reference the planned tool and its design location.  
  `Verification Command(s):` `rg -n "history-docs|HISTORY_DOCS.md" README.md ARCHITECTURE.md docs/EXTENDING.md`  
  `Result:` pass

- [x] **H0-03**  
  `Outcome:` Create initial package scaffolding for the future tool and prompt namespace.  
  `Definition of Done:` `src/engllm/tools/history_docs/` and `src/engllm/prompts/history_docs/` exist and are importable.  
  `Verification Command(s):` `pytest tests/test_imports.py`  
  `Result:` pass

- [x] **H0-04**  
  `Outcome:` Define a narrow first implementation slice instead of attempting the full tool at once.  
  `Definition of Done:` The design doc and task board both identify explicit target-commit traversal + shared history manifests as the first slice.  
  `Verification Command(s):` `rg -n "First-slice scope|explicit target-commit|shared history artifacts" docs/HISTORY_DOCS.md TASKS.md`  
  `Result:` pass

### History Phase 1 — Checkpoint Selection And Git History Traversal

`Objective:` Resolve one explicit checkpoint commit at a time and map commits into deterministic checkpoint windows.
`Why:` Everything else depends on stable checkpoint identity and interval boundaries.
`Dependencies:` History Phase 0.
`Completion Criteria:` The tool can resolve target commits, derive prior boundaries, and emit canonical history manifests without mutating the working tree.

- [x] **H1-01**  
  `Outcome:` Define explicit checkpoint identity and prior-boundary rules for manual-first history traversal.  
  `Definition of Done:` Checkpoint IDs, timestamps, explicit previous overrides, and artifact-derived ancestor lookup rules are deterministic and documented.  
  `Verification Command(s):` `.venv/bin/pytest -q --no-cov tests/test_history_docs_h1.py -k "checkpoint_id_for or uses_latest_ancestor or explicit_previous_override"`  
  `Result:` pass

- [x] **H1-02**  
  `Outcome:` Implement read-only git history traversal for target commits and interval enumeration.  
  `Definition of Done:` `engllm history-docs build` resolves target commits, validates prior boundaries, and produces ordered commit windows without checking out historical trees.  
  `Verification Command(s):` `.venv/bin/pytest -q --no-cov tests/test_history_docs_h1.py -k "git_history_helpers or rejects_equal_previous or rejects_non_ancestor_previous or cli_build"`  
  `Result:` pass

- [x] **H1-03**  
  `Outcome:` Persist checkpoint and interval manifests under shared history artifacts.  
  `Definition of Done:` `shared/history/checkpoint_plan.json` and `shared/history/intervals.jsonl` are canonically rewritten, upsert safely on rerun, and are test-covered.  
  `Verification Command(s):` `.venv/bin/pytest -q --no-cov tests/test_history_docs_h1.py -k "initial_run_writes_manifests or rerun_is_idempotent or workspace_override_changes_artifact_location"`  
  `Result:` pass

### History Phase 2 — Checkpoint Snapshot Analysis

`Objective:` Analyze the repository as it existed at each checkpoint.
`Why:` The final docs must describe the checkpoint snapshot holistically, not just the deltas.
`Dependencies:` History Phase 1.
`Completion Criteria:` Snapshot analysis can emit structural inventories, subsystem candidates, and build/dependency source detection for a checkpoint commit.

- [ ] **H2-01**  
  `Outcome:` Materialize or inspect checkpoint snapshots without disturbing the user’s working tree.  
  `Definition of Done:` Snapshot analysis can run against checkpoint-resolved repo state safely and deterministically.

- [ ] **H2-02**  
  `Outcome:` Reuse or adapt existing scanner infrastructure for checkpoint-specific structural analysis.  
  `Definition of Done:` Snapshot structural models include files, symbols, module candidates, and subsystem signals.

- [ ] **H2-03**  
  `Outcome:` Detect build/package infrastructure and dependency sources at the checkpoint.  
  `Definition of Done:` Snapshot artifacts record dependency-manifest sources such as `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, or similar files when present.

### History Phase 3 — Interval Delta Analysis

`Objective:` Analyze the commit window between checkpoints for design-meaningful changes.
`Why:` Smaller change windows are the core mechanism for improving documentation quality over time.
`Dependencies:` History Phase 1.
`Completion Criteria:` Interval analysis produces structured change candidates rather than only raw changed-file lists.

- [ ] **H3-01**  
  `Outcome:` Classify commits and diffs for architectural, interface, algorithmic, dependency, and infrastructure signals.  
  `Definition of Done:` Interval analysis emits typed change records instead of prose-only summaries.

- [ ] **H3-02**  
  `Outcome:` Detect candidate subsystem, interface, and dependency changes from interval evidence.  
  `Definition of Done:` Interval artifacts include structured design-change candidates with evidence links.

- [ ] **H3-03**  
  `Outcome:` Add algorithm-emergence and strategy-variant signals without overclaiming.  
  `Definition of Done:` The tool surfaces candidate algorithm/variant evidence conservatively for later capsule generation.

### History Phase 4 — Structured Documentation Model

`Objective:` Build the internal checkpoint model that represents documentation-worthy concepts at each checkpoint.
`Why:` The tool should update structured state over time rather than patch prose directly.
`Dependencies:` History Phases 2-3.
`Completion Criteria:` A checkpoint model can merge prior checkpoint state, current snapshot evidence, and interval delta evidence deterministically.

- [ ] **H4-01**  
  `Outcome:` Define the first concrete checkpoint documentation models.  
  `Definition of Done:` Initial models exist for checkpoints, subsystems/modules, dependencies, sections, and evidence links.

- [ ] **H4-02**  
  `Outcome:` Implement deterministic merge/update rules for checkpoint concepts.  
  `Definition of Done:` Stable concepts persist, changed concepts revise cleanly, and stale concepts retire conservatively.

- [ ] **H4-03**  
  `Outcome:` Persist versioned checkpoint models and evidence links per checkpoint.  
  `Definition of Done:` Tool artifacts include inspectable checkpoint model JSON for every built checkpoint.

### History Phase 5 — Section Inference And Inclusion Rules

`Objective:` Decide which sections should appear for a checkpoint and how deep they should be.
`Why:` The rendered docs should use a stable core plus optional evidence-driven sections, not a rigid filler template.
`Dependencies:` History Phase 4.
`Completion Criteria:` Section plans are evidence-scored, confidence-weighted, and bounded in depth.

- [ ] **H5-01**  
  `Outcome:` Implement stable-core section planning for the first slice.  
  `Definition of Done:` The tool can always plan Introduction, Architectural Overview, Subsystems and Modules, Dependencies, and Build/Development Infrastructure.

- [ ] **H5-02**  
  `Outcome:` Add optional-section scoring with explicit thresholds.  
  `Definition of Done:` Optional sections appear only when evidence warrants them.

- [ ] **H5-03**  
  `Outcome:` Scale section depth using evidence strength instead of fixed prose quotas.  
  `Definition of Done:` Section plans include confidence and depth metadata for rendering.

### History Phase 6 — Algorithm Knowledge Capsules

`Objective:` Create focused internal algorithm artifacts for meaningful algorithmic clusters.
`Why:` One of the main product bets is that history windows improve algorithm documentation quality.
`Dependencies:` History Phases 3-5.
`Completion Criteria:` The tool can emit algorithm capsules when evidence supports them and link them back into the checkpoint model.

- [ ] **H6-01**  
  `Outcome:` Detect algorithm families and meaningful algorithm clusters conservatively.  
  `Definition of Done:` The tool avoids labeling every changed module as an algorithm.

- [ ] **H6-02**  
  `Outcome:` Capture shared abstractions, strategy variants, data structures, phases, and assumptions when evidence exists.  
  `Definition of Done:` Capsules preserve structured algorithm evidence rather than only prose summaries.

- [ ] **H6-03**  
  `Outcome:` Link algorithm capsules into checkpoint sections and concept models.  
  `Definition of Done:` Final checkpoint docs can incorporate algorithm capsules without requiring every capsule to become a top-level section.

### History Phase 7 — Dependency Documentation Pipeline

`Objective:` Document important direct dependencies for each checkpoint.
`Why:` Dependency infrastructure is a meaningful part of project understanding and should evolve with checkpoints.
`Dependencies:` History Phases 2, 4, and 5.
`Completion Criteria:` Dependencies are extracted from build/package infrastructure and rendered using the required two-short-paragraph format.

- [ ] **H7-01**  
  `Outcome:` Parse and normalize direct dependencies from build/package files.  
  `Definition of Done:` The tool can emit a dependency inventory per checkpoint from project manifests and lockfiles where appropriate.

- [ ] **H7-02**  
  `Outcome:` Classify important dependencies for documentation inclusion.  
  `Definition of Done:` Weak or incidental dependencies can be omitted when they would only create noise.

- [ ] **H7-03**  
  `Outcome:` Generate concise general-purpose and project-specific dependency summaries.  
  `Definition of Done:` Each documented dependency renders as two short paragraphs, not a file-usage audit.

### History Phase 8 — Rendering Engine

`Objective:` Render full checkpoint docs from the structured checkpoint model.
`Why:` The final deliverable must be a holistic design document for each checkpoint, not a patch report.
`Dependencies:` History Phases 4-7.
`Completion Criteria:` Each checkpoint can render to a complete present-state Markdown document plus structured debug outputs.

- [ ] **H8-01**  
  `Outcome:` Implement checkpoint document rendering from section plans and checkpoint models.  
  `Definition of Done:` Rendered docs follow present-state, design-document style without release-note framing.

- [ ] **H8-02**  
  `Outcome:` Support stable core sections plus evidence-driven optional sections.  
  `Definition of Done:` Omitted sections stay omitted instead of turning into boilerplate.

- [ ] **H8-03**  
  `Outcome:` Emit structured debug artifacts alongside Markdown renders.  
  `Definition of Done:` Render inputs and outputs remain inspectable for later agent work.

### History Phase 9 — Validation And Quality Evaluation

`Objective:` Validate style, evidence coverage, and regression quality across checkpoints.
`Why:` The tool must stay coherent and avoid drifting into filler, speculation, or release-note phrasing.
`Dependencies:` History Phases 1-8.
`Completion Criteria:` The tool has dedicated tests and validation checks for checkpoint quality, evidence coverage, and style.

- [ ] **H9-01**  
  `Outcome:` Add deterministic tests for checkpoint planning, interval analysis, and checkpoint-model persistence.  
  `Definition of Done:` History traversal and model-building logic are testable without live LLM access.

- [ ] **H9-02**  
  `Outcome:` Add rendering and style checks that detect release-note phrasing and weak filler sections.  
  `Definition of Done:` Rendered checkpoint docs are validated as standalone present-state documents.

- [ ] **H9-03**  
  `Outcome:` Add targeted quality sampling for algorithm and dependency sections.  
  `Definition of Done:` The highest-risk sections have explicit regression coverage and review criteria.

### History Phase 10 — Future Extensions

`Objective:` Capture non-blocking future directions without mixing them into the first implementation arc.
`Why:` The tool has a large extension surface, but the initial build should stay narrow and prove the architecture first.
`Dependencies:` History Phases 1-9.
`Completion Criteria:` Future work remains documented and bounded instead of leaking into the first implementation slice.

- [ ] **H10-01**  
  `Outcome:` Document optional evolution-report outputs separate from the main checkpoint docs.  
  `Definition of Done:` Future diff/timeline views are clearly separated from the core holistic-doc workflow.

- [ ] **H10-02**  
  `Outcome:` Document future directions for richer rationale extraction, diagrams, checkpoint browsing, and confidence visualization.  
  `Definition of Done:` Extension ideas remain explicit but non-blocking.

## 5) Testing And Validation Requirements (For All Phases)

- Unit tests for symbol extraction, edge construction, graph loading, traversal, and rerank math.
- Integration tests for `generate`, `propose-updates`, and `ask` with and without graph artifacts.
- Deterministic artifact tests for graph IDs, counts, ordering, and full-vs-incremental equivalence.
- Regression tests for fallback behavior when hierarchy/graph artifacts are missing or corrupt.
- Commit-aware Q&A tests that assert grounded citations and conservative TBD behavior.
- History-docs phases should add deterministic tests for checkpoint selection, interval metadata, checkpoint-model persistence, and present-state rendering quality.
- Quality gates required on every phase PR:
  - `ruff check src tests`
  - `mypy src`
  - `pytest -q`
  - coverage must remain `>= 90%`

## 6) Nice-To-Have / Later (Non-Blocking)

- [ ] Real embedding generation and local vector index implementation (still file-backed, deterministic build metadata).
- [ ] Advanced cross-language semantic/call-graph inference beyond import/dependency-level relations.
- [ ] Richer section impact inference at symbol-flow level, not only evidence overlap.
- [ ] Graph visualization/export tooling for artifact inspection (CLI report or static visualization).
- [ ] Optional benchmark harness for large mixed-language repos with memory/runtime trend reporting.

## 7) Future Tool Tracks (Planned, Not Implemented Yet)

- [ ] CI log summarizer built on shared repo analysis plus `CiLogClient` integrations.
- [ ] Code review automation built on `RepoHostClient` and `IssueTrackerClient`, with vendor adapters such as Atlassian and GitLab living behind those interfaces.
- [ ] Release-note generation from commit ranges with optional issue-enrichment support.

## 8) Historical Completed Milestones (Preserved)

- [x] **T-001..T-006** v1 hardening baseline (task board, CLI override parity, error handling, regression coverage).
- [x] **T-007..T-010** v1 acceptance-proofing baseline (extensibility docs, Gemini optional dependency/docs, acceptance tests, quality gate pass).
- [x] Graph baseline implemented: graph artifacts + ask graph augmentation + deterministic reranking + CLI graph flags + fallback behavior.

## 9) Blocked

- [ ] None currently.
