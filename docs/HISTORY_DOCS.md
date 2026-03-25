# History-Walk Documentation Tool Design

This document defines the design direction for the history-walk
project-documentation tool in EngLLM.

The goal of the tool is to generate **holistic documentation snapshots** for a
repository at historical checkpoints. Each snapshot should describe the project
**as it existed at that point in time**.

The tool uses deltas internally, but its rendered output must not read like
release notes or a changelog.

## Purpose

The history-walk documentation tool exists to improve project documentation by
reasoning over smaller windows of change instead of reconstructing a mature code
base all at once.

The tool should:

- generate multiple historical checkpoint documents for one repository
- use repo snapshots, interval commits, diffs, and prior checkpoint models as evidence
- preserve architectural and algorithmic understanding that may fade in mature snapshots
- produce complete, coherent, standalone docs for each checkpoint
- stay conservative, evidence-backed, and inspectable

## Non-Goals

This tool is not:

- a release-notes generator
- a changelog generator
- a delta-first renderer
- a prose-only patching system that blindly carries text forward
- a rigid template filler that emits weak sections by default
- a file-by-file dependency audit

Future enhancements such as diagrams, evolution timelines, and UI browsing are
valuable, but they are not part of the initial implementation slice.

## Tool Placement In EngLLM

This tool lives in the EngLLM package as a first-class tool namespace.

Planned package placement:

- `src/engllm/tools/history_docs/`: tool orchestration, tool-facing models, and rendering
- `src/engllm/prompts/history_docs/`: history-docs-specific prompt builders and templates
- `src/engllm/core/analysis/`: shared history traversal, checkpoint selection, snapshot analysis, delta analysis, and section-inference helpers that may later be reused by other tools

Planned future CLI surface:

- `engllm history-docs build`

History Phases 1 through 11-02 now implement that command end to end, from
single-checkpoint traversal through semantic checkpoint advisories,
checkpoint-scoped semantic subsystem/capability maps, final rendered checkpoint
Markdown, and build-integrated validation. H10 adds an internal benchmark and
evaluation harness around those rendered artifacts while keeping the public CLI
unchanged.

## Current Implemented Slice

The current implementation covers History Phases 1 through 11-02:

- explicit target-commit selection via `engllm history-docs build`
- optional explicit previous-checkpoint override
- artifact-derived previous-checkpoint lookup using the latest valid ancestor
- shared checkpoint and interval manifests under `shared/history/`
- checkpoint-scoped `semantic_checkpoint_plan.json` artifacts with:
  - first-parent ancestry analysis up to the requested target commit
  - deterministic candidate detection from tags, interface/dependency shifts,
    build-manifest changes, broad changes, new top-level areas, and merge anchors
  - one structured LLM planning pass over the bounded candidate set
  - heuristic fallback when planner generation fails or returns no judgments
- temporary checkpoint snapshot export via `git archive` into a disposable temp
  directory
- structural scanning of the checkpoint snapshot without mutating the user's
  working tree
- tool-scoped snapshot structural models with files, code summaries, symbol
  summaries, subsystem candidates, and build-source metadata
- checkpoint-scoped `semantic_structure_map.json` artifacts with:
  - one structured LLM clustering pass over compact H2 subsystem/module evidence
  - exclusive semantic subsystem partitions over active modules
  - non-exclusive capability labels attached to modules and semantic subsystems
  - heuristic path-based fallback when clustering fails or validates poorly
  - internal path-based baseline control still available for H10 comparisons
  - plain semantic clustering without H11-03 remains shadow-only
- checkpoint-scoped `semantic_context_map.json` artifacts with:
  - one structured LLM extraction pass over compact snapshot plus semantic-structure evidence
  - exactly one system node plus evidence-backed context nodes and interface candidates
  - heuristic fallback that always preserves a conservative system boundary node
  - public `engllm history-docs build` now renders `System Context` and
    `Interfaces` through the promoted H11-03 path
- tool-scoped `interval_delta_model.json` artifacts with:
  - first-parent commit diff semantics
  - diff-only fallback when the previous snapshot artifact is unavailable
  - structured subsystem, interface, dependency, and algorithm candidate signals
- checkpoint-scoped `interval_interpretation.json` artifacts with:
  - one structured LLM interpretation pass over compact H3/H2 evidence
  - design-change insights, rationale clues, and significant change windows
  - validation that prevents invented commits, change ids, subsystem ids, or evidence links
  - conservative heuristic fallback when interpretation fails or returns empty results
- tool-scoped `checkpoint_model.json` artifacts with:
  - active and retired subsystem/module/dependency-source concepts
  - optional semantic display names, summaries, capability labels, and baseline
    subsystem references on subsystem concepts
  - deterministic core section stubs
  - previous-model fallback when the prior checkpoint model artifact is missing
- tool-scoped `section_outline.json` artifacts with:
  - fixed-order core and optional section plans
  - conservative evidence-scored inclusion decisions
  - confidence and depth metadata for later rendering
- tool-scoped `algorithm_capsules/` artifacts with:
  - `index.json` plus one JSON file per capsule
  - deterministic capsule linking back into checkpoint concepts and sections
  - an evidence-gated `algorithms_core_logic` section in the final section plan
- tool-scoped `dependencies.json` artifacts with:
  - parsed and normalized direct dependency inventories aggregated by
    ecosystem plus dependency name
  - conservative warnings for unsupported or ambiguous manifest syntax
  - LLM-assisted two-paragraph dependency summaries
  - links back into checkpoint dependency-source concepts
- deterministic final `checkpoint.md` rendering plus `render_manifest.json`
  trace artifacts built only from H4-H7 structured models
- build-integrated `validation_report.json` artifacts with:
  - hard structural/evidence checks that fail the build after the report is
    written
  - soft style/quality warnings for `TBD` dependency summaries, thin algorithm
    capsules, and release-note phrasing
- internal H10 benchmark/evaluation artifacts with:
  - reusable benchmark case manifests covering small, medium,
    algorithm-heavy, dependency-heavy, and architecture-heavy histories
  - LLM-judged `quality_report.json` artifacts per benchmarked variant
  - deterministic `comparison_report.json` artifacts that compare variants from
    structured rubric outputs without another model call
  - top-level benchmark `suite_manifest.json` artifacts that aggregate
    per-case results and coverage tags while leaving `engllm history-docs build`
    unchanged
  - internal `real_benchmark` runs plus `promotion_gate_report.json` artifacts
    for deciding whether shadow-mode semantic variants beat the path baseline on
    real sibling repositories

Quarterly checkpoint auto-selection is still deferred. H11-01 keeps the
explicit target commit authoritative and uses the semantic checkpoint plan as
an advisory artifact only. H11-02 similarly keeps semantic subsystem/capability
clustering advisory in normal builds while enabling H10 to compare a semantic
grouping variant against the path-based baseline.

## Terminology

Use these terms consistently during implementation.

- `checkpoint`: one selected point in repository history that gets a full standalone documentation snapshot
- `checkpoint window`: the commit interval between the prior checkpoint and the current checkpoint
- `snapshot analysis`: structural analysis of the repository as it exists at the checkpoint commit
- `interval delta analysis`: analysis of the commits and diffs inside the checkpoint window
- `checkpoint model`: the structured internal documentation model representing the project at one checkpoint
- `algorithm capsule`: a focused internal artifact describing one meaningful algorithm or algorithm family
- `section plan`: the evidence-scored outline describing which sections should appear and how deep they should go
- `render mode`: final holistic rendering of a checkpoint document
- `update mode`: internal model-update logic that merges prior checkpoint state, current snapshot evidence, and interval deltas

## Core Product Hypothesis

Documentation quality can improve when the system documents a repository
incrementally over time because smaller windows of change are easier to analyze
accurately than reconstructing years of design history from a single mature
snapshot.

This is especially important for:

- architectural decomposition that becomes implicit over time
- algorithm introductions and reshapes
- strategy families and variant emergence
- dependency and build-infrastructure evolution
- rationale clues that show up in commits, tests, and focused code windows

## Rendering Philosophy

Every checkpoint document must be:

- standalone
- holistic
- coherent
- present-state for that checkpoint
- written as if it were the current design document at that moment

Internally, the tool should reason over:

- prior checkpoint docs and models
- interval commits and diffs
- the repo snapshot at the checkpoint

Rendered output should avoid phrases such as:

- "since last version"
- "new this quarter"
- "recently added"

unless the system is explicitly generating a separate history/evolution artifact,
which is not part of the main checkpoint document.

## Section Inclusion Philosophy

The final checkpoint document should use:

- a stable core
- optional evidence-driven sections

### Stable core sections

These are expected in most checkpoints:

- Introduction
- System Context
- Architectural Overview
- Subsystems and Modules
- Interfaces
- Algorithms and Core Logic
- Dependencies
- Build and Development Infrastructure

### Optional evidence-driven sections

These should appear only when evidence supports them:

- Strategy Variants and Design Alternatives
- Data and State Management
- Error Handling and Robustness
- Performance Considerations
- Security Considerations
- Design Notes and Rationale
- Limitations and Constraints

### Inclusion rule

A section must be:

- evidence-triggered
- confidence-weighted
- scope-bounded

Implementation should support section scoring using signals such as:

- structural signals
- behavioral signals
- historical signals
- usage signals

A weak section should be omitted rather than rendered as filler.

## Dependency Documentation Rule

Dependencies should come from build and package infrastructure at the checkpoint,
not from ad hoc token matching.

Each dependency subsection should have exactly two short paragraphs:

1. a short paragraph describing what the dependency is and what it is generally used for
2. a short paragraph describing what the project uses it for

Dependency documentation should stay general and explanatory. It should not turn
into a file-by-file or method-by-method usage audit.

## Algorithm Documentation Rule

Algorithm documentation is one of the main reasons to analyze history
incrementally.

The tool should eventually support detection and documentation of:

- algorithmic modules introduced during a checkpoint window
- major reshapes of existing algorithms
- strategy or variant families for the same capability
- shared abstractions among variants
- meaningful data structures
- execution phases
- invariants or assumptions
- evidence-backed tradeoff clues from code, tests, and commits

To support that, the architecture should allow internal algorithm capsules that
feed the main checkpoint model without forcing every algorithm observation
straight into top-level prose.

## Core Architecture

The tool should separate **update mode** from **render mode**.

### Update mode

Update mode reasons over time.

Responsibilities:

1. select checkpoints
2. resolve repo snapshot for the current checkpoint
3. analyze interval commits and diffs since the prior checkpoint
4. load the prior checkpoint model when available
5. merge prior model + snapshot evidence + interval deltas into the current checkpoint model
6. derive section inclusion and depth decisions

### Render mode

Render mode produces the final holistic document.

Responsibilities:

1. load the checkpoint model and section plan
2. render a complete checkpoint document in present-state style
3. keep optional sections evidence-bounded
4. emit human-readable docs plus structured artifacts for inspection

## Planned Artifact Layout

The history-walk tool should use the EngLLM workspace split between shared
artifacts and tool-specific artifacts.

### Shared history artifacts

These are reusable by future tools such as release or evolution tooling.

- `artifacts/workspaces/<workspace_id>/shared/history/checkpoint_plan.json`
- `artifacts/workspaces/<workspace_id>/shared/history/intervals.jsonl`
- `artifacts/workspaces/<workspace_id>/shared/history/checkpoints/<checkpoint_id>/snapshot_manifest.json`

H1 and H2 currently implement these shared artifacts:

- `checkpoint_plan.json` as the authoritative checkpoint registry
- `intervals.jsonl` as the canonical ordered commit windows per checkpoint
- `snapshot_manifest.json` as the checkpoint-scoped record of temporary export,
  source-root mapping, manifest-search scope, and structural counts

### Tool-specific history-docs artifacts

- `artifacts/workspaces/<workspace_id>/tools/history_docs/manifest.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/snapshot_structural_model.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/interval_delta_model.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/interval_interpretation.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/checkpoint_model.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/section_outline.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/algorithm_capsules/index.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/dependencies.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/algorithm_capsules/*.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/checkpoint.md`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/render_manifest.json`
- `artifacts/workspaces/<workspace_id>/tools/history_docs/checkpoints/<checkpoint_id>/validation_report.json`

The exact file names can evolve, but the split between shared history traversal
artifacts and tool-specific rendered/model artifacts should remain.

Current H2 behavior intentionally keeps snapshot export temporary:

- the checkpoint tree is exported into a disposable temp directory
- the exported tree is deleted after structural analysis completes
- only inspectable manifests and structured models are persisted

Current H3 behavior adds one tool-scoped interval artifact:

- `interval_delta_model.json` as the deterministic checkpoint-window delta model
  derived from interval commits, first-parent diffs, current snapshot evidence,
  and previous-snapshot comparison when available

Current H12-01 behavior adds one checkpoint-scoped interval interpretation
artifact:

- `interval_interpretation.json` as the structured LLM-assisted interpretation
  layer over H3 evidence, with design-change insights, rationale clues,
  significant windows, and conservative heuristic fallback

Current H4 behavior adds one tool-scoped checkpoint-state artifact:

- `checkpoint_model.json` as the deterministic present-state model for one
  checkpoint, including active and retired concepts plus fixed core section
  records

Current H7 behavior adds one tool-scoped dependency artifact:

- `dependencies.json` as the canonical direct-dependency inventory for one
  checkpoint, including aggregated declarations, section-target metadata,
  summary text, and non-fatal warnings

Current H8 behavior adds final render outputs:

- `checkpoint.md` as the deterministic present-state checkpoint document built
  from `checkpoint_model.json`, `section_outline.json`, H6 capsules, and H7
  dependency summaries
- `render_manifest.json` as the structured trace of rendered sections, source
  artifact usage, and subsection counts

Current H9 behavior adds build-integrated validation output:

- `validation_report.json` as the deterministic validation artifact for final
  rendered checkpoint output
- hard structural/evidence violations fail `engllm history-docs build` only
  after the validation report is persisted
- soft quality/style findings remain warnings and do not fail the build

## Major Internal Models

These models are planned conceptual anchors for implementation. They do not all
need to be coded immediately, but the design should stay aligned with them.

### Repository history and checkpoint models

- `CheckpointSelectionPlan`
  - checkpoint cadence
  - ordered checkpoint list
  - commit resolution details
- `CheckpointDescriptor`
  - checkpoint id
  - timestamp
  - commit hash
  - prior checkpoint reference
- `CommitInterval`
  - start and end checkpoint references
  - included commits
  - diff metadata

### Snapshot and delta models

- `CheckpointSnapshotModel`
  - file inventory
  - symbol/module inventories
  - subsystem candidates
  - build/dependency manifests
  - structural summaries
- `IntervalDeltaModel`
  - commit classifications
  - design-change candidates
  - interface changes
  - dependency changes
  - algorithm-emergence signals

### Documentation concept models

- `CheckpointDocumentationModel`
  - project concept state at one checkpoint
  - structured references to sections and concepts
- `SubsystemConcept`
- `ModuleConcept`
- `InterfaceConcept`
- `AlgorithmConcept`
- `StrategyVariantConcept`
- `DependencyConcept`
- `BuildInfrastructureConcept`
- `SectionPlan`
- `EvidenceLink`
- `AlgorithmCapsule`

## Recommended First Implementation Slice

The first implementation slice should prove the end-to-end shape with a narrow,
useful subset rather than attempting the full system.

### First-slice scope

- explicit target-commit selection via `engllm history-docs build`
- artifact-derived previous-checkpoint lookup, with optional explicit override
- deterministic checkpoint/interval manifest generation
- read-only git history traversal with no checkout or worktree mutation
- temporary checkpoint snapshot export and structural analysis with no checkout
  or worktree mutation
- shared and tool-scoped artifacts only; no interval-delta reasoning,
  structured checkpoint model merge, rendering, or LLM calls yet

### First-slice rationale

This slice validates the main architecture without overcommitting early to the
hardest parts, especially algorithm capsules and deep section-inference logic.
It also creates reusable shared artifacts and checkpoint-structural models that
later phases can enrich.

## Phased Implementation Plan

### Phase 0 — Concept Capture And Scaffolding

Goal:

- capture the concept in repo docs
- define terms, artifact vocabulary, and non-goals
- create initial package scaffolding for the future tool

Expected outputs:

- this design doc
- `TASKS.md` history-docs roadmap
- initial `tools/history_docs/` and `prompts/history_docs/` package skeletons

### Phase 1 — Checkpoint Selection And Git History Traversal

Goal:

- support explicit target-checkpoint selection over git history
- map commits into checkpoint windows using the latest prior ancestor checkpoint
  by default
- establish deterministic checkpoint identifiers and shared manifests

Expected capabilities:

- explicit `--checkpoint-commit` selection
- optional `--previous-checkpoint-commit` override
- artifact-derived previous-checkpoint lookup
- initial checkpoint handling
- interval metadata between checkpoints
- commit resolution for each checkpoint without checking out historical trees

Expected outputs:

- checkpoint plan manifest
- interval metadata artifacts

Current implementation status:

- implemented
- single-checkpoint, manual-first
- quarterly checkpoint auto-selection deferred

### Phase 2 — Checkpoint Snapshot Analysis

Goal:

- analyze the repository as it exists at each checkpoint
- identify structure, modules, interfaces, dependencies, and likely subsystems

Expected capabilities:

- language-aware scanning
- symbol and module extraction
- build and package file detection
- dependency source identification
- subsystem clustering heuristics

Expected outputs:

- checkpoint structural model
- symbol/module inventories
- build/dependency manifests

Current implementation status:

- implemented via temporary `git archive` export
- scans configured source roots that still exist at the checkpoint
- records missing historical roots instead of failing
- searches build/dependency manifests only within source roots plus their
  ancestor chain to repo root

### Phase 3 — Interval Delta Analysis

Goal:

- analyze commits and diffs between checkpoints to find design-meaningful changes

Expected capabilities:

- commit classification
- diff classification
- new subsystem detection
- interface change detection
- dependency change detection
- algorithm/strategy emergence signals

Expected outputs:

- interval change model
- commit evidence summaries
- design-change candidates
- algorithm candidate detections

Current implementation status:

- implemented in `engllm history-docs build`
- uses first-parent diff semantics for merge commits and empty-tree diff basis
  for root commits
- compares against the previous checkpoint snapshot when available
- falls back to diff-only aggregation with `observed` status when the previous
  snapshot artifact is missing

### Phase 4 — Structured Documentation Model

Goal:

- maintain a structured model of the project at each checkpoint

Expected capabilities:

- merge prior checkpoint model + current snapshot + interval deltas
- preserve stable concepts
- revise changed concepts
- introduce new concepts
- retire stale concepts conservatively

Expected outputs:

- checkpoint documentation model
- evidence links between concepts and source evidence

Current implementation status:

- implemented in `engllm history-docs build`
- persists `checkpoint_model.json` after H3
- models subsystems, modules, dependency-source concepts, sections, and
  evidence links
- carries retired concepts forward for lineage while keeping section records
  active-only
- persists `section_outline.json` after H4
- keeps H4 checkpoint-model sections as fixed core stubs while H5 owns the
  scored section outline
- scores optional sections conservatively from H3/H4 evidence plus token
  heuristics
- persists `algorithm_capsules/index.json` plus one per-capsule JSON artifact
- links capsule ids into checkpoint concepts, section stubs, and the scored
  `algorithms_core_logic` / strategy-variant section plans
- persists deterministic `checkpoint.md` render output after H7
- persists `render_manifest.json` with section-level trace/debug metadata
- renders only `included` sections from the scored outline, in outline order
- carries dependency prose through from `dependencies.json` without new LLM
  calls during rendering
- keeps standalone interface concepts deferred

### Phase 5 — Section Inference And Inclusion Rules

Goal:

- determine which sections should appear and how deep they should be

Expected capabilities:

- evidence scoring
- threshold-based inclusion
- section depth selection
- omission of weak sections

Expected outputs:

- section inclusion engine
- section confidence metadata
- section outline for each checkpoint

### Phase 6 — Algorithm Knowledge Capsules

Goal:

- generate focused internal algorithm artifacts for meaningful algorithmic clusters

Expected capabilities:

- detect algorithm families
- detect strategy variants
- infer shared abstractions
- capture important data structures and phases
- preserve links back to code, tests, and commits

Expected outputs:

- algorithm capsule artifacts
- links from capsules into the checkpoint model and section outline

### Phase 7 — Dependency Documentation Pipeline

Goal:

- extract direct dependencies and document them concisely

Expected capabilities:

- parse build and dependency infrastructure
- identify direct dependencies
- classify important dependencies
- generate the required two-paragraph summaries
- attach dependencies to the checkpoint model

Expected outputs:

- dependency inventory per checkpoint
- dependency summaries for rendering

### Phase 8 — Rendering Engine

Goal:

- render full checkpoint docs from the structured model

Expected capabilities:

- present-state holistic rendering
- stable core sections plus optional evidence-driven sections
- avoidance of release-note framing
- emission of Markdown plus structured debug output

Expected outputs:

- final checkpoint documentation files
- optional JSON render/debug artifacts

Current implementation status:

- implemented in `engllm history-docs build`
- renders `checkpoint.md` deterministically with no additional provider calls
- uses `section_outline.json` as the source of truth for final section
  inclusion and ordering
- keeps omitted optional sections out of the final Markdown instead of filling
  them with boilerplate
- writes `render_manifest.json` as the structured render trace for later agent
  inspection

### Phase 9 — Validation And Quality Evaluation

Goal:

- validate that checkpoint docs stay coherent, evidence-backed, and style-correct

Expected capabilities:

- style validation
- section-presence validation
- evidence coverage checks
- regression comparisons across checkpoints
- focused sampling of algorithm and dependency quality

Expected outputs:

- deterministic tests
- validation reports
- quality gates for the tool

Expected quality gates:

- `.venv/bin/python -m black --check src tests`
- `.venv/bin/python -m isort --check-only src tests`
- `.venv/bin/ruff check src tests`
- `.venv/bin/mypy src`
- `.venv/bin/pytest -q`

### Phase 10 — Future Extensions

This is not an immediate implementation phase. It captures future directions.

Possible future work:

- separate history/evolution reports
- richer rationale extraction
- diagram generation
- deeper dependency role inference
- checkpoint browsing UI
- checkpoint-to-checkpoint diff views
- confidence visualization

## Long-Term Extension Points

The design should leave room for:

- alternate checkpoint cadences beyond calendar quarters
- history-derived graph or traceability artifacts
- richer rationale capture from commit messages and tests
- reuse of checkpoint artifacts by release or review tooling
- alternate renderers beyond Markdown

## Planning Status

This document is the planning baseline for the history-walk documentation tool.
The next implementation step should be the narrow first slice described above,
starting with explicit checkpoint selection and interval metadata rather than a
full history-docs engine.
