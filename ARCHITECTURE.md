# EngLLM Architecture

This document describes the internal architecture of EngLLM.

It explains how the system is organized internally and how the major subsystems interact.

Functional requirements are defined in `SPEC.md`.
Development rules are defined in `AGENTS.md`.

---

# 1. System Overview

EngLLM is a multi-tool repository analysis toolkit.

Today it ships four tool namespaces:

1. `engllm sdd ...` for SDD generation and update proposals
2. `engllm ask answer ...` / `engllm ask interactive` for grounded repository Q&A
3. `engllm repo ...` for shared repository utilities such as diff inspection
4. `engllm history-docs build` for checkpoint selection, history traversal,
   snapshot structural analysis, interval delta analysis, and checkpoint-state
   modeling

`engllm history-docs build` now also emits advisory semantic checkpoint-planning
artifacts, checkpoint-scoped semantic subsystem/capability maps,
checkpoint-scoped dependency documentation artifacts, final checkpoint
Markdown, and build-integrated validation over those historical snapshots. An
internal H10 benchmark harness now evaluates those rendered outputs with
structured LLM judging while keeping the public CLI unchanged.
The design for that tool lives in `docs/HISTORY_DOCS.md`.

The system intentionally performs deterministic analysis first and LLM generation second.

---

# 2. Core Pipeline Model

All current tools share the same high-level pattern:

```text
Toolkit Config + Tool-Specific Target/Inputs
↓
Repository Analysis
↓
Shared Artifact Build / Reuse
↓
Tool-Specific Evidence Construction
↓
Prompt Construction
↓
LLM Generation
↓
Tool-Specific Rendering / Reporting
```

Each stage is implemented by a separate subsystem.

---

# 3. Subsystems

## 3.1 Toolkit Layout

The package is organized around a shared platform plus tool namespaces:

* `domain/`: shared base models and error types
* `core/`: deterministic repository analysis, graph/retrieval/hierarchy, artifact persistence, workspace helpers, and tool registration contracts
* `llm/`: provider abstraction and concrete adapters only
* `integrations/`: external-system capability interfaces for future repo host, issue tracker, and CI tooling
* `prompts/`: centralized prompt namespaces split by `core`, `sdd`, `ask`, and scaffolded future namespaces such as `history_docs`
* `tools/`: tool-specific workflows, renderers, and canonical tool model namespaces
* `cli/`: a thin tool-first command router

## 3.2 Config System

Loads and validates all user-supplied configuration.

Inputs:

* toolkit configuration
* tool-specific target files
* SDD templates

Outputs:

* validated configuration objects

Responsibilities:

* schema validation
* default values
* normalization

This subsystem performs no repository inspection and no generation logic.

---

## 3.3 Repository Analysis

Extracts structured information from the repository.

Inputs:

* source roots
* include/exclude rules
* commit references

Outputs:

* source file inventory
* code summaries
* symbol summaries
* dependency summaries
* diff results

The goal of this subsystem is to convert raw repository data into structured facts.

This subsystem must not perform documentation generation.

---

## 3.4 Commit Impact Analyzer

Converts raw diff data into a higher-level change model.

Input:

* git diff
* file summaries

Output:

* `CommitImpact` model

The impact model classifies changes such as:

* interface changes
* logic changes
* dependency changes
* documentation-only changes

It also maps changes to candidate SDD sections.

---

## 3.4 Evidence Builder

Constructs section-scoped evidence packs.

Evidence packs are the primary input to generation.

Each evidence pack contains:

* CSC metadata
* section specification
* relevant code summaries
* symbol summaries
* dependency summaries
* commit impact summary, if applicable
* existing section text, if applicable

Evidence packs must be deterministic and inspectable.

---

## 3.5 Prompt Builder

Transforms an evidence pack into an LLM prompt.

The prompt builder is responsible for:

* selecting the correct prompt template
* injecting evidence data
* attaching response schemas
* producing a structured generation request

The prompt builder must not call provider SDKs.

---

## 3.6 LLM Adapter

Handles communication with language models.

Responsibilities:

* send structured generation requests
* receive responses
* validate structured outputs
* perform retry logic if necessary

The rest of the system interacts only with the abstract LLM interface.

Concrete providers like Gemini and future providers are implemented behind that interface.

---

## 3.7 Tool Orchestrators

Current orchestration is split by tool namespace:

### `tools/sdd/`

Coordinates initial SDD generation and commit-based update proposals.

### `tools/ask/`

Coordinates standard retrieval-backed Q&A and intensive whole-repo screening.

### `tools/repo/`

Hosts shared repository utilities such as diff inspection and legacy retrieval
index migration.

Tool orchestrators compose shared `core/` services but should keep heavy logic
in reusable deterministic modules.

### `tools/history_docs/`

The history-walk documentation tool now implements H1-H11-02 of its current
roadmap. It combines:

* checkpoint selection and history traversal
* advisory semantic checkpoint planning
* semantic subsystem and capability clustering
* checkpoint snapshot analysis
* interval delta analysis
* structured checkpoint documentation models
* final holistic rendering for each checkpoint
* build-integrated rendered-artifact validation
* an internal benchmark/evaluation harness for comparing future variants,
  including semantic clustering against the path-based baseline

Its key architectural rule is that deltas are used to improve generation
internally, while rendered checkpoint docs remain standalone present-state
documents rather than release notes.

---

## 3.8 Renderers

Convert structured results into user-facing outputs.

Outputs include:

* Markdown SDD documents
* JSON review artifacts
* update proposal reports

Renderers must operate only on structured domain models.

---

## 3.9 Engineering Graph Layer

EngLLM now builds a deterministic engineering/documentation graph that augments
repository Q&A and traceability.

Graph artifacts are file-backed and inspectable:

* `artifacts/workspaces/<workspace_id>/shared/graph/manifest.json`
* `artifacts/workspaces/<workspace_id>/shared/graph/nodes.jsonl`
* `artifacts/workspaces/<workspace_id>/shared/graph/edges.jsonl`
* `artifacts/workspaces/<workspace_id>/shared/graph/symbol_index.json`
* `artifacts/workspaces/<workspace_id>/shared/graph/adjacency.json`

Node types include:

* `directory`
* `file`
* `symbol`
* `chunk`
* `sdd_section`
* `commit`

Edge types include:

* `contains`
* `defines`
* `references`
* `documents`
* `parent_of`
* `imports`
* `changed_in`
* `impacts_section`

IDs are deterministic and stable for unchanged inputs.

`imports` edges are language-aware across supported analyzers (Python, Java,
JavaScript/TypeScript, Go, Rust, C#, C/C++) and use conservative repo-local
resolution. Ambiguous/external dependencies are intentionally left unresolved.

The graph manifest also carries build-planning metadata for inspectability:

* `planner_decision`
* `input_fingerprint`
* `fragment_stats`
* `fragments_path`

Graph build currently uses fragment-based composition. Canonical graph outputs
(`nodes.jsonl`, `edges.jsonl`, `symbol_index.json`, `adjacency.json`) are the
query-facing artifacts, while `graph/fragments/` is an internal build cache used
to support deterministic `no_op`, `partial`, and `full` graph builds.

Current fragment categories are:

* `structure`
* `file::<path>`
* `section::<section_id>`
* `commit::<commit_range>`
* `misc`

Graph build lifecycle:

1. prepare scan, retrieval, hierarchy, section, and commit-impact inputs
2. compute deterministic input and fragment fingerprints
3. choose planner decision (`no_op`, `partial`, or `full`)
4. rebuild only impacted fragments when safe
5. compose canonical graph artifacts from all current fragments
6. write manifest metadata describing what happened

Planner decision meanings:

* `no_op`: prior graph artifacts and fingerprints still match current inputs
* `partial`: a compatible prior graph exists and only a bounded subset of fragments must be rebuilt
* `full`: first build, incompatible prior state, corrupt/missing fragments, or unsafe partial scope

The graph layer is built during both `generate` and `propose-updates`. The
commit fragment is only present when commit-impact inputs are available, which is
why commit-aware `ask` questions are most informative against artifacts produced
by `propose-updates`.

---

## 3.10 Ask Retrieval Pipeline

`ask` uses a deterministic multi-stage retrieval flow:

1. lexical retrieval over the retrieval store
2. optional hierarchy expansion
3. graph anchor extraction from primary evidence
4. bounded graph neighborhood expansion
5. deterministic re-ranking
6. structured evidence pack assembly
7. grounded answer generation

Text-candidate orchestration is unified behind candidate-source plumbing:

* lexical source
* hierarchy source
* graph expansion source
* vector placeholder source

Lexical retrieval remains the baseline. Hierarchy and graph sources only enrich
that baseline; they do not replace it.

Graph anchor extraction is symbol-first: related symbols come from traversed
graph symbol nodes and symbol index lookups tied to anchor files/chunks rather
than token sweeps over chunk text.

For commit-oriented questions, intent routing also supports a dedicated
`change_impact` path. That path can auto-anchor the single commit node in the
graph store, prefer `changed_in` and `impacts_section` traversal, and surface
`related_commits` in the evidence pack when commit-aware context is available.

Intent-specific graph emphasis is currently:

* `implementation`: `defines`, `contains`, `references`, `parent_of`
* `dependency`: `imports`, `contains`, `defines`, `references`
* `documentation`: `documents`, `impacts_section`, `contains`, `references`
* `architecture`: balanced structural/documentation edges
* `change_impact`: `changed_in`, `impacts_section`, `documents`, `contains`, `references`

Re-ranking combines lexical + anchor + graph signals with explicit deterministic
weights (`0.65 lexical`, `0.20 anchor`, `0.15 graph`) plus source-type bias.
Tie-breaks are stable by `source_path`, `line_start`, and `chunk_id`.

Inclusion reasons include optional structured graph-path rationale for auditability
(edge type, source node, target node, and traversal distance).

The query evidence pack can include:

* primary chunks
* selected chunks
* citations
* related files
* related symbols
* related sections
* related commits
* inclusion reasons with score breakdown and graph-path rationale

If hierarchy or graph artifacts are missing/corrupt, `ask` falls back to the
available deterministic stages and adds an uncertainty note instead of failing.

---

## 3.11 Vector-Ready Candidate Sources

Retrieval orchestration is structured around candidate sources:

* lexical source (implemented)
* graph expansion source (implemented)
* vector source (interface placeholder)

This keeps current behavior stable while allowing future vector retrieval
integration without reworking workflow interfaces.
`ask` resolves vector settings from optional project config generation defaults
plus CLI overrides, but vector retrieval remains disabled-by-default and
placeholder-backed in this phase (no embedding/index backend yet).

Future extension seams:

* richer symbol and dependency fidelity in `repo/` and graph build
* additional graph-driven evidence types in `ask`
* real vector retrieval behind the existing candidate-source abstraction
* richer commit-to-section impact inference without changing the current file-backed graph model

---

## 3.12 History-Walk Documentation Tool

The history-walk documentation tool extends EngLLM beyond current-state
documentation and Q&A by generating complete documentation snapshots for
historical checkpoints.

The tool should separate:

* update mode: merge prior checkpoint model + current snapshot analysis + interval deltas
* render mode: emit a complete present-state document for the checkpoint

Expected reusable shared capabilities include:

* explicit checkpoint selection
* git history traversal
* temporary checkpoint snapshot export
* checkpoint snapshot structural analysis
* interval diff analysis
* checkpoint metadata and interval manifests

Expected tool-specific capabilities include:

* checkpoint structural models
* checkpoint documentation models
* section-inference logic
* algorithm capsule handling
* holistic checkpoint rendering

The detailed phased plan, artifact vocabulary, section-inclusion rules, and
first implementation slice for this tool are defined in `docs/HISTORY_DOCS.md`.

Current implemented slice:

* manual-first `engllm history-docs build`
* shared `checkpoint_plan.json` and `intervals.jsonl`
* temporary snapshot export via `git archive`
* manifest search limited to analyzed source roots plus their ancestor chain to
  repo root
* tool-scoped `snapshot_structural_model.json`
* tool-scoped `semantic_structure_map.json`
* tool-scoped `interval_delta_model.json`
* tool-scoped `checkpoint_model.json`
* tool-scoped `section_outline.json`
* tool-scoped `algorithm_capsules/index.json` plus one JSON file per capsule
* tool-scoped `dependencies.json`
* deterministic `checkpoint.md` output plus `render_manifest.json` debug trace
* build-integrated `validation_report.json` output for rendered checkpoint docs
* first-parent diff semantics for merge commits
* diff-only fallback with `observed` statuses when the previous snapshot
  artifact is unavailable
* active and retired checkpoint concepts with active-only core section stubs in
  `checkpoint_model.json`
* semantic display names, summaries, and capability labels that can be threaded
  into subsystem concepts when the internal semantic grouping mode is enabled
* separate conservative section planning with scored inclusion and depth
  metadata in `section_outline.json`
* deterministic algorithm capsule linking into checkpoint concepts and the
  evidence-gated `algorithms_core_logic` / strategy-variant sections
* deterministic final rendering driven by `section_outline.json` inclusion
  order and existing structured artifacts only
* deterministic post-render validation with hard-error build failure and
  warning-only quality findings
* quarterly auto-selection still deferred

---

# 4. Data Model Strategy

The system uses structured domain models for all major artifacts.

Important models include:

* `CSCDescriptor`
* `SDDTemplate`
* `SDDSectionSpec`
* `CodeUnitSummary`
* `SymbolSummary`
* `CommitImpact`
* `SectionEvidencePack`
* `SectionDraft`
* `SectionUpdateProposal`

These models serve as the system's internal language.

Modules communicate through these models rather than raw dictionaries.

---

# 5. Generation Strategy

SDD generation is section-scoped.

Instead of generating a full document in one model call, the system:

1. builds evidence for one section
2. generates the section
3. stores structured output
4. proceeds to the next section

Advantages:

* better prompt control
* smaller context
* easier debugging
* safer generation

---

# 6. Update Strategy

Update proposals are impact-driven.

Instead of regenerating an entire document:

1. analyze commit
2. detect impacted sections
3. generate updates only for those sections

This reduces unnecessary changes and improves reviewability.

---

# 7. Dependency Rules

Subsystem dependencies must follow this direction:

```text
CLI
↓
Workflows
↓
Config / Repo / Analysis / Prompts / LLM / Render
↓
Domain
```

Key constraints:

* domain models have no dependencies on other modules
* LLM providers are isolated
* repo analysis does not call the LLM
* renderers do not inspect repository data

---

# 8. Deterministic vs Generative Responsibilities

The system separates deterministic and generative work.

Deterministic:

* repo scanning
* code summary extraction
* diff parsing
* commit impact classification
* evidence construction
* section mapping

Generative:

* section drafting
* design narrative
* update text proposals

This separation is essential for maintainability.

---

# 9. Error Handling Model

Errors should be categorized by subsystem:

| Subsystem  | Example Errors         |
| ---------- | ---------------------- |
| config     | invalid YAML           |
| repo       | missing files          |
| git        | invalid commit spec    |
| analysis   | unsupported file       |
| llm        | provider error         |
| validation | schema mismatch        |
| render     | invalid document model |

Workflows should propagate errors with clear messages.

---

# 10. Extension Points

The architecture is designed to allow:

* additional LLM providers
* richer language parsers
* additional output formats
* more advanced change detection
* additional repo-focused tools
* CI / review / release integrations

The key extension boundaries are:

* `llm/`
* `integrations/`
* `tools/`
* `core/repo/`
* `core/analysis/`

---

# 11. Architectural Priorities

The system should prioritize:

1. clarity
2. testability
3. modularity
4. traceability
5. conservative generation

The goal is not to produce the most sophisticated AI pipeline.

The goal is to produce a reliable documentation generation tool that engineers can trust and extend.
