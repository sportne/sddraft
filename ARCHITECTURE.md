# SDDraft Architecture

This document describes the internal architecture of SDDraft.

It explains how the system is organized internally and how the major subsystems interact.

Functional requirements are defined in `SPEC.md`.
Development rules are defined in `AGENTS.md`.

---

# 1. System Overview

SDDraft is a documentation generation pipeline.

It converts repository state plus configuration into SDD documentation artifacts.

The pipeline has two modes:

1. Initial SDD generation
2. Commit-based update proposals

The system intentionally performs deterministic analysis first and LLM generation second.

---

# 2. Core Pipeline Model

All workflows follow the same high-level pipeline:

```text
Config + CSC Descriptor
↓
Repository Analysis
↓
Retrieval + Hierarchy + Graph Artifact Build
↓
Evidence Construction
↓
Prompt Construction
↓
LLM Generation
↓
Document Rendering
```

Each stage is implemented by a separate subsystem.

---

# 3. Subsystems

## 3.1 Config System

Loads and validates all user-supplied configuration.

Inputs:

* project configuration
* CSC descriptors
* SDD templates

Outputs:

* validated configuration objects

Responsibilities:

* schema validation
* default values
* normalization

This subsystem performs no repository inspection and no generation logic.

---

## 3.2 Repository Analysis

Extracts structured information from the repository.

Inputs:

* source roots
* include/exclude rules
* commit references

Outputs:

* source file inventory
* code summaries
* interface summaries
* dependency summaries
* diff results

The goal of this subsystem is to convert raw repository data into structured facts.

This subsystem must not perform documentation generation.

---

## 3.3 Commit Impact Analyzer

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
* interface summaries
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

## 3.7 Workflow Orchestrators

Two orchestrators exist:

### Generate SDD

Coordinates initial document generation.

### Propose Updates

Coordinates commit-based documentation updates.

Orchestrators perform the pipeline sequence but do not contain heavy logic.

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

SDDraft now builds a deterministic engineering/documentation graph that augments
repository Q&A and traceability.

Graph artifacts are file-backed and inspectable:

* `artifacts/<CSC>/graph/manifest.json`
* `artifacts/<CSC>/graph/nodes.jsonl`
* `artifacts/<CSC>/graph/edges.jsonl`
* `artifacts/<CSC>/graph/symbol_index.json`
* `artifacts/<CSC>/graph/adjacency.json`

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
* batch workflows
* CI integration

The key extension boundaries are:

* `llm/`
* `render/`
* `repo/`
* `analysis/section_mapper`

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
