# EngLLM Usage Guide

This guide covers the public CLI, configuration shape, artifact layout, and the
main operating modes shipped in EngLLM today.

## Command Layout

EngLLM now uses a tool-first CLI:

- `engllm sdd validate-config`
- `engllm sdd generate`
- `engllm sdd propose-updates`
- `engllm ask answer`
- `engllm ask interactive`
- `engllm repo inspect-diff`
- `engllm repo migrate-index`

Runtime model settings can still be overridden per command with
`--provider`, `--model`, and `--temperature` where those flags apply.

## Config Files

### Project config

Project config defines repo scope, output layout, model defaults, and per-tool
settings.

```yaml
project_name: ExampleProject

workspace:
  output_root: ../artifacts

sources:
  roots:
    - src/
  include:
    - "**/*.py"
  exclude:
    - "**/tests/**"

llm:
  provider: mock
  model_name: mock-engllm
  temperature: 0.2

generation:
  max_files: 500
  code_chunk_lines: 40
  retrieval_top_k: 6
  write_batch_size: 200
  max_in_memory_records: 2000
  index_shard_size: 1000

tools:
  sdd:
    template: ../templates/sdd_default.yaml
  ask:
    interactive_top_k: 6
```

Key fields:

- `workspace.output_root`: root directory for workspace artifacts.
- `sources`: deterministic repo discovery rules used by shared analysis.
- `llm`: default provider/model settings.
- `generation`: shared build settings for retrieval, chunking, and indexing.
- `tools.sdd`: SDD-specific defaults such as the template path.
- `tools.ask`: ask-specific defaults such as interactive retrieval size.

### SDD target config

SDD generation still targets a CSC-style descriptor.

```yaml
csc_id: NAV_CTRL
title: Navigation Control
purpose: Executes route sequencing and navigation control.

source_roots:
  - src/nav_ctrl/

key_files:
  - src/nav_ctrl/controller.py
```

This repo ships an example target at `examples/sdd_target_core.yaml`.

## Quick Start

### Offline / mock provider

```bash
make setup-venv
make install-dev

engllm sdd validate-config --config examples/project.yaml --target examples/sdd_target_core.yaml
engllm sdd generate --config examples/project.yaml --target examples/sdd_target_core.yaml --repo-root . --provider mock
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "What are the main workflows in this project?" --provider mock
engllm repo inspect-diff --commit-range HEAD~1..HEAD --repo-root .
engllm sdd propose-updates --config examples/project.yaml --target examples/sdd_target_core.yaml --existing-sdd artifacts/workspaces/SDDRAFT_CORE/tools/sdd/sdd.md --commit-range HEAD~1..HEAD --repo-root . --provider mock
```

### Ollama / local model

```bash
engllm sdd validate-config --config examples/project_ollama.yaml --target examples/sdd_target_core.yaml
engllm sdd generate --config examples/project_ollama.yaml --target examples/sdd_target_core.yaml --repo-root . --provider ollama --model qwen2.5:14b-instruct-q4_K_M
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "What are the main workflows in this project?" --provider ollama --model qwen2.5:14b-instruct-q4_K_M
```

## Workspace Artifact Layout

EngLLM writes artifacts under a workspace/tool split:

- shared analysis: `artifacts/workspaces/<workspace_id>/shared/`
- tool outputs: `artifacts/workspaces/<workspace_id>/tools/<tool_name>/`

For the example target in this repo (`workspace_id = SDDRAFT_CORE`), the main
paths are:

- `artifacts/workspaces/SDDRAFT_CORE/shared/retrieval/`
- `artifacts/workspaces/SDDRAFT_CORE/shared/hierarchy/`
- `artifacts/workspaces/SDDRAFT_CORE/shared/graph/`
- `artifacts/workspaces/SDDRAFT_CORE/tools/sdd/sdd.md`
- `artifacts/workspaces/SDDRAFT_CORE/tools/sdd/review_artifact.json`
- `artifacts/workspaces/SDDRAFT_CORE/tools/sdd/run_metrics.json`
- `artifacts/workspaces/SDDRAFT_CORE/tools/ask/run_metrics_ask.json`
- `artifacts/workspaces/SDDRAFT_CORE/tools/ask/intensive/`

### Shared retrieval artifacts

- `shared/retrieval/manifest.json`
- `shared/retrieval/chunks-*.jsonl`
- `shared/retrieval/postings-*.jsonl`
- `shared/retrieval/docstats.jsonl`

### Shared hierarchy artifacts

- `shared/hierarchy/manifest.json`
- `shared/hierarchy/file_summaries.jsonl`
- `shared/hierarchy/directory_summaries.jsonl`
- `shared/hierarchy/nodes.jsonl`
- `shared/hierarchy/edges.jsonl`
- `shared/hierarchy/**` rendered file and directory summaries

Directory summaries are subtree-first: each directory summary describes the
entire subtree rooted at that directory, not just immediate children.

### Shared graph artifacts

- `shared/graph/manifest.json`
- `shared/graph/nodes.jsonl`
- `shared/graph/edges.jsonl`
- `shared/graph/symbol_index.json`
- `shared/graph/adjacency.json`

Artifact purposes:

- `manifest.json`: build metadata, counts, planner decision, and relative file paths.
- `nodes.jsonl`: one deterministic node record per graph node.
- `edges.jsonl`: one deterministic edge record per relationship.
- `symbol_index.json`: compact symbol lookup used by graph-aware retrieval.
- `adjacency.json`: cached inbound/outbound edge lists for traversal.

### Intensive ask artifacts

- `tools/ask/intensive/corpus/manifest.json`
- `tools/ask/intensive/corpus/chunks.jsonl`
- `tools/ask/intensive/runs/<question_hash>/manifest.json`
- `tools/ask/intensive/runs/<question_hash>/screenings.jsonl`
- `tools/ask/intensive/runs/<question_hash>/selected_excerpts.json`
- `tools/ask/intensive/runs/<question_hash>/run_metrics.json`

## Graph Layer

Graph node ID shapes are deterministic:

- directory: `dir::<path>`
- file: `file::<path>`
- symbol: `sym::<path>::<kind>::<qualified_name_or_name>`
- chunk: `chunk::<chunk_id>`
- section: `sdd_section::<section_id>`
- commit: `commit::<commit_range>`

Graph edge IDs use:

- `edge::<type>::<source_id>::<target_id>`

Current node types:

- `directory`
- `file`
- `symbol`
- `chunk`
- `sdd_section`
- `commit`

Current edge types:

- `contains`
- `defines`
- `references`
- `documents`
- `parent_of`
- `imports`
- `changed_in`
- `impacts_section`

`imports` edges are multi-language and conservative. EngLLM emits them for the
supported analyzers when repo-local resolution is reliable, and leaves ambiguous
or external dependencies unresolved instead of guessing.

### Inspect graph artifacts

```bash
python -m json.tool artifacts/workspaces/SDDRAFT_CORE/shared/graph/manifest.json
head -n 5 artifacts/workspaces/SDDRAFT_CORE/shared/graph/nodes.jsonl
head -n 5 artifacts/workspaces/SDDRAFT_CORE/shared/graph/edges.jsonl
rg -n '"edge_type": "(changed_in|impacts_section)"' artifacts/workspaces/SDDRAFT_CORE/shared/graph/edges.jsonl
python -m json.tool artifacts/workspaces/SDDRAFT_CORE/shared/graph/manifest.json | rg 'planner_decision|input_fingerprint|fragment_stats|fragments_path'
rg -n '"node_id": "(dir::|file::|sym::|chunk::|sdd_section::|commit::)' artifacts/workspaces/SDDRAFT_CORE/shared/graph/nodes.jsonl
rg -n '"edge_id": "edge::' artifacts/workspaces/SDDRAFT_CORE/shared/graph/edges.jsonl
```

### Graph-aware ask flow

Standard `ask` uses a deterministic staged pipeline:

1. lexical retrieval from `shared/retrieval/`
2. hierarchy expansion when hierarchy artifacts exist
3. graph anchor extraction and bounded neighborhood expansion when graph artifacts exist
4. deterministic reranking
5. final grounded answer generation

The evidence pack can include:

- primary chunks
- selected chunks
- citations
- related files
- related symbols
- related sections
- related commits
- inclusion reasons with score breakdown and graph-path rationale

If hierarchy or graph artifacts are missing or corrupt, `ask` falls back to the
available deterministic stages and adds uncertainty instead of failing.

Change-impact questions are most useful when `ask` points at shared artifacts
produced by `engllm sdd propose-updates`, because those runs also emit commit
nodes plus `changed_in` and `impacts_section` edges.

## Ask Modes

### Standard mode

Use standard mode for the normal lexical/hierarchy/graph workflow.

```bash
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "Where is graph expansion implemented?" --provider ollama --graph-depth 2 --graph-top-k 16
```

Vector flags are available in standard mode as forward-looking placeholders:

```bash
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "Where is dependency resolution implemented?" --vector-enabled --vector-top-k 12
```

Current vector behavior is intentionally a no-op placeholder. The abstraction is
there so a real backend can be added later without redesigning `ask`.

### Intensive mode

Use intensive mode when you want whole-repo screening rather than lexical or
graph retrieval.

```bash
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --config examples/project.yaml --repo-root . --mode intensive --question "Where is Ollama support implemented?" --provider mock
```

Intensive mode:

- rebuilds repo scope from the configured roots/include/exclude rules
- builds or reuses a deterministic structured corpus
- packs whole files across chunk boundaries when they fit the token budget
- splits a file only when that file alone exceeds the chunk budget
- sends screening payloads as structured JSON with ordered file/line segments
- validates returned excerpts against the chunk segment map
- returns a conservative `TBD` answer when screening finds nothing relevant

Intensive mode requires `--config` because it needs repo discovery rules.
Graph and vector flags are accepted for CLI compatibility, but they apply only
to standard mode.

## Repo Utilities

Inspect diff impact:

```bash
engllm repo inspect-diff --commit-range HEAD~1..HEAD --repo-root .
```

Migrate a legacy single-file retrieval index into the current sharded retrieval
store layout:

```bash
engllm repo migrate-index --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval_index.json
```

After migration, point `ask` at the retrieval directory:

```bash
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "What are the main workflows?" --provider mock
```

## Troubleshooting

- Graph disabled: `engllm sdd generate --no-graph`, `engllm sdd propose-updates --no-graph`, and `engllm ask answer --no-graph` force graph-free behavior.
- Graph missing or corrupt: standard `ask` falls back gracefully and records uncertainty instead of failing.
- Planner expectations: unchanged shared inputs usually yield `planner_decision: "no_op"`; scoped update runs often yield `partial`; first-time or unsafe rebuilds yield `full`.
- Retrieval path errors: point `--index-path` at `artifacts/workspaces/<workspace_id>/shared/retrieval/`, not at the tool output directory.
- Intensive mode needs config: pass `--config` so EngLLM can rebuild repo scope deterministically.
- Large intensive chunks: the default is `8192` approximate tokens, but the corpus builder is designed for much larger settings such as `131072`.

## Quality Gates

```bash
make format-check
make lint
make typecheck
make test
```
