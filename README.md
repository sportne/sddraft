# EngLLM

EngLLM is a deterministic repository-analysis toolkit with LLM-assisted tools on top.

The next planned tool family is history-walk documentation generation, which is
documented in [docs/HISTORY_DOCS.md](docs/HISTORY_DOCS.md).

Today the repo ships three tool namespaces:
- `engllm sdd ...`: generate and update Software Design Description artifacts
- `engllm ask answer ...`: answer grounded repository questions with citations
- `engllm repo ...`: shared repository utilities such as diff inspection and index migration

## Quick Start

```bash
make setup-venv
make install-dev

engllm sdd validate-config --config examples/project.yaml --target examples/sdd_target_core.yaml
engllm sdd generate --config examples/project.yaml --target examples/sdd_target_core.yaml --repo-root . --provider mock
engllm ask answer --index-path artifacts/workspaces/SDDRAFT_CORE/shared/retrieval --question "What are the main workflows?" --provider mock
```

## Command Layout

- `engllm sdd validate-config`
- `engllm sdd generate`
- `engllm sdd propose-updates`
- `engllm ask answer`
- `engllm ask interactive`
- `engllm repo inspect-diff`
- `engllm repo migrate-index`

## Artifact Layout

Shared analysis artifacts live under:
- `artifacts/workspaces/<workspace_id>/shared/`

Tool-specific outputs live under:
- `artifacts/workspaces/<workspace_id>/tools/sdd/`
- `artifacts/workspaces/<workspace_id>/tools/ask/`

For the built-in EngLLM example target, the main paths are:
- `artifacts/workspaces/SDDRAFT_CORE/shared/retrieval/`
- `artifacts/workspaces/SDDRAFT_CORE/shared/hierarchy/`
- `artifacts/workspaces/SDDRAFT_CORE/shared/graph/`
- `artifacts/workspaces/SDDRAFT_CORE/tools/sdd/sdd.md`
- `artifacts/workspaces/SDDRAFT_CORE/tools/ask/intensive/`

## Documentation

- Usage and operations: [docs/USAGE.md](docs/USAGE.md)
- History-walk tool design: [docs/HISTORY_DOCS.md](docs/HISTORY_DOCS.md)
- Extension guide: [docs/EXTENDING.md](docs/EXTENDING.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Product spec: [SPEC.md](SPEC.md)
- Agent rules: [AGENTS.md](AGENTS.md)
- Task board: [TASKS.md](TASKS.md)

## Quality Gates

```bash
make ci
```
