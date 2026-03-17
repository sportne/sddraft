# SDDraft

SDDraft is a deterministic CLI for generating and maintaining
MIL-STD-498-style Software Design Description (SDD) documents from source code
and git history.

## What It Does

- Generates initial SDD drafts from repository evidence.
- Proposes section updates from commit impact.
- Answers grounded repository questions with citations via `sddraft ask`.
- Builds inspectable retrieval, hierarchy, and engineering-graph artifacts.

## Quick Start

```bash
make setup-venv
make install-dev

sddraft validate-config --project-config examples/project.yaml --csc examples/csc_sddraft.yaml
sddraft generate --project-config examples/project.yaml --csc examples/csc_sddraft.yaml --repo-root . --provider mock
sddraft ask --index-path artifacts/SDDRAFT_CORE/retrieval --question "What are the main workflows?" --provider mock
```

## Commands

- `sddraft generate`
- `sddraft propose-updates`
- `sddraft validate-config`
- `sddraft inspect-diff`
- `sddraft ask`
- `sddraft migrate-index`

## Documentation

- Usage and operations: [docs/USAGE.md](docs/USAGE.md)
- Extension guide: [docs/EXTENDING.md](docs/EXTENDING.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Product spec: [SPEC.md](SPEC.md)
- Agent rules: [AGENTS.md](AGENTS.md)
- Task board: [TASKS.md](TASKS.md)

## Quality Gates

```bash
make ci
```
