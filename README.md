# SDDraft

SDDraft is a deterministic Python CLI that helps teams generate and maintain
MIL-STD-498-style Software Design Description (SDD) documents.

It supports:
- Initial SDD draft generation from repository evidence.
- Commit-driven update proposals for impacted SDD sections.
- Grounded project Q&A on the command line using generated docs and code chunks.
- Multi-language repository analysis for Python, Java, and C++.

## Commands

- `sddraft generate`
- `sddraft propose-updates`
- `sddraft validate-config`
- `sddraft inspect-diff`
- `sddraft ask`

## Language Support

- Supported source languages: Python, Java, C++, JavaScript, TypeScript, Go, Rust, and C#.
- Supported extensions: `.py`, `.java`, `.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.js`, `.mjs`, `.cjs`, `.ts`, `.tsx`, `.go`, `.rs`, `.cs`.
- Parsing uses tree-sitter analyzers in the `repo` layer.
- Unsupported extensions are ignored by language analysis unless included in future analyzers.

## Quick Start

1. Create a virtual environment and install dev dependencies:

```bash
make setup-venv
make install-dev
```

2. Validate configuration:

```bash
sddraft validate-config --project-config examples/project.yaml --csc examples/csc_nav_ctrl.yaml
```

3. Generate an initial SDD:

```bash
sddraft generate --project-config examples/project.yaml --csc examples/csc_nav_ctrl.yaml --repo-root . --provider mock
```

4. Ask grounded questions:

```bash
sddraft ask --index-path artifacts/NAV_CTRL/retrieval_index.json --question "What interfaces are exposed?" --provider mock
```

## Quality Gates

The project includes `kleuw`-style quality infrastructure:
- `black` + `isort` formatting
- `ruff` linting
- `mypy` type checking
- `pytest` + `pytest-cov` with coverage threshold

Run all checks with:

```bash
make ci
```
