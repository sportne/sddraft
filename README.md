# SDDraft

SDDraft is a deterministic Python CLI that helps teams generate and maintain
MIL-STD-498-style Software Design Description (SDD) documents.

It supports:
- Initial SDD draft generation from repository evidence.
- Commit-driven update proposals for impacted SDD sections.
- Grounded project Q&A on the command line using generated docs and code chunks.
- Multi-language repository analysis for Python, Java, C++, JavaScript, TypeScript, Go, Rust, and C#.

## Commands

- `sddraft generate`
- `sddraft propose-updates`
- `sddraft validate-config`
- `sddraft inspect-diff`
- `sddraft ask`

Runtime LLM settings can be overridden per command via `--provider`, `--model`, and
`--temperature` (for `generate`, `propose-updates`, and `ask`).

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

## Gemini Setup (Optional)

Gemini is supported through the provider abstraction and is optional for local/offline development.

1. Install Gemini support:

```bash
pip install -e .[gemini]
```

2. Set API key:

```bash
export GEMINI_API_KEY="your-key"
```

3. Run with Gemini:

```bash
sddraft generate --project-config examples/project.yaml --csc examples/csc_nav_ctrl.yaml --repo-root . --provider gemini --model gemini-2.5-flash
```

If Gemini dependencies are not installed or `GEMINI_API_KEY` is missing, the CLI exits with a clear `Error:` message.

## Ollama Setup (Local, Optional)

Ollama is supported through the same provider abstraction and uses the local
HTTP API (`/api/chat`) with schema-constrained JSON output.

1. Start Ollama:

```bash
ollama serve
```

2. Pull the recommended local model:

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

3. Run SDDraft commands with Ollama:

```bash
sddraft generate --project-config examples/project.yaml --csc examples/csc_nav_ctrl.yaml --repo-root . --provider ollama --model qwen2.5:14b-instruct-q4_K_M
sddraft propose-updates --project-config examples/project.yaml --csc examples/csc_nav_ctrl.yaml --existing-sdd artifacts/NAV_CTRL/sdd.md --commit-range HEAD~1..HEAD --repo-root . --provider ollama --model qwen2.5:14b-instruct-q4_K_M
sddraft ask --index-path artifacts/NAV_CTRL/retrieval_index.json --question "What interfaces are exposed?" --provider ollama --model qwen2.5:14b-instruct-q4_K_M
```

4. Optional endpoint override (default is `http://127.0.0.1:11434`):

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

If Ollama is unreachable, the CLI exits with `Error: Cannot connect to Ollama ...`.

Example Ollama config is provided in `examples/project_ollama.yaml`.

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

## Task Board

In-repo implementation tracking is maintained in [`TASKS.md`](TASKS.md).

## Extending SDDraft

### Add an LLM provider

1. Implement a provider client in `src/sddraft/llm/` that satisfies `LLMClient` from `llm/base.py`.
2. Ensure all provider responses are parsed as structured JSON and validated against `response_model`.
3. Register the provider in `llm/factory.py` without importing SDKs outside the `llm` module.
4. Add provider tests using fakes/mocks so tests run without network access.

### Add a language analyzer

1. Add a `LanguageAnalyzer` implementation in `src/sddraft/repo/language_analyzers.py`.
2. Map file extensions to a normalized `SourceLanguage` and register the analyzer.
3. Ensure analyzer output feeds existing `CodeUnitSummary` and `InterfaceSummary` models.
4. Add scan/diff tests to validate deterministic extraction and change classification.

### Add a renderer/output format

1. Implement rendering in `src/sddraft/render/` from structured domain models only.
2. Keep repository inspection out of renderers; workflows provide all input models.
3. Add tests for deterministic output formatting and error behavior.
