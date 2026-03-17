# Extending SDDraft

This guide describes how to add new capabilities while preserving SDDraft
architecture boundaries.

## Add an LLM Provider

1. Implement a provider client in `src/sddraft/llm/` that satisfies `LLMClient` from `llm/base.py`.
2. Ensure provider responses are parsed as structured JSON and validated against `response_model`.
3. Register the provider in `llm/factory.py` without importing provider SDKs outside `llm/`.
4. Add provider tests with mocks/fakes so tests remain network-independent.

## Add a Language Analyzer

1. Add a `LanguageAnalyzer` implementation in `src/sddraft/repo/language_analyzers.py`.
2. Map file extensions to a normalized `SourceLanguage` and register the analyzer.
3. Ensure analyzer output feeds `CodeUnitSummary` and `SymbolSummary`.
4. Add scan/diff tests to validate deterministic extraction and classification.

## Add a Renderer Output

1. Implement rendering in `src/sddraft/render/` from structured domain models only.
2. Keep repository inspection out of renderers; workflows provide input models.
3. Add tests for deterministic output formatting and error behavior.

## Design Rules To Preserve

- Keep provider-specific code in `src/sddraft/llm/` only.
- Keep repository logic out of workflows and renderers.
- Keep deterministic analysis outside the LLM boundary.
- Keep structured JSON validation mandatory for LLM outputs.
