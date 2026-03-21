# AGENTS.md — Guidelines for AI Coding Agents

This repository is designed for AI-assisted development. Coding agents working on this project must follow the guidelines in this document.

This file defines **development rules and behavioral expectations** for agents contributing to EngLLM.

If conflicts occur:
1. `SPEC.md` defines required functionality.
2. `ARCHITECTURE.md` defines system structure.
3. `AGENTS.md` defines how agents should implement changes.

---

# 1. Core Principles

Agents must adhere to the following principles when working on this project:

1. Maintain a **clean layered architecture**.
2. Prefer **typed models over dynamic dictionaries**.
3. Keep logic **deterministic outside the LLM boundary**.
4. Ensure all outputs are **reviewable and traceable**.
5. Avoid speculative or fabricated documentation content.

---

# 2. Layer Boundaries

Agents must preserve the separation between layers.

The repository is structured as:

src/engllm/
- domain/
- core/
- prompts/
- llm/
- integrations/
- tools/
- cli/

Rules:

• `domain/` must not depend on any other project modules.  
• `core/repo/` must not call LLMs.  
• `core/analysis/` must not call provider SDKs.  
• `core/render/` must not inspect repository files.  
• `tools/` should compose shared services and prompt builders, but must not call provider SDKs directly.  
• `cli/` must remain thin and route into tool entrypoints.  

---

# 3. LLM Provider Isolation

LLM integration must be provider-abstracted.

The system interacts with models only through the abstract interface defined in `llm/base.py`.

Provider implementations (such as Gemini) must exist only inside the `llm/` module.

No other module may directly reference provider SDKs.

---

# 4. Structured Outputs Only

LLM outputs must be **validated structured data**.

Agents must not accept free-form responses without validation.

All generation tasks must:

1. Define a response schema.
2. Request structured output.
3. Validate the response.
4. Handle schema failures gracefully.

---

# 5. Conservative Documentation Generation

The system must behave conservatively when generating documentation.

Agents must ensure that:

- interfaces are never invented
- requirement IDs are never fabricated
- design intent is not asserted without evidence
- missing information is marked as `TBD`

Documentation must reference the evidence used to generate it.

---

# 6. Incremental Development

Agents should implement features in **small logical increments**.

Recommended order:

1. domain models
2. configuration loading
3. repository analysis
4. LLM abstraction
5. generation workflow
6. update workflow
7. rendering
8. CLI integration

Do not attempt to implement the entire system in a single step.

---

# 7. Testing Requirements

All deterministic logic must be testable without LLM access.

Agents must:

• write unit tests for core modules  
• use a **mock LLM provider** for workflow tests  
• avoid network calls in automated tests  

Tests should focus on:

- configuration loading
- repo scanning
- diff parsing
- evidence building
- rendering
- workflow orchestration

---

# 8. Prompt Management

Prompts must be centralized.

Rules:

• Prompts live in `prompts/`.  
• Workflows must not embed prompt text.  
• Prompt builders should construct requests deterministically.  

Prompt templates may be modified without altering workflow logic.

---

# 9. Coding Standards

Agents should follow these coding practices:

- Python 3.11+
- use type hints everywhere
- prefer Pydantic models for structured data
- prefer pathlib over raw paths
- keep functions short and focused
- write docstrings for public APIs

Avoid unnecessary frameworks or heavy dependencies.

---

# 10. Error Handling

Errors must be explicit and informative.

Agents must avoid silent failures.

Errors should be categorized as:

- configuration errors
- repository errors
- git errors
- analysis errors
- LLM errors
- validation errors
- rendering errors

CLI commands must return meaningful error messages.

---

# 11. Anti‑Patterns to Avoid

Agents must not:

- generate entire SDD documents in a single LLM call
- bypass the provider abstraction
- mix repository logic with generation logic
- hide failures inside generic exception handlers
- rely on global state
- couple workflows directly to CLI arguments

---

# 12. Design Philosophy

EngLLM is designed to be a **deterministic repository-analysis toolkit with generative stages**.

Agents should prioritize:

clarity > cleverness  
simplicity > abstraction  
explicit models > implicit structures  

When uncertain, choose the simplest design that maintains architectural boundaries.
