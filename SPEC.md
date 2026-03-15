# SPEC.md — SDDraft Project Specification

## 1. Project Overview

**SDDraft** is a Python CLI tool that generates **initial Software Design Description (SDD) drafts** for software components and proposes **documentation updates based on git commits**.

The tool targets **MIL-STD-498-style SDDs**, but it is not intended to implement the full MIL-STD-498 documentation suite. Instead, it focuses on practical generation of design documentation from existing repositories.

The system must support two primary workflows:

1. **Initial SDD generation**
2. **Commit-driven SDD update proposals**

LLM usage must be **provider-abstracted**. The initial provider will be **Gemini**, but the rest of the application must not depend directly on Gemini APIs.

---

# 2. Core Goals

## 2.1 Functional Goals

The system must:

* generate initial SDD drafts for a CSC
* analyze git commits or commit ranges
* detect design-relevant changes
* propose updates to existing SDDs
* produce Markdown documentation
* emit structured JSON review artifacts

---

## 2.2 Architectural Goals

The project must:

* separate domain logic from LLM providers
* separate repository analysis from generation logic
* use structured domain models
* support future LLM providers
* support future output formats

The system should function primarily as a **deterministic documentation pipeline with an LLM generation stage**.

---

## 2.3 Quality Goals

The system should:

* avoid hallucinating design claims
* explicitly mark missing information as `TBD`
* record evidence sources used during generation
* produce reviewable documentation outputs
* remain maintainable and testable

---

# 3. Non-Goals (Version 1)

The first version does **not** need to support:

* DOCX export
* PDF export
* diagrams
* issue-tracker integrations
* heavy static analysis
* vector databases or RAG infrastructure
* full requirements traceability inference
* a web UI

The first deliverable is a **Python CLI tool**.

---

# 4. Core Use Cases

## 4.1 Generate an SDD

Input:

* project configuration
* CSC descriptor
* SDD template
* source repository

Output:

* Markdown SDD document
* JSON artifact containing evidence, assumptions, and missing information

---

## 4.2 Propose Documentation Updates

Input:

* project configuration
* CSC descriptor
* existing SDD document
* git commit or commit range

Output:

* proposed updates to affected sections
* human-readable update report
* JSON artifact describing the proposals

---

## 4.3 Batch Generation

The system should optionally generate SDDs for multiple CSC descriptors in one run.

---

# 5. Repository Structure

Recommended structure:

```
sddraft/
  README.md
  SPEC.md
  ARCHITECTURE.md
  AGENTS.md
  pyproject.toml

  src/sddraft/
    domain/
    config/
    repo/
    analysis/
    prompts/
    llm/
    workflows/
    render/
    cli/

  templates/
  examples/
  tests/
```

---

# 6. Configuration Files

## 6.1 Project Configuration

Defines:

* source roots
* file inclusion/exclusion patterns
* SDD template location
* LLM configuration
* generation options

Example:

```yaml
project_name: ExampleProject

sources:
  roots:
    - src/
  include:
    - "**/*.py"
  exclude:
    - "**/tests/**"

sdd_template: templates/sdd_default.yaml

llm:
  provider: gemini
  model_name: gemini-default
  temperature: 0.2
```

---

## 6.2 CSC Descriptor

Describes a component whose SDD should be generated.

Example:

```yaml
csc_id: NAV_CTRL
title: Navigation Control
purpose: Executes route sequencing and navigation control.

source_roots:
  - src/nav_ctrl/

key_files:
  - src/nav_ctrl/controller.py

provided_interfaces:
  - NavControlService

used_interfaces:
  - PositionProvider

requirements:
  - SYS-NAV-001
```

---

## 6.3 SDD Template

Defines the document structure.

Example:

```yaml
document_type: sdd

sections:
  - id: "1"
    title: "Scope"
    instruction: "Describe the CSC and document purpose."
    evidence_kinds:
      - csc_descriptor

  - id: "3"
    title: "Design Overview"
    instruction: "Summarize CSC responsibilities."
    evidence_kinds:
      - code_summary
      - dependencies
```

---

# 7. Domain Models

The system must define structured models for the following concepts:

CSCDescriptor
SDDTemplate
SDDSectionSpec
CodeUnitSummary
InterfaceSummary
CommitImpact
SectionEvidencePack
SectionDraft
SectionUpdateProposal

All models must use strong typing and validation (preferably Pydantic).

---

# 8. Repository Analysis

The repository analysis subsystem must support:

### Source Scanning

Recursive file discovery using configured roots and glob rules.

### Code Summaries

Extract lightweight information such as:

* functions
* classes
* docstrings
* imports

### Interface Extraction

Where feasible:

Python:

* public classes
* public methods
* module-level functions

C/C++:

* headers
* prototypes
* public methods

### Dependency Extraction

Detect imports and include relationships.

---

# 9. Git Diff Analysis

The system must support:

* commit ranges
* changed file detection
* added/removed line counts
* detection of signature changes
* detection of dependency changes
* detection of comment-only edits

The result should be normalized into a **CommitImpact** model.

---

# 10. Change-to-Section Mapping

Commit impact analysis must map detected changes to SDD sections.

Example mapping:

| Change               | Section          |
| -------------------- | ---------------- |
| API signature change | Interface Design |
| major logic changes  | Detailed Design  |
| dependency changes   | Design Overview  |

This mapping may initially be rule-based.

---

# 11. LLM Interface

The system must define a provider-neutral interface.

Example:

```python
class LLMClient(Protocol):
    def generate_structured(
        self,
        request: StructuredGenerationRequest
    ) -> StructuredGenerationResponse:
        ...
```

The rest of the system must only interact with this interface.

---

# 12. Prompt Strategy

Three prompt layers must exist:

### System Prompt

Defines generation rules:

* use only supplied evidence
* never invent requirements or interfaces
* mark missing information as `TBD`
* return structured JSON

---

### Section Generation Prompt

Inputs:

* section specification
* evidence pack
* CSC descriptor

Outputs:

* structured section draft

---

### Update Proposal Prompt

Inputs:

* existing section text
* commit impact summary
* evidence pack

Outputs:

* proposed section update
* rationale
* uncertainty list

---

# 13. Rendering

The system must produce the following outputs.

## Markdown SDD

Example structure:

```
# NAV_CTRL Software Design Description

## 1 Scope
...

## 2 Referenced Documents
...
```

---

## Review JSON Artifact

Must include:

* evidence references
* assumptions
* missing information
* confidence

---

## Update Proposal Report

Should include:

* impacted sections
* proposed revised text
* rationale
* review priority

---

# 14. CLI Commands

Required commands:

```
sddraft generate
sddraft propose-updates
sddraft validate-config
sddraft inspect-diff
```

---

# 15. Testing Requirements

The project must include automated tests.

### Unit Tests

Cover:

* configuration loading
* repository scanning
* diff parsing
* evidence building
* rendering

### Provider Abstraction Tests

Use a mock provider so tests do not require network access.

### Integration Tests

Verify workflows using the mock provider.

---

# 16. Implementation Phases

Recommended development order:

Phase 1
Project skeleton and domain models

Phase 2
Configuration loading and validation

Phase 3
Repository scanning and diff parsing

Phase 4
LLM abstraction and mock provider

Phase 5
SDD generation workflow

Phase 6
Commit update workflow

Phase 7
Gemini provider implementation

Phase 8
Testing and documentation

---

# 17. Acceptance Criteria

Version 1 is complete when:

* `sddraft generate` produces a Markdown SDD
* `sddraft propose-updates` produces update proposals
* LLM integration is provider-abstracted
* tests run without network access
* documentation explains how to extend the system

The system must remain modular, testable, and easy for future contributors to extend.
