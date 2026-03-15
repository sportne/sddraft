# SDDraft Task Board

## Update Rules

- Use one task format: `ID`, `Status`, `Outcome`, `Definition of Done`, `Verification Command(s)`.
- Allowed statuses: `[ ]` not started, `[~]` in progress, `[x]` done.
- Update task status in the same PR that changes implementation.
- Keep `Now (V1 Gate)` focused on release hardening and stability.

## Now (V1 Gate)

- [x] **T-001**  
  `Outcome:` Add and standardize this repository task board; link it from README.  
  `Definition of Done:` `TASKS.md` exists with fixed sections and task format; README points to it.  
  `Verification Command(s):` `rg -n "Task Board|TASKS.md" README.md TASKS.md`

- [x] **T-002**  
  `Outcome:` Make CLI LLM overrides fully consistent for `generate`, `propose-updates`, and `ask`.  
  `Definition of Done:` Runtime `provider`/`model` overrides are propagated end-to-end in workflows.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_workflow_generate_and_ask.py tests/test_workflow_propose_updates.py`

- [x] **T-003**  
  `Outcome:` Add `--temperature` parity for generation/update commands and honor resolved runtime values.  
  `Definition of Done:` `generate` and `propose-updates` accept `--temperature` and pass it to structured requests.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_workflow_generate_and_ask.py tests/test_workflow_propose_updates.py`

- [x] **T-004**  
  `Outcome:` Harden user-facing errors for missing files and invalid runtime inputs.  
  `Definition of Done:` Missing existing-SDD path and missing retrieval index produce categorized, clear CLI errors.  
  `Verification Command(s):` `pytest tests/test_cli_additional.py tests/test_workflow_propose_updates.py`

- [x] **T-005**  
  `Outcome:` Update docs/examples to reflect current multi-language support and provider setup expectations.  
  `Definition of Done:` README includes runtime override guidance and task-board reference.  
  `Verification Command(s):` `rg -n "provider|model|temperature|Task Board" README.md`

- [x] **T-006**  
  `Outcome:` Add regression tests for override propagation and error paths while maintaining coverage >= 90%.  
  `Definition of Done:` New tests cover override and error behavior; test suite remains above threshold.  
  `Verification Command(s):` `pytest`

## Next

- [ ] Establish a lockfile workflow (`uv.lock` or equivalent) for reproducible local/CI installs.
- [ ] Add release notes/changelog process for milestone-based development.

## Later

- [ ] Add optional machine-readable CLI report mode for easier automation integration.
- [ ] Add expanded multi-language fixtures for larger mixed-language repositories.

## Done

- [x] Core deterministic pipeline for `generate`, `propose-updates`, `inspect-diff`, and `ask`.
- [x] Tree-sitter multi-language analyzer coverage for Python, Java, C++, JavaScript, TypeScript, Go, Rust, and C#.

## Blocked

- [ ] None currently.
