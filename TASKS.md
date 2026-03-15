# SDDraft Task Board

## Update Rules

- Use one task format: `ID`, `Status`, `Outcome`, `Definition of Done`, `Verification Command(s)`.
- Allowed statuses: `[ ]` not started, `[~]` in progress, `[x]` done.
- Update task status in the same PR that changes implementation.
- Keep `Now (V1 Gate)` focused on SPEC acceptance criteria for v1.

## Now (V1 Gate)

- [x] **T-007**  
  `Outcome:` Add explicit "Extending SDDraft" documentation for provider, language analyzer, and renderer extension paths.  
  `Definition of Done:` README contains concrete extension steps with architecture boundary reminders.  
  `Verification Command(s):` `rg -n "Extending SDDraft|Add an LLM provider|Add a language analyzer|Add a renderer" README.md`

- [x] **T-008**  
  `Outcome:` Add optional Gemini dependency surface and concise Gemini setup guidance.  
  `Definition of Done:` `pyproject.toml` includes `project.optional-dependencies.gemini`; README documents install, API key, and expected error behavior.  
  `Verification Command(s):` `rg -n "optional-dependencies|gemini|GEMINI_API_KEY" pyproject.toml README.md`

- [x] **T-009**  
  `Outcome:` Add acceptance tests for required CLI commands and batch generation behavior.  
  `Definition of Done:` Tests cover `validate-config`, `generate`, `propose-updates`, `inspect-diff`, and multi-CSC generation in offline/mock mode.  
  `Verification Command(s):` `pytest tests/test_cli_acceptance_v1.py`
  `Result:` `tests/test_cli_acceptance_v1.py` added and passing.

- [x] **T-010**  
  `Outcome:` Run final v1 acceptance checklist and record results in this task board.  
  `Definition of Done:` `ruff`, `mypy`, and full `pytest` pass with coverage >= 90% and status updated to done.  
  `Verification Command(s):` `ruff check src tests && mypy src && pytest`
  `Result:` `ruff` passed, `mypy` passed, `pytest` passed (`54 passed`, `92.48%` coverage).

## Next

- [ ] Establish a lockfile workflow (`uv.lock` or equivalent) for reproducible local/CI installs.
- [ ] Add release notes/changelog process for milestone-based development.

## Later

- [ ] Add optional machine-readable CLI report mode for easier automation integration.
- [ ] Add expanded multi-language fixtures for larger mixed-language repositories.

## Done

- [x] **T-001** Add and standardize in-repo task board; link from README.
- [x] **T-002** Align runtime `provider`/`model` override behavior across `generate`, `propose-updates`, and `ask`.
- [x] **T-003** Add `--temperature` parity for generation/update workflows.
- [x] **T-004** Harden user-facing error handling for missing files and invalid runtime inputs.
- [x] **T-005** Update docs for runtime override behavior and board usage.
- [x] **T-006** Add regression tests for override propagation and error paths while maintaining coverage >= 90%.
- [x] Core deterministic pipeline for `generate`, `propose-updates`, `inspect-diff`, and `ask`.
- [x] Tree-sitter multi-language analyzer coverage for Python, Java, C++, JavaScript, TypeScript, Go, Rust, and C#.

## Blocked

- [ ] None currently.
