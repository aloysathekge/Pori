# Current State

## Active Task

Codebase-review follow-ups. Full suite now 180 passed; black/isort clean; no
new mypy errors introduced (all remaining mypy errors are pre-existing).

Security (done earlier this session):
- Replaced unsafe `eval()` in `pori/tools/standard/math_tools.py` with an AST-based
  safe arithmetic evaluator (`evaluate_expression`), with an exponent guard.
- Hardened `pori/sandbox/path_resolution.py` against path traversal: added
  `_safe_join`, so `replace_virtual_path` now rejects `../` escapes and
  `replace_virtual_paths_in_command` leaves traversal tokens unrewritten.
- Added `tests/test_math_tools.py` and traversal tests in
  `tests/test_sandbox_integration.py`.

Quality / maintainability (this round):
- `pori/agent.py`: extracted `Agent._reject_action(...)` helper and replaced 5
  duplicated rejection blocks in `execute_actions` (behavior-preserving).
- `pori/agent.py`: replaced 6 silent `except Exception: pass` with `logger.debug`.
- `pori/memory.py`: added module logger; extracted `_SEMANTIC_WEIGHT`/
  `_LEXICAL_WEIGHT` + `AgentMemory._blend_scores` and replaced the 3 hardcoded
  `0.75*sem + 0.25*lex` sites; added debug logging on embedding fallbacks.
- Added `tests/test_config.py` (12 tests) covering config resolution order,
  validation failures, back-compat aliases, and the `create_llm` factory.

Not done (deferred, larger/riskier): full decomposition of the 486-line
`execute_actions` and the Agent god-object; LLM-provider and API-layer tests;
enabling mypy enforcement in CI (currently `|| true`).

## Previously: Active Task

Enabled multi-agent team mode in `config.yaml` for CLI use.

## Decisions Made

- AGENTS.md is the canonical repo instruction file.
- `.agent/` holds Aloy repo-local rules, commands, skills, and progress memory.
- CLI team mode: `research-team` with `delegate` mode and three members (researcher, analyst, writer).

## Important Discoveries

- CLI only uses team mode when `config.yaml` has a non-empty `team.members` list.
- Stale SQLite session memory can cause single-agent CLI to answer wrong prior questions; `/new` clears transient context.

## Blockers

- None.

## Next Session Should Start With

Restart `uv run pori` to pick up team config. Run `/new` before unrelated tasks to avoid memory contamination.
