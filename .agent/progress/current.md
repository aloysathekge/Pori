# Current State

## Active Task

Fixed two security issues found during a codebase review:
- Replaced unsafe `eval()` in `pori/tools/standard/math_tools.py` with an AST-based
  safe arithmetic evaluator (`evaluate_expression`), with an exponent guard.
- Hardened `pori/sandbox/path_resolution.py` against path traversal: added
  `_safe_join`, so `replace_virtual_path` now rejects `../` escapes and
  `replace_virtual_paths_in_command` leaves traversal tokens unrewritten.
- Added `tests/test_math_tools.py` and traversal tests in
  `tests/test_sandbox_integration.py`. Full suite: 168 passed.

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
