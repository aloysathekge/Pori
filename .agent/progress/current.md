# Current State

## Active Task

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
