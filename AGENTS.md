# Pori Core

Core Pori AI agent framework repo.

- Stack: Python / AI agent framework.
- Safe verification: Python tests likely; prefer uv run pytest, pytest, or repo README command when dependencies are available..
- Source of truth: Python source, pyproject.toml, README/docs, env-key names only..
- Never touch: secrets, credentials, generated dependency folders, deployment config, or production-impacting settings without approval.
- Current active work: unknown.

## Aloy Agent Layer

Use .agent/agent.json for repo-local Aloy setup. Keep this file concise and put deeper workflows in .agent.

Before repo work, read:

- .agent/rules/repo-safety.md
- .agent/rules/source-of-truth.md
- .agent/rules/verification.md
- .agent/skills/repo-workflow/SKILL.md

Useful local commands:

- .agent/commands/explore.md
- .agent/commands/verify.md
