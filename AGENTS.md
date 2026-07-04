# Pori Core

Kernel + products monorepo. **Pori** is an eval-native, memory-native agent
**kernel** (`pori/`, `import pori`, product-agnostic). **Aloy** is the first
product built on it (`products/aloy/`).

- Stack: Python kernel (uv) + TypeScript surfaces (bun workspace).
- **Layout & the one-way dependency rule:** see [`MONOREPO.md`](./MONOREPO.md)
  (`products → extensions → pori`, never upward; surfaces reach the backend only
  over REST + SSE). Products are self-contained + extractable.
- **Where things are:** kernel `pori/` (the agent is a package, `pori/agent/`);
  shared TS client `packages/pori-client` (`@pori/client`); Aloy's stack under
  `products/aloy/{backend,app,desktop,website}`.
- Safe verification: `uv run --no-sync pytest tests/ -q` and
  `uv run mypy pori/ --ignore-missing-imports` (both green on `main`).
- Source of truth: Python source, `pyproject.toml`, README/docs; env-key names only.
- Never touch: secrets, credentials, generated dependency folders, deployment
  config, or production-impacting settings without approval. **No Claude/AI
  attribution on commits or PRs.**
- **Current active work + next steps: [`.agent/progress/current.md`](./.agent/progress/current.md)**
  (read this first). Product plan: [`docs/Aloy.md`](./docs/Aloy.md); boot guide:
  [`products/aloy/BOOT.md`](./products/aloy/BOOT.md).

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
