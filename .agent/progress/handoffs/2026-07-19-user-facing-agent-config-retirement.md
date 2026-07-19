# User-facing AgentConfig retirement - 2026-07-19

## Outcome

Aloy remains one customer-facing assistant. The app no longer exposes an
Agents navigation item or a page for creating model/provider/prompt/tool
configurations. Existing `/agents` bookmarks redirect to Today.

## Runtime boundary

- Product specialist roles remain developer-owned through `model_roles.py` and
  `aloy.models.yaml`.
- Legacy `AgentConfig` rows remain intact for Conversations that already
  reference them.
- AgentConfig CRUD and provider/capability diagnostics require
  `policy:manage`.
- An ordinary member may create a Conversation without an AgentConfig but may
  not select one explicitly.
- User profile updates can no longer select `default_agent_config_id`.

## Verification

- Focused backend RBAC/config tests: `10 passed`.
- Aloy app ESLint: passed.
- Aloy app production build: passed with the existing large-chunk warning.
- Changed backend files: Ruff and Black passed.
- Focused backend mypy: passed.

The first backend test attempt was blocked by the machine's shared pytest temp
directory permissions. Re-running with a repository-local `--basetemp` passed.
