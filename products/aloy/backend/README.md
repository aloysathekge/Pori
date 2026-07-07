# `products/aloy/backend` — Aloy backend

The Aloy backend, **adopted from the existing `aloy_backend` service** (our own
FastAPI product) rather than rebuilt. It **composes the Pori kernel** and adds the
product plane: tenancy, auth, persistence, and the surface the web/desktop apps
talk to over REST + SSE.

## Stack

FastAPI · SQLAlchemy 2 · Alembic · asyncpg (**PostgreSQL**) · Pydantic ·
uvicorn · Docker. Auth is **Supabase JWT** (verified server-side via JWKS in
`aloy_backend/auth.py`). Composes `pori` (the kernel) via
`[tool.uv.sources] pori = { path = "../../..", editable = true }`.

## Routes (`aloy_backend/routes/`)

`organizations`, `users` (**tenancy**), `conversations`, `memory`, `teams`,
`traces`, `usage`, `skills`, `evolution`, `agent_configs`, `runs`.

## Dependency rule

Imports `pori` (kernel) + (later) `extensions/pori-*`; **never imported by them.**
Surfaces (`products/aloy/app`, `products/aloy/desktop`) reach it only over REST + SSE.

## Migration status (docs/Aloy.md — "adopt aloy_backend, unify on PoriEvent")

- [x] **Stage 3.1** — copy `aloy_backend` → here; wire the kernel path to the repo
  root (`../../..`); drop `aloy_backend`'s AI-tooling cruft; all Python
  syntax-compiles clean.
- [ ] **Stage 3.2 — boot** — bring the stack up locally. Copy-paste guide in
  [`../BOOT.md`](../BOOT.md): defaults to **SQLite** (no Postgres needed), needs
  only a free Supabase project (auth) + an LLM key. `uv sync` → `alembic upgrade
  head` → `uvicorn aloy_backend.api:app`. *(Written; run it to verify end to end.)*
- [x] **Stage 3.3 — unify on `PoriEvent`** — `streaming.py` now **relays the
  kernel's live `PoriEvent` stream** (`run_start`/`step_*`/`text_delta`/
  `thinking_delta`/`tool_call_start|end`/`run_end`) via `execute_task(on_event=…)`,
  replacing the step-polling `status/step/message`. Contract now matches
  `@pori/client`. A final `message` frame is kept for DB persistence. (Delegation
  already surfaces as `delegate_task` tool events.)
- [x] **Stage 3.3b — clarify buttons** — `streaming.py` now runs the agent in a
  **worker thread** with a **`ClarifyBridge`** (harvested from `pori/api`), emits
  `clarification_request` frames, and `POST /v1/conversations/clarify/{id}`
  (`resolve_clarification`) resumes the paused run. Frontend renders the buttons.
- [ ] `pori/api` → trivial reference server once the frontend consumes this.
- [ ] Reconcile the two `config.yaml` / duplicate settings with the kernel.

Gateway (Slack/Telegram) will **harvest Hermes's gateway architecture**
(`references/hermes-agent-deep-dives/gateway-messaging.md`) when we add it.
