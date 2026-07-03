# `products/aloy/backend` — Aloy backend

The Aloy backend, **adopted from the existing `pori_cloud` service** (our own
FastAPI product) rather than rebuilt. It **composes the Pori kernel** and adds the
product plane: tenancy, auth, persistence, and the surface the web/desktop apps
talk to over REST + SSE.

## Stack

FastAPI · SQLAlchemy 2 · Alembic · asyncpg (**PostgreSQL**) · Pydantic ·
uvicorn · Docker. Auth is **Supabase JWT** (verified server-side via JWKS in
`pori_cloud/auth.py`). Composes `pori` (the kernel) via
`[tool.uv.sources] pori = { path = "../../..", editable = true }`.

## Routes (`pori_cloud/routes/`)

`organizations`, `users` (**tenancy**), `conversations`, `memory`, `teams`,
`traces`, `usage`, `skills`, `evolution`, `agent_configs`, `runs`.

## Dependency rule

Imports `pori` (kernel) + (later) `extensions/pori-*`; **never imported by them.**
Surfaces (`apps/web`, `apps/desktop`) reach it only over REST + SSE.

## Migration status (docs/Aloy.md — "adopt pori_cloud, unify on PoriEvent")

- [x] **Stage 3.1** — copy `pori_cloud` → here; wire the kernel path to the repo
  root (`../../..`); drop `pori_cloud`'s AI-tooling cruft; all Python
  syntax-compiles clean.
- [ ] **Stage 3.2 — boot** — `uv sync` the backend deps, provide `.env`
  (Supabase + `DATABASE_URL`), run Alembic migrations, boot uvicorn against a
  local Postgres.
- [ ] **Stage 3.3 — unify on `PoriEvent`** — the `conversations` streaming
  currently emits its own `status/tool/step/message` SSE. **Harvest the kernel
  `pori/api`'s `PoriEvent` mapping + clarify bridge + delegation** into it, so the
  contract matches `@aloy/shared`. Then `pori/api` shrinks to a reference server.
- [ ] Reconcile the two `config.yaml` / duplicate settings with the kernel.

Gateway (Slack/Telegram) will **harvest Hermes's gateway architecture**
(`references/hermes-agent-deep-dives/gateway-messaging.md`) when we add it.
