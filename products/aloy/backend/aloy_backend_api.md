# Aloy Backend — API & Architecture Reference

_Verified against commit `7f067f9` (2026-07-12). This document describes the
architecture and every route module accurately; for exhaustive
endpoint/schema detail, the live source of truth is `GET /openapi.json`
(FastAPI generates it from the code — it cannot drift). If this document
contradicts the code, the code wins and this file needs a new verification
pass._

## Architecture

- **Framework**: FastAPI (async), served by uvicorn — **exactly one API
  worker process** (ADR 0009: live-run/clarify/stop/resume registries are
  in-process; scale the durable worker service, never API processes).
- **Database**: PostgreSQL (Supabase) in prod, SQLite in dev — SQLModel +
  asyncpg/aiosqlite. Alembic migrations (`create_all` covers new tables in
  dev but never ALTERs — run migrations for column changes).
- **Auth**: Supabase JWT (verified via JWKS). Every request resolves to an
  **OrganizationContext**.
- **Tenancy — the load-bearing concept**: Aloy is **org-scoped multi-tenant
  with RBAC**. Personal accounts are single-member orgs (`user:<uuid>` org
  ids). `OrganizationContext` carries org, user, role-derived
  `Permission`s, and the org's `OrganizationPolicy` (limits, allowed
  tools/models/providers). Routes gate via
  `require_permission(Permission.X)`; run-starting endpoints compose
  permission + rate limit (`rate_limited_permission`). Every data access
  filters by `organization_id` — the org boundary is applied in SQL, never
  post-hoc.
- **The kernel boundary**: the backend imports ONLY the `pori` front door
  (ADR 0008, CI-enforced). Kernel = agent loop, memory, tools, teams;
  backend = tenancy, persistence, transport, product tools.
- **Streaming**: kernel `PoriEvent`s relayed as SSE. A background pump owns
  each run's event lifecycle; HTTP responses are subscribers — so clients
  can disconnect, re-attach live (`GET /conversations/{id}/live`), stop
  (`POST .../stop`), and continue interrupted runs.
- **Persistence discipline**: ONE finalizer (`persist_run_outcome`,
  idempotent by run_id) writes message + run + usage + trace + event log +
  memory + stored artifacts in one transaction (ADR 0007).
- **Run assembly**: `run_surface.py::resolve_run_surface()` is the single
  service that gives ANY run path (chat, worker) the caller's connections,
  MCP servers, file library, and capability-gated tool denials.
- **Storage**: durable blobs behind the `ObjectStore` seam (local disk dev /
  S3-compatible prod incl. Supabase Storage); per-conversation sandbox jails
  under `SANDBOX_BASE_DIR`; uploads eagerly provisioned into the sandbox;
  artifacts extracted to storage in the finalizer.

## Route modules (all under `/v1`)

| Module | Prefix | What it owns |
|---|---|---|
| `conversations` | `/conversations` | Chat threads: CRUD, branch/export, search; **send_message** (streaming + blocking + team + durable modes); live re-attach / stop / clarify; per-conversation artifacts; durable file uploads |
| `runs` | `/runs` | Standalone durable runs (created here, executed by the worker service via DB leases) |
| `files` | `/files` | Durable file downloads; the user file library (save/remove — writes the memory pointer) |
| `memory` | `/memory` | CoreMemory blocks (persona/human/notes) + typed knowledge entries (org→team→personal scoped) |
| `agent_configs` | `/agent-configs` | Per-user LLM/agent settings + tool/model info endpoints |
| `teams` | `/teams` | Multi-agent team blueprints (router/broadcast/delegate) |
| `skills` | `/skills` | Reusable instruction skills (DB-backed catalog) |
| `connections` | `/connections` | Native OAuth connect-engine (Google: Gmail+Calendar); encrypted token custody; user- or org-scoped |
| `mcp_servers` | `/mcp-servers` | Remote MCP server registry (user/org scoped); tools join runs via the run surface |
| `evolution` | `/evolution` | Governed self-evolution proposals (review/approve/activate) |
| `cron` | `/cron` | Scheduled recurring runs (executed by the worker's cron tick) |
| `organizations` | `/organizations` | Org CRUD, membership, roles, policy |
| `users` | `/me` | Profile + usage history |
| `usage` | `/usage` | Token/cost records per org/user |
| `traces` | `/traces` | Execution traces (span trees) per run/conversation |
| `gateway` | `/gateway` | Messaging-gateway pairing (Telegram) |
| `system` | `/system` | Operator-facing execution info (sandbox backend, isolation) |

## Key flows

**Send message (the money path)** — `POST /conversations/{id}/messages`:
requires `RUN_CREATE` + rate limit → durable upload refs resolved → task
assembled (inline text / native image+PDF blocks / extracted docx-xlsx /
uploads reference block) → run surface resolved → one of four modes:
durable-enqueue (worker executes), team-inline, **streaming** (SSE; pump +
subscriber model), or blocking. All modes persist through the single
finalizer.

**Live-run control** — re-attach: `GET /conversations/{id}/live` (replays
buffered frames, then follows); stop: `POST /conversations/{id}/stop`
(cooperative cancel — in-flight LLM call aborted, partial text kept, run
marked stopped); continue: send with `resume_run_id` (warm resume from the
kernel checkpoint when cached; graceful re-prompt otherwise).

**Files** — attach in composer → small files ride inline/native; everything
also/or gets a durable copy (`POST /conversations/{id}/files`, streamed,
quota'd) → eagerly provisioned into the conversation sandbox → agents work
on real bytes → artifacts extracted back to storage in the finalizer →
downloadable (`GET /files/{id}`), bookmarkable into the library
(`POST /files/{id}/library` — creates the cross-chat memory pointer +
gates the `fetch_my_file` tool on).

## Operational invariants

- One API worker (ADR 0009). No `--reload` on Windows dev (broken worker
  spawns — restart manually, then verify one listener + expected routes).
- Tests run with empty API keys (mocks only); 60% coverage floor; mypy
  gates both packages.
- In-process registries (`live_runs`, `CLARIFY_BRIDGES`, `resumable_runs`,
  rate limiter) are the single-worker constraint; Redis/DB is the
  documented scale-out path.
