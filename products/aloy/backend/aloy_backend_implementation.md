# Pori Cloud — Implementation Plan

## Current State

- FastAPI app with Supabase JWT auth (JWKS-based ES256 verification)
- Async Postgres (Supabase) via SQLAlchemy + asyncpg
- Run model (id, user_id, task, status, max_steps, success, steps_taken, final_answer, reasoning, metrics, created_at)
- Endpoints: `POST /v1/runs`, `GET /v1/runs`, `GET /v1/runs/{id}` — all scoped to authenticated user
- Pori engine integrated (Orchestrator, Agent, Team, Tools, Memory)
- Pori supports 3 LLM providers: Anthropic, OpenAI, Google Gemini

### Current Files

```
aloy_backend/
  __init__.py
  api.py              # Routes + lifespan
  auth.py             # JWT verification via Supabase JWKS
  config.py           # Pydantic-settings
  database.py         # Async engine + session
  models.py           # Run table
  schemas.py          # RunRequest, RunResponse
  orchestrator.py     # Builds pori Orchestrator
  main.py             # Uvicorn entry
  prompts/            # Bundled agent prompts
```

---

## Phase 1: Foundation

Make the existing API production-ready. Cannot connect a frontend without these.

### 1.1 CORS Middleware

**File:** `api.py`

Add `CORSMiddleware` right after app creation. Configure `allow_origins` from a new `CORS_ORIGINS` setting (default: `["http://localhost:3000"]` for Next.js dev). Allow credentials, all methods, all headers.

### 1.2 Health Check

**File:** `api.py`

Add `GET /v1/health` returning `{"status": "ok", "version": "0.1"}`. No auth required. Needed for deployment health probes and frontend connectivity checks.

### 1.3 Error Handling Middleware

**File:** new `middleware.py`

- Catch unhandled exceptions, return structured JSON errors (`{"detail": "...", "request_id": "..."}`)
- Assign a UUID request ID to every request, return in `X-Request-ID` header
- Log errors with request context

### 1.4 Structured Logging

**File:** new `logging_config.py`

Set up structured JSON logging via `logging.config.dictConfig`. Include request_id in log context. Replace all `print()` with proper logger calls.

### 1.5 Config Expansion

**File:** `config.py`

Add:
- `cors_origins: list[str]` (default `["http://localhost:3000"]`)
- `log_level: str` (default `"INFO"`)
- `rate_limit_rpm: int` (default `60`)
- `max_concurrent_runs: int` (default `5`)

### 1.6 Alembic Migrations

Replace `SQLModel.metadata.create_all` with proper Alembic migrations so the schema can evolve without data loss. Run `alembic init aloy_backend/alembic`, configure `env.py` for async engine.

---

## Phase 2: Conversations

The core user experience. Users chat with agents, not fire-and-forget tasks. The entire frontend UX depends on this.

### 2.1 Models

**File:** `models.py`

**Conversation:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `title` (str | None — auto-generated from first message, editable)
- `agent_config_id` (FK to AgentConfig, nullable)
- `created_at`, `updated_at`

**Message:**
- `id` (uuid, PK)
- `conversation_id` (FK, indexed)
- `role` ("user" | "assistant" | "system")
- `content` (str)
- `metadata` (JSON — tool calls, step details, metrics)
- `created_at`

Add nullable `conversation_id` FK to existing `Run` model.

### 2.2 Schemas

**File:** `schemas.py`

- `ConversationCreate` (optional title, optional agent_config_id)
- `ConversationResponse` (id, title, created_at, updated_at, message_count)
- `ConversationDetail` (includes list of messages)
- `MessageResponse` (id, role, content, metadata, created_at)
- `SendMessageRequest` (content: str, stream: bool = False)

### 2.3 Endpoints

**File:** new `routes/conversations.py`

- `POST /v1/conversations` — create conversation
- `GET /v1/conversations` — list (paginated: `?limit=20&offset=0`)
- `GET /v1/conversations/{id}` — get with messages
- `DELETE /v1/conversations/{id}` — delete
- `PATCH /v1/conversations/{id}` — update title
- `POST /v1/conversations/{id}/messages` — send message (triggers agent)

The message endpoint:
1. Saves user message to DB
2. Loads conversation history, seeds `AgentMemory`
3. Creates Orchestrator with that memory
4. Executes task
5. Saves assistant response to DB
6. Returns response

### 2.4 Refactor to Routers

Move existing run endpoints to `routes/runs.py`. Mount all routers from `api.py`:
- `/v1/runs` — existing
- `/v1/conversations` — new
- `/v1/health` — health check

---

## Phase 3: Streaming (SSE)

Without streaming, the frontend shows a loading spinner for 10–60+ seconds.

### 3.1 SSE Endpoint

**File:** `routes/conversations.py`

When `POST /v1/conversations/{id}/messages` receives `stream: true`, return a `StreamingResponse` with `media_type="text/event-stream"`.

Uses pori's `on_step_start` / `on_step_end` callbacks to push events into an `asyncio.Queue`. A generator yields SSE events.

SSE event format:
```
event: step_start
data: {"step": 1, "status": "thinking"}

event: step_end
data: {"step": 1, "action": "web_search", "result": "..."}

event: message
data: {"role": "assistant", "content": "Here is what I found..."}

event: done
data: {"message_id": "...", "metrics": {...}}
```

Step-level streaming, not token-level. No changes to pori's LLM protocol needed.

### 3.2 SSE Helper

**File:** new `streaming.py`

- `async def stream_agent_execution(orchestrator, task, memory, conversation_id) -> AsyncGenerator[str, None]`
- Handles queue, callback wiring, SSE formatting
- Sends `event: error` on failure before closing

---

## Phase 4: Agent Configuration

Users choose their model, customize system prompts, select which tools agents can use.

### 4.1 Model

**File:** `models.py`

**AgentConfig:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `name` (str — "My Custom Agent", "Code Helper")
- `provider` ("anthropic" | "openai" | "google")
- `model` (str)
- `temperature` (float, default 0.0)
- `max_steps` (int, default 15)
- `system_prompt` (str | None — custom override)
- `tools` (JSON list[str] | None — tool name filter, null = all)
- `is_default` (bool)
- `created_at`

### 4.2 Endpoints

**File:** new `routes/agent_configs.py`

- `GET /v1/agent-configs` — list user's configs
- `POST /v1/agent-configs` — create
- `PATCH /v1/agent-configs/{id}` — update
- `DELETE /v1/agent-configs/{id}` — delete
- `GET /v1/models` — list available models/providers
- `GET /v1/tools` — list available tools from pori registry

### 4.3 Wire into Conversations

Look up the conversation's `agent_config_id`, use `create_llm()` for the right provider, build filtered tool registry, inject custom system prompt.

---

## Phase 5: Background Tasks

`POST /v1/runs` currently blocks the HTTP connection. Long-running agents will time out.

### 5.1 Background Run Manager

**File:** new `background.py`

- On submission, create `Run` with `status="running"`
- Launch via `asyncio.create_task()`
- On completion, update `Run` with results
- Track task references in a dict for status polling

### 5.2 Updated Run Flow

- `POST /v1/runs` returns immediately with `status: "running"` and run ID
- `GET /v1/runs/{id}` shows current status (running/completed/failed)
- Streaming conversations (Phase 3) handle the real-time use case; background runs are for fire-and-forget API usage

---

## Phase 6: User Profiles

### 6.1 Model

**File:** `models.py`

**UserProfile:**
- `id` (user_id from Supabase, PK)
- `display_name` (str | None)
- `avatar_url` (str | None)
- `default_agent_config_id` (FK | None)
- `preferences` (JSON — theme, etc.)
- `created_at`, `updated_at`

Auto-create on first authenticated request.

### 6.2 Endpoints

**File:** new `routes/users.py`

- `GET /v1/me` — get current user profile (auto-creates if missing)
- `PATCH /v1/me` — update profile
- `GET /v1/me/usage` — usage stats

---

## Phase 7: Multi-Agent Teams

Pori already has a full Team system. Expose it via API.

### 7.1 Model

**File:** `models.py`

**TeamConfig:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `name` (str)
- `mode` ("router" | "broadcast" | "delegate")
- `members` (JSON — list of MemberConfig dicts)
- `max_delegation_steps` (int)
- `created_at`

### 7.2 Endpoints

**File:** new `routes/teams.py`

- `POST /v1/teams` — create team config
- `GET /v1/teams` — list user's teams
- `POST /v1/teams/{id}/run` — execute task with team
- `POST /v1/conversations/{id}/messages` with `team_id` param — route through team

Uses `pori.team.Team` directly, constructing from stored config.

---

## Phase 8: Rate Limiting + Usage Tracking

### 8.1 Rate Limiter

**File:** new `rate_limit.py`

In-memory sliding window per user. FastAPI dependency. Returns `429 Too Many Requests` with `Retry-After` header. Swap to Redis-backed for multi-instance.

### 8.2 Usage Model

**File:** `models.py`

**UsageRecord:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `run_id` (FK | None)
- `conversation_id` (FK | None)
- `provider`, `model`
- `input_tokens`, `output_tokens`
- `estimated_cost` (float)
- `created_at`

Populated from pori's metrics (TokenUsage per step).

### 8.3 Usage Endpoints

- `GET /v1/me/usage` — aggregate stats
- `GET /v1/me/usage/history?days=30` — daily breakdown

---

## Phase 9: File Uploads

### 9.1 Model

**File:** `models.py`

**File:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `filename`, `content_type`, `size_bytes`
- `storage_path` (Supabase Storage bucket path)
- `created_at`

### 9.2 Endpoints

**File:** new `routes/files.py`

- `POST /v1/files` — upload (multipart, store in Supabase Storage)
- `GET /v1/files` — list user's files
- `DELETE /v1/files/{id}` — delete

### 9.3 Wire into Agent Context

When a message references a `file_id`, load file content and inject into agent context.

---

## Phase 10: API Keys

For programmatic access without Supabase auth.

### 10.1 Model

**File:** `models.py`

**ApiKey:**
- `id` (uuid, PK)
- `user_id` (str, indexed)
- `key_hash` (SHA-256, never store plaintext)
- `name` (str)
- `last_used_at`, `expires_at`, `created_at`

### 10.2 Dual Auth

**File:** `auth.py`

Accept either:
1. `Authorization: Bearer <supabase_jwt>` (existing)
2. `X-API-Key: pk_...` (new — hash and look up in DB)

Both return the same `user_id`.

### 10.3 Endpoints

**File:** new `routes/api_keys.py`

- `POST /v1/api-keys` — create (returns plaintext once)
- `GET /v1/api-keys` — list (masked)
- `DELETE /v1/api-keys/{id}` — revoke

---

## Final Directory Structure

```
aloy_backend/
  __init__.py
  main.py                 # Uvicorn entry
  api.py                  # App factory, mounts routers, middleware
  config.py               # Settings
  database.py             # Async engine + session
  auth.py                 # JWT + API key auth
  models.py               # All SQLModel tables
  schemas.py              # All request/response schemas
  middleware.py            # Error handling, request ID
  logging_config.py        # Structured logging
  streaming.py            # SSE helper
  background.py           # Background task manager
  rate_limit.py           # Rate limiting dependency
  orchestrator.py         # Orchestrator builder (per-user config)
  routes/
    __init__.py
    runs.py
    conversations.py
    agent_configs.py
    teams.py
    users.py
    files.py
    api_keys.py
  alembic/
    env.py
    versions/
  prompts/
    system/
      agent_core.md
```

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| SSE over WebSocket | Simpler, works with standard HTTP, auto-reconnects, sufficient for one-way streaming |
| Step-level streaming (not token-level) | Uses existing pori `on_step_start`/`on_step_end` callbacks, no framework changes needed |
| Postgres is source of truth for conversations | Not pori's memory system — load history from DB, seed AgentMemory per request |
| Single process for MVP | `asyncio.create_task()` for background work, in-memory rate limiting. Add Redis/Celery when scaling |
| Users use your API keys (not BYOK) | Simplifies billing. Bring-your-own-key is a future concern requiring encrypted key storage |
| Alembic over create_all | Schema will evolve across all 10 phases — need migrations that preserve data |

---

## Pori Framework Fixes Needed

### Wire `on_step_start`/`on_step_end` callbacks into `agent.run()`

**Problem:** The orchestrator's `execute_task()` defines `on_step_start` and `on_step_end` callback parameters, but `agent.run()` never calls them. The step loop in `agent.py` calls `self.step()` directly without any hooks. This means SSE streaming in aloy_backend can only poll agent state, not receive rich step data (tool inputs/outputs, reasoning, errors, memory summaries).

**What the CLI shows that the API doesn't:**
- Tool input parameters and raw results
- Agent reasoning/reflection after each step
- Error details per step
- Memory summaries
- Step duration

**Fix:** In `pori/agent.py`, modify the `run()` method to accept and call `on_step_start`/`on_step_end` callbacks around each `self.step()` call, passing the agent instance so the callback can read `agent.state`, `agent.memory.tool_call_history`, etc. Then wire the orchestrator's callbacks through to `agent.run()`.

**Impact:** Once fixed, aloy_backend's SSE streaming can emit rich step events with full tool I/O, reasoning, and error details — matching what the CLI shows.

### Persistent Memory per User (CRITICAL)

**Problem:** Currently aloy_backend creates a fresh `AgentMemory` per request, seeds it with conversation history, and throws it away. This means the agent has NO persistent memory — it doesn't remember user preferences, prior context, or anything across conversations. This defeats the purpose of Pori's memory system, which is the core differentiator.

**What Pori's memory system provides:**
- `CoreMemory` blocks (persona, human, notes) — editable, persistent knowledge about the user and agent
- `memory_insert` / `memory_rethink` tools — the agent can update its own memory
- `ToolCallRecord` history — what tools were used and when
- Archival memory — long-term searchable storage

**What needs to happen:**
- Build a `PostgresMemoryStore` (implementing pori's `MemoryStore` protocol) that stores memory state in Supabase Postgres, scoped per user
- Each user gets persistent `CoreMemory` blocks — the agent remembers preferences, context, and learned information across ALL conversations
- The `memory_insert` and `memory_rethink` tools work as designed — when the agent decides to remember something, it persists
- This is NOT optional — it's the moat. Without persistent memory, aloy_backend is just an API wrapper around an LLM
