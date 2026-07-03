# Pori Cloud API Reference

Complete API documentation for the pori_cloud backend. This document describes every endpoint, request/response schema, authentication, and architectural decisions needed to build the frontend.

---

## Architecture Overview

- **Framework**: FastAPI (async Python)
- **Database**: PostgreSQL (Supabase) via SQLAlchemy + asyncpg
- **Auth**: Supabase JWT (ES256 via JWKS)
- **Streaming**: Server-Sent Events (SSE) for real-time agent output
- **Memory**: Letta-inspired persistent memory per user (CoreMemory shared across conversations, messages per conversation)
- **Base URL prefix**: `/v1`

### Key Concepts

- **Conversation** — a chat thread. Contains messages. Optionally linked to an AgentConfig.
- **Message** — a single user or assistant message within a conversation.
- **Run** — a record of an agent execution. Created per message send (within conversations) or standalone via `/runs`.
- **AgentConfig** — user-customizable LLM settings (provider, model, temperature, tools, system prompt).
- **TeamConfig** — multi-agent team blueprint (members, mode: router/broadcast/delegate).
- **UserProfile** — auto-created on first auth. Stores display name, avatar, preferences.
- **Memory** — persistent agent memory per user. CoreMemory blocks (persona, human, notes) shared across all conversations. Knowledge entries store long-term facts; archival routes remain as backward-compatible aliases.
- **UsageRecord** — per-request token counts and cost tracking.
- **TraceRecord** — full agent execution trace (span tree) for debugging.

---

## Authentication

All endpoints except those marked "Public" require a Supabase JWT.

```
Authorization: Bearer <supabase_jwt_token>
```

The JWT `sub` claim is used as the `user_id`. All data is scoped to the authenticated user — users can only see/modify their own resources.

**Public endpoints** (no auth):
- `GET /v1/health`
- `GET /v1/agent-configs/info/models`
- `GET /v1/agent-configs/info/tools`

---

## Rate Limiting

Applied to `POST /v1/conversations/{id}/messages` (the most expensive endpoint).

- Default: 60 requests per minute per user
- Returns `429 Too Many Requests` with `Retry-After` header when exceeded
- In-memory sliding window (resets on server restart)

---

## Error Format

All errors return JSON:
```json
{
  "detail": "Human-readable error message"
}
```

Standard HTTP codes: `400` (bad request), `401` (unauthorized), `404` (not found / not owned), `429` (rate limited), `500` (server error).

---

## Endpoints

### Health

#### `GET /v1/health`
**Auth**: None

**Response** `200`:
```json
{"status": "ok", "version": "0.1"}
```

---

### Conversations

#### `POST /v1/conversations`
Create a new conversation.

**Request Body**:
```json
{
  "title": "string | null",
  "agent_config_id": "string | null"
}
```
Both fields are optional. Title auto-generates from the first message if not set.

**Response** `201` — `ConversationResponse`:
```json
{
  "id": "string",
  "title": "string | null",
  "agent_config_id": "string | null",
  "created_at": "2026-03-27T12:00:00Z",
  "updated_at": "2026-03-27T12:00:00Z",
  "message_count": 0
}
```

---

#### `GET /v1/conversations`
List user's conversations.

**Query Params**: `limit` (default 20), `offset` (default 0)

**Response** `200` — `List[ConversationResponse]`

Ordered by `updated_at` DESC (most recently active first). Includes `message_count`.

---

#### `GET /v1/conversations/{conversation_id}`
Get conversation with all messages.

**Response** `200` — `ConversationDetail`:
```json
{
  "id": "string",
  "title": "string | null",
  "agent_config_id": "string | null",
  "created_at": "datetime",
  "updated_at": "datetime",
  "messages": [
    {
      "id": "string",
      "role": "user | assistant",
      "content": "string",
      "metadata": {"reasoning": "...", "steps_taken": 3, "metrics": {...}} | null,
      "created_at": "datetime"
    }
  ]
}
```

Messages ordered by `created_at` ASC (chronological).

---

#### `PATCH /v1/conversations/{conversation_id}`
Update conversation title.

**Request Body**:
```json
{"title": "New title"}
```

**Response** `200` — `ConversationResponse`

---

#### `DELETE /v1/conversations/{conversation_id}`
Delete conversation and all related data (messages, runs, usage records, traces).

**Response** `204`

---

#### `POST /v1/conversations/{conversation_id}/messages`
Send a message and get an agent response. This is the core endpoint.

**Request Body** — `SendMessageRequest`:
```json
{
  "content": "string (1-100,000 chars)",
  "max_steps": 15,
  "stream": false,
  "team_id": "string | null"
}
```

**Execution paths**:

1. **`team_id` set** — routes through a multi-agent team. The team executes (router picks one member, broadcast runs all, delegate creates a multi-step plan), and the synthesized answer is saved.

2. **`stream: false`** (default) — runs agent synchronously, returns when done.

3. **`stream: true`** — returns an SSE stream immediately.

**Non-streaming response** `201` — `MessageResponse`:
```json
{
  "id": "string",
  "role": "assistant",
  "content": "The agent's response",
  "metadata": {
    "reasoning": "string | null",
    "steps_taken": 3,
    "metrics": {...} | null
  },
  "created_at": "datetime"
}
```

**Streaming response** `200` — `text/event-stream`:
```
event: status
data: {"status": "running", "task": "..."}

event: step
data: {"step": 1, "max_steps": 15, "tool": {"tool": "web_search", "success": true}, "plan": "..."}

event: step
data: {"step": 2, "max_steps": 15, "tool": {"tool": "answer", "success": true}, "plan": "..."}

event: message
data: {"role": "assistant", "content": "Here is what I found...", "reasoning": "...", "steps_taken": 2, "success": true, "metrics": {...}}

event: done
data: {}
```

On error during streaming:
```
event: error
data: {"detail": "Error description"}

event: done
data: {}
```

**Side effects**: Saves user message, auto-titles conversation, executes agent with persistent memory, saves assistant message + Run + UsageRecord + TraceRecord, flushes memory to Postgres.

---

### Runs (Standalone)

Fire-and-forget agent execution without conversations.

#### `POST /v1/runs`
**Request Body**:
```json
{"task": "string", "max_steps": 15}
```

**Response** `202` — `RunResponse`:
```json
{
  "id": "string",
  "status": "pending",
  "success": false,
  "steps_taken": 0,
  "final_answer": null,
  "reasoning": null,
  "metrics": null,
  "created_at": "datetime"
}
```

Returns immediately. Poll `GET /v1/runs/{id}` for status.

#### `GET /v1/runs`
List all runs. **Response** `200` — `List[RunResponse]`

#### `GET /v1/runs/{run_id}`
Get run status/result. **Response** `200` — `RunResponse`

Status values: `"pending"` | `"running"` | `"completed"` | `"failed"`

---

### Agent Configs

#### `POST /v1/agent-configs`
**Request Body** — `AgentConfigCreate`:
```json
{
  "name": "My Custom Agent",
  "provider": "google",
  "model": "gemini-2.5-flash",
  "temperature": 0.0,
  "max_steps": 15,
  "system_prompt": "You are a helpful coding assistant" | null,
  "tools": ["web_search", "code_execution"] | null,
  "is_default": false
}
```

Provider must be `"anthropic"`, `"openai"`, or `"google"`.

**Response** `201` — `AgentConfigResponse`:
```json
{
  "id": "string",
  "name": "string",
  "provider": "string",
  "model": "string",
  "temperature": 0.0,
  "max_steps": 15,
  "system_prompt": "string | null",
  "tools": ["string"] | null,
  "is_default": false,
  "created_at": "datetime"
}
```

#### `GET /v1/agent-configs`
List configs. **Response** `200` — `List[AgentConfigResponse]`

#### `GET /v1/agent-configs/{id}`
Get one config.

#### `PATCH /v1/agent-configs/{id}`
Update config. All fields optional.

#### `DELETE /v1/agent-configs/{id}`
Delete config. **Response** `204`

#### `GET /v1/agent-configs/info/models` (Public)
Returns available models grouped by provider:
```json
{
  "anthropic": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
  "openai": ["gpt-4o", "gpt-4o-mini"],
  "google": ["gemini-2.5-flash", "gemini-2.5-pro"]
}
```

#### `GET /v1/agent-configs/info/tools` (Public)
Returns available tools from Pori's registry:
```json
[
  {"name": "web_search", "description": "Search the web"},
  {"name": "code_execution", "description": "Execute Python code"},
  ...
]
```

---

### Teams

#### `POST /v1/teams`
**Request Body** — `TeamConfigCreate`:
```json
{
  "name": "Research Team",
  "mode": "delegate",
  "members": [
    {
      "name": "researcher",
      "description": "Deep web research and fact-finding",
      "llm_config": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"} | null,
      "agent_settings": {"max_steps": 20} | null,
      "tools": ["web_search"] | null
    }
  ],
  "max_delegation_steps": 10,
  "max_concurrent_members": 5
}
```

Mode: `"router"` (pick best member), `"broadcast"` (all in parallel), `"delegate"` (multi-step plan).

**Response** `201` — `TeamConfigResponse`

#### `GET /v1/teams`
List teams.

#### `GET /v1/teams/{id}`
Get team.

#### `PATCH /v1/teams/{id}`
Update team. All fields optional.

#### `DELETE /v1/teams/{id}`
Delete team. **Response** `204`

#### `POST /v1/teams/{id}/run`
Execute a task with the team (standalone, outside conversations).

**Request Body**: `{"task": "Compare Rust vs Go"}`

**Response** `200` — `TeamRunResponse`:
```json
{
  "task": "string",
  "completed": true,
  "steps_taken": 12,
  "final_answer": "Synthesized answer from all members...",
  "mode": "delegate",
  "metrics": null
}
```

Teams can also be used inside conversations by passing `team_id` in the send message request.

---

### User Profile

#### `GET /v1/me`
Get profile (auto-creates on first call).

**Response** `200` — `UserProfileResponse`:
```json
{
  "id": "supabase-user-id",
  "display_name": "string | null",
  "avatar_url": "string | null",
  "default_agent_config_id": "string | null",
  "preferences": {} | null,
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

#### `PATCH /v1/me`
Update profile. All fields optional.

**Request Body** — `UserProfileUpdate`:
```json
{
  "display_name": "Aloy",
  "avatar_url": "https://...",
  "default_agent_config_id": "config-id",
  "preferences": {"theme": "dark"}
}
```

#### `GET /v1/me/usage`
Basic usage stats.

**Response** `200` — `UsageStatsResponse`:
```json
{
  "total_conversations": 42,
  "total_messages": 156,
  "total_runs": 38,
  "member_since": "datetime"
}
```

---

### Memory

User's persistent agent memory (Letta-style). CoreMemory blocks are shared across all conversations — the agent remembers who you are.

#### `GET /v1/me/memory`
Get all CoreMemory blocks.

**Response** `200`:
```json
{
  "blocks": [
    {"label": "persona", "value": "I am a helpful assistant", "limit": 2000, "read_only": false},
    {"label": "human", "value": "User is a senior engineer who prefers concise answers", "limit": 2000, "read_only": false},
    {"label": "notes", "value": "Working on pori_cloud project", "limit": 2000, "read_only": false}
  ]
}
```

#### `GET /v1/me/memory/{block_label}`
Get one block. Labels: `persona`, `human`, `notes`.

#### `PATCH /v1/me/memory/{block_label}`
Edit a block manually.

**Request Body**: `{"value": "New block content"}`

Returns `400` if value exceeds the block's character limit (2000 chars default).

#### `DELETE /v1/me/memory`
Reset all persistent memory (CoreMemory blocks and knowledge entries). **Response** `204`

#### `GET /v1/me/memory/knowledge`
List knowledge entries.

**Query Params**: `limit` (default 50), `offset` (default 0)

**Response** `200`:
```json
[
  {
    "id": "knowledge_abc123",
    "content": "The user mentioned they're building a SaaS product",
    "tags": ["project", "context"],
    "importance": 3,
    "source": "agent",
    "created_at": "2026-03-27T12:00:00"
  }
]
```

#### `POST /v1/me/memory/knowledge`
Manually add a knowledge entry.

**Request Body**:
```json
{"content": "Important long-term fact", "tags": ["project"], "importance": 3, "source": "user"}
```

**Response** `201` — `KnowledgeEntryResponse`

#### `GET /v1/me/memory/archival`
Backward-compatible alias for `GET /v1/me/memory/knowledge`.

#### `POST /v1/me/memory/archival/search`
Search knowledge entries. Current implementation filters by user and optional tags; vector semantic search is a future upgrade.

**Request Body**:
```json
{"query": "what project is the user working on", "k": 10, "tags": null}
```

**Response** `200` — `List[KnowledgeEntryResponse]`

#### `DELETE /v1/me/memory/knowledge/{entry_id}`
Delete a knowledge entry. **Response** `204`

#### `DELETE /v1/me/memory/archival/{passage_id}`
Backward-compatible alias for deleting a knowledge entry. **Response** `204`

---

### Usage & Billing

Token and cost tracking per request.

#### `GET /v1/me/usage` (under `/me/usage`)
Aggregated usage summary.

**Query Params**: `days` (default 30, range 1-365)

**Response** `200`:
```json
{
  "total_tokens": 45230,
  "total_cost": 0.1234,
  "total_requests": 42,
  "by_model": {
    "google/gemini-2.5-flash": {"tokens": 30000, "cost": 0.05, "requests": 30},
    "anthropic/claude-sonnet-4-5": {"tokens": 15230, "cost": 0.0734, "requests": 12}
  }
}
```

Note: This endpoint is at `/v1/me/usage` and is separate from the basic `/v1/me/usage` stats endpoint under User Profile. The router ordering means the usage router's `GET /v1/me/usage` takes precedence.

#### `GET /v1/me/usage/history`
Daily breakdown.

**Query Params**: `days` (default 30)

**Response** `200`:
```json
[
  {"date": "2026-03-25", "tokens": 12000, "cost": 0.04, "requests": 15},
  {"date": "2026-03-26", "tokens": 18000, "cost": 0.05, "requests": 12}
]
```

#### `GET /v1/me/usage/records`
Individual usage records.

**Query Params**: `limit` (default 50), `offset` (default 0)

**Response** `200`:
```json
[
  {
    "id": "string",
    "run_id": "string | null",
    "conversation_id": "string | null",
    "provider": "google",
    "model": "gemini-2.5-flash",
    "input_tokens": 800,
    "output_tokens": 200,
    "total_tokens": 1000,
    "estimated_cost": 0.003,
    "created_at": "datetime"
  }
]
```

---

### Traces

Agent execution traces for debugging. Stored automatically when an agent runs within a conversation.

#### `GET /v1/traces`
List traces (summary only, no full span tree).

**Query Params**: `limit` (default 50), `offset` (default 0)

**Response** `200`:
```json
[
  {
    "id": "string",
    "run_id": "string | null",
    "conversation_id": "string | null",
    "duration_seconds": 3.21,
    "total_spans": 6,
    "status": "ok",
    "created_at": "datetime"
  }
]
```

#### `GET /v1/traces/{trace_id}`
Get full trace with span tree.

**Response** `200`:
```json
{
  "id": "string",
  "run_id": "string",
  "conversation_id": "string",
  "trace_data": {
    "trace_id": "abc123",
    "name": "Agent.run",
    "run_id": "...",
    "status": "ok",
    "duration": "3.210s",
    "total_spans": 6,
    "input": "What is...",
    "output": "The answer is...",
    "tree": [
      {
        "span_id": "...",
        "name": "step_1",
        "type": "agent",
        "status": "ok",
        "duration": "1.200s",
        "children": [
          {
            "span_id": "...",
            "name": "gemini-2.5-flash.invoke",
            "type": "llm",
            "status": "ok",
            "duration": "0.800s",
            "attributes": {"model": "gemini-2.5-flash"},
            "children": []
          }
        ]
      }
    ]
  },
  "duration_seconds": 3.21,
  "total_spans": 6,
  "status": "ok",
  "created_at": "datetime"
}
```

#### `DELETE /v1/traces/{trace_id}`
Delete a trace. **Response** `204`

---

## Database Schema

```
conversations
  id (PK), user_id (idx), title, agent_config_id, created_at, updated_at

messages
  id (PK), conversation_id (idx), role, content, metadata (JSON), created_at

runs
  id (PK), user_id (idx), conversation_id (idx), status, task, max_steps,
  success, steps_taken, final_answer, reasoning, metrics (JSON), created_at

agent_configs
  id (PK), user_id (idx), name, provider, model, temperature, max_steps,
  system_prompt, tools (JSON), is_default, created_at

team_configs
  id (PK), user_id (idx), name, mode, members (JSON), max_delegation_steps,
  max_concurrent_members, created_at

user_profiles
  id (PK = user_id), display_name, avatar_url, default_agent_config_id,
  preferences (JSON), created_at, updated_at

usage_records
  id (PK), user_id (idx), run_id (idx), conversation_id, provider, model,
  input_tokens, output_tokens, total_tokens, estimated_cost, created_at

trace_records
  id (PK), user_id (idx), run_id (idx), conversation_id, trace_data (JSON),
  duration_seconds, total_spans, status, created_at

core_memory_blocks
  id (PK), user_id (idx), label, value, char_limit, created_at, updated_at

knowledge_entries
  id (PK), user_id (idx), content, tags (JSON), importance, source,
  metadata (JSON), created_at
```

---

## Memory Architecture

Pori uses a Letta-inspired memory system:

- **CoreMemory** (3 blocks: persona, human, notes) — persists per user, shared across ALL conversations. The agent reads and writes these blocks to remember who you are.
- **Knowledge Entries** — long-term facts stored as searchable records. The current search path supports tag filtering; semantic/vector search is planned.
- **Messages** — per conversation, loaded from the database each time.

Core memory is stored in `core_memory_blocks` rows keyed by `user_id` and label. Long-term facts are stored in `knowledge_entries`; `/archival` endpoints are compatibility aliases over those records.

When a team executes, team members get the user's CoreMemory as **read-only** — they can see user context but can't modify it. Only the main conversation agent can update persistent memory.

---

## Streaming Protocol

SSE (Server-Sent Events) via `text/event-stream`. Connect with:
```javascript
const response = await fetch('/v1/conversations/{id}/messages', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ content: 'Hello', stream: true }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
// Parse SSE events from the stream
```

Event sequence: `status` -> `step` (repeated) -> `message` | `error` -> `done`

The `done` event always fires last, even after errors. Use it to clean up.
