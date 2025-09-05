# FastAPI Integration Roadmap  
*Pori AI Agent Framework*  

## 1. Objectives
1. Provide a robust HTTP interface so external services, front-end apps, or other back-end jobs can drive agents programmatically.  
2. Preserve current CLI workflow while adding stateless **request/response** and optional **long-running task** patterns.  
3. Ensure security, scalability, observability, and minimal disruption to existing code.

---

## 2. Target Architecture (Textual View)

```
┌────────────┐   HTTP    ┌────────────────┐    Async Calls     ┌────────────┐
│  Clients   │ ────────▶ │  FastAPI App   │ ──────────────────▶│ Orchestrator│
│(Web / svc) │           │  (pori.api)    │                    │   (core)   │
└────────────┘           │ • Routing      │                    └────┬───────┘
                         │ • AuthZ/AuthN  │                          │
                         │ • Validation   │              Shared Mem │
                         └──────┬─────────┘                          │
                                │                        ┌──────────▼─────────┐
                                │                        │   Agent Runtime     │
                                │                        └─────────┬──────────┘
                                │                                  │
                        Observability                 Tools Registry│
                     (logging, metrics)                            ▼
                                                     ┌────────────────────────┐
                                                     │Tool Executors & Memory │
                                                     └────────────────────────┘
```

Key points:  
• **pori.api** becomes a new package containing FastAPI app, routers, and schemas.  
• Existing `Orchestrator`, `Agent`, `EnhancedAgentMemory`, and `ToolRegistry` stay in core and are imported.  
• One shared asyncio event-loop maintained by FastAPI (uvicorn).  

---

## 3. New / Updated Components

| Component | Type | Description |
|-----------|------|-------------|
| `pori/api/__init__.py` | package | Exports `create_app()` factory |
| `pori/api/routers/agents.py` | router | Endpoints to submit tasks, check status, get results |
| `pori/api/routers/tools.py` | router | (Optional) Introspect available tools |
| `pori/api/models.py` | Pydantic | Request / response schemas reused across routers |
| `pori/api/deps.py` | helpers | Dependency functions (auth, inject orchestrator, etc.) |
| `pori/api/security.py` | helpers | API-key or JWT verification utilities |
| `pori/api/background.py` | helpers | Manage background task execution & cancellation |
| `pori/config.py` | config | Centralised settings via `pydantic.BaseSettings` |
| `pori/cli.py` | script | Thin wrapper to `uvicorn pori.api:app` for dev |

Modifications:  
• `orchestrator.py` – expose a **singleton** or factory suited for FastAPI dependency injection.  
• `main.py` (CLI) – no change, but share orchestrator instance to avoid duplicate memory.  
• `requirements.txt` – add `fastapi`, `uvicorn[standard]`, `python-multipart` (if file upload).  

---

## 4. Phased Implementation Plan

### Phase 0 – Foundation (1 week)
1. Add FastAPI & uvicorn to deps.  
2. Create `pori/config.py` (env-driven).  
3. Implement `create_app()` returning FastAPI instance with CORS, exception handlers, logging middleware.

### Phase 1 – Core Task API (2 weeks)
1. Build `routers/agents.py`:
   - `POST /v1/tasks` – submit task (sync or async).  
   - `GET  /v1/tasks/{id}` – fetch metadata & status.  
   - `GET  /v1/tasks/{id}/result` – final results & reasoning.  
   - `DELETE /v1/tasks/{id}` – cancel / stop agent.
2. Background execution:
   - Use `asyncio.create_task` with internal registry; map to orchestrator `execute_task`.  
   - Persist results in shared dict or DB (optional in phase1: in-mem).  

### Phase 2 – Tool & Memory Endpoints (1 week)
1. `GET /v1/tools` – list tools with descriptions.  
2. `GET /v1/memory/{agent_id}` – (optional, gated) introspect memory for debugging.

### Phase 3 – Security & Rate Limiting (1 week)
1. Implement API-Key header `X-API-Key` (env list of keys).  
2. Optional JWT/OAuth2 path via FastAPI’s security utilities.  
3. Basic per-IP rate limiting (Starlette-middleware or Redis-based).

### Phase 4 – Persistence & Scaling (2 weeks)
1. Replace in-memory task registry with Redis/PostgreSQL.  
2. Containerize with Docker; `docker-compose` service for API + worker (if we offload heavy jobs to Celery).  
3. Helm chart for Kubernetes if target infra requires.

### Phase 5 – Observability & Testing (ongoing)
1. Structured JSON logs integrated with existing `logging_config`.  
2. Prometheus metrics middleware.  
3. Tests:  
   - Unit tests for routers & security.  
   - Integration test spinning up app with `httpx.AsyncClient`.

---

## 5. API Design

### 5.1 Endpoint Summary

| Method | Path | Purpose |
|--------|------|---------|
| POST | /v1/tasks | Submit new agent task |
| GET | /v1/tasks/{task_id} | Retrieve task metadata & progress |
| GET | /v1/tasks/{task_id}/result | Get final answer (200 if done, 202 if running) |
| DELETE | /v1/tasks/{task_id} | Cancel / stop running task |
| GET | /v1/tools | List registered tools |
| GET | /v1/health | Readiness probe |

### 5.2 Schema Sketches

```
TaskCreateRequest:
  task: str
  max_steps: int = 50
  stream: bool = false   # if true -> Server-Sent Events (phase later)

TaskCreateResponse:
  task_id: str
  status: "queued" | "running" | "completed" | "failed"
  submitted_at: datetime

TaskStatusResponse:
  task_id: str
  status: ...
  steps_taken: int
  success: bool | null
  started_at: datetime
  updated_at: datetime

TaskResultResponse:
  task_id: str
  success: bool
  final_answer: str | null
  reasoning: str | null
```

Common error payload:
```
ErrorResponse:
  detail: str
  code: str
```

Pagination (for /tools) via `limit` & `offset` query params.

---

## 5.3 Real-Time Agent Monitoring Endpoints

Real-time transparency is critical for debugging, auditing and rich UI experiences.  
We support **both polling** (HTTP GET) **and streaming** (WebSocket/SSE) models.

### Endpoint Matrix

| Method  | Path | Purpose |
|---------|------|---------|
| GET | /v1/tasks/{task_id}/steps | Paginated list of executed steps |
| GET | /v1/tasks/{task_id}/plan  | Current agent plan / reflection |
| GET | /v1/tasks/{task_id}/tools | Tool-call history (latest N) |
| GET | /v1/tasks/{task_id}/memory | (Debug) Snapshot of working memory |
| POST | /v1/tasks/{task_id}/control | Pause / resume / stop agent |
| WS / SSE | /v1/stream/tasks/{task_id} | Live updates: steps, plan revisions, tool calls |

All endpoints are protected by the same `X-API-Key` scheme and respect rate-limits.

### 5.3.1 Polling APIs

```
GET /v1/tasks/{task_id}/steps?limit=20&offset=0

StepListResponse:
  task_id: str
  total: int
  items: List[StepInfo]

StepInfo:
  step_number: int
  started_at: datetime
  duration_seconds: float
  success: bool
  state_snapshot: dict   # optional mini-state for UI
```

```
GET /v1/tasks/{task_id}/plan

PlanResponse:
  task_id: str
  current_plan: List[str]
  last_reflection: str | null
  updated_at: datetime
```

```
GET /v1/tasks/{task_id}/tools?limit=50

ToolCallListResponse:
  task_id: str
  total: int
  items: List[ToolCall]

ToolCall:
  id: str
  tool_name: str
  parameters: dict
  result: dict | str
  success: bool
  timestamp: datetime
```

```
GET /v1/tasks/{task_id}/memory

MemorySnapshotResponse:
  task_id: str
  working_messages: List[Message]  # last N messages
  summary: str | null              # last summarization text
  vector_hits: int | null          # debug info
```

```
POST /v1/tasks/{task_id}/control
{
  "action": "pause" | "resume" | "stop"
}

ControlResponse:
  task_id: str
  new_status: str    # paused / running / stopped
  detail: str
```

### 5.3.2 Streaming API (WebSocket/SSE)

Endpoint: `GET /v1/stream/tasks/{task_id}`  
Clients negotiate either:
• `Accept: text/event-stream`  → Server-Sent Events  
• `Upgrade: websocket`         → WebSocket

#### SSE Message Format
```
event: step
data: {
  "step_number": 3,
  "success": true,
  "duration_seconds": 1.42,
  "tool_calls": [
    {"tool_name": "search_web", "success": true}
  ]
}

event: plan_update
data: {
  "current_plan": ["gather data", "summarize", "done"]
}

event: memory_summary
data: {
  "summary": "We have collected facts about topic X..."
}
```

#### WebSocket JSON Envelope
```
{
  "type": "step" | "plan_update" | "tool_call" | "memory_summary" | "log",
  "payload": { ... }   # schema mirrors SSE data above
}
```

The backend pushes messages from the `Orchestrator` callbacks (`on_step_end`, plan reflection hooks, tool executor events).

### 5.3.3 Implementation Notes

1. **Data Source** – The existing `EnhancedAgentMemory` already stores step metadata and tool call history. Expose read-only accessors for router dependencies.<br>
2. **Back-Pressure** – For high-frequency steps, batch updates or throttle pushes (max 5 msgs/sec).<br>
3. **Security** – The same API key must be supplied as query param (`?api_key=`) for SSE, or via WebSocket sub-protocol header.<br>
4. **Retention** – Step/tool logs kept in memory until task completion; optional persistence via Redis/Postgres in Phase 4.<br>
5. **Client SDK** – Provide TypeScript helper to connect to streaming endpoint, auto-reconnect, and expose typed events.

With these monitoring endpoints, consumers can build dashboards that show each agent's thought-process, enabling powerful human-in-the-loop workflows.

## 6. Security

1. **Transport** – TLS termination handled by ingress / load balancer.  
2. **Authentication** –  
   • Header `X-API-Key`.  
   • Keys stored in `PORI_API_KEYS` env (comma-sep) or DB table.  
3. **Authorization** – simple: key grants full access; later, scopes per endpoint.  
4. **Rate Limiting** – sliding window per key/IP.  
5. **CORS** – allow configurable origins.  
6. **Sensitive Data** – redact logs (e.g., Anthropic API keys, user prompts).

---

## 7. Deployment Strategy

| Environment | Method | Notes |
|-------------|--------|-------|
| Local dev | `uvicorn pori.api:create_app --reload` | Hot reload, logs to console |
| Docker | `docker build -t pori-api .` | Copy only runtime deps |
| Prod | Kubernetes / ECS | Use liveness & readiness probes on `/v1/health` |
| Serverless | (optional) | FastAPI + Mangum or Lambda handler |

Environment variables:
```
PORI_API_HOST=0.0.0.0
PORI_API_PORT=8000
PORI_LOG_LEVEL=info
PORI_API_KEYS=key1,key2
...
```

---

## 8. Observability

1. **Logging** – integrate existing `setup_logging` with request-ID and task-ID correlation IDs.  
2. **Metrics** – expose `/metrics` for Prometheus: request latency, agent count, tool call counts.  
3. **Tracing** – optional OpenTelemetry instrumentation.

---

## 9. Migration & Compatibility

1. CLI (`python pori/main.py`) remains untouched; both interfaces can coexist.  
2. Shared `EnhancedAgentMemory` allows results from API and CLI in same session when app run inside same process.  
3. Deprecation path: after adoption, gradually encourage consumers to move from CLI to HTTP.  
4. Provide utility script to post tasks via API so existing shell workflows convert easily.

---

## 10. Timeline & Staffing (Rough)

| Phase | Duration | Roles |
|-------|----------|-------|
| 0 | 1 week | 1 back-end |
| 1 | 2 weeks | 2 back-end, 1 QA |
| 2 | 1 week | 1 back-end |
| 3 | 1 week | 1 back-end, 1 DevSecOps |
| 4 | 2 weeks | 1 back-end, 1 DevOps |
| 5 | Parallel | QA + SRE |

Buffer 1 week for hardening & docs. Total ≈ 8 weeks.

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Blocking async tasks exhaust worker threads | High | Use proper asyncio & tune `uvicorn --workers`; offload heavy jobs to Celery/Taskiq |
| Unbounded memory growth (shared memory) | Medium | TTL pruning, persistent vector store with size cap |
| LLM latency causes HTTP timeouts | High | Offer async polling model; default timeout 5 min |
| Key leakage in logs | Medium | Redaction filters, structured logging |

---

## 12. Next Steps

1. Approve architecture & timeline.  
2. Spin off branch `feature/fastapi-integration`.  
3. Phase 0 kickoff: add dependencies, scaffolding, CI job for API tests.  
4. Schedule security review at Phase 3.  
5. Prepare documentation site with OpenAPI schema and quick-start examples.

---
