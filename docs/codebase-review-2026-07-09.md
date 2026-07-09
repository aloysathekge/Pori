# Codebase review — 2026-07-09

_Three parallel verified review passes (backend, frontend, architecture) after the
connections/MCP arc, the send_message single-finalizer refactor, and the chat-UX
fixes. Every finding below was verified against source (file:line), not
speculated. This file is the tracking list; strike items as they're fixed._

## Ranked: the most important problems

### 1. The durable worker still has its own duplicate persistence (HIGH)
`background.py:292-382` (`execute_claimed_run`) hand-builds UsageRecord /
TraceRecord / Message and flushes memory itself — it never calls
`persist_run_outcome`. The drift the refactor retired is alive on the worker
path, and it already disagrees with the main path:
- no `_json_safe` on `artifacts/plan/selected_skills` (`background.py:302-304`)
  → the exact "TokenUsage not serializable → assistant message silently lost"
  bug, waiting on the durable path;
- no `RunEventLog` → durable runs have no replay;
- `_serialize_metrics` (`background.py:38-53`) is a verbatim copy of `_json_safe`.
**Fix: route the worker through `persist_run_outcome`.**

### 2. Worker poison-pill crash loop (HIGH)
`background.py:394-408`: the final commit runs OUTSIDE the try/except, and
`worker.py:63-86` doesn't wrap `execute_claimed_run`. One run whose commit
raises (e.g. #1's serialization) crashes the worker; the run's lease expires,
it's re-claimed, crashes the next worker — forever. Cron rides the same loop
and dies with it. **Fix: guard the commit; wrap run_once in serve().**

### 3. Frontend: no stream abort → cross-conversation bleed (CRITICAL, app)
`ChatPage.tsx:234` + `sse.ts:45-61`: no AbortController anywhere. Switch
conversations mid-stream and conversation A's tokens + final bubble render into
conversation B. Navigate away entirely → reader leaks + setState on unmounted
component. **Fix: AbortController per send, abort on conv switch + unmount, tag
callbacks with their conversation id.**

### 4. CLARIFY_BRIDGES breaks under >1 worker (HIGH at scale)
`streaming.py:33,179-184`: the clarify answer is routed via an in-process
global set. With 2+ uvicorn workers or pods, the resolve POST can land on a
worker that doesn't hold the bridge → the paused run never resumes. Also no
ownership check: any authenticated user who learns a clarification id can
answer another user's clarification (uuid entropy is the only guard).
**Fix: scope bridges by (org,user) + reject cross-owner submits now; move the
registry to the DB/Redis when scaling past one worker.**

### 5. Frontend: stream end-cases hang or drop the reply (HIGH, app)
- `sse.ts:49-61`: leftover buffer discarded at stream end — a final frame
  without the trailing `\n\n` silently drops the assistant reply.
- No read watchdog: a stalled-open connection leaves "sending" spinning
  forever with the input locked (`ChatPage.tsx:372`).
- `onError` doesn't clear `streaming/sending` (`ChatPage.tsx:208`).
**Fix: flush trailing buffer; idle-timeout watchdog; reset state in onError.**

### 6. Missing indexes on the worker's hot query (HIGH, silent)
`worker.py:29-49` filters runs on status/cancel_requested/lease_expires_at and
orders by created_at — none indexed (`models.py:412-467`). Full table scan per
poll tick; also the send_message admission query. **Fix: composite index.**
Related: `with_for_update(skip_locked=True)` is Postgres-only (`worker.py:46`)
— on SQLite two workers can claim the same run (dev/prod behavioral split).

### 7. Artifact drawer bugs (MEDIUM, app)
- `ArtifactDrawer.tsx:24`: `useState(openPath)` ignores later prop changes —
  clicking a second artifact does nothing. Fix: sync via useEffect.
- `artifactPath` not cleared on conversation switch → drawer queries the new
  conversation for the old path.
- Shared-cwd caveat (backend): the allowlist gates the path STRING; in
  shared-process mode two orgs writing the same filename read each other's
  bytes (`conversations.py:574-619`). Consequence of shared cwd, noted for the
  sandbox follow-up.

### 8. Streaming persist rides fragile FastAPI teardown ordering (MEDIUM)
`conversations.py:842-891` uses the request-scoped AsyncSession inside the
StreamingResponse generator's `finally`, under BaseHTTPMiddleware
(`middleware.py:15`) — the stack with a history of teardown-ordering bugs, and
a disconnect-as-cancellation can interrupt the awaited persist. **Fix: open a
fresh session inside the generator for the finalizer.** Also:
`persist_run_outcome`'s idempotency fall-through can attempt a duplicate-PK
insert (`run_outcome.py:185-226`), and interrupted streams persist no Run row
(no billing for partial work; streaming also skips the max_concurrent_runs
admission check).

## Architecture assessment

### The kernel/product boundary: direction enforced, surface not
import-linter only checks direction (product may import kernel, never reverse).
It does NOT stop deep imports — and the product is FORCED past the front door
for six load-bearing seams the public API omits: the LLM factory
(`pori.config`), the event contract (`PoriEvent`/`RUN_END` from
`pori.observability`), clarify (`pori.clarify`), MCP config (`pori.mcp`),
sandbox hooks (`pori.sandbox`), prompt config (`pori.utils.prompt_loader`).
Plus four gratuitous deep imports that could use the front door today
(`teams.py:7,9,10`, `tools/__init__.py:9`).
**Fix: export the six seams from `pori/__init__.py`; add an import-linter
contract forbidding deep imports; clean the four redundant ones.**

### The seams are untyped dict-passing
`tool_context_extra` keys are magic strings (a typo = silent "not connected");
`clarify_handler` falls back to stdin `input()` when mis-wired (a hung server
thread, not an error); `stream_agent_execution` widens `mcp_servers` back to
`Optional[list]`; `result_holder` is an untyped out-param dict. None have
schemas. **Direction: a typed RunSeams/ToolContext contract.**

### Dead code: `pori/api/` is an unwired fork of the streaming bridge
`pori/api/routers/agents.py` duplicates `aloy_backend/streaming.py` (which was
harvested from it) and is referenced nowhere. Two copies of the subtlest
concurrency code in the system, already drifted. **Fix: delete `pori/api/` (or
make it the canonical helper the product imports).**

### Other
- `TOOL_REGISTRY` global is mutated on every request before `.filtered()`
  (`orchestrator.py:59-65`) — idempotent today, race-smell forever.
- Rate limiter is per-process (limit × N workers; resets on deploy).
- `pori/agent/core.py` is an 1866-line god-class (loop + quality gate +
  artifacts + receipts + journal + metrics). The loop logic itself is
  well-gated; the size is the maintainability risk. (Note: the Evaluator is
  deterministic — NOT an LLM call per step; planning/reflection/compression
  calls are all conditional.)
- Untested surfaces: clarify bridge routing, auth/JWKS, rate limiting,
  connection crypto, multi-worker lease races.

## Frontend: remaining verified bugs (beyond #3/#5/#7)
- Out-of-order conversation loads can show the wrong history
  (`ChatPage.tsx:67-96`, no cancellation flag).
- `onDone` fires twice (`sse.ts:60` + `114-116`) → double loadConversations.
- Streamed messages lack `run_id` → Replay button missing until reload.
- CRLF SSE framing unsupported (`\r\n\r\n` never matches `\n\n`).
- Dead streaming UI: plan/step/status setters never populated (no
  onStep/onPlan in `sse.ts`) — StreamingIndicator's plan checklist never shows.
- No 401 handling — expired session renders silent empty lists.
- Cosmetics: no break-words on bubbles; Date.now() React keys; O(n²)
  re-markdown per token.
- Checked-and-OK: no XSS surface (no rehype-raw), multi-line SSE data frames,
  keepalive comments, RunReplay cleanup, double-send guard.

## Suggested fix order
1. Worker → `persist_run_outcome` + crash guard + runs index (backend #1/#2/#6)
2. Frontend stream lifecycle: AbortController + trailing-buffer flush +
   watchdog + onError reset + conv-switch guards (app #3/#5, H3/H5)
3. Clarify ownership check (+ document the single-worker constraint)
4. Artifact drawer prop-sync + reset (app #7)
5. Kernel public-API exports + import-linter deep-import contract; delete
   `pori/api/`
6. The rest as chores (indexes done in 1; 401 handling; run_id in stream
   message frame; CRLF framing)
