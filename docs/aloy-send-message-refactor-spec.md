# Aloy `send_message` refactor — one run-outcome, one persist path

_Design spec (2026-07-09). SPEC ONLY — no code. Retires a whole bug class: the
streaming and non-streaming paths in `send_message` each persist their own subset
of a run's outcomes, and they drifted — memory, then traces, then usage were each
"forgotten in the streaming path" and fixed separately (PRs #121, #121, #122).
Grounded in the Hermes architecture (`references/hermes-agent-deep-dives/
streaming-persistence-architecture.md`): converge both transports to one
normalized object, then persist once, downstream, through one shared finalizer._

## The problem (root cause, not symptoms)

`routes/conversations.py::send_message` has two branches:
- **Streaming** (`req.stream`): `stream_agent_execution` runs the agent in a
  worker thread and yields SSE frames; the route's `_stream()` generator then
  **re-parses the outcome from the final SSE `message` frame** and persists
  Message + Run + TraceRecord + UsageRecord + RunEventLog + core-memory **inline,
  with its own commits.**
- **Non-streaming**: calls `orchestrator.execute_task`, then `_save_result`
  persists Message + Run + UsageRecord + TraceRecord + core-memory.

Two persist sites, maintained independently → they drift. Every "the Traces/
Usage/Memory view is empty" bug this session was the streaming branch missing a
persist the non-stream branch had. Persisting **inside a transport handler** is
the defect; adding the missing lines is treating symptoms.

Two extra hazards in the current streaming branch:
- It reconstructs the outcome by `json.loads`-ing the SSE `message` frame — lossy
  and fragile (the wire shape and the persisted shape are coupled).
- On **client disconnect** mid-SSE, the generator is cancelled and the inline
  persist may never run — the whole turn (message, usage, everything) is lost.

## The contract

> **The SSE handler's only job is to emit bytes and fill a `RunOutcome`. The
> single finalizer both paths share is the only thing allowed to persist.**

Three rules, straight from Hermes:
1. **Normalize first.** Both transports produce ONE canonical `RunOutcome` built
   from `execute_task`'s return value (authoritative) — never re-parsed from SSE.
2. **Persist once, downstream.** One `persist_run_outcome(...)` writes everything
   as a unit, called at the same altitude by both paths.
3. **Persist survives disconnect.** The finalizer runs in a `finally`, so a
   dropped SSE client still records what happened.

## Design

### `RunOutcome` (a dataclass, `aloy_backend/run_outcome.py`)
Everything needed to persist a finished run, built once:
- `final_answer`, `reasoning`, `success`, `steps_taken`
- `metrics` (tokens in/out/total, cost, model), `trace` (dict), `artifacts`,
  `plan`, `selected_skills`
- `prompt_fingerprint`, `tool_surface_fingerprint`, `execution_receipts`
- `memory` (the run's `AgentMemory`, for the core-memory flush)
- `run_id`, `conversation_id`, `task` (the user's message)
- `events` (the coalesced `EventLogCollector` output — streaming only; None otherwise)

### `build_run_outcome(agent_result, memory, run_context, task, events=None)`
Pure function: maps `execute_task`'s result dict (`final_answer`, `success`,
`steps_taken`, `metrics`, `trace`, `artifacts`, …) + the run's `memory` into a
`RunOutcome`. **This is the single place that knows the result shape.** The
streaming path passes the SAME result object the agent produced, plus its
collector's events — not a re-parsed SSE frame.

### `persist_run_outcome(session, context, conv, outcome)`
The ONE finalizer. In a **single transaction** it writes:
- the assistant `Message` (content = `final_answer`, metadata = reasoning/steps/
  metrics/artifacts/plan/run_id),
- the `Run` row (success, steps, fingerprints, receipts, …),
- the `UsageRecord` (from `metrics`),
- the `TraceRecord` (from `trace`, when present),
- the `RunEventLog` (from `outcome.events`, when present),
- the core-memory flush (`_flush_memory_to_db`) + the typed memory records,
- `conv.updated_at`,
then **one `await session.commit()`**.

Idempotent by run_id where relevant (a retry/checkpoint can't double-write).
`_save_result` and the inline streaming persistence are **both deleted** and
replaced by this — a single implementation, so there is nothing left to drift.

### The two callers, after the refactor
- **Non-stream:** `result = await execute_task(...)` → `outcome =
  build_run_outcome(result, memory, …)` → `await persist_run_outcome(...)`.
- **Stream:** `stream_agent_execution` **returns the agent result** (not just
  yields frames); the route accumulates it, and in a `finally` after the SSE loop
  builds the outcome and calls the same `persist_run_outcome`. The generator
  performs **no** DB writes.

### Disconnect handling
Wrap the stream body so the finalizer runs on cancellation too:
```
try:
    async for frame in stream_agent_execution(...): yield frame
finally:
    if result_captured:
        await persist_run_outcome(build_run_outcome(result, memory, …))
```
Mirror Hermes: a client disconnect should also **cooperatively interrupt** the
run (cancel the agent) rather than let it run headless — a follow-up once the
kernel exposes a cancel hook through `execute_task`.

## The regression guard (cheap, structural)
One test that runs the **same task through both paths** and asserts the persisted
row sets are **identical** (same Message/Run/Usage/Trace present, same field
values). This makes drift a test failure, not a production bug found by clicking
an empty page. This is the durable fix — the finalizer plus the test.

## Phase 2 (noted, not this refactor): event-log as source of truth
Longer term, make `RunEventLog` the ordered, `seq`-stamped stream that (a) the
browser tails as SSE, (b) is the replay capture, and (c) whose terminal
`run.completed` event carries the outcome to persist — one pipeline, per the
Hermes synthesis. This refactor is the prerequisite (a single finalizer); the
event-log unification is a separate, larger step and gets its own spec.

## Rollout
1. Add `RunOutcome` + `build_run_outcome` + `persist_run_outcome` (no behavior
   change yet).
2. Switch the **non-stream** path to them; confirm parity with tests.
3. Switch the **streaming** path: `stream_agent_execution` returns the result;
   move persistence out of `_stream()` into the `finally` finalizer.
4. Delete `_save_result` and the inline streaming persist.
5. Add the both-paths-parity regression test + the disconnect-persists test.
Each step is independently testable; the existing 100+ backend tests are the net.

## Open decisions
1. **Where `build_run_outcome` lives** — a new `run_outcome.py`, or inside
   `conversations.py`? (Lean: its own module; it's reused and worth unit-testing.)
2. **Should `stream_agent_execution` return the result, or expose it via a
   holder?** (Lean: return it — the generator can `return value` and the route
   reads it, cleaner than a mutable holder.)
3. **Disconnect → interrupt the run now or later?** Persisting-on-disconnect is
   this spec; cooperatively cancelling the agent needs a kernel cancel hook —
   defer to a follow-up unless cheap.
4. **Idempotency key** — dedupe persistence by `run_id` (one Run per turn)?
   (Lean: yes; guards retries and the disconnect-finally double-firing.)
