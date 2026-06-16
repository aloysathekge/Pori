# Roadmap: Adopting Agno's Best Ideas into Pori

Derived from the Agno study (`references/agno`) and its companion notes. Companion:
`.agent/reference-studies/workflow-adaptation.md`.

Principle: extract interfaces, not code (Agno is Apache-2.0). Each phase is
independently shippable, test-backed, and behavior-preserving for existing APIs.
Order is by **leverage × dependency**, not by Agno's size.

```
Phase 0  RunOutput + event stream   ──┐ (unblocks streaming, pause/resume)
Phase 1  Workflow primitive          ─┤ depends on Phase 0
Phase 2  Session→Run→Message model    ┘ depends on Phase 0
Phase 3  Reasoning subsystem          (independent, but emits Phase 0 events)
Phase 4  Safety guardrails (PII/PI)   (independent quick win — do anytime)
Phase 5  Knowledge/RAG upgrade        (optional, roadmap-dependent)
Phase 6+ AgentOS / OpenTelemetry      (deferred — product-scope decision)
```

Legend — effort: S ≈ <1d, M ≈ 1–3d, L ≈ 1wk+.

---

## Phase 0 — Structured `RunOutput` + event stream  (effort: M) ★ start here

**Why first:** Highest ROI and a prerequisite for Phases 1–3. Today `Agent.run()`
(`pori/agent.py:1539`) returns an ad-hoc dict and the answer hides behind
`result_summary()`. A typed result makes Agent/Team/Workflow interchangeable and
makes streaming/pause-resume first-class.

**Tasks**
1. Add `pori/run.py` with:
   - `RunStatus` enum: `PENDING | RUNNING | PAUSED | COMPLETED | ERROR`.
   - `RunOutput` dataclass: `content` (final answer), `messages`, `run_id`,
     `status`, `metrics` (reuse `metrics.RunMetrics`), `trace`, `steps_taken`,
     `reasoning_steps` (Phase 3), `requirements` (HITL pause), `created_at`.
     Methods: `to_dict()`, `from_dict()`.
   - `RunEvent` enum + lightweight event dataclasses. Start with a **minimal** set
     (not Agno's 35): `RunStarted, StepStarted, StepCompleted, ToolCallStarted,
     ToolCallCompleted, RunCompleted, RunError`. Extend later.
2. Make `Agent.run()` return `RunOutput` **while keeping dict compatibility**:
   add `RunOutput.to_dict()` and have callers (CLI, `Orchestrator.execute_task`,
   `Team`) read typed fields. Keep a `Dict` shim for one release.
3. Add an optional `on_event: Callable[[RunEvent], None]` to `Agent.run()` and emit
   events at existing span boundaries (we already build `Span`s in the loop).
4. Add `Agent.print_response(stream=False)` convenience (mirrors Agno ergonomics).

**Files:** new `pori/run.py`; edit `pori/agent.py`, `pori/orchestrator/core.py`,
`pori/team/core.py`, `pori/cli.py`, `pori/__init__.py` (export `RunOutput`).

**Tests:** `tests/test_run_output.py` — round-trip `to_dict/from_dict`; event order
on a mocked run; dict-shim back-compat. Reuse `conftest.py` `MockLLM`.

**Acceptance:** existing suite stays green via the dict shim; `RunOutput.content`
== old `result_summary()["final_answer"]`; events fire in correct order.

---

## Phase 1 — `Workflow` primitive  (effort: M)  depends on Phase 0

Implement the design already specified in
`.agent/reference-studies/workflow-adaptation.md`.

**Tasks (incremental sub-steps from that doc):**
1. `pori/workflow/context.py` (`WorkflowContext`) + `steps.py`
   (`FunctionStep`, `AgentStep`→delegates to `Orchestrator.execute_task`,
   `TeamStep`) + sequential `Workflow.arun()` returning a **`RunOutput`** (Phase 0).
2. Control-flow steps: `ConditionStep`, `LoopStep` (mandatory `max_iterations`),
   `ParallelStep` (`asyncio.gather` on deep-copied contexts, merge later-wins).
3. Emit `SpanType.workflow`/`SpanType.step` into the existing `Trace`; aggregate
   child `RunMetrics`.
4. `WorkflowConfig` in `pori/config.py` (mirror `TeamConfig`) + CLI wiring.

**Files:** new `pori/workflow/`; edit `pori/config.py`, `pori/__init__.py`, `pori/cli.py`,
`pori/observability/trace.py` (new span types).

**Tests:** per-step unit tests with mocked Orchestrator/Team; end-to-end
Research→Draft→Critique→Loop example against `MockLLM`.

**Acceptance:** sequential pipeline runs end-to-end; loop respects `max_iterations`;
parallel step merges correctly; result is a valid `RunOutput`.

---

## Phase 2 — Session → Run → Message persistence  (effort: L)  depends on Phase 0

**Why:** Pori has no concept of a session containing multiple runs containing
messages with per-message metrics. `MemoryStore` is tied to memory blocks only.

**Tasks**
1. Define `Session` (id, user_id, agent_id, list[RunOutput], created/updated) and a
   `SessionStore` protocol with `save_run`, `load_session`, `list_sessions`.
2. Implement `InMemorySessionStore` + `SQLiteSessionStore` (extend the existing
   SQLite plumbing in `pori/memory.py`). Store runs as JSON blobs (Agno pattern) to
   avoid premature normalization.
3. Thread `session_id` through `Orchestrator.execute_task` / `Agent` and persist a
   `RunOutput` per run.
4. Add per-message metrics to `AgentMessage` (`memory.py`) — token counts per turn.

**Files:** new `pori/session.py`; edit `pori/memory.py`, `pori/orchestrator/core.py`,
`pori/agent.py`, `pori/config.py` (session backend selection).

**Tests:** `tests/test_session.py` — persist/load a multi-run session; SQLite
round-trip; metrics aggregation across runs.

**Acceptance:** a session accumulates runs across calls; reload reconstructs typed
`RunOutput`s; no change required for callers that don't pass `session_id`.

---

## Phase 3 — Reasoning subsystem (native extended thinking)  (effort: M)

**Why:** Pori's Plan→Reflect is hardcoded; it ignores model-native reasoning.
Pori targets Claude/OpenAI/Google directly, so this is a natural fit.

**Tasks**
1. `pori/reasoning/` with `ReasoningStep` + `ReasoningConfig` (`min_steps`,
   `max_steps`, `enabled`, `mode: native|cot`).
2. In `pori/llm/anthropic.py` (and openai/google), surface native reasoning/thinking
   tokens from responses; capture into `ReasoningStep`s.
3. Populate `RunOutput.reasoning_steps` (Phase 0) and account reasoning tokens
   separately in `RunMetrics` (`metrics.py`).
4. CoT fallback for non-reasoning models (reuse the existing plan path).

**Files:** new `pori/reasoning/`; edit `pori/llm/*.py`, `pori/metrics.py`,
`pori/agent.py`, `pori/run.py`.

**Tests:** `tests/test_reasoning.py` — reasoning tokens accounted separately;
steps captured; CoT fallback path with a non-reasoning `MockLLM`.

**Acceptance:** extended-thinking tokens appear in metrics; reasoning steps land in
`RunOutput`; non-reasoning models unaffected.

> If LLM-provider work feels risky, do the provider-test backfill (deferred earlier)
> first — it de-risks touching `pori/llm/*.py`.

---

## Phase 4 — Safety guardrails: PII + prompt-injection  (effort: S)  independent

**Why:** Pori has content/factuality/topic guardrails but no PII or prompt-injection
guard. Each is small and self-contained on the existing `BaseEval`/guardrail
interface. Can be done anytime (no dependency on Phases 0–3).

**Tasks**
1. `PIIGuardrail` in `pori/eval/guardrails.py` — regex for SSN/credit-card/email/
   phone; `mode: block | mask`.
2. `PromptInjectionGuardrail` — curated jailbreak-pattern list; configurable.
3. Wire both through `Agent(guardrails=[...])` (already supported).

**Files:** edit `pori/eval/guardrails.py`, `pori/eval/__init__.py`.

**Tests:** `tests/test_guardrails_safety.py` — PII detected/masked; injection
patterns caught; clean input passes.

**Acceptance:** both attachable like existing guardrails; no false-positive on
benign text in tests.

---

## Phase 5 — Knowledge/RAG upgrade  (effort: L)  optional

Only if RAG is on the roadmap. Pori already blends `0.75*semantic + 0.25*lexical`
(`memory.py`) — upgrade path:
1. Replace the linear blend with **Reciprocal Rank Fusion** for hybrid search.
2. Introduce a `Reranker` interface (sentence-transformers cross-encoder default).
3. Optionally separate a `Knowledge` abstraction (chunking + embedder + retriever)
   from raw archival memory; keep Chroma as the first backend.

**Acceptance:** retrieval quality measurably improves on a small eval set; existing
archival API preserved.

---

## Phase 6+ — Deferred (product-scope decisions, not quick wins)

- **AgentOS-style runtime**: scope-based RBAC + JWT, per-user multi-tenant isolation,
  cron scheduling, chat interfaces. Revisit when Pori Cloud needs it.
- **OpenTelemetry tracing**: export `Trace`/`Span` to OTel. Revisit when external
  APM integration is required.

---

## Suggested execution order

1. **Phase 0** (foundation) → 2. **Phase 4** (quick safety win, parallelizable) →
3. **Phase 1** (Workflow) → 4. **Phase 2** (sessions) → 5. **Phase 3** (reasoning) →
6. Phase 5 / 6+ as roadmap dictates.

Each phase: branch from `main`, keep CI green (black/isort/mypy-non-blocking/pytest),
update `.agent/progress/current.md`, and add the phase's tests before merging.
```
