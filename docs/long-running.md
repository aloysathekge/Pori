# Long-Running Tasks — Hermes Harvest Plan

_2026-07-06. Source: two deep audits — Hermes marathon mechanisms (references/hermes-agent)
crossed against Pori's current state. Pattern-harvest only (kernel never pastes Hermes code)._

**Status (2026-07-06, same session):** Phases 1–3 are implemented across three
stacked PRs (#95 Phase 1, #96 Phase 2, Phase 3 follows #96). Implemented:
write-ahead tool journal, per-step checkpoint, `Agent(resume_task_id=…)`,
salvage summary, compression-on-by-default, sqlite config default, wall-clock
budget, worker resume-not-restart with checkpoint-as-heartbeat lease renewal,
docker-compose worker service, cron engine (at-most-once → run queue) with
CRUD routes, delivery into conversations via the existing Message path.
**Still open:** wiring the kernel's `delegate_task(background=true)` to enqueue
backend child Runs (API + worker support exist; the tool→queue bridge needs its
own focused pass), and team-run checkpointing.

## The verdict in one line

Pori is resilient **within** a process (LLM retry+jitter, layered loop-detection
guardrails, AC-3 compression) but **nothing about a run survives a restart**: loop
counters, the plan, and partial progress are all in-memory; even the Aloy durable
worker re-runs a crashed task from scratch, and partial work persists only if a run
fully completes. Hermes's marathon superpower is the opposite discipline:
**write-ahead everything, reconcile on boot, never hard-delete, bound every horizon.**

## Pori's five structural gaps (from the kernel audit)

1. **No persisted loop state → no resume.** `AgentState` (n_steps, failures, plan)
   and `PlanStore` are in-memory only (`pori/agent/schemas.py:8-20`,
   `pori/planning.py:1-7`); every `Agent.__init__` mints a new task_id, so recovery
   is always restart-from-scratch.
2. **Partial progress is completion-gated.** Aloy flushes conversation/artifacts only
   on `status == "completed"` (`background.py:227-266`); a timeout discards hours of
   work. Kernel default memory backend is in-memory (`PORI_MEMORY_BACKEND=memory`).
3. **Hard, ungraceful stops.** At `max_steps` the run returns `completed=False` with
   nothing delivered (`core.py:1556`). `ExecutionBudget.max_duration_seconds` exists
   but is consumed nowhere.
4. **Background work dies with the process.** Daemon-thread delegation
   (`background_delegation.py:54`), no scheduler/cron anywhere.
5. **Learning compounds knowledge, not execution.** Experiences/skills make the
   *next* task smarter; nothing makes the *current* task recoverable.

## The harvest roadmap

### Phase 1 — Never lose work (kernel; small, high leverage)

| # | Borrow (Hermes mechanism) | Lands in Pori as |
|---|---|---|
| 1.1 | **Forced summary on budget exhaustion** (`turn_finalizer`: one tools-stripped call when iterations run out) | At `max_steps`/failure-stop, one final tools-stripped LLM call that synthesizes a best-effort answer + handoff. EASY, transforms marathon UX. |
| 1.2 | **Persist the loop skeleton** (Hermes stamps session state to SQLite) | Add `AgentState` + `PlanStore` + step counter to the `AgentMemory` snapshot (persist hooks already exist — `memory.py:489`). Accept an existing `task_id` in `Agent.__init__` for continuity. |
| 1.3 | **Write-ahead the tool-call turn** (`conversation_loop.py:4303` persists the assistant tool-call message *before* executing tools) | Guarantee the same ordering in `core.py` so a mid-tool crash leaves an accurate journal. |
| 1.4 | **Plan survives compression** (todo re-injected post-compaction, completed items dropped — `conversation_compression.py:544`) | Re-inject the `update_plan` plan (pending/in-progress only) after AC-3 compaction; and **turn `compress_context` on by default**. |
| 1.5 | SQLite as default memory backend for anything non-ephemeral (CLI + Aloy). | Config default flip + docs. |

### Phase 2 — Resume, don't restart (kernel + Aloy backend)

| # | Borrow | Lands as |
|---|---|---|
| 2.1 | **resume_pending marker + boot reconciliation** (`gateway/run.py:5809`: durable flag distinct from transcript; startup pass dispatches synthetic resume turns; flag cleared only on success) | Aloy: on worker re-claim, if the run has persisted loop state → `Agent.resume(task_id)` re-enters the loop at the stored step with the stored plan, instead of `execute_task` from zero. |
| 2.2 | **Incremental flush** (Hermes persists every turn) | Aloy: flush messages/artifacts per-step (or per-N-steps), not completion-gated. A timeout keeps everything up to the last step. |
| 2.3 | **Heartbeat staleness over wall-clock kills** (`delegate_tool.py:603`: 30s heartbeats; "stale" = 450s idle between turns or 1200s inside one tool — distinguishes slow-API from stuck) | Replace/augment `asyncio.wait_for(timeout_seconds)` for long runs: heartbeat from the step loop; the worker lease renews while heartbeats are fresh; kill only on staleness. Also wire `max_duration_seconds` as a real (generous) ceiling. |
| 2.4 | **Iteration budget with refund** (`iteration_budget.py`: parent 90, each child independent 50; programmatic sub-calls refunded) | Extend `BudgetLedger`; per-child budgets in `subagents.py`. |

### Phase 3 — Marathon infrastructure (mostly Aloy / extension)

| # | Borrow | Lands as |
|---|---|---|
| 3.1 | **Cron engine** (JSON job store + advisory file lock; advance `next_run_at` for all due jobs under the lock *before* executing any — at-most-once; claim-TTL for multi-replica; heartbeat for status) | Aloy backend scheduler service feeding the existing run queue (the Cron control screen is already on the product roadmap). Kernel stays cron-free. |
| 3.2 | **Durable background delegation** | Replace daemon-thread background children with enqueued runs on Aloy's existing lease-based worker — persistence and retry come free. Kernel keeps thread mode for CLI/local. |
| 3.3 | **Results re-enter as deliverables** (DeliveryRouter; a tool may only promise async delivery if the channel supports it) | PoriEvent notification on background/cron completion; app surfaces it. |
| 3.4 | **File-state discipline** (atomic temp+rename, boot reconciliation, epoch-stamped markers) | Adopt as the standing pattern for any new kernel state file. |

### Parked (good, not now)

Dual-flag transcript rewind (active/compacted soft-archive), git filesystem
checkpoints, scale-to-zero hibernation, full ContextCompressor 5-phase port (AC-3
already covers the core; borrow its *ideas*: prune old tool results first, protect
head/tail by token budget, fold previous summaries, abort-don't-destroy on aux failure).

## Sequencing note

Phase 1 is almost entirely kernel and mostly EASY-rated; it makes every surface
(CLI, Aloy, future products) lose-nothing by default. Phase 2 is the true
"world-class" line — resume-not-restart — and its Aloy half depends on the durable
worker actually being deployed (see the 2026-07-06 audit: the worker service is
missing from docker-compose). Phase 3 rides on Phase 2's machinery.
