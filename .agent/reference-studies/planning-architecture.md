# Reference Study: Planning Architecture for Pori

Source studied: `references/hermes-agent` (Hermes) and `references/claude-code`
(Claude Code public distribution), plus Pori's own `pori/agent.py`.

Status: **Phase 0 and Phase 1 implemented**; Phase 2 (plan-mode permission gate)
remains a proposal. See `pori/planning.py`, `pori/tools/standard/planning_tools.py`
(the `update_plan` tool), and `tests/test_planning.py`.

License note: per the reference rules we extract *principles and interfaces*,
not code. Nothing below is copied from Hermes or Claude Code source; the
skeletons are written against Pori's own APIs.

---

## 1. Why

Pori's `Agent` runs a **Plan → Act → Reflect → Evaluate** loop. Today the Plan and
Reflect stages are **framework-driven**: the runtime makes *separate* LLM calls to
generate and revise a plan, independent of the main reasoning call that actually
selects tools.

- `pori/agent.py:1338` `_plan_if_needed()` — one side LLM call, prompted from the
  task string.
- `pori/agent.py:1393` `_reflect_and_update_plan()` — another side LLM call, fed
  only the task, the current plan, and a bare `"success" / "fail: <err>"` summary
  of the last results (`agent.py:1396-1407`). It receives **no memory, no system
  prompt, and not the actual tool calls/outputs**.

The resulting plan is a `List[str]` with no status, injected into the step prompt
as advisory text (`agent.py:1051`) but never enforced.

### Observed failure (the motivating bug)

Task: *"write and run a program that list only vowels of my name."* The agent knew
the user's name from the core-memory `human` block and answered correctly
(`Aloy / Aloysius`). But the **plan** said:

```
- Ask the user for their name.
- Write a Python script that extracts only vowels from the provided name.
- Execute the script against the given name.
- Return the resulting list of vowels.
```

Two defects, both structural:

1. **Plans for already-known facts.** `_plan_if_needed`'s prompt hard-codes "if the
   task requires info the user has not provided, the FIRST step must be `ask_user`."
   It reasons off the *task string* ("my name") and never reconciles against memory,
   so it plans to ask for a name the agent already has.
2. **Plan decoupled from reality.** The plan was *re-generated after* a failed
   `answer` call, with no record of the work already done — so it described steps 1–2
   (already completed) as still pending, and the model ignored it entirely.

These are not prompt bugs to patch; they follow directly from generating plans in a
side call that is blind to memory and execution state.

---

## 2. How Hermes and Claude Code plan

Both systems converge on the same architecture: **planning is a tool the main model
calls, not a framework-side LLM call.** The model plans with the exact context it
acts with; the framework only persists and re-injects the list so it stays truthful.

### Hermes — `todo` tool (`references/hermes-agent/tools/todo_tool.py`)

- A single stateful tool `todo(todos?, merge?)`: read when args are omitted, write
  otherwise; returns the full list + summary every call.
- Per-session `TodoStore`: an ordered list (position = priority) of flat items
  `{id, content, status}` with `status ∈ {pending, in_progress, completed,
  cancelled}` (`todo_tool.py:22`). Bounded (`MAX_TODO_ITEMS = 256`).
- Write modes: `merge=False` replaces the whole list; `merge=True` updates by `id`
  and appends new items.
- **All behavioral guidance lives in the tool schema description** (cacheable,
  static): "use for 3+ step tasks; list order is priority; only ONE item
  in_progress at a time; mark completed immediately when done; if something fails,
  cancel it and add a revised item."
- Advisory and **model-owned** — nothing gates execution on the plan. The framework's
  only active roles: (a) re-inject **only pending/in_progress** items after context
  compression so finished work isn't redone (`conversation_compression.py:501`), and
  (b) hydrate the store from history when a fresh agent instance starts
  (`run_agent.py:3253`). Kept **separate from long-term memory** (its own `memory`
  tool).

### Claude Code — `TodoWrite` tool + plan mode (`references/claude-code`)

- `TodoWrite` is a model-callable tool (granted per-agent like any other —
  `plugins/feature-dev/agents/code-architect.md:4`). The model calls it with the
  **entire** todo array each time (full-state replace). Items
  `{id, content, activeForm, status: pending|in_progress|completed}`, exactly one
  `in_progress`.
- The list is **session state, re-injected on every change** and specially
  preserved across compaction (`CHANGELOG.md:4311`, `:2788`); surfaced to the user
  as a live checklist and via `/todos` (`:3943`).
- **Plan mode** is a separate, optional feature: a *permission mode* in which the
  permission layer **blocks all mutating tools** regardless of allow rules
  (`CHANGELOG.md:969`). The model writes a plan and calls the **`ExitPlanMode`**
  tool (`:1043`), which triggers a **human approval gate** (`:3451`, `:2269`); on
  approval the mode flips to `default`/`acceptEdits` and execution proceeds.
- Planning is **model-driven**; the framework authors no plan — it only enforces
  permissions and the approval UI.

### The one architectural sentence

> Planning = a tool the main model calls + (optionally) a permission gate — **not** a
> separate orchestration LLM that emits a plan the executor follows.

---

## 3. Recommended architecture for Pori

Fold Plan and Reflect into the **main reasoning loop** via a model-driven plan tool.
The model plans with the same context it acts with (which already includes core
memory), and owns its own updates.

### Phase 1 — model-driven `update_plan` tool (implemented)

Shipped as: `PlanStore`/`PlanItem` (`pori/planning.py`), the always-available
`update_plan` kernel tool (`pori/tools/standard/planning_tools.py`), wired into the
agent's tool context and step prompt; `planning_mode`/`reflection_mode` now default
to `"never"` (the legacy side LLM calls remain opt-in via `"auto"`/`"always"`).

Original design notes:

- New stateful tool registered in `pori/tools/standard/`:
  `update_plan(todos, merge=False)` backed by a per-run `PlanStore` (mirrors the
  existing per-run receipt/skill state on `Agent`).
- Item shape: `{id: str, content: str, status: pending|in_progress|completed|cancelled}`.
  Replace vs merge-by-id. Bounded count/length. **One `in_progress` at a time.**
- **All guidance in the tool's schema description** (Hermes pattern), so it is part
  of the cached tool surface and needs no system-prompt mutation: when to plan (3+
  steps), one in_progress, mark done immediately, cancel + add revised item on
  failure, do not plan to gather facts already present in context/memory.
- Retire the two side LLM calls: delete/disable `_plan_if_needed`
  (`agent.py:1338`) and `_reflect_and_update_plan` (`agent.py:1393`). Keep the
  "Current Plan" block in the step prompt (`agent.py:1051`) but source it from
  `PlanStore` and render **only pending/in_progress** items.
- Keep the plan **advisory and separate from long-term memory**, consistent with
  Pori's existing `AgentMemory` boundary.

Net effect: removes 2 LLM calls per planned task; the plan is built and revised by
the model that actually acts, so it cannot ask for known facts or describe completed
work. This directly fixes both observed defects.

### Phase 2 — plan mode as a permission mode (high synergy)

Reuse the machinery already merged in `feat/tool-authorization-policy`:

- A `plan` run mode in which `ToolAuthorizationPolicy` denies every tool carrying
  `SideEffect.FILESYSTEM_WRITE` (and future side-effect classes) until the model
  calls a new `exit_plan_mode` tool and the user approves via the existing **HITL**
  handler.
- On approval, flip the mode and proceed. This is Claude Code's plan-mode mapped
  onto Pori's `SideEffect` taxonomy + `HITLConfig` — no new enforcement layer
  needed.

Phase 2 is optional and gated behind config; Phase 1 is the structural fix.

---

## 4. Mapping to existing Pori code

| Concern | Today | After Phase 1 |
| --- | --- | --- |
| Plan generation | `_plan_if_needed` side LLM call | model calls `update_plan` |
| Plan revision | `_reflect_and_update_plan` side LLM call (context-starved) | model calls `update_plan(merge=True)` with full context |
| Plan item | `List[str]` (no status) | `{id, content, status}` in `PlanStore` |
| Plan in prompt | `agent.py:1051`, all steps | sourced from `PlanStore`, pending/in_progress only |
| Behavior guidance | planner prompt | tool schema description |
| Enforcement | none | none (Phase 1) / permission gate (Phase 2) |

### Non-goals

- Do not make the plan gate ordinary execution in Phase 1 (keep it advisory, like
  both references).
- Do not store plan progress in long-term memory.
- Do not transplant a DAG/dependency model — a flat priority-ordered list is what
  both references use and is sufficient.

---

## 5. Phase 0 (shipped with this doc) — answer artifact-reference crash

Orthogonal to the planning redesign but discovered in the same trace: the `answer`
tool's `AnswerArtifactReference.path` was a *required* field, while the runtime
validator `_artifact_reference_errors` (`agent.py`) accepts **path or receipt_id**.
When the model referenced an artifact by `receipt_id` only, Pydantic raised a hard
`ValidationError`, crashing the `answer` call and triggering a spurious re-plan.

Fix: make `path` optional (and accept an optional `description`) in
`pori/tools/standard/core_tools.py`, aligning the schema with the "path or
receipt_id" contract. A genuinely empty reference now becomes a graceful,
model-correctable rejection instead of a crash.

---

## 6. Verification plan

- Phase 0: unit tests in `tests/test_agent_completion.py` — `answer` with
  receipt_id-only reference is accepted (receipt exists); reference with neither
  path nor receipt_id is rejected gracefully (no `ValidationError`).
- Phase 1: tests that a task referencing a fact present in core memory yields a plan
  that does **not** call `ask_user`; that completed items are not re-injected; that
  `update_plan` enforces a single `in_progress`.
- Standard gates: `uv run pytest`, `black`, `isort`, `mypy`.
