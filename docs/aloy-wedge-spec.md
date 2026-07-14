# Aloy Wedge — Implementation Spec (V1)

_Status: **DRAFT for build**, 2026-07-14. Builds directly on
`docs/aloy-vision.md` (canonical product definition) — read that first; this
spec turns its §6 wedge into a buildable plan. The four foundational decisions
below are **settled** (founder ideation, closed 2026-07-14) and are premises,
not open questions. Related: `CLAUDE.md` (kernel map),
`aloy_backend_api.md` (backend architecture), `docs/adr/` (ADRs)._

**How to read this if you are a coding agent:** §0 is the settled premises —
do not relitigate them. §1 is the data model (the tables you create/alter).
§2 is the migration. §3 is the Proposal system — the deepest part; read it
twice. §4–5 are the visible surfaces. §6 is memory. §7 is what NOT to build.
§8 is the build order + acceptance gates. Use the vision's primitive names
exactly (Event, Session, Proposal, Task, Surface, Trail).

---

## 0. Scope and the settled premises

The wedge proves the **Event loop end-to-end on one thin slice**: a Life Event
+ one **Project** Event, with a templated Surface, the Agent Trail, the
Proposal system (async object + sync fast-path), Tasks, and the Today lens.

The four locked decisions (from `aloy.txt`, do not reopen):

1. **Event ↔ Session ↔ Conversation.** A Conversation *becomes* a **Session**;
   a Session belongs to exactly one **Event**; there is always a default
   **Life** Event. All durable state (memory, files) belongs to the **Event**.
   The Session is a **stateless lens** — a grouping of messages, nothing more.
2. **Proposals are persistent async objects.** A gated tool call is **never
   executed inline** — it is intercepted, converted into a Proposal (storing
   the exact `tool + args`), and executed **later by a dumb executor** on
   approval. The synchronous HITL flow we already shipped becomes a *fast-path
   view* over this object, not a separate mechanism.
3. **The wedge Event type is Project** (dogfood: "Building Aloy"). Its Surface
   is **templated** (Status · Tasks · Activity · Notes), not model-composed.
4. **Cuts:** no Reality Objects, no auto/emergent event detection, no
   model-composed UI, no cross-event memory retrieval. Keep: manual event
   creation, Today, event Surfaces, the Proposal system.

**Invariant that governs the whole spec (vision §4.1, §3.8):**
> No agent reality change without a **committed** Proposal. If anything ever
> executes a gated action before a Proposal is persisted and committed, the
> system is broken.

---

## 1. Data model

New tables and column additions. All are org-scoped multi-tenant like the rest
of the backend (`organization_id` on every row; personal accounts are
`user:<uuid>` orgs — see `aloy_backend_api.md`). Timestamps are tz-aware UTC.

### 1.1 `events` (new — the aggregate root)

| column | type | notes |
|---|---|---|
| `id` | str pk | `evt_<uuid>` |
| `organization_id` | str, index | tenant boundary |
| `user_id` | str, index | owner |
| `type` | str | `"life"` \| `"project"` (wedge set; open vocabulary later) |
| `title` | str | "Life", "Building Aloy" |
| `status` | str | Machine 2 lifecycle: `emerging`\|`active`\|`dormant`\|`concluded`\|`archived`. Wedge uses `active`. |
| `phase` | str | free, model-defined; empty in wedge |
| `summary` | str | model-maintained one-liner (Surface header) |
| `is_life` | bool | exactly one `true` per user; the default root |
| `metadata` | JSON | extensibility |
| `created_at`/`updated_at` | datetime | |

Rules: **exactly one `is_life=true` Event per user**, created lazily on first
use (mirrors `ensure_personal_organization`). Event creation is a **direct
user action, never a Proposal** (vision §3.1, invariant #1).

### 1.2 Session = the existing `conversations` table + `event_id`

**Decision (pragmatic, stated so no one "fixes" it):** we do **not** physically
rename the `conversations` table. Renaming a table that `messages`, `runs`,
`stored_files`, memory, and the sandbox all key into is a high-blast-radius
change for zero functional gain. Instead:

- **Add `event_id` (FK → `events.id`, NOT NULL, index)** to `conversations`.
- The **domain term is Session**; the API and UI say "session"/"event"; the
  physical table stays `conversations`. A Session *is* a conversation row.
- `messages` and `runs` keep `conversation_id` (= session id). They reach the
  Event via `conversation.event_id` — retrieval joins through it.

### 1.3 `proposals` (new — the execution primitive)

| column | type | notes |
|---|---|---|
| `id` | str pk | `prop_<uuid>` |
| `organization_id`/`user_id` | str, index | tenant + owner |
| `event_id` | str, index | the Event this acts on |
| `origin_session_id` | str \| null | session that produced it |
| `origin_run_id` | str \| null | run that produced it |
| `tool` | str | **exact tool name**, e.g. `gmail_send` |
| `args` | JSON | **exact validated args** — the executable payload |
| `reason` | str | agent's justification (for the card + Trail) |
| `impact` | str | what changes if committed |
| `risk` | str | `low`\|`medium`\|`high` (drives routing) |
| `routing` | str | `auto`\|`notify`\|`ask` (wedge: fixed policy → always `ask` for the write set) |
| `status` | str | state machine below |
| `expires_at` | datetime \| null | Ask expiry (vision §3.6) |
| `safe_default` | JSON \| null | action on expiry (`{decision: "reject"}` in wedge) |
| `decided_by`/`decided_at` | str/datetime \| null | who resolved the Ask |
| `receipt` | JSON \| null | `ToolExecutionReceipt` on commit (Verifiable Reality) |
| `error` | str \| null | on failure |
| `created_at`/`updated_at` | datetime | |

### 1.4 `tasks` (new — event-owned Working state, NOT gated)

| column | type | notes |
|---|---|---|
| `id` | str pk | `task_<uuid>` |
| `organization_id`/`user_id`/`event_id` | str, index | |
| `title` | str | |
| `status` | str | `open`\|`done` (wedge); |
| `order` | int | manual/agent ordering |
| `created_by` | str | `"user"` or agent id |
| `created_at`/`updated_at` | datetime | |

Creating/completing a Task is **Working state** — reversible, internal, cheap —
so it does **not** go through the Proposal system (vision §3.7, §3.8). Both the
user (UI) and the agent (a `task_create`/`task_update` tool) mutate tasks
directly.

---

## 2. Migration plan (conversation → Event/Session)

Additive and reversible where possible. One Alembic migration + a data
backfill. **No message/run/file rows move** — only new keys are added.

1. **Create** `events`, `proposals`, `tasks` tables.
2. **Add** `event_id` to `conversations`, plus `event_id` to `stored_files` and
   the memory tables (`knowledge_entries`, core-memory as applicable).
3. **Backfill Life:** for each user, create their `is_life=true` Event.
4. **Assign every existing conversation to that user's Life Event**
   (`conversations.event_id = life.id`). Every past chat becomes a Life
   Session; nothing is lost, nothing dangles.
5. **Re-key memory + files to Event:** set `stored_files.event_id` and
   knowledge-entry `event_id` from their conversation's new `event_id`. Global
   (user-level) memory keeps `event_id = null` (see §6).
6. **Sandbox re-key.** Path changes from `sandbox/threads/{conversation_id}/`
   to **`sandbox/events/{event_id}/`** for the persistent event workspace, plus
   **`sandbox/events/{event_id}/runs/{run_id}/`** for per-run scratch (so
   concurrent Sessions in one Event can't collide). Migration moves existing
   `threads/{cid}/` dirs under the conversation's Life Event workspace, best-
   effort (files are also durable in object storage; the sandbox is a cache).
7. **`conversation_search` → event history.** The tool (kernel
   `core_tools.py`) searches the messages loaded into the run's `AgentMemory`.
   The change is **what a run loads**: the run loads its **Event's** recent
   messages (across all the Event's Sessions), windowed to the context budget —
   not one Session's. The tool then searches/pages the Event's full history and
   becomes the **context page-fault** (vision §5). Rename to
   `search_event_history` (alias the old name one release for safety).

**Rollback:** drop the three tables + the added columns; conversations behave
as before (event_id ignored). Because no rows moved, rollback is clean.

---

## 3. The Proposal system (the deep part)

### 3.1 The contract

```
agent calls a gated tool (e.g. gmail_send)
        │
        ▼
INTERCEPT — do NOT execute. Create a Proposal {tool, args, status: proposed}.
        │
        ▼
Agent receives: {"status": "staged", "proposal_id": "prop_…"}   ← not the tool result
        │
        ├── Path A (user present): approval card shown now; quick approve →
        │                          EXECUTOR runs it, receipt written, Surface updates.
        └── Path B (no user):     Proposal waits in the inbox; run ENDS;
                                   later the user approves → EXECUTOR runs it.
```

Execution is **always** by a **dumb executor** — `run(tool, args)` → store
receipt → mark committed. **No reasoning, no re-interpretation, no agent loop
at execution time.** The Proposal is *serialized intent + executable payload*;
that is what makes async, safe execution, and replay/audit possible.

### 3.2 State machine (reconciled to vision §3.7)

```
Working → Proposed → Routed ─┬─ auto ──────────────► Executing → Committed(+Receipt)
                             ├─ notify ─(tell)─────► Executing → Committed(+Receipt)
                             └─ ask ─► Pending ─┬─ approved ─► Executing → Committed(+Receipt)
                                                ├─ rejected ─► Withdrawn
                                                └─ expired ──► apply safe_default
                                        Executing ─(on tool error)─► Failed
```

Wedge simplification: the write set routes **`ask`** (fixed policy — external
messages always Ask, vision §3.6); `auto`/`notify` exist in the model but are
not exercised until learned routing (deferred). `safe_default` in the wedge is
`{decision: "reject"}` (never send on timeout — matches the guardrail we
shipped).

### 3.3 The interception seam (kernel touch-point)

Recommended, minimal kernel change: **add a `defer` decision** to the HITL
vocabulary (`approve | edit | reject | defer`). The product's approval handler
(evolve `aloy_backend/approvals.py::ApprovalBridge`):

1. On a gated call, **persists a Proposal row** (`status=proposed→routed→
   pending`), then returns `Decision(type="defer", proposal_id=…)`.
2. The kernel gate, on `defer`, **skips execution** and returns
   `{"status": "staged", "proposal_id": …}` as the tool result (a new branch
   next to the existing reject/edit handling in `pori/agent/dispatch.py`).

This preserves the kernel's product-agnosticism (it knows "defer", not
"Proposal") and reuses everything we built. The **sync fast-path**: the same
Proposal is surfaced immediately over SSE (the existing `approval_request`
frame, now carrying `proposal_id`); a quick approve calls the executor. If no
answer before the run's window closes, the run ends and the Proposal persists
in the inbox — the interactive and background paths are **one object**.

### 3.4 The executor (new, product-side)

A standalone component — **not** an agent run:

```
execute_proposal(proposal_id):
    load Proposal (must be status=approved/routed-auto)
    resolve tool context (connections for user+event) via resolve_run_surface
    result = ToolExecutor.execute(proposal.tool, proposal.args, context)   # single call
    write receipt (ToolExecutionReceipt) → proposal.receipt
    status = committed (or failed + error)
    emit a Trail entry + Surface refresh
```

Reuses: `ToolExecutor` (kernel), `resolve_run_surface` (connections/token),
`ToolExecutionReceipt` (Verifiable Reality). Triggered by (a) the approve
endpoint (fast-path) and (b) a worker tick over `status=approved` proposals
(background path). Idempotent by `proposal_id` (mirror the single-finalizer
discipline, ADR 0007) — an approve + a worker tick must never double-execute.

### 3.5 Endpoints

- `POST /v1/events` — create an Event (direct action, no Proposal).
- `GET /v1/events` / `GET /v1/events/{id}` — list / read (+ Surface payload).
- `POST /v1/events/{id}/proposals/{pid}/decision` — `{approve|reject|edit}` →
  routes to the executor on approve. (The existing
  `/conversations/approve/{id}` becomes the fast-path alias.)
- `GET /v1/today` — the Today aggregation (§5).
- `POST /v1/events/{id}/tasks`, `PATCH …/tasks/{tid}` — Task CRUD (ungated).

### 3.6 Safety tests (acceptance gates for §3)

- **No inline execution:** a gated tool call in a run produces a Proposal row
  and a `staged` result, and the external side effect has **not** happened
  until a commit. (Assert the tool's real call count is 0 pre-approval.)
- **Idempotent commit:** approve + worker tick → exactly one execution, one
  receipt.
- **Expiry → safe default:** an unanswered Ask past `expires_at` applies
  `safe_default` (reject) and never executes.

---

## 4. The Project Event Surface (templated)

A **schema payload** (typed JSON, vision §3.3 / ADR 0010) the app renders with
trusted components — for the wedge it is **hand-authored per the Project type**,
not model-composed. `GET /v1/events/{id}` returns:

```jsonc
{
  "event": { "id", "title", "status", "summary" },
  "surface": {
    "type": "project",
    "sections": [
      { "kind": "status",   "summary": "…", "phase": "…" },
      { "kind": "tasks",    "tasks": [ {id,title,status,order} ] },      // interactive
      { "kind": "activity", "entries": [ …Trail/receipt entries… ] },   // read-only
      { "kind": "notes",    "notes": "…" }                              // event-scoped memory
    ],
    "proposals": [ …pending proposals for this event… ]                 // approval cards inline
  }
}
```

Rendering rules (Verifiable Reality, invariant #2): the Surface shows
**committed state + relevant working state + pending Proposals** — **never an
unsupported claim**. The `activity` section renders **receipts/Trail entries**,
not agent prose. Tasks are directly interactive (toggle done, add) and mutate
Working state without Proposals. The Surface is **recomputed from Event state on
read** (no separate stored surface artifact in V1).

---

## 5. The Today view (aggregation)

`GET /v1/today` — a lens across the user's Event graph (vision §3.3):

- **Needs a decision:** all `status=pending` Proposals across the user's Events
  (the attention inbox — falls straight out of the `proposals` table).
- **Changed:** recently `committed` Proposals + recent Trail entries (last 24h).
- **Upcoming:** open Tasks flagged/soon (wedge: just open tasks; time-based
  triggers later).
- Grouped by Event; Life first. Home screen is this lens, not any one Surface.

Purely a query over existing tables — no new state.

## 6. Memory model

Three layers, event-primary retrieval (vision §5, `aloy.txt`):

- **Global (user):** preferences, identity — `MemoryScope` with `event_id=null`.
  Small; always loaded (like CoreMemory).
- **Event:** everything relevant to the Event — `event_id` set. Primary scope.
- **Session:** **none.** Sessions carry no durable memory (decision 1).

`MemoryScope` (kernel `memory_contracts.py`) gains **`event_id`** alongside
`organization_id/user_id/agent_id`; drop reliance on `session_id` for durable
memory. **Retrieval for a run in Event E:** load Global + Event-E memory +
Event-E recent messages (windowed); `search_event_history` pages older Event
history on demand. **Cross-event retrieval is deferred** (§7) — with Life + one
Project there is nothing to page across.

## 7. Non-goals (explicitly deferred — do not build)

- **Reality Objects** (Calendar/Money/People). The Project Event is entirely
  event-local (tasks + notes + trail); it references no shared objects. Zero
  Reality-Object infra in the wedge.
- **Auto / emergent event detection** (soft clusters, worthiness, "Aloy
  proposes an Event"). Manual creation only.
- **Model-composed / free-form Surfaces.** Templated Project Surface only.
- **Cross-event memory retrieval** and the Life graph-coordinator.
- **Learned routing** (`auto`/`notify` earned autonomy) — the model exists;
  the wedge hard-codes `ask` for the write set.
- **Time/incoming-data triggers** beyond manual + the existing cron chassis.

## 8. Build sequence and acceptance gates

Each phase is independently shippable and CI-green; later phases don't start
until the earlier gate passes.

1. **Data model + migration** (§1–2). Gate: migration runs; every existing
   conversation has an `event_id` = its user's Life Event; a
   **leakage=0 / recall=1** scoping test — an agent scoped to Event A loads
   A's + global memory and **never** Event B's — passes. (This is the
   "is memory good enough" gate.)
2. **Proposal object + executor + interception** (§3), reusing the HITL bridge
   as the fast-path. Gate: the three §3.6 safety tests pass; a live
   `gmail_send` stages → approve → executor sends → receipt, with **zero**
   inline execution.
3. **Tasks + Project Surface + Event/Today endpoints** (§1.4, §4, §5). Gate:
   create a Project Event, see its Surface, add/complete tasks, and Today lists
   its pending proposals.
4. **The loop, end-to-end** (hero flow, vision §6): in the "Building Aloy"
   Event — agent works → Trail explains → Proposal appears → user decides →
   reality changes (email sent, receipt) → Surface + Today update. Gate: the
   60-second demo runs on the founder's own account.

## 9. What's reused vs new (map)

| Piece | Status |
|---|---|
| Agent loop, tools, streaming, sandbox, run-surface, finalizer | **Reused** |
| HITL approval bridge (`approvals.py`) | **Reused** as the Proposal fast-path |
| Receipts / traces / run-event-log | **Reused** as the Trail + Surface `activity` |
| `conversation_search` | **Reused**, re-scoped to the Event |
| `events` / `proposals` / `tasks` tables | **New** |
| `event_id` on conversations/files/memory + sandbox re-key | **New (migration)** |
| Proposal executor + `defer` kernel decision + interception | **New** |
| Project Surface schema + Today aggregation | **New** |

---

_Open the build against §8 phase 1. The founder decides the "Building Aloy"
Event's exact task seed and Surface copy at phase 3._
