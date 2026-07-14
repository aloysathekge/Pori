# Aloy Wedge — Implementation Spec (V1)

_Status: **DRAFT for build (architecture-hardened)**, 2026-07-14. Builds directly on
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
   **Life** Event. All durable state (memory, files, artifacts, Tasks,
   Proposals, and Trail) belongs to the **Event**. The Session is a
   **stateless lens** — a grouping of messages, nothing more. A Session may
   introduce a file and a Run may create one, but those links are provenance,
   not ownership.
2. **Proposals are persistent async objects.** A gated tool call is **never
   executed inline** — it is intercepted, converted into a Proposal (storing
   the exact tool plus normalized, validated args), and executed **later by a
   dumb executor** on approval. The synchronous HITL flow we already shipped
   becomes a *fast-path view* over this object, not a separate mechanism.
3. **The wedge Event type is Project** (dogfood: "Building Aloy"). Its Surface
   is **templated** (Status · Tasks · Activity · Notes), not model-composed.
4. **Cuts:** no Reality Objects, no auto/emergent event detection, no
   model-composed UI, no cross-event memory retrieval. Keep: manual event
   creation, Today, event Surfaces, the Proposal system.

**Invariant that governs the whole spec (vision §4.1, §3.8):**
> No agent reality change without a **durably persisted and authorized**
> Proposal. A gated action may execute only after the Proposal reaches
> `Approved`; it reaches `Committed` only after execution produces a receipt.
> If a gated tool runs before durable approval, or the product represents it
> as committed without evidence, the system is broken.

### 0.1 Identity and ownership (do not collapse these)

Four identities coexist deliberately:

| identity | meaning | owns |
|---|---|---|
| `event_id` | durable product aggregate | memory, Tasks, Proposals, Trail, durable files/artifacts |
| `session_id` | one conversational lens (physical `conversation.id`) | messages only |
| `run_id` | one agent execution | execution record, trace, receipts, temporary outputs |
| `workspace_id` | persistent sandbox lane | equals `event_id` in the wedge |

Do **not** redefine `session_id` to mean Event. Existing resume, live-stream,
branching, and conversation semantics remain session-scoped. Event scope is
added orthogonally. Run scratch is per-run; persistent workspace state is
per-Event.

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
| `lifecycle` | str | Machine 2: `emerging`\|`active`\|`dormant`\|`concluded`\|`archived`. Wedge uses `active`. |
| `phase` | str | free, model-defined; empty in wedge |
| `summary` | str | model-maintained one-liner (Surface header) |
| `is_life` | bool | exactly one `true` per organization + user; the default root |
| `metadata` | JSON | extensibility |
| `created_at`/`updated_at` | datetime | |

Rules: **exactly one `is_life=true` Event per organization + user**, created
lazily on first use (mirrors `ensure_personal_organization`). Enforce this
with a database partial unique index over `(organization_id, user_id)` where
`is_life=true`; application checks alone are insufficient under concurrent
first requests. Event creation is a **direct user action, never a Proposal**
(vision §3.1, invariant #1).

### 1.2 Session = the existing `conversations` table + `event_id`

**Decision (pragmatic, stated so no one "fixes" it):** we do **not** physically
rename the `conversations` table. Renaming a table that `messages`, `runs`,
`stored_files`, memory, and the sandbox all key into is a high-blast-radius
change for zero functional gain. Instead:

- **Add `event_id` (FK → `events.id`, NOT NULL, index)** to `conversations`.
- The **domain term is Session**; the API and UI say "session"/"event"; the
  physical table stays `conversations`. A Session *is* a conversation row.
- `messages` keep `conversation_id` (= session id) and reach the Event through
  the Session.
- `runs` keep `conversation_id` for provenance **and gain direct `event_id`**.
  Event agents/background work need an Event identity even when no Session is
  present. `run_event_logs`, traces, and context artifacts should likewise
  carry `event_id` when they are queried as Event activity.
- A branched Session inherits its parent's `event_id`; gateway-created and
  cron-created Sessions are assigned explicitly to Life or a chosen Event.

### 1.3 `proposals` (new — the execution primitive)

| column | type | notes |
|---|---|---|
| `id` | str pk | `prop_<uuid>` |
| `organization_id`/`user_id` | str, index | tenant + owner |
| `event_id` | str, index | the Event this acts on |
| `origin_session_id` | str \| null | session that produced it |
| `origin_run_id` | str \| null | run that produced it |
| `tool` | str | **exact tool name**, e.g. `gmail_send` |
| `args` | JSON | **normalized, schema-validated args** — the executable payload |
| `tool_schema_fingerprint` | str | fingerprint captured at staging; incompatible deployments refuse execution |
| `reason` | str | agent's justification (for the card + Trail) |
| `impact` | str | what changes if committed |
| `risk` | str | `low`\|`medium`\|`high` (drives routing) |
| `routing` | str | `auto`\|`notify`\|`ask` (wedge: fixed policy → always `ask` for the write set) |
| `status` | str | `proposed`\|`routed`\|`pending`\|`approved`\|`executing`\|`committed`\|`withdrawn`\|`failed`\|`indeterminate` |
| `expires_at` | datetime \| null | Ask expiry (vision §3.6) |
| `safe_default` | JSON \| null | action on expiry (`{decision: "reject"}` in wedge) |
| `decided_by`/`decided_at` | str/datetime \| null | who resolved the Ask |
| `receipt` | JSON \| null | `ToolExecutionReceipt` on commit (Verifiable Reality) |
| `execution_attempt_id` | str \| null | stable idempotency/reconciliation key for the claimed attempt |
| `provider_operation_id` | str \| null | provider-side message/event/object id when available |
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
directly. Every mutation appends a Trail entry in the **same transaction**.

### 1.5 `event_trail_entries` (new — append-only activity truth)

The Trail is a constitutional primitive, not a UI projection over convenient
logs. Receipts, traces, and run-event logs feed it but do not replace it.

| column | type | notes |
|---|---|---|
| `id` | str pk | `trail_<uuid>` |
| `organization_id`/`user_id`/`event_id` | str, index | tenant + Event boundary |
| `actor_id` | str | user, agent, worker, or system actor |
| `kind` | str | `run_progress`\|`task_changed`\|`proposal_staged`\|`proposal_decided`\|`proposal_committed`\|`artifact_added` |
| `summary` | str | short human-readable narrative |
| `run_id`/`proposal_id`/`task_id` | str \| null | typed provenance links |
| `evidence_refs` | JSON | receipt/artifact/trace references backing the entry |
| `payload` | JSON | typed detail for rendering/debugging |
| `created_at` | datetime | append time; entries are never updated in place |

All Event state mutations append their corresponding Trail entry atomically.
This makes “complete by construction” testable and lets Today query activity
without scanning JSON run logs.

### 1.6 Event files and artifacts (`stored_files` evolution)

Durable bytes belong to the Event. Evolve `stored_files` as follows:

| column | type | notes |
|---|---|---|
| `event_id` | str, index, non-null | durable owner |
| `origin_session_id` | str \| null | Session that introduced the upload/output |
| `run_id` | str \| null | Run that produced the artifact |
| `kind` | str | `upload`\|`artifact`; unchanged |

The existing `conversation_id` may remain as a compatibility alias during the
migration, but it is provenance, not ownership. Deleting a Session must not
delete Event files, artifacts, or the Event workspace. A run scratch file only
becomes an Event artifact when the single finalizer stores it durably, writes
the `StoredFile` row, and appends `artifact_added` to the Trail.

Global library files remain intentionally cross-Event. Their Event link records
origin/provenance; library visibility is governed separately by the existing
library flag and tenant policy.

---

## 2. Migration plan (conversation → Event/Session)

Additive and reversible where possible. Use staged Alembic changes: add
nullable keys, backfill and verify, then enforce non-null/unique constraints.
**No message/run/file rows move** — ownership keys are added; existing blob
keys remain valid opaque storage locators.

1. **Create** `events`, `proposals`, `tasks`, and `event_trail_entries` tables,
   including the partial unique Life-Event index.
2. **Add nullable `event_id`** to `conversations`, `runs`, `run_event_logs`,
   traces/context artifacts used by the Trail, `stored_files`, and
   `knowledge_entries`. Add `origin_session_id` to `stored_files`. CoreMemory
   remains global in the wedge and does **not** gain `event_id`.
3. **Backfill Life:** for each user, create their `is_life=true` Event.
4. **Assign every existing conversation to that user's Life Event**
   (`conversations.event_id = life.id`). Every past chat becomes a Life
   Session; nothing is lost, nothing dangles.
5. **Backfill provenance:** populate direct `runs.event_id` and related audit
   rows through their Session; populate `stored_files.event_id` and
   `origin_session_id`; populate event-scoped knowledge through provenance
   where unambiguous. User-global knowledge keeps `event_id=null` (see §6).
6. **Verify, then constrain:** assert zero dangling Sessions/files/runs, then
   make required Event ownership columns non-null. Rows legitimately global
   remain nullable by contract.
7. **Introduce workspace identity.** Add `event_id`/`workspace_id` to the run
   context and tool context without changing `session_id`. The local layout is:

   ```text
   sandbox/events/{event_id}/user-data/{workspace,uploads,outputs}
   sandbox/events/{event_id}/runs/{run_id}/scratch
   ```

   E2B reuses one persistent sandbox per `workspace_id`; per-run scratch avoids
   concurrent Session collisions. Existing `threads/{conversation_id}` data is
   moved into the owning Life Event workspace best-effort. The sandbox is a
   cache; object storage is authoritative. **Do not move existing object-store
   blobs solely to make their key contain `event_id`.**
8. **Change deletion semantics.** Deleting a Session removes its messages and
   session-scoped runtime records, not Event-owned files/workspace. Event
   deletion/archival owns durable cleanup and must preserve global-library
   files according to existing library policy.
9. **`conversation_search` → event history.** The tool (kernel
   `core_tools.py`) searches the messages loaded into the run's `AgentMemory`.
   The change is **what a run loads**: the run loads its **Event's** recent
   messages (across all the Event's Sessions), windowed to the context budget —
   not one Session's. The tool then searches/pages the Event's full history and
   becomes the **context page-fault** (vision §5). Rename to
   `search_event_history` (alias the old name one release for safety).

**Rollback:** before any Event contains state that did not originate in exactly
one legacy Session, the new tables/columns can be dropped and conversations
behave as before. After users create cross-Session Event state, rollback is a
data projection/export operation, not a claim of lossless automatic reversal.

---

## 3. The Proposal system (the deep part)

### 3.1 The contract

```
agent calls a gated tool (e.g. gmail_send)
        │
        ▼
VALIDATE + INTERCEPT — do NOT execute. Persist normalized tool + args + schema
fingerprint as a Proposal; append `proposal_staged` to the Trail.
        │
        ▼
Agent receives: {"status": "staged", "proposal_id": "prop_…"}   ← not the tool result
        │
        ├── Path A (user present): approval card shown now; quick approve calls
        │                          the executor asynchronously.
        └── Path B (no user):     Proposal waits in Today/inbox; the run can end;
                                   later approval calls the same executor.
```

Execution is **always** by a **dumb executor** — `run(tool, args)` → store
receipt → mark committed. **No reasoning, no re-interpretation, no agent loop
at execution time.** The Proposal is *serialized intent + executable payload*;
that is what makes async, safe execution, and replay/audit possible.

The **fast-path is UI latency, not inline execution**. The originating agent
continues from a staged outcome; it does not block waiting for the external
result and must not claim the action happened. Commitment updates the Proposal,
Trail, Today, and Surface. Work that must react to the committed result is a
later follow-up run/event, not a continuation hidden inside the approval route.

### 3.2 State machine (reconciled to vision §3.7)

```
Working → Proposed → Routed ─┬─ auto ──────────────► Approved → Executing → Committed(+Receipt)
                             ├─ notify ─(tell)─────► Approved → Executing → Committed(+Receipt)
                             └─ ask ─► Pending ─┬─ approved ─► Approved → Executing → Committed(+Receipt)
                                                ├─ rejected ─► Withdrawn
                                                └─ expired ──► apply safe_default
                                        Executing ─┬─ definite tool error ─► Failed
                                                   └─ outcome uncertain ───► Indeterminate
```

Wedge simplification: the write set routes **`ask`** (fixed policy — external
messages always Ask, vision §3.6); `auto`/`notify` exist in the model but are
not exercised until learned routing (deferred). `safe_default` in the wedge is
`{decision: "reject"}` (never send on timeout — matches the guardrail we
shipped).

`Indeterminate` is mandatory for external systems: if the provider may have
accepted the action but the process died before storing its receipt, the
executor must **not retry automatically** and risk duplicating reality. It
reconciles through a provider operation/idempotency key when possible;
otherwise it surfaces the uncertainty for review. The system promises
**effectively-once where a provider supports idempotency/reconciliation, and
at-most-once with explicit uncertainty otherwise** — never mathematically
“exactly once” across a database and an external API.

### 3.3 The interception seam (kernel touch-point)

Recommended, minimal kernel change: **add a `defer` decision with a structured
result** to the HITL vocabulary (`approve | edit | reject | defer`). `defer`
means “the consequence was durably staged elsewhere,” not success or failure.
The product's approval handler (evolve
`aloy_backend/approvals.py::ApprovalBridge`):

1. Resolve the registered tool and validate/normalize args with its Pydantic
   schema **before** persistence. Reject invalid calls without creating a
   Proposal.
2. Persist a Proposal (`proposed → routed → pending`) with the tool-schema
   fingerprint and append its Trail entry in one transaction.
3. Return `Decision(type="defer", result={status:"staged", proposal_id:…})`.
4. The kernel skips execution and returns the structured staged result beside
   the existing reject/edit branches in `pori/agent/dispatch.py`.

This preserves kernel product-independence: Pori understands a deferred
consequence, not Aloy's Proposal model. Add `ReceiptStatus.STAGED` (or an
equivalent non-terminal outcome), so staged is never recorded as `SUCCEEDED`
or `FAILED`. Completion validation must refuse claims such as “email sent”
unless a committed receipt backs them.

The Proposal is surfaced immediately over SSE (the existing
`approval_request` frame, now carrying `proposal_id`) and also persists in the
inbox. Interactive and background paths therefore produce **one object** and
neither depends on an in-memory Future surviving.

### 3.4 The executor (new, product-side)

A standalone component — **not** an agent run:

```
execute_proposal(proposal_id):
    lock/claim Proposal (Approved → Executing, one winner)
    re-authorize current org membership, policy, Event lifecycle, and tool access
    re-resolve live connections/MCP context through resolve_run_surface
    verify tool schema fingerprint and normalized args
    execute once with execution_attempt_id/idempotency key where supported
    persist provider operation id + receipt → Committed
    OR persist Failed / Indeterminate
    append the matching Trail entry in the same transaction
```

Reuses: `ToolExecutor` (kernel), `resolve_run_surface` (connections/token),
`ToolExecutionReceipt` (Verifiable Reality). Triggered by (a) the approve
endpoint enqueueing/claiming the Proposal (fast-path) and (b) a worker tick
over `status=approved` Proposals (background path). A transactional compare-
and-set or row lock makes exactly one caller win the `Approved → Executing`
transition, so approve + worker tick cannot both start execution.

The executor uses an explicit allowlist of Proposal-safe tools; a stored tool
name is never arbitrary code authority. Current authorization is checked again
at execution because membership, policy, credentials, Event lifecycle, and
tool availability may have changed since staging. An `edit` may change args
for the **same tool only**, then revalidates, recomputes risk/routing, refreshes
the fingerprint, and requires approval again when material.

### 3.5 Endpoints

- `POST /v1/events` — create an Event (direct action, no Proposal).
- `GET /v1/events` / `GET /v1/events/{id}` — list / read (+ Surface payload).
- `POST /v1/events/{id}/proposals/{pid}/decision` — `{approve|reject|edit}` →
  stores the decision; approve claims/enqueues the executor. (The existing
  `/conversations/approve/{id}` becomes a compatibility fast-path alias.)
- `GET /v1/today` — the Today aggregation (§5).
- `POST /v1/events/{id}/tasks`, `PATCH …/tasks/{tid}` — Task CRUD (ungated).

### 3.6 Safety tests (acceptance gates for §3)

- **No inline execution:** a gated tool call in a run produces a Proposal row
  and a `staged` result, and the external side effect has **not** happened
  until a commit. (Assert the tool's real call count is 0 pre-approval.)
- **Staged is not committed:** the kernel records a staged outcome and refuses
  an unsupported “sent/booked/published” completion claim.
- **Validated payload:** invalid args produce no Proposal; edits are
  revalidated and cannot swap the tool.
- **Single claim:** approve + worker tick → one `Executing` claimant and at
  most one provider call.
- **Crash window:** simulate provider success followed by process failure
  before DB commit; the Proposal becomes/reconciles as `Indeterminate` and is
  never blindly retried.
- **Execution-time revocation:** membership/policy/credential removal after
  staging causes a safe failure before the provider call.
- **Schema drift:** a changed tool fingerprint refuses stale execution.
- **Expiry → safe default:** an unanswered Ask past `expires_at` applies
  `safe_default` (reject) and never executes.

---

## 4. The Project Event Surface (templated)

A **schema payload** (typed JSON, vision §3.3 / ADR 0010) the app renders with
trusted components — for the wedge it is **hand-authored per the Project type**,
not model-composed. `GET /v1/events/{id}` returns:

```jsonc
{
  "event": { "id", "title", "lifecycle", "summary" },
  "surface": {
    "type": "project",
    "sections": [
      { "kind": "status",   "summary": "…", "phase": "…" },
      { "kind": "tasks",    "tasks": [ {id,title,status,order} ] },      // interactive
      { "kind": "activity", "entries": [ …Trail/receipt entries… ] },   // read-only
      { "kind": "notes",    "notes": "…" },                             // Event memory
      { "kind": "files",    "files": [ {id,name,kind,origin_run_id} ] }  // durable Event files
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

Every rendered assertion carries an explicit state/evidence posture:

- Working state is visibly working/draft and links to its Trail provenance.
- Pending reality change links to its Proposal.
- Committed reality links to a receipt/provider operation id.
- Model-written `summary` and notes are working state, never implicit proof of
  an external fact.
- Files appear only from durable `StoredFile` rows; raw sandbox paths never
  become Surface artifacts.

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
- **Session:** **no durable semantic memory.** Sessions retain messages and
  runtime identity, but do not own knowledge/files/workspace state.

`MemoryScope` (kernel `memory_contracts.py`) gains **`event_id`** alongside
`organization_id/user_id/agent_id`; `session_id` remains available for runtime
and provenance but is not the durable knowledge boundary. CoreMemory stays
global in the wedge.

Consolidate the backend's current memory-assembly paths into **one**
`load_event_memory()` service used by streaming, blocking, durable worker,
gateway, and cron runs. For a run in Event E it loads:

1. Global CoreMemory/preferences.
2. Global + Event-E knowledge (never Event B).
3. Event state needed for the task (summary, Tasks, pending Proposals, file
   manifest), compactly rendered.
4. A token-bounded recent message window across Event-E Sessions, ordered by
   timestamp with Session provenance.
5. `search_event_history` as the page fault for older Event-E messages.

**Cross-event retrieval is deferred** (§7). The acceptance contract is
`leakage=0`; recall tests cover global + same-Event records and messages.

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

1. **Aggregate + migration** (§1–2). Create Event/Task/Proposal/Trail schema,
   perform the nullable→backfill→constrain migration, and update every Session
   creation path. Gate: exactly one Life Event per org+user under concurrent
   creation; every existing Session/Run/file has valid Event ownership; branch,
   gateway, cron, and deletion semantics pass.
2. **Event context + workspace/files** (§0.1, §1.6, §2, §6). Land the unified
   `load_event_memory()`, keep Event/Session/Run identities separate, provision
   the Event workspace, and promote run outputs through the single finalizer.
   Gate: **leakage=0** — an Event-A run loads global + A memory/messages/files
   and never B; two Sessions share A's durable artifacts while concurrent run
   scratch cannot collide; deleting one Session preserves them.
3. **Proposal object + staged kernel outcome** (§3.1–3.3). Validate before
   persistence, return a first-class staged result, and append Trail entries.
   Gate: invalid calls stage nothing; valid calls execute zero side effects;
   the model cannot represent staged as committed.
4. **Executor + commit rail** (§3.4–3.6). Add execution-time authorization,
   allowlisting, fingerprint checks, single-claim locking, expiry, receipts,
   and `Indeterminate` reconciliation. Gate: every §3.6 safety test passes; a
   live `gmail_send` stages → approves → sends → commits with provider evidence
   and zero inline execution.
5. **Tasks + Project Surface + Today** (§1.4–1.6, §4–5). Gate: create a Project
   Event, work across two Sessions, add/complete Tasks, see Event files and the
   complete Trail, and resolve a pending Proposal from both its Surface and
   Today.
6. **The loop, end-to-end** (hero flow, vision §6): in the "Building Aloy"
   Event — agent works → Trail explains → Proposal appears → user decides →
   reality changes (email sent, receipt) → Surface + Today update. Gate: the
   60-second demo runs on the founder's own account, plus a deliberate crash-
   window drill proves no blind duplicate send.

## 9. What's reused vs new (map)

| Piece | Status |
|---|---|
| Agent loop, tools, streaming, sandbox, run-surface, finalizer | **Reused** |
| HITL approval bridge (`approvals.py`) | **Evolved** from blocking Future to durable staging fast-path |
| Receipts / traces / run-event-log | **Reused** as evidence feeding first-class Trail entries |
| `conversation_search` | **Reused**, re-scoped to the Event |
| `events` / `proposals` / `tasks` / `event_trail_entries` tables | **New** |
| `event_id` on Sessions/Runs/files/memory/audit + workspace identity | **New (migration)** |
| Event-owned file promotion + Session-safe deletion | **Evolves storage/finalizer** |
| Proposal executor + staged receipt/defer decision + interception | **New** |
| Project Surface schema + Today aggregation + Event files section | **New** |

---

_Open the build against §8 phase 1. The founder decides the "Building Aloy"
Event's exact task seed and Surface copy at phase 5._
