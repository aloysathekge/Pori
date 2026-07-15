# Aloy — product vision

_Canonical product definition, version 2.1. Revised 2026-07-15 after the
Event-workspace, Task-execution, and Life-Conversation reviews. Product specs
and implementation plans must agree with this document. The active V1 delivery
sequence lives in
[`aloy-v1-plan.md`](./aloy-v1-plan.md)._

## 1. Thesis

**Aloy is a persistent workspace for meaningful work and life events, where
agents make durable progress and people remain in control of consequences.**

Most assistants reduce work to:

```text
conversation → response → forgotten scrollback
```

Aloy turns an intention into a durable operating loop:

```text
intention → Event → Task → agent work → trusted state → next action
                                   └→ Proposal → decision → receipt
```

The conversation remains important, but it is no longer the only place where
work exists. Tasks, files, decisions, memory, evidence, and history belong to
the Event and survive individual Runs, app restarts, and context-window limits.

> **Life is the permanent user–Aloy space. A dedicated Event is the durable
> home for a meaningful outcome. Tasks are how intention becomes work.**

The user controls intent, initiation, priorities, decisions, and review. Aloy
plans, researches, drafts, tracks, executes permitted work, and adapts. V1
starts with explicit initiation and bounded execution; autonomy is granted
deliberately and earned later.

## 2. The product model

```text
USER ↔ ALOY
       │
       ├── LIFE ── permanent personal space
       │     ├── Conversation A
       │     ├── Conversation B
       │     └── loose Tasks, files, memory, and Trail
       │
       ├── TODAY ── attention across Life and Events
       │
       └── DEDICATED EVENT
             durable identity, outcome, context, files, and policy
             ├── one canonical continuous Conversation
             ├── Tasks, Surface, Trail, and Triggers
             └── RUN ── bounded agent execution
                         ├── Working state and artifacts
                         └── Proposal → decision → receipt
```

Three different things must never be collapsed:

| identity | meaning | lifetime |
|---|---|---|
| **Event** | durable product aggregate | until archived/deleted |
| **Conversation** | one user-visible chat thread | until archived/deleted |
| **Run** | one bounded agent execution | minutes or hours |

“Session” is a product role, not another aggregate. In a dedicated Event, its
canonical Conversation is the continuous Session. In Life, each user-started
Conversation is an independent Session over shared personal state.

## 3. The primitives

### 3.1 Event — the durable home

An Event represents something with a future: _Career OS, Building Aloy,
University, Trip to San Francisco, Weekly Review_. It owns the context needed
to make progress over time:

- identity, goal, lifecycle, phase, and summary;
- one canonical continuous Conversation for a dedicated Event;
- Tasks and their execution history;
- working state, files, and artifacts;
- pending and resolved Proposals;
- Event memory and connected services;
- the Trail and evidence references;
- triggers, limits, and autonomy settings.

An Event is not a page and not a chat folder. The UI is a lens over the Event;
the Event remains authoritative when the UI is closed.

The lifecycle is deliberately small:

```text
Emerging → Active → Dormant → Concluded → Archived
```

- **Emerging:** a possible Event awaiting user acceptance. Deferred after V1.
- **Active:** work and explicitly configured triggers may run.
- **Dormant:** state is retained; proactive work and triggers are paused.
- **Concluded:** the intended outcome is complete; the Event may be reopened.
- **Archived:** hidden from normal navigation and operationally inactive.

Users control lifecycle directly. Aloy may recommend a transition, but V1 does
not silently make Events dormant, concluded, or archived. Phase names inside an
active Event remain model- or user-defined.

Manual Event creation is always available and commits directly because the
user is the authority. Aloy-initiated Event creation is a Proposal and comes
after the V1 loop is proven.

### 3.2 Life — the permanent personal space

Life is the permanent system Event representing the user's ongoing relationship
with Aloy. It is a personal workspace, not the user identity and not the agent
identity. Unlike a dedicated Event, Life may own many Conversations.

It provides:

1. unstructured capture before a dedicated Event exists;
2. multiple fresh Conversation threads without creating Event clutter;
3. one-off Tasks that do not deserve their own Event;
4. routing into an existing or new Event;
5. future cross-Event coordination;
6. the foundation for Today.

**New conversation** creates a Conversation in Life. **New Event** creates a
dedicated Event and its canonical Conversation. Life Conversations do not
appear as Events in the Event rail. Tasks, files, artifacts, and Trail entries
created from a Life Conversation belong to Life and retain their originating
Conversation as provenance.

Life must be useful with zero user-created Events. Cross-Event reasoning,
emergent clusters, and automatic Event proposals are later capabilities; V1
keeps dedicated Event context isolated and creation manual. A user may create
an Event from a Life Conversation through an explicit action that preserves an
origin reference; Aloy never silently reclassifies the Conversation.

### 3.3 Conversations and continuous Event Sessions

A dedicated Event has exactly one canonical user-facing Conversation. Opening
the Event always resumes it, like reopening a long-running coding project. It
cannot be independently deleted while the Event exists.

Life is intentionally different. The user may start many Conversations to get
a clean transcript for a new topic. Deleting a Life Conversation deletes that
thread's messages, not Life-owned Tasks, files, memory, receipts, or Trail. If
the deleted thread is the stored Life default, the system safely selects another
recent Conversation or returns Life to an empty-chat state.

Starting a fresh Life Conversation resets transcript context, not the
relationship or accepted memory. A Run loads:

1. the current Conversation's bounded recent messages;
2. accepted global and Life or dedicated-Event memory;
3. the owning Event's relevant Tasks, files, decisions, and working state;
4. older Conversations only through explicit, scoped retrieval when relevant.

The runtime must not inject every Life transcript into every new Conversation.
Messages remain durable and pageable, summaries preserve accepted meaning, and
`search_event_history` pages older evidence into context on demand.

Legacy, gateway, branch, or transport Conversations may remain as provenance,
but a dedicated Event never presents them as competing user-facing Sessions.

Background Task Runs report to the Conversation selected for that execution.
For a dedicated Event this is its canonical Conversation; for Life it is the
Conversation where work was initiated or the one the user explicitly selected.
The Task keeps an optional originating Conversation for provenance, while the
Run stores the exact Conversation that receives progress and results.
V1 permits one active foreground Run per Conversation and one active Task Run
per owning Event, subject to a small account-wide concurrency cap.

### 3.4 Task — durable executable work

A Task is not merely a checklist item. It is the durable contract that turns
an intention into bounded agent work.

A Task carries:

- title and instructions;
- definition of done;
- status, priority, and optional due date;
- execution mode (`manual`, later `scheduled` or `triggered`);
- assigned agent/configuration;
- current Run and execution history;
- optional originating Conversation for provenance;
- progress, result summary, blocker, and produced artifacts;
- budget limits for steps, time, tools, and cost.

The V1 state machine is:

```text
open → queued → in_progress
                 ├→ blocked
                 ├→ waiting_approval
                 ├→ failed
                 ├→ cancelled
                 └→ done
```

The default start mechanism is an explicit **Work on this** action. Creating a
Task does not silently spend money or begin work. Aloy may create and recommend
Tasks, but it may only start them automatically when the user explicitly grants
that Event an automation mode.

One Task execution creates one Run. The worker claims it with a durable lease,
persists progress, and supports stop, retry, and resume. Clarification moves the
Task to `blocked`; an external consequence moves it to `waiting_approval`;
completion requires the Task's definition of done, not merely an agent saying
“done.” Users may always reopen a Task.

Task results appear consistently:

1. a compact result in the Conversation selected for that Run;
2. a result summary on the Task;
3. durable Event artifacts when files were produced;
4. an expandable Trail group linked to the Run and evidence.

### 3.5 Run — bounded execution

A Run is one attempt by an agent to perform a conversational turn, Task, cron
job, gateway request, or follow-up. It is finite, budgeted, checkpointed, and
observable.

Runs may Observe, Compute, and Stage inside the Event without approval. A Run
must never confuse staged work with an external consequence. Long or complex
Tasks begin with a plan; progress is persisted at meaningful milestones rather
than logging every token.

Sub-agents may help, but the parent Run owns validation, synthesis, budget, and
the final Task outcome.

### 3.6 Surface — trusted state beside the conversation

The Surface is the living, interactive representation of an Event. In the
desktop workspace it is the collapsible, resizable context pane beside the
continuous Session. It is always available; it does not suddenly replace the
conversation.

The V1 component vocabulary is trusted and finite:

- Status
- Tasks
- Decisions
- Files
- Trail
- Notes

The Surface is recomputed from durable rows and updated live through the same
REST + SSE boundary used by the conversation. New data updates badges without
unexpectedly changing the selected tab.

Every displayed assertion has a visible posture:

- **Working:** research, drafts, and model-maintained summaries;
- **Blocked:** work waiting on information or tooling;
- **Pending:** a Proposal awaiting a decision;
- **Committed:** a consequence backed by a receipt;
- **Failed:** a definite unsuccessful attempt;
- **Indeterminate:** an external outcome that must be reconciled.

V1 uses one Project template. Model-composed Surface layouts may come later,
but only through an allowlisted schema vocabulary—never arbitrary UI code as
trusted product state.

### 3.7 Trail — the durable narrative

The Trail explains what happened, why Aloy acted, what changed, and what
evidence exists. It belongs to the Event and continues across Sessions, Runs,
workers, and app restarts.

Trail records semantic activity:

- Event lifecycle and trigger changes;
- Task creation, claims, milestones, blockers, completion, and failure;
- Run start and terminal outcome;
- artifact creation and promotion;
- Proposal, decision, execution, and reconciliation transitions;
- important memory and phase changes.

Conversation messages are not duplicated into Trail. Low-level tool and model
events remain in Run Replay. Trail groups one Task execution into an expandable
narrative and links to the relevant Session message, Run Replay, artifact,
Proposal, or receipt.

Trail is append-only. Corrections append correcting entries. Privacy redaction
is an exceptional audited operation. The Event narrative lasts with the Event;
raw debug data may have a shorter retention policy.

### 3.8 Proposal — consent before consequence

Agents may freely research, compare, simulate, draft, and stage inside an
Event. Crossing into shared or external reality requires a durable Proposal:

```text
staged intent → Proposal → routing → decision → execution → receipt
```

Always-Ask V1 actions include sending messages, spending, publishing, booking,
deleting important external data, and changing account permissions. Approval
never executes unvalidated prose: the Proposal stores the normalized tool and
arguments, then a non-agent executor rechecks authorization, policy,
credentials, schema, and Event lifecycle before calling the provider.

Approved execution is idempotent or reconcilable. If the provider may have
accepted an action before the receipt was stored, the Proposal becomes
`indeterminate` and is never blindly retried.

Auto and Notify remain the same Proposal mechanism with pre-granted consent.
They are not exercised in V1. Learned autonomy never overrides fixed safety
policy.

### 3.9 Files, artifacts, and memory

Durable bytes belong to the Event. The Surface groups them as Uploads, Working
files, and Outputs; these are product categories, not fragile physical folder
semantics. A scratch file becomes an artifact only when the finalizer stores it
durably and records provenance.

Working files may evolve through immutable versions. Files link to Tasks,
Runs, Proposals, and Trail entries. Shared library blobs may be referenced by
several Events without duplicating bytes.

Memory is an index over durable things:

- **Global:** identity and stable preferences;
- **Event:** accepted facts, decisions, summaries, results, and file pointers;
- **Transcript:** durable messages that are searched on demand, not treated as
  automatically true memory.

Receipts outrank model memory. Tenant and Event filtering occurs before
ranking or semantic retrieval; cross-Event retrieval is deferred until
isolation is proven.

### 3.10 Triggers — explicit reasons Aloy wakes

Every proactive Run begins from a durable, Event-scoped trigger and writes why
it woke to Trail. Rollout order:

1. user presses **Work on this**;
2. explicit schedule;
3. incoming email or webhook;
4. external state change;
5. model-detected pattern.

A trigger queues durable work before execution. Idempotency keys and leases
prevent duplicates. Dormant Events have no active triggers. Users can pause
proactive work globally or per Event.

### 3.11 Today — the attention lens

Today is not another Event Surface. It is the cross-Event view of what needs
the user now:

1. waiting decisions;
2. overdue or time-critical work;
3. blocked work requiring input;
4. user-pinned priorities;
5. stale Tasks;
6. recent meaningful changes.

Prioritization is deterministic and explainable. Model judgment may break ties
but cannot silently override due dates, approvals, blockers, or user pins.

Ordinary progress is ambient. V1 uses badges and Today, not proactive push.
Digest, push, reach-out, and interruption budgets come after the core loop.

## 4. The constitution

These invariants override feature convenience:

1. **No agent consequence without a durable Proposal.**
2. **No committed claim without evidence.** A checkmark means a receipt exists.
3. **No invisible work.** Every Run and meaningful state transition is durable
   and explainable through Task, Trail, and Run Replay.
4. **No accidental execution.** Task creation is not Task initiation unless an
   explicit automation grant says otherwise.
5. **Dedicated Events have one canonical Conversation; Life may have many.**
   Runs are temporary; Conversations and durable Event state continue.
6. **Tasks are executable contracts.** Completion is tied to a definition of
   done and resulting state, not model prose.
7. **Context is a cache, not a home.** Durable truth lives in Event state,
   artifacts, memory, Trail, and receipts.
8. **Events own context; shared reality is referenced, not copied.**
9. **Scope before retrieval.** Tenant and Event boundaries are applied before
   ranking, search, or model context assembly.
10. **Bounded agency.** Every Run has explicit time, step, tool, and cost limits.
11. **Dormant means quiet.** No triggers or background agents run for a dormant
    Event.
12. **The user can stop and recover.** Work is cancellable, retryable, and
    reconcilable without duplicating consequences.

## 5. Product experience

Desktop is the primary Aloy experience; web remains a complete fallback. The
main shell has three flexible regions:

```text
navigation | current conversation | owning Life/Event Surface
```

- **New conversation** opens a fresh Life thread.
- **New Event** creates a dedicated workspace and canonical Conversation.
- Life Conversation history is separate from the dedicated Event rail.
- The Event rail shows dedicated Events, not Life chats or every Task.
- The conversation is the place to instruct, think, clarify, and review.
- The Surface shows trusted current state and controls.
- Task detail and files open in the context pane or drawer; rich artifacts may
  expand into the main canvas.
- On mobile, the Surface becomes a full-screen sheet.
- Admin capabilities—Agents, Skills, Connections, Memory, Usage, Settings—are
  secondary to the Event workspace.

The two entry loops are intentionally small:

```text
New conversation → talk or create a loose Life Task

New Event → create Task → Work on this → watch progress
→ review artifact → approve consequence → see receipt
```

## 6. The V1 proof: Career OS

The V1 hero flow uses a real Event and a real Task:

> Career OS → “Research US companies for startup jobs” → **Work on this**

The complete proof is:

```text
open Career OS
→ start the research Task
→ Aloy plans and performs sourced web research
→ progress streams into the continuous Session and Trail
→ a cited report becomes an Event artifact
→ the Task reaches done from its definition of done
→ Aloy stages an email summary
→ the user approves the Proposal
→ Gmail sends to the founder's own address
→ a provider receipt commits the consequence
→ Surface and Today update live
```

The crash-window drill repeats the final action while killing the process after
provider acceptance but before database commit. Reconciliation must find the
provider operation and must not send a duplicate.

This flow is the acceptance test for the product thesis. A static Task list or
a convincing chat response is not enough.

## 7. V1 scope

V1 includes:

- Life and manually created Project Events;
- multiple user-started Conversations in Life;
- one continuous canonical Conversation per dedicated Event;
- executable Tasks with explicit initiation;
- durable worker execution, stop/retry/resume, and bounded budgets;
- trusted templated Surface with live updates;
- complete semantic Trail plus Run Replay;
- Event files, artifacts, and scoped memory;
- sourced web research;
- Ask-routed Proposals and receipt-backed Gmail execution;
- Today decisions, blockers, stale Tasks, and meaningful changes;
- crash recovery, idempotency/reconciliation, and context isolation tests.

V1 deliberately excludes:

- automatic/emergent Event detection;
- cross-Event Life coordination and retrieval;
- learned Auto/Notify autonomy;
- push-notification and interruption-budget learning;
- Calendar/Money/People Reality Objects and object guardians;
- arbitrary model-generated application UI;
- cross-user shared Events and multi-agent negotiation;
- unrestricted concurrent Runs inside one Event;
- local-folder desktop access and full mobile-native clients.

## 8. Substrate and status

| Product capability | Current substrate | V1 status |
|---|---|---|
| Agent execution | Pori loop, tools, teams, streaming, stop/resume | built foundation |
| Event aggregate | Event ownership on Runs/files/memory/audit | built foundation |
| Life Conversations | singleton Life assignment + Conversation CRUD | substrate built; scoped UI/context/deletion is R1 |
| Dedicated Event Session | `Event.primary_conversation_id` | built foundation; R1 locks product boundary |
| Task working state | Task CRUD + agent mutation tools | built foundation; executable model is R2 |
| Durable execution | worker leases, checkpoints, cron chassis | built foundation; Task claim/link is R2–R3 |
| Surface | Event workspace + trusted Task/Proposal/File/Trail reads | built foundation; live Task progress is R4 |
| Trail | Event entries, receipts, traces, run replay | partial; semantic Task grouping next |
| Proposals | durable staging, decisions, executor, receipts | built foundation |
| Files/artifacts | object storage, finalizer, library pointers | built foundation |
| Event memory | scoped loader and history-search seams | Life transcript isolation R1; compaction R7 |
| Web research | tool ecosystem/MCP seams | provider-neutral product tools needed |
| Gmail proof | connection + Proposal-safe execution rail | integration drill needed |
| Today | Event aggregation of decisions/activity/open Tasks | built foundation; blockers/staleness R4 |
| Desktop | shell + web workspace | responsive QA R0/R7; native packaging later |

## 9. Delivery rule

The active sequence and acceptance gates are in
[`docs/aloy-v1-plan.md`](./aloy-v1-plan.md). Each phase gets its own branch and
PR into the `aloy-v1` integration branch. `main` remains protected until the
Career OS loop, visual QA, reliability drill, and required tests pass.

Architecture remains one-way:

```text
products → extensions → pori
```

The Pori kernel stays product-agnostic. Aloy surfaces reach the backend only
through REST + SSE. Product state and policy remain in `products/aloy`.

Related documents:

- [`aloy-v1-plan.md`](./aloy-v1-plan.md) — active execution plan
- [`aloy-wedge-spec.md`](./aloy-wedge-spec.md) — implemented Event/Proposal foundation
- [`Aloy.md`](./Aloy.md) — monorepo/product architecture
- [`engineering-excellence-spec.md`](./engineering-excellence-spec.md) — quality bar
- [`../products/aloy/BOOT.md`](../products/aloy/BOOT.md) — local boot guide
- [`../.agent/progress/current.md`](../.agent/progress/current.md) — live state
