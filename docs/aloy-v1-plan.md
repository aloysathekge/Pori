# Aloy V1 — reset delivery plan

_Active plan, 2026-07-15. This plan begins from the Event, Proposal, workspace,
and initial Surface foundation already built on `aloy-v1`. It supersedes the
remaining phase order in `aloy-wedge-spec.md`. Product decisions come from
`aloy-vision.md`._

## 1. V1 outcome

V1 must prove both ways a user begins with Aloy:

```text
New conversation → fresh Life thread → talk or create loose work

New Event → continuous Event conversation → executable Task → durable Run
          → evidence + artifact → Proposal → decision → receipt
```

The Career OS proof is complete only when its research Task produces sourced
work and one approved Gmail consequence without losing continuity or
duplicating the action after a crash.

## 2. Locked product contract

1. The user and Aloy exist above the Event model; neither is an Event.
2. Life is one permanent, system-created personal Event per user and may own
   many user-visible Conversations.
3. **New conversation** creates a fresh Conversation in Life and does not add an
   item to the Event rail.
4. A dedicated Event owns exactly one canonical user-visible Conversation for
   its lifetime. Other conversation rows are provenance, not competing product
   Sessions.
5. **New Event** creates the Event and its canonical Conversation atomically.
6. Life and dedicated Events own durable Tasks, files, artifacts, memory,
   Proposals, and Trail. Conversations own their messages and provide origin
   provenance.
7. A new Life Conversation receives shared accepted Life memory and relevant
   durable state, not every previous Life transcript.
8. Every Task Run stores the Conversation that receives progress,
   clarification, and results. Dedicated Event Tasks use the canonical
   Conversation; Life work uses the current or explicitly selected thread.
9. Creating a Task does not start it. V1 execution begins with explicit
   **Work on this**.
10. External consequences still require Proposal → decision → executor →
    receipt.

## 3. Branch and merge policy

- `main` remains protected.
- `aloy-v1` is the integration branch.
- Each reset phase uses its own branch based on the latest `aloy-v1`.
- Each phase is independently reviewed, tested, and merged into `aloy-v1`.
- No phase begins from an unmerged sibling branch.
- No V1 merge to `main` until R7 passes.
- Branch names never use an agent or tool prefix.

## 4. Baseline already built

The following foundation is retained:

- Event aggregate and Event ownership across Runs, files, memory, and audit;
- singleton Life Event assignment when a Conversation is created without an
  explicit Event;
- `Conversation.event_id` and `Event.primary_conversation_id`;
- canonical Conversation provisioning for dedicated Events;
- Event workspace with conversation and context pane;
- initial Task CRUD and agent Task mutation tools;
- Event files/artifacts and persistent workspace;
- durable Proposal staging, decisions, executor, receipts, and reconciliation;
- Today aggregation;
- durable worker, Run checkpoints, streaming, stop/resume, traces, and replay.

The substrate already permits several Conversation rows in Life, but the app,
deletion rules, context loader, and product contract do not yet handle that
topology correctly. R1 closes that gap; it is not considered shipped merely
because the foreign key permits it.

## 5. Reset phases

### R0 — close the existing Event workspace foundation

**Branch:** current `aloy-v1-phase-5-surfaces`

Scope:

- apply the formatting-only migration correction;
- complete signed-in visual QA of the three-region dedicated Event workspace;
- verify canonical Conversation reopen and delete protection;
- verify streaming and Surface loading against the local stack;
- update draft PR #168 and merge it into `aloy-v1` when green.

Gate:

- CI is green;
- no browser console errors on the Event route;
- conversation output streams visibly;
- leaving and reopening Career OS returns the same Conversation;
- Tasks, Decisions, Files, and Trail render in the context pane at desktop and
  mobile widths;
- the visual QA report has `final result: passed`.

R0 does not redesign Life or add executable Tasks. It closes and merges the
existing branch so R1 starts from a clean integration point.

### R1 — Life conversations and dedicated Event sessions

**Branch:** `aloy-v1-r1-life-conversations`

Scope:

- make **New conversation** create and open a Life Conversation;
- make **New Event** create a dedicated Event and canonical Conversation;
- show Life Conversation history separately from the dedicated Event rail;
- scope the Chat list/API to Life so dedicated Event Conversations never leak
  into its switcher;
- keep dedicated Event navigation locked to its canonical Conversation;
- define Life default-conversation deletion: select a recent remaining thread
  or return to a safe empty state without deleting Life-owned durable state;
- keep dedicated Event canonical-Conversation deletion protected;
- load the current Conversation transcript plus accepted owning-Event state;
  do not hydrate a Run with every Conversation transcript in Life;
- add scoped retrieval for older Life Conversations when explicitly relevant;
- support explicit **Create Event from this conversation** with durable origin
  provenance and a user-confirmed seed summary; preserve the source Life thread
  and do not silently move Tasks or files;
- add backend and UI language that distinguishes Conversation, dedicated Event
  Session, and Run.

Gate:

- two new Life Conversations have separate transcripts and shared accepted
  Life memory;
- a prompt in Life Conversation B does not automatically receive transcript A;
- the Chat switcher contains Life Conversations only;
- dedicated Event navigation always returns its canonical Conversation;
- deleting any Life Conversation leaves Life valid and preserves Life Tasks,
  files, Trail, and receipts;
- deleting a dedicated Event's canonical Conversation returns `409`;
- creating an Event from a Life Conversation preserves both the source thread
  and inspectable origin provenance;
- tenant and dedicated Event isolation remain zero-leakage.

### R2 — executable Task model

**Branch:** `aloy-v1-r2-task-model`

Scope:

- evolve Task status to
  `open|queued|in_progress|blocked|waiting_approval|done|failed|cancelled`;
- add instructions, definition of done, priority, optional due date,
  execution mode, assigned agent, current Run, result summary, blocker, and
  budget policy;
- add nullable `origin_conversation_id` to Task for provenance and nullable
  `task_id` to Run; the existing Run Conversation identifies its report target;
- require the origin Conversation to belong to the Task's owning Event;
- add a safe additive migration/backfill from existing `open|done` rows;
- add Task claim/transition service with legal-transition validation;
- preserve atomic Task + Trail writes.

Gate:

- old Tasks migrate without loss and receive a valid origin when available;
- a Life Task can originate from either of two Life Conversations;
- an Event Task cannot reference a Conversation owned by Life or another Event;
- illegal transitions fail without partial writes;
- concurrent claims yield one winner;
- Task state changes always produce their semantic Trail entry.

### R3 — Work on this and durable Task execution

**Branch:** `aloy-v1-r3-task-execution`

Scope:

- add **Work on this**, Stop, Retry, and Resume controls;
- add `POST /events/{event_id}/tasks/{task_id}/work` and related controls;
- queue a Task Run through the durable worker rather than an in-memory job;
- assemble Task instructions from its owning Event and selected Conversation;
- enforce one active Task Run per owning Event in V1 and queue additional work;
- allow at most one foreground conversational Run per Conversation, with a
  small account-wide concurrency cap;
- persist start, milestone, blocked, approval, and terminal state;
- write compact lifecycle messages to the Run's selected Conversation;
- handle clarification as `blocked` and external consequences as
  `waiting_approval`.

Gate:

- the stale Career OS research Task starts from one click;
- a Life Task reports to the Life Conversation that created it;
- a dedicated Event Task reports to its canonical Conversation;
- app closure does not stop the worker Run;
- worker restart resumes or safely retries from durable state;
- Stop and Retry do not duplicate Runs or consequences;
- the UI explains why work is queued, running, blocked, or failed.

### R4 — live Surface and semantic Trail

**Branch:** `aloy-v1-r4-live-surface-trail`

Scope:

- extend SSE with Task and owning-Event state updates;
- refresh Life and dedicated Event Surfaces without polling or tab switching;
- group one Task execution into an expandable Trail narrative;
- link Trail to origin Conversation, Run Replay, artifacts, Proposals, and
  receipts;
- add cursor pagination for long Trail and Conversation histories;
- add reconnect/refetch behavior and explicit stale/offline states;
- update Today with blocked and stale Life and Event Tasks.

Gate:

- Task progress appears live in the correct Conversation and Surface;
- progress never appears in a sibling Life Conversation;
- reconnecting during a Run loses no terminal state;
- no duplicate or missing semantic Trail transitions occur;
- long histories do not require loading the full transcript or Trail.

### R5 — sourced web research and artifacts

**Branch:** `aloy-v1-r5-research-tools`

Scope:

- add provider-neutral `web_search` and `read_web_page` product tools;
- require source URL, retrieval timestamp, title, and evidence provenance;
- block or clearly degrade when research tooling is unavailable;
- build a Career OS research instruction profile without coupling it to one
  search vendor;
- generate a cited report as a durable Event artifact;
- index the result in Event memory with links to evidence and the Task.

Gate:

- the Career OS Task finds a defined set of current US startup opportunities;
- every reported company has inspectable source evidence;
- unsupported or inaccessible claims are marked rather than invented;
- the report survives app and worker restarts and appears under Event Files;
- cross-Event retrieval leakage remains zero.

### R6 — Career OS decision and receipt loop

**Branch:** `aloy-v1-r6-career-os-loop`

Scope:

- let the research Task stage a Gmail summary Proposal linked to the Task;
- keep the Task `waiting_approval` until the Proposal resolves;
- approve from the Event Surface or Today;
- send to the founder's own address through the existing safe executor;
- commit the receipt and finish the Task only after evidence exists;
- reflect the result live in Conversation, Surface, Today, and Trail.

Gate:

- no email is sent before durable approval;
- rejection produces no provider call;
- approval produces one provider call and one committed receipt;
- Task completion cannot precede receipt persistence;
- the complete Career OS loop works on the founder account.

### R7 — reliability, context longevity, and release

**Branch:** `aloy-v1-r7-release-gate`

Scope:

- run the provider-success/database-crash reconciliation drill;
- add Task/Run watchdogs for expired leases and stuck work;
- verify time, step, tool-call, cost, and account-concurrency budgets;
- paginate Conversation history and add summary compaction thresholds;
- verify Life retrieval, dedicated Event history page faults, and isolation;
- finish responsive and accessibility QA at required viewports;
- update boot, operator, architecture, and demo documentation;
- run the complete safe verification suite.

Gate:

- crash after provider acceptance does not duplicate the email;
- stuck Tasks are detected and recoverable;
- a long dedicated Event reopens quickly and recalls older evidence on demand;
- a fresh Life Conversation stays transcript-clean while retaining accepted
  personal memory;
- required viewports pass visual and interaction QA;
- kernel tests, backend tests, mypy, app build/lint, import boundaries, and CI
  are green;
- both entry flows and the 60-second Career OS demonstration are repeatable.

## 6. Required V1 evals

1. **Conversation topology:** New conversation enters Life; New Event creates
   one canonical dedicated Conversation.
2. **Transcript isolation:** a fresh Life Conversation does not automatically
   ingest sibling transcripts.
3. **Task completion:** definition-of-done satisfaction and correct terminal
   state.
4. **Task routing:** progress, clarification, and results return to the Run's
   selected Conversation exactly once.
5. **Research quality:** evidence coverage, freshness, and unsupported-claim
   rate.
6. **Event isolation:** dedicated Event-A leakage into Event-B equals zero.
7. **Trail completeness:** every required semantic transition appears once.
8. **Proposal enforcement:** no protected provider call before approval.
9. **Effectively-once execution:** no duplicate consequence in the crash window.
10. **Recovery:** queued/running/blocked work survives process restarts.
11. **Budget:** Runs stop at configured step, time, tool, cost, and concurrency
    ceilings.
12. **UX continuity:** reopening a dedicated Event preserves its Conversation;
    reopening a Life Conversation preserves that thread.

## 7. Explicitly deferred until after V1

- automatic Task selection and learned autonomy;
- scheduled and incoming-data triggers beyond the existing chassis;
- automatic or model-initiated promotion of Conversations into Events;
- automatic transfer of existing Life Tasks/files during Event creation;
- cross-Event Life coordination and retrieval;
- emergent Event detection;
- Auto/Notify routing;
- push notifications and learned attention budgets;
- free-form model-composed Surfaces;
- Reality Objects beyond Documents/Accounts/Preferences;
- shared cross-user Events;
- unrestricted concurrent Runs per Event or account;
- desktop local-folder integration and native mobile clients.

## 8. Immediate next action

Finish R2 on `aloy-v1-r2-task-model`: run draft-PR CI and verify the additive
migration, provenance isolation, legal transitions, atomic claims, and
Task/Trail consistency. Merge R2 into `aloy-v1` only when those checks pass.
Then create `aloy-v1-r3-task-execution` from the updated integration branch;
do not add **Work on this** or worker execution on the unmerged R2 branch.
