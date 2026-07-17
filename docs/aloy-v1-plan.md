# Aloy V1 — reset delivery plan

_Active plan, revised 2026-07-17 after the Event bootstrap and scoped-memory review. This
plan begins from the Event, Proposal, workspace, and initial Surface foundation
already built on `aloy-v1`. It supersedes the
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
11. An Event Surface is a versioned React application authored by Aloy for one
    Event and executed outside Aloy's trusted application boundary.
12. Conversation and Surface are peer workspace regions with user-controlled
    conversation-focus, split, and Surface-focus modes.
13. Generated source, durable Event data, and local presentation state are
    separate. Generated UI cannot replace Tasks, Proposals, receipts, files,
    Runs, evidence, or Trail as sources of truth.
14. Meaningful Surface interactions feed the canonical Event Session through
    validated intents; presentation-only interaction remains local.
15. Generated code receives only a capability-scoped SDK. External actions
    remain Proposal → decision → executor → receipt.

## 3. Branch and merge policy

- `main` remains protected.
- `aloy-v1` is the integration branch.
- Each reset phase uses its own branch based on the latest `aloy-v1`.
- Each phase is independently reviewed, tested, and merged into `aloy-v1`.
- No phase begins from an unmerged sibling branch.
- No V1 merge to `main` until R8 passes.
- Branch names never use an agent or tool prefix.

## 4. Baseline already built

The following foundation is retained:

- Event aggregate and Event ownership across Runs, files, memory, and audit;
- singleton Life Event assignment when a Conversation is created without an
  explicit Event;
- `Conversation.event_id` and `Event.primary_conversation_id`;
- canonical Conversation provisioning for dedicated Events;
- Event workspace with conversation and initial fixed context pane;
- initial Task CRUD and agent Task mutation tools;
- Event files/artifacts and persistent workspace;
- durable Proposal staging, decisions, executor, receipts, and reconciliation;
- Today aggregation;
- durable worker, Run checkpoints, streaming, stop/resume, traces, and replay.

The fixed pane and R4 live invalidation are foundations, not the completed
Surface architecture. R5 replaces the one-template assumption with the
model-authored application runtime in
[`aloy-surface-spec.md`](./aloy-surface-spec.md).

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

R4 supplies the durable live projection transport and preserves the initial
system panels. It does not claim that the fixed right pane is the final Surface
composition model.

Gate:

- Task progress appears live in the correct Conversation and Surface;
- progress never appears in a sibling Life Conversation;
- reconnecting during a Run loses no terminal state;
- no duplicate or missing semantic Trail transitions occur;
- long histories do not require loading the full transcript or Trail.

### R5 — model-authored Event Surface runtime

**Branch:** `aloy-v1-r5-composable-surfaces`

Scope:

- implement Event-owned Surface Project, immutable source Revision, isolated
  Build, structured Interaction, and provenance-bearing Data Record concepts
  from [`aloy-surface-spec.md`](./aloy-surface-spec.md);
- define the restricted React project/manifest contract, version-locked import
  allowlist, and `@aloy/surface` SDK;
- add authenticated, optimistic, idempotent authoring tools for reading,
  patching, building, previewing, publishing, and rolling back Surface source;
- build generated source in an isolated fixed toolchain with limits,
  diagnostics, immutable bundles, and last-good recovery;
- require an explicit atomic publication record and exact revision/build
  pointer so successful drafts cannot silently replace the live Surface;
- expose host-owned publication history and rollback to a previously published
  last-good build without changing canonical Event data;
- execute published bundles in a sandboxed separate/opaque-origin iframe with
  strict CSP, a schema-validated `MessageChannel`, and no host credentials,
  cookies, storage, navigation, or direct network by default;
- add tenant/Event-scoped SDK reads, reactive Event data, structured intents,
  explicit model turns, and host-validated consequential requests;
- separate code revision, data revision, interaction history, and personal
  workspace preferences;
- add the Aloy Surface Builder skill for project, SDK, interaction, truth,
  accessibility, preview, and repair guidance;
- implement the brief, required viewport/state capture, deterministic audit,
  independent Surface Critic, primary user-job simulation, bounded repair, and
  quality scorecard contract in §13 of
  [`aloy-surface-spec.md`](./aloy-surface-spec.md);
- choose and document the privileged Map widget adapter, tiles/geocoding
  boundary, attribution, privacy posture, credential isolation, and fallback;
- package University first and Madrid second as installable showcase templates
  for onboarding and live marketing, using only the ordinary Event/Surface
  authoring, build, SDK, interaction, and publication paths;
- add conversation-focus, resizable split, and Surface-focus workspace modes;
- carry code/data revision and interaction status over the R4 Event SSE
  connection without moving focus, scroll, or local presentation state;
- append semantic Trail entries for publish and meaningful interactions while
  excluding presentation-only and preference noise;
- keep the existing Event surface response compatible during migration.

Gate:

- the University showcase template renders navigation, timetable, courses,
  assessments, provenance, live work, and study actions from Event data;
- after the dedicated widget phase, the Madrid showcase template renders a
  map, flight/hotel choices, Schengen readiness, budget, itinerary,
  uncertainty, comparison intent, and protected payment intent;
- installing either template creates normal tenant-owned Event data and
  Surface revisions; no University/travel conditional exists in the app,
  backend, host runtime, or SDK;
- meaningful interactions reach the canonical Session exactly once while
  filters, sorting, and navigation remain local;
- Tasks, decisions, files, receipts, evidence, and Trail remain canonical
  whether or not generated code displays them;
- undeclared imports, direct network/host access, cross-Event reads, storage,
  navigation, and iframe escape attempts fail closed;
- failed builds and runtime crashes retain the last-good revision;
- intentionally weak University/Madrid candidates fail the quality gate, the
  Builder repairs seeded findings, and exhausted repair retains last-good;
- live updates preserve focus and local interaction state in all three
  workspace modes and at narrow widths.

### R5.5 — Event setup, context bootstrap, and scoped memory

**Branch:** `aloy-v1-event-setup`

**R5.5a shipped on the branch:** durable tenant/user-scoped setup drafts;
autosaved name, description, and mode; typed file, link, template, and existing
connection context; object-store staging; idempotent promotion into one Event
and canonical Conversation; Event-scoped knowledge/provenance, file-library
transfer, connection grants, and Trail; context ingestion failure cannot block
creation. The remaining bullets are R5.5b onward.

Scope:

- add a resumable, host-owned `EventSetupDraft` with typed notes, files, links,
  template seeds, and narrowly scoped connection grants;
- keep explicit Event creation available with only a name while transferring
  accepted draft context atomically into the Event and canonical Conversation;
- ingest setup sources asynchronously with visible status, provenance,
  sensitivity, freshness, failure, revocation, and retry semantics;
- produce a typed, versioned, evidence-linked `EventBrief` only after a
  host-owned readiness gate finds sufficient context;
- use name-only and little-context fallbacks that keep Conversation primary,
  ask at most one material question at a time, and never generate a confident
  but speculative Surface;
- invoke the existing Surface Builder and quality/publication pipeline for a
  deliberately small first Surface derived from the Event Brief;
- generate the Event cover asynchronously from a sanitized visual brief rather
  than asking the user to design it or sending raw sensitive context to the
  image model;
- formalize global versus Event memory, require tenant/user/Event isolation,
  and encode Event-over-global plus personal-over-team-over-organisation
  conflict precedence;
- consolidate evidence-backed semantic, episodic, and procedural memory with
  provenance, confidence, sensitivity, retention, and supersession;
- add host-owned Event memory controls for inspect, correct, forget, and
  explicit promotion to global memory;
- update the Event Brief incrementally and consider Surface evolution only
  after meaningful source, goal, phase, user-request, or usability changes.

Gate:

- an Event created with only a name is immediately usable and produces no
  invented Surface facts;
- a file, link, user note, or scoped connection can survive draft promotion,
  ingestion, restart, and provenance inspection;
- sufficient context creates an evidence-linked Event Brief and one
  quality-gated starting Surface without University/travel/career branching;
- one Event cannot retrieve another Event's memory, files, connection data, or
  transcript history under direct, semantic, or adversarial retrieval tests;
- canonical Tasks, receipts, files, provider evidence, and current Event state
  override conflicting model memory;
- corrections supersede earlier memory without erasing provenance, and the
  user can inspect and forget the accepted record;
- cover and Surface work remain asynchronous, idempotent, recoverable, and
  unable to block or roll back Event creation.

### R6 — sourced web research and artifacts

**Branch:** `aloy-v1-r6-research-tools`

Scope:

- add provider-neutral `web_search` and `read_web_page` product tools;
- require source URL, retrieval timestamp, title, and evidence provenance;
- block or clearly degrade when research tooling is unavailable;
- build a Career OS research instruction profile without coupling it to one
  search vendor;
- generate a cited report as a durable Event artifact;
- index the result in Event memory with links to evidence and the Task;
- persist sourced company records and report artifacts for use through the R5
  Surface SDK rather than hardcoding a Career-OS product page.

Gate:

- the Career OS Task finds a defined set of current US startup opportunities;
- every reported company has inspectable source evidence;
- unsupported or inaccessible claims are marked rather than invented;
- the report survives app and worker restarts and appears under Event Files;
- the same sourced companies appear in a model-authored Career OS Surface
  without changing canonical artifact or evidence truth;
- cross-Event retrieval leakage remains zero.

### R7 — Career OS decision and receipt loop

**Branch:** `aloy-v1-r7-career-os-loop`

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

### R8 — reliability, context longevity, and release

**Branch:** `aloy-v1-r8-release-gate`

Scope:

- run the provider-success/database-crash reconciliation drill;
- add Task/Run watchdogs for expired leases and stuck work;
- verify time, step, tool-call, cost, and account-concurrency budgets;
- add summary compaction thresholds over the R4-paginated Conversation history;
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
- generated Surface projects pass build isolation, iframe escape, SDK schema,
  keyboard, accessibility, recovery, and responsive safety checks;
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
13. **Surface authoring:** independent generated applications render trusted
    Event data, meaningful intents reach the canonical Session exactly once,
    code/data revisions remain separate, escape attempts fail closed, and live
    updates do not steal focus or destroy local workspace state.

## 7. Explicitly deferred until after V1

- automatic Task selection and learned autonomy;
- scheduled and incoming-data triggers beyond the existing chassis;
- automatic or model-initiated promotion of Conversations into Events;
- automatic transfer of existing Life Tasks/files during Event creation;
- cross-Event Life coordination and retrieval;
- emergent Event detection;
- Auto/Notify routing;
- push notifications and learned attention budgets;
- arbitrary npm dependencies, full-stack generated services, direct generated
  network/provider access, user-installed Surface plugins, and unsandboxed
  model-generated code;
- Reality Objects beyond Documents/Accounts/Preferences;
- shared cross-user Events;
- unrestricted concurrent Runs per Event or account;
- desktop local-folder integration and native mobile clients.

## 8. Immediate next action

R4 is merged into `aloy-v1` as PR #172. The authoring-harness, virtual
filesystem, and Surface persistence slices are merged through PR #175. The
isolated build + diagnostics + preview-metadata slice is implemented on
`aloy-v1-r5-surface-build-preview`.

Complete R5.5 on `aloy-v1-event-setup`: draft context, scoped ingestion,
EventBrief readiness, Event-memory isolation and consolidation, and the first
evidence-grounded bootstrap Surface. Then create the University showcase
through that ordinary path, followed by the reviewed widget phase and Madrid.
Do not add provider-specific research or Gmail behavior to this phase.
