# Aloy V1 — reset delivery plan

_Active plan, revised 2026-07-19 after the Event bootstrap and scoped-memory review. This
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

The V1 proof is complete only when the same domain-neutral Event loop works for
University, travel, career, and future Events: understand context, perform
bounded work, present it in Conversation or a Surface, accept a user decision,
stage any consequence for approval, execute it once, and reconcile evidence
back into the Event without losing continuity.

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
16. Aloy is one user-facing assistant. Specialist model roles and legacy
    AgentConfig infrastructure are operator-owned, absent from customer
    navigation, and never user-selected when starting an ordinary Conversation.
17. A trusted Surface trigger contains an interaction ID, not model-trusted
    form contents. Aloy reads that interaction through an Event-scoped tool and
    treats its payload as user input, never as system instructions.
18. Accepted Surface work has one durable lifecycle visible to generated UI:
    queued, running, waiting for approval, executing, and a terminal outcome.
    A transport acknowledgement is not a completion claim.
19. Surface commands do not manufacture Tasks. State-only changes persist
    without a Run; explicit reasoning starts a Run; external consequences stage
    a Proposal. A canonical Task is created or changed only when genuine work
    with a useful definition of done exists.

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

**Model-native request handoff implemented on
`aloy-v1-surface-requests`:** an ordinary Event model may submit a structured
Surface experience brief based on request meaning and durable product value.
The host queues an idempotent `surface_builder` Run with the product-owned
profile and skill. That Run receives bounded Event context and the current
draft as trusted prompt input, then returns one complete structured candidate
with no model-visible tools. The host owns persistence, build, inspection, and
publication. Completion requires an exact live publication receipt; the
Conversation card reads only the published runtime.

R5 keeps one React App Surface runtime; there is no separate HTML/Surface Lite
lane in V1. Simple experiences are small React projects using the same SDK,
sandbox, publication, and quality contract as rich experiences. Delivery is
split into the following independently merged branches:

1. **`aloy-v1-r5-builder-control-plane`** - developer-owned Surface Builder and
   optional advisory Critic roles kept independent from ordinary Conversation
   AgentConfig; immutable assignment provenance; capability/policy checks;
   skill version; model usage, cost, and timing diagnostics. A Builder role
   must be explicitly configured and qualified before it can receive a Run.
2. **`aloy-v1-r5-host-build-pipeline`** - replace model-orchestrated file,
   build, preview, publish, and answer calls with one complete candidate
   submission through provider structured output and an empty model tool
   surface. The host persists, validates, builds, inspects, and publishes,
   returning structured repair diagnostics for one bounded full resubmission.
   Publication alone advances the live pointer atomically; failed candidates
   remain inspectable drafts/builds and never replace last-good. Long
   non-streaming generation writes a durable heartbeat, every host stage updates
   the Run, and the Conversation card plus Surface Workbench expose queued,
   generation, validation, compilation, inspection, repair, publication, and
   terminal failure before a build row exists. A provider response that fails
   candidate parsing retains its exact error, usage, length, hash, React-source
   signal, and bounded raw head/tail across retries without storing an
   unbounded model response in the Run row. Structured-output compatibility is
   a kernel provider contract rather than Surface-specific prompting: each
   provider declares its accepted JSON Schema dialect, prompt-envelope needs,
   strictness, and model-family request options. The frozen Builder assignment
   also carries a per-generation deadline independent from the larger durable
   Run budget. Schema-invalid or timed-out submissions fail closed instead of
   repeating the same expensive request through an unchanged model assignment.
   Provider parsing accepts only the candidate envelope shape; Aloy's host then
   applies authoritative path, file, size, manifest, and source policy. A
   shape-valid candidate that violates those rules receives deterministic
   diagnostics through the same bounded repair submission rather than failing
   before repair. Host-owned files such as `index.html` remain forbidden.
3. **`aloy-v1-r5-surface-command-runtime`** - introduce the versioned
   host-owned Surface command contract behind V1 compatibility. Define typed
   entities and explicit `create`/`replace`/`merge`/`delete` semantics;
   host-owned query and command hooks with pending/error/conflict/retry states;
   one canonical data projection shared by Surface and Event Conversation; an
   Event-scoped detailed-state read tool; and validated `local`, `state`,
   `reasoning`, `external_action`, `automation`, and `source_change` routing.
   Host-generated interaction tests, rather than model-authored checks alone,
   must cover success, rejection, stale revision, reconnect, empty, populated,
   partial, and permission-denied states. Migrate Career OS first and remove
   model-owned persistence wrappers before other showcase work continues.
   The first slice establishes command contract v1, strict state operations, a
   compatibility-only legacy dispatch path, fixed effect-to-wake mapping, the
   shared bounded Event-context projection, an Event-scoped detailed read
   tool, and snapshot-bound trusted reasoning triggers. The second slice makes
   generated controls use a host-owned `useSurfaceCommand()` lifecycle with
   duplicate-submit suppression, immutable exact-action retry, structured
   conflict and
   failure states, and accessible feedback metadata. The trusted host delivers
   refreshed canonical context before resolving a committed command, while the
   browser publication gate now rejects command paths that merely emit an SDK
   message without visibly rendering the acknowledged outcome. Existing
   immutable publications remain unchanged until rebuilt through the normal
   Builder pipeline. The remaining slice makes conflict/rejection attempts
   durable, enables governed automation and source-change routing, and migrates
   Career OS from its old generated wrappers to the new command lifecycle.
4. **`aloy-v1-r5-fast-build-runtime`** - run every candidate in a fresh,
   short-lived Surface Build Sandbox created from a pinned E2B template with
   the fixed React compiler, Aloy SDK, and browser gate already installed. The
   host uploads validated source, invokes one fixed compile-and-inspect command,
   retrieves immutable bundles, diagnostics, captures, and receipts, and then
   releases the sandbox. Record the template identity, limits, readiness,
   acquisition/compile/inspection timings, hashes, and termination reason. No
   per-Surface dependency install, public sandbox URL, sandbox credential,
   model-owned command, plugin, config, or HTML shell is allowed. Add
   content-addressed build reuse and remote cold/warm benchmarks; introduce a
   warm pool only if measured acquisition misses the target. Developer
   workstations may explicitly select the same fixed host-local compiler while
   remote sandboxes are unavailable, but local execution is forbidden as a
   hosted default. This slice does not create a long-lived sandbox per Event;
   the richer future Event Execution Workspace is a separate provider-neutral
   runtime with its own capabilities and security policy.
5. **`aloy-v1-r5-live-surface-ux`** - upgrade the durable polled Builder state
   from Phase 2 to Event SSE progress; add automatic published-Surface handoff,
   richer diagnostics, last-good continuity, immutable runtime preparation,
   and authenticated private cache.
6. **`aloy-v1-r5-surface-quality`** - deterministic authority/build checks,
   required viewport/state evidence, accessibility and overflow checks,
   primary-job simulation, bounded repair, feedback,
   pinning, revision history, and rollback.
7. **`aloy-v1-r5-university-proof`** - repeatable natural-language University
   generation without the word Surface, fake Tasks, or hardcoded domain logic;
   later revision such as a grade calculator; at least twenty benchmark prompts
   with routing, build, repair, latency, quality, and cost results.
8. **`aloy-v1-r5-widgets-madrid`** - trusted maps and other high-value widgets,
   then the Madrid showcase through the same ordinary Surface pipeline and
   protected action boundary.

Performance targets are evidence gates, not promises hidden in prose: an
existing published Surface should reopen under `500 ms` P50; warm sandbox
acquisition under `800 ms` P50; React compilation under `2 s` P50; and the
complete non-model candidate pipeline under `5 s` P50. The user receives a
real host-owned building state within one second even when frontier-model
generation takes longer.

The initial E2B Hobby test environment is sufficient for correctness and
latency exercises: its documented/current dashboard limits include twenty
concurrent sandboxes, twenty concurrent template builds, up to 8 vCPU and 8 GB
RAM, 10 GB sandbox disk, and a one-hour maximum continuous sandbox lifetime,
with pause/resume available and usage billed per second. The current account
also has the provider's one-time USD 100 credit. These are observed test
capacity, not Aloy architecture constants: deployment reads provider limits
from configuration and monitoring, and Event longevity never depends on an
individual sandbox lifetime. Reconfirm changing limits against the [E2B
documentation](https://e2b.dev/docs) before capacity or cost decisions.

Scope:

- implement Event-owned Surface Project, immutable source Revision, isolated
  Build, structured Interaction, and provenance-bearing Data Record concepts
  from [`aloy-surface-spec.md`](./aloy-surface-spec.md);
- define the restricted React project/manifest contract, version-locked import
  allowlist, and `@aloy/surface` SDK;
- add authenticated, optimistic, idempotent authoring tools for reading,
  patching, building, previewing, publishing, and rolling back Surface source;
- give ordinary Event Runs only a host-controlled Surface-request tool and
  execute accepted briefs through a separate purpose-scoped Builder Run;
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
- implement the brief, required viewport/state evidence, deterministic audit,
  primary user-job simulation, bounded repair, and
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
- a normal request such as creating a useful timetable can cause the model to
  queue a Surface without the user knowing Aloy's internal term or using a
  keyword, while a queued/draft-only build never appears as ready;
- timetable rows, itinerary items, navigation, and other presentation records
  do not become canonical Tasks unless they are independently actionable work;
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

**Branches:** `aloy-v1-event-setup`, `aloy-v1-event-context-ingestion`, then
`aloy-v1-event-context-pack`, then `aloy-v1-event-bootstrap`

**R5.5a shipped on the branch:** durable tenant/user-scoped setup drafts;
autosaved name, description, and mode; typed file, link, template, and existing
connection context; object-store staging; idempotent promotion into one Event
and canonical Conversation; Event-scoped knowledge/provenance, file-library
transfer, connection grants, and Trail; context ingestion failure cannot block
creation.

**R5.5b shipped on the branch:** promoted files and public links are durable
worker-leased ingestion jobs with visible state, bounded automatic and manual
retry, expired-lease recovery, and semantic Trail. Model-independent
extractors write provenance-, freshness-, sensitivity-, and retention-bearing
Event knowledge for text/HTML/JSON/CSV/XML/YAML/PDF/DOCX/XLSX sources. Public
link retrieval is size/time bounded and rejects private/local network targets,
unsafe redirects, and unsupported content. The Event Workbench shows live
source readiness and errors. Internal storage and knowledge ids remain hidden.
R5.5c implements content-addressed canonical Event context snapshots, safe
trusted-context prompt placement and cache boundaries, deterministic readiness,
Event-over-global conflict precedence, and typed/versioned/evidence-validated
Event Brief persistence. The Event Workbench exposes safe readiness status.

**R5.5d implemented on the branch:** sufficient trusted context queues one
idempotent, purpose-scoped `event_bootstrap` Run for an immutable snapshot.
The Run uses the developer-configured default model, a no-tool product profile,
bounded frozen evidence, structured `EventBriefPayload` output, tenant and
membership revalidation, evidence-reference validation, bounded retry, and
stale-snapshot replacement. Creation and ingestion trigger it automatically;
the Workbench exposes waiting, queued, running, ready, and failed states plus a
safe manual retry. Fake structured-model tests cover the flow without model
credits. Connection synchronization and bootstrap Surface/cover work remain.

**R5.5e implemented on `aloy-v1-event-memory-settings`:** the Event's own
settings, opened by the settings icon directly after Trail in the right Event
context dock, now inspect active Event memory separately from inherited global
memory. Memory is deliberately nested within settings instead of being a peer
operational tab.
Host-owned endpoints enforce tenant/user/Event scope for reads and mutations;
user corrections supersede rather than overwrite, forgetting soft-deletes,
and explicit promotion creates an idempotent provenance-linked global record.
Corrections and forgetting retire a potentially stale derived Event Brief,
refresh the immutable Event-context snapshot, and reuse the existing readiness
gate for any replacement bootstrap. All three actions write content-free
semantic Trail evidence, and another Event's record cannot be read or mutated
through the API. Inherited global records are read-only inside an Event and
canonical Event state remains outside the memory editor. Automatic
evidence-gated consolidation and richer global-memory management remain
separate work.

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

Implementation contract (R6):

- Pori owns the provider-neutral `web_search` and `read_web_page` contracts;
  Aloy supplies an Event-scoped evidence recorder and its SSRF-safe public-page
  reader through tool context. Search vendors never enter product records.
- Every successful observation carries URL, title, retrieval time, provider,
  content hash, and—inside an Event—a committed evidence ID before the model
  receives it. Repeated observations in one Run are idempotent.
- Raw fetched excerpts, canonical records, and report indexes remain durable in
  Event memory but are excluded from mutable automatic prompt hydration. Models
  read their compact projection on demand, preserving provenance, context
  budgets, and prompt-cache reuse.
- `event_record_upsert` is generic by namespace. `observed` and `inferred`
  records fail closed without same-user, same-organization, same-Event evidence;
  unsupported facts use `unverified`. Revisions supersede rather than erase.
- A Task explicitly freezes `execution_profile=sourced_research`; this choice is
  made semantically by Aloy's Task tool call, not by a title/keyword patch. Its
  immutable Run profile requires search, page reading, durable records, and a
  cited Markdown report.
- The worker completion gate rejects a research Run that lacks committed web
  evidence, at least one observed/inferred Event record grounded in that Run's
  evidence, a stored Markdown artifact, or a citation to an observed source
  URL. An `unverified` placeholder cannot satisfy the gate. Gate evaluation can
  recover evidence and records from durable Run provenance after a worker
  restart rather than depending on process-local collectors. Accepted reports
  are indexed with Task, Run, file, record, and evidence links.
- Surfaces declare `records:<namespace>` and read the host-owned projection with
  `useEventRecords(namespace)`. This is read-only evidence-backed truth and is
  separate from model-authored source and mutable `data:<namespace>` UI state.
- `/events/{event_id}/evidence` and `/events/{event_id}/records` provide bounded,
  tenant-scoped inspection. The Trail records evidence batches, record revisions,
  report indexing, and the Run's durable research quality-gate receipt. Generic
  memory views, reset, delete, search, export, and retention controls exclude
  canonical evidence, Event records, and report indexes.

Gate:

- the Career OS Task finds a defined set of current US startup opportunities;
- every reported company has inspectable source evidence;
- unsupported or inaccessible claims are marked rather than invented;
- the report survives app and worker restarts and appears under Event Files;
- the same sourced companies appear in a model-authored Career OS Surface
  without changing canonical artifact or evidence truth;
- cross-Event retrieval leakage remains zero.

### R7 — generic Event operating loop

**Branch:** `aloy-v1-r7-event-operating-loop`

Scope:

- let Aloy retrieve the exact accepted Surface interaction from its trusted
  interaction ID without placing untrusted generated-UI payloads in the system
  prompt;
- expose the durable interaction lifecycle through the Surface SDK so controls
  can render queued, running, approval, execution, completion, rejection,
  failure, cancellation, and indeterminate states from host truth;
- keep local presentation, durable state, reasoning, external action,
  automation, and source-change effects distinct;
- reconcile reasoning outcomes and Proposal receipts into the canonical Event
  Conversation, Surface projection, Today, and Trail;
- keep `automation` and `source_change` fail-closed until their separately
  governed Schedule and Builder routes are complete;
- exercise the same contract with University, Madrid, and Career acceptance
  cases; no domain name or payload shape may be compiled into the runtime.

Gate:

- a Surface reasoning command can retrieve its exact validated payload only
  inside its tenant, user, and Event scope;
- a Surface-triggered Run cannot succeed or retain artifacts unless its durable
  receipt proves that it read the exact originating interaction; that proof
  survives worker restart and checkpoint resume;
- generated UI follows the accepted interaction beyond the initial request and
  never presents `accepted` as `completed`;
- a state-only command persists once without waking a model;
- a reasoning command starts one canonical Event Run and records one outcome;
- an external action produces no provider call before approval, rejection
  produces none, and approval produces at most one call plus one receipt;
- Conversation, Surface, Today, and Trail converge on the same terminal state;
- the tests pass with domain-neutral fixtures and at least one acceptance case
  each from University, Madrid, and Career.

### R8 — reliability, context longevity, and release

**Branch:** `aloy-v1-r8-release-gate`

**First reliability slice implemented:** the worker now runs bounded,
row-lock-aware Run and Task watchdogs before claiming new work. An expired Run
with attempts remaining is reclaimed through the existing checkpoint-resume
path and records a recovery Trail entry. An expired final attempt or an
interrupted cancellation terminalizes once, clears its lease, reconciles its
Task, Conversation, Surface, Schedule, and Trail projections, and becomes
explicitly retryable where appropriate. A queued/in-progress Task whose Run is
missing, mismatched, or already terminal is repaired from host truth. Stale
provider executions remain `indeterminate` and are never submitted again
blindly.

**Second reliability slice implemented:** every Run producer resolves and
freezes host-owned step, tool-call, token, cost, and active-duration ceilings
before queueing; the worker re-clamps them under an organization row lock before
claiming. A single kernel budget ledger follows an ordinary Agent, every hidden
model call, Team coordination, members, and nested Teams, and restores its
usage after checkpoint resume. Active duration excludes time pending in the
queue or waiting on a user. Event bootstrap and Surface Builder specialist Runs
use the same contract. Exhaustion is terminal and non-retryable, with durable
usage, receipt, and Trail evidence; a failed Surface build retains the last
working publication. Unknown model pricing fails closed whenever a cost ceiling
is configured.

Token and cost truth arrives after provider calls complete, so one call, or
already in-flight parallel Team calls, can cross a ceiling; Aloy records the
actual overage and permits no new model or tool action. Exact pre-call spend
guarantees require a governed price catalog, conservative concurrent
reservations, and provider output limits derived from the remaining budget.
Those provider controls are a release-hardening follow-up, not a reason to
undercount actual usage.

**Third reliability slice implemented:** Conversation history now has a stable
host-owned token allowance independent of a model's advertised context size.
When that allowance is crossed, Pori rolls the already accepted summary and
the next contiguous transcript prefix into one replacement summary. Aloy stores
it as an immutable, versioned `ContextArtifact` with first/last message,
timestamps, covered count, and content fingerprint; only a gap-free prefix can
advance the boundary. Reopening hydrates the latest verified summary plus a
bounded current-Conversation tail. It no longer loads up to 5,000 Event messages
into every Run. `search_event_history` now page-faults a bounded candidate set
through an async tenant/user/Event-scoped database handler, so older or sibling
evidence remains available without automatic prompt injection. A fresh Life
Conversation consequently starts transcript-clean while accepted personal
memory still loads. Prompt caching remains an optimization over this stable
prefix, never the source of durable truth.

**Fourth reliability slice implemented:** provider execution and recovery are
now distinct rails. A tool may expose a typed, read-only reconciler that
survives capability snapshot filtering without exposing another model tool.
`indeterminate` Proposals receive durable inspection leases and bounded
exponential backoff; an unknown lookup never retries the write. Gmail sends are
correlated by a deterministic RFC822 Message-ID, while Calendar inserts use a
caller-chosen deterministic event ID. A provider success followed by simulated
database commit loss now resolves to a receipt-backed `committed` state through
provider lookup with exactly one send. Tools without proof remain visibly
`indeterminate`.

**Fifth release-readiness slice implemented:** the existing boot, product,
backend, frontend, architecture, operator, and 60-second Career OS guidance now
describes the durable Event/Surface system rather than the legacy Chat product.
The generated-Surface browser gate no longer races Chrome navigation: it waits
for the host-owned runtime document before transferring the secure
`MessageChannel`, and each accessible interaction check receives its own
bounded deadline. Pori now exports the shared budgeted-model wrapper through
its public front door, so Event bootstrap and Surface Builder no longer import
kernel internals. Kernel, backend, app, Python typing, and all three import
boundary contracts pass on the final code. Responsive/accessibility evidence
and real-provider University, Madrid, and Career acceptance remain explicit
manual gates; this slice does not claim them from static inspection.

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

### R9 - Surface quality and showcase foundation

**Branch:** `aloy-v1-r9-surface-quality`

R9 turns Surface quality from a trusted-pipeline convention into durable,
build-bound evidence. It then layers responsive evidence, reviewed building
blocks, primary-job simulation, and repeatable showcases onto that same
publication contract. R9 does not merge Aloy V1 to `main`; `aloy-v1` remains
the integration branch until the complete product acceptance gate passes.

Delivery slices:

1. **Exact-build quality receipt:** trusted preview records a fingerprinted
   receipt bound to the build, source revision, source checksum, and retained
   bundle. New publication fails closed without that receipt. Rollback remains
   available for a previously published legacy last-good build.
2. **Viewport and state evidence:** render the required wide, split, tablet,
   and mobile compositions plus applicable populated, loading, empty, stale,
   error, long-content, approval, and indeterminate states. Retain the five
   baseline captures plus compact state observations, overflow/focus/
   accessibility diagnostics, timings, and fingerprints as build evidence.
3. **Optional asynchronous visual review:** defer a vision-capable Critic until
   measured Builder quality demonstrates a need. If introduced, it receives
   only selected representative captures, remains advisory and asynchronous,
   and never becomes a V1 publication dependency or waiver mechanism.
4. **Primary-job simulation and bounded repair:** freeze the requested user
   jobs, execute their accessible pointer and keyboard paths, validate typed
   outcomes against Event truth, and allow only bounded complete-candidate
   repairs. Exhaustion retains the last-good publication.
5. **Reviewed SDK foundations and widgets:** provide calm responsive tokens,
   primitives, and a versioned host registry. Privileged widgets such as Map
   are host-owned adapters with explicit data, credential, attribution,
   privacy, fallback, and mobile contracts; generated code receives no direct
   provider or network authority.
6. **Generic showcase proofs:** create University first and Madrid second as
   installable onboarding/marketing seeds using ordinary Event context,
   Builder, SDK, quality, interaction, Proposal, and publication paths. No
   University, travel, or Career conditional may enter Aloy runtime code.

**Implemented viewport and state foundation:** the trusted local browser now renders
1440px wide, 640px split, 768px tablet, 390px mobile, and 360px narrow-mobile
compositions. It blocks page overflow, horizontally clipped controls, missing
main landmarks, unnamed controls, missing image alternatives, keyboard-
unreachable custom controls, duplicate ids, missing captures, and capture
storage failure. Capture hashes and deterministic DOM observations are bound
into the exact-build receipt; local PNGs are retained beside the immutable
bundle. The real runtime context now carries a versioned host-owned state for
each capability-scoped resource, and `useSurfaceResourceState` exposes the same
contract to generated React. The publication browser drives loading, empty,
stale, error, permission-denied, pending, and indeterminate states plus dense
long-content and approval-required scenarios through that public context at
wide and mobile sizes. Long content populates only ordinary capability-scoped
Event shapes. Approval uses a pending Proposal and `waiting_approval`
Interaction; `useSurfaceApprovalState` binds a visible summary while decision
controls remain host-owned. A visible primary region must bind the SDK state
and transition with it; inspection-only branches cannot satisfy the gate. Those
18 state observations retain compact trusted fingerprints while only the five
baseline viewport PNGs are stored. The browser also tabs through every visible enabled
control at all five baseline sizes, blocks unreachable controls, premature focus
cycles, and missing visible focus indicators, and records stronger 2px/3:1
outline evidence without falsely claiming the optional AAA criterion. Text is
measured against its effective solid backdrop in all five baseline and 18 state
compositions; normal text must reach 4.5:1 and large text 3:1, while an
unresolvable image/gradient backdrop fails closed. Small compact touch targets
are recorded for later scoring. Remote evidence transport remains a follow-up.

**Implemented primary-job simulation:** every ordinary Surface request now
freezes a versioned, fingerprinted job contract before the Builder starts. Job
ids and descriptions are copied into the Builder task and must match both the
complete candidate envelope and `surface.json`; a candidate that drops,
renames, reorders, or substitutes an easier job is rejected before source is
persisted. The manifest maps each job to bounded accessible click, fill, and
select steps plus host-observable assertions for named visible UI, exactly one
typed SDK request, committed capability-scoped Surface data, or approval state.
The trusted browser resets canonical Event context, executes every job against
the real compiled bundle, validates intent payload schemas and outcomes, and
binds per-job fingerprints and timings into exact-build quality policy `@5`.
Read-only jobs can prove a useful named view without synthetic interaction.
Interactive jobs must converge through the SDK; model-owned selectors, scripts,
and self-reported success cannot satisfy the gate. These deterministic proofs
run against handcrafted candidates without model credits; live Builder quality
still requires later provider-backed acceptance.

Gate:

- a build that was not inspected, whose receipt was altered, or whose source or
  bundle differs from the receipt cannot be newly published;
- every required viewport and applicable state has trusted retained evidence,
  and blocking accessibility, overflow, focus, runtime, or interaction findings
  prevent publication;
- primary user jobs succeed through accessible UI paths and converge with
  canonical Event state exactly once;
- failed repair, widget, or evidence infrastructure leaves the current
  last-good Surface live and exposes an actionable diagnostic;
- University and Madrid pass through the same domain-neutral contracts and
  remain useful at mobile, split-pane, and wide-desktop widths.

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
- richer trigger types and incoming-data automation beyond Event Schedules;
- automatic or model-initiated promotion of Conversations into Events;
- automatic transfer of existing Life Tasks/files during Event creation;
- cross-Event Life coordination and retrieval;
- emergent Event detection;
- Auto/Notify routing;
- push notifications and learned attention budgets;
- authenticated browser connections, user login/takeover, private-account
  monitoring, adaptive browser control, and browser-submitted external actions;
- arbitrary npm dependencies, full-stack generated services, direct generated
  network/provider access, user-installed Surface plugins, and unsandboxed
  model-generated code;
- Reality Objects beyond Documents/Accounts/Preferences;
- shared cross-user Events;
- unrestricted concurrent Runs per Event or account;
- desktop local-folder integration and native mobile clients.

### Post-V1 browser delivery track

Authenticated browser work is deliberately outside the current R0-R8 release
gate. Its architecture and provider research live in
[`aloy-browser-agent-spec.md`](./aloy-browser-agent-spec.md). It may be promoted
into an active release only through an explicit scope decision; current Surface
work must not absorb browser credentials or general web authority.

The planned sequence is:

1. **B0 — contracts and fake provider:** provider-neutral browser Session,
   Context, evidence, action, and error contracts; durable connection/grant/Run
   state; injection, concurrency, and crash tests.
2. **B1 — Browserbase connection and read-only Session:** Context creation,
   exclusive connection leases, backend-mediated Live View, login/takeover,
   reauthentication, and one allowlisted deterministic read canary.
3. **B2 — research ladder and authenticated observations:** API/Search/Fetch
   routing before browser allocation, typed Stagehand atomic operations,
   read-only monitoring, evidence, and Today attention.
4. **B3 — adaptive reliability:** reviewed recipes, page fingerprints, repair
   limits, checkpoints, reconnect, provider queues, budgets, privacy, proxy,
   and prompt-injection gates.
5. **B4 — staged external actions:** Proposal-bound stage/execute/reconcile
   flow, frozen action fingerprints, and one reversible non-payment write
   canary with effectively-once drills.
6. **B5 — hardened release:** controlled file transfer, quarantine, retention,
   multi-site evals, and measured reliability, latency, cost, and support gates.

Browserbase is the first provider, but Aloy owns the agent loop. The hosted
Browserbase Agents abstraction is not the canonical executor because it owns a
parallel model/tool loop and currently couples browser access with fixed shell
and filesystem tools. The initial path uses Browserbase Sessions plus
deterministic Playwright and bounded Stagehand operations behind Pori contracts.

## 8. Immediate next action

R8 is merged into `aloy-v1` as PR #198 and the initial R9 quality-state slice is
merged as PR #199. R9 primary-job simulation is active on
`aloy-v1-r9-primary-job-simulation`. Exact-build receipts, the five-viewport
baseline, the public lifecycle, long-content and approval state matrix,
keyboard-focus traversal, deterministic text contrast, frozen requested jobs,
semantic browser workflow execution, and build-bound job evidence are complete.
A visual Critic is deferred and optional, not a publication gate. Remote
evidence transport and reviewed SDK primitives/widgets follow; University and
Madrid are proofs of the ordinary pipeline, not special runtime behavior. Keep
`main` untouched until the R9 gate, manual product QA, live-provider proofs,
remote sandbox acceptance, and dogfooding exit criteria all pass.
