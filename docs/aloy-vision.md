# Aloy — product vision

_Canonical product definition, version 3.3, revised 2026-07-17. This document
defines what Aloy is, how its core product concepts fit together, and what V1
must prove. Detailed contracts live in the linked child specifications; live
implementation status lives in [`.agent/progress/current.md`](../.agent/progress/current.md)._

## 1. Thesis

**Aloy is a persistent workspace for meaningful work and life events, where
agents make durable progress and people remain in control of consequences.**

Most assistants reduce work to a transient exchange:

```text
conversation → response → forgotten scrollback
```

Aloy turns an intention into a durable operating loop:

```text
intention → Event → Task → agent work → evidence → useful state → next action
                              └→ Proposal → decision → receipt
```

Conversation remains central, but it is no longer the only place where work
exists. Tasks, files, artifacts, decisions, memory, evidence, and history belong
to a durable Event. They survive individual model Runs, app restarts, worker
restarts, and context-window limits.

The division of responsibility is deliberate:

- the user defines intent, starts work, sets priorities, supplies judgment,
  and controls consequential actions;
- Aloy plans, researches, drafts, organizes, tracks, and executes permitted
  work within explicit limits;
- the system preserves truth, provenance, consent, recovery, and isolation even
  when a model is wrong or a process crashes.

V1 begins with explicit initiation and bounded execution. Proactivity and
autonomy are capabilities the user grants later; they are not defaults Aloy
quietly assumes.

> **Life is the permanent user–Aloy space. A dedicated Event is the durable
> home for a meaningful outcome. Tasks turn intention into executable work.
> Runs do the temporary computation.**

## 2. The product model

```text
USER ↔ ALOY
       │
       ├── LIFE — permanent personal space
       │     ├── Conversation A
       │     ├── Conversation B
       │     └── loose Tasks, files, memory, and Trail
       │
       ├── TODAY — attention across Life and dedicated Events
       │
       └── DEDICATED EVENT
             durable identity, outcome, context, files, and policy
             ├── one canonical continuous Conversation
             ├── Tasks, Surface, Trail, and Triggers
             └── RUN — bounded agent execution
                   ├── working state and artifacts
                   └── Proposal → decision → executor → receipt
```

Three identities must never be collapsed:

| identity | meaning | lifetime |
|---|---|---|
| **Event** | durable product aggregate and source-of-context boundary | until archived or deleted |
| **Conversation** | user-visible dialogue and instruction history | until archived or deleted |
| **Run** | one bounded attempt to reason or perform work | seconds, minutes, or hours |

“Session” describes a product role, not a fourth aggregate. A dedicated
Event's canonical Conversation is its continuous Session. In Life, every
user-started Conversation is a separate Session over shared accepted personal
state.

The product has two equally important entry points:

```text
New conversation → fresh Life Conversation

New Event → explicit setup → dedicated Event + its lifetime Conversation
```

A conversation does not automatically become a dedicated Event. An Event is a
deliberate commitment to durable context and an outcome. Aloy may suggest that
a Life Conversation deserves an Event later, but it never silently moves or
reclassifies the user's work. The user always reviews the setup and explicitly
creates the Event.

## 3. The primitives

### 3.1 Event — the durable home

An Event represents something with a future: _Career OS, University, Building
Aloy, Trip to Madrid, Weekly Review_. It owns the context required to make
progress over time:

- identity, goal, lifecycle, phase, and summary;
- the canonical Conversation for a dedicated Event;
- Tasks and their execution histories;
- working state, uploads, files, and artifacts;
- pending and resolved Proposals and their receipts;
- Event memory, evidence, and connected-service references;
- the semantic Trail;
- triggers, limits, and autonomy settings;
- one model-authored Event Surface and its revision history.

An Event is not a page, a folder of chats, or a permanently running model. The
UI is a lens over it. The Event remains authoritative when its screen is closed
and when no agent process is active.

The first-use product explanation is benefit-led rather than architectural:

> **An Event is a dedicated, ongoing space where Aloy helps you manage
> something important over time.**

The practical test is whether the subject will still matter tomorrow and Aloy
should keep track without the user explaining it again. A question such as
“Explain recursion” belongs in a Life Conversation; “Help me succeed this
university semester” can justify a University Event. In Aloy, Event therefore
means an ongoing goal, responsibility, journey, or life chapter—not merely a
calendar appointment.

The lifecycle is intentionally small:

```text
Emerging → Active → Dormant → Concluded → Archived
```

- **Emerging:** a possible Event awaiting user acceptance; deferred beyond V1.
- **Active:** user-started work and explicitly configured triggers may run.
- **Dormant:** state is retained while proactive work and triggers are paused.
- **Concluded:** the intended outcome is complete and may later be reopened.
- **Archived:** hidden from normal navigation and operationally inactive.

Users control lifecycle changes. Aloy may recommend a change, but V1 never
silently makes an Event dormant, concluded, or archived. Manual Event creation
is a direct user action, not a Proposal.

### 3.2 Life — the permanent personal space

Life is the system-created personal Event representing the user's ongoing
relationship with Aloy. It is neither the user nor the agent. It is the place
for work that does not yet deserve a dedicated Event.

Life provides:

1. unstructured capture and ordinary conversation;
2. multiple fresh Conversation threads without Event clutter;
3. one-off or loose Tasks;
4. shared accepted personal memory and preferences;
5. files and artifacts that belong to personal work;
6. an explicit path into a new or existing dedicated Event;
7. the personal foundation for Today.

**New conversation** creates a new Conversation inside Life. Life Conversations
do not appear as Events in the Event rail. Deleting one removes its messages,
not Life-owned Tasks, files, artifacts, receipts, Trail, or accepted memory.

A new Life Conversation receives relevant accepted Life state, but it does not
receive every older transcript. Older conversations are retrieved explicitly
when they are relevant. Life must remain useful even when the user creates no
dedicated Events.

#### Life product experience

Life is presented as a first-class personal space, not as a generic “Chat”
destination and not as another card in the dedicated Event list. The global
navigation names it **Life**. **New conversation** always creates a fresh Life
thread, while **New Event** remains a visually and conceptually distinct action
that creates a durable workspace.

Today keeps Life close through a compact personal-space band: the user can
continue a recent Life Conversation, start a fresh one, or capture a loose
thought without pretending Life is an attention-ranked Event. Opening Life
shows its Conversation canvas and thread switcher while Life-owned Tasks,
files, artifacts, memory, receipts, and Trail remain durable outside any one
thread. Life may gain a user-shaped Surface later, but V1 does not force it into
a fixed personal dashboard.

### 3.3 Conversations and continuous Event Sessions

A dedicated Event has exactly one canonical user-facing Conversation for its
lifetime. Opening the Event always resumes that Conversation, like reopening a
long-running coding project. The transcript grows, becomes pageable, and is
summarized when necessary, but the product does not ask the user to select a
new Session inside the Event.

The Conversation persists; a model process does not. Every user turn or
meaningful interaction starts a finite Run, which may finish, block, or hand
work to the durable worker. Closing the app ends neither the Conversation nor
properly queued background work.

Life is intentionally different: a user may create many Conversations for
clean topical starts. A Run loads only:

1. bounded recent messages from the active Conversation;
2. accepted global and owning-Event memory;
3. relevant Tasks, files, decisions, evidence, and working state;
4. older Event history through explicit scoped retrieval when needed.

Legacy, branch, gateway, or transport Conversation rows may remain for
provenance, but a dedicated Event never presents them as competing Sessions.

Background Task progress, clarification, and results return to the Conversation
selected for that Run. Dedicated Event Tasks use the canonical Conversation;
Life Tasks use the originating or explicitly selected Life Conversation.

### 3.4 Task — durable executable work

A Task is not a stale checklist item and not a sentence the model once wrote.
It is the durable contract that turns an intention into bounded work.

A Task carries:

- title, instructions, and definition of done;
- status, priority, and optional due date;
- execution mode and assigned agent profile;
- owning Event and originating Conversation provenance;
- current Run and execution history;
- progress, result summary, blocker, and produced artifacts;
- limits for steps, time, tools, and cost.

The V1 state machine is:

```text
open → queued → in_progress
                  ├→ blocked
                  ├→ waiting_approval
                  ├→ failed
                  ├→ cancelled
                  └→ done
```

Creating a Task does not silently start it. V1 begins execution through an
explicit **Work on this** action. A later schedule, incoming-data trigger, or
user-granted automation policy may start eligible work, but the wake reason is
always durable and visible.

One Task execution creates one Run. The worker claims it with a lease,
checkpoints meaningful progress, and supports Stop, Retry, and Resume.
Clarification moves work to `blocked`; a pending external consequence moves it
to `waiting_approval`; completion requires the definition of done and resulting
evidence, not an unsupported “done” in model prose.

Results appear consistently in four places:

1. a compact update in the Run's selected Conversation;
2. a durable Task result and status;
3. Event artifacts when files were produced;
4. an expandable Trail narrative linked to the Run and evidence.

### 3.5 Run — bounded execution

A Run is one finite attempt to answer a Conversation turn, execute a Task,
react to a trigger, reconcile a Proposal, or perform a follow-up. It is
budgeted, checkpointed, observable, cancellable, and recoverable.

Runs may observe, compute, organize, and stage inside the Event without
approval. They may use sub-agents, tools, skills, and an isolated sandbox, but
the parent Run owns synthesis, validation, budget, and final outcome.

A Run is never the durable home of user work. Useful results are promoted into
Event state, artifacts, memory, Proposals, or Trail before scratch state can be
discarded.

### 3.6 Event Surface — the model-authored Event application

The Event Surface is a durable, live application authored and evolved by Aloy
for one Event. It is the visual operating layer over trusted Event state—not
the Aloy app itself, not a fixed dashboard, and not a predefined collection of
Tasks, Files, or Trail blocks.

> **Event data is permanent truth. A Surface is a versioned, replaceable
> application Aloy builds over that truth.**

A University Event may become a timetable and assessment workspace. A Madrid
trip may become a map, flight, visa, hotel, budget, and itinerary application.
Career OS may become a sourced company-research and opportunity-tracking tool.
These Surfaces share a runtime and safety contract, not a page template.

#### A real application, not one screen

A Surface may have the complete information architecture its Event requires:
local navigation, tabs and sub-tabs, routes, sticky headers, scrollable views,
search, filters, forms, tables, boards, calendars, timelines, maps, charts,
galleries, comparison tools, and responsive layouts. It must include useful
loading, empty, error, stale, and unavailable states. The model may author
custom React components when trusted primitives are insufficient.

There are two deliberately separate navigation levels:

1. **Aloy workspace navigation** is trusted host chrome. It controls
   Conversation, Workbench items, opened files and artifacts, Run Replay, and
   Event context.
2. **Surface navigation** is generated for the Event. A Madrid Surface may
   contain Overview, Flights, Stay, Match, Itinerary, Budget, and Documents. A
   University Surface may contain Today, Timetable, Courses, Assignments,
   Exams, Study, and Documents, with course-level sub-navigation beneath it.

The Surface fills its Workbench pane. Ordinarily it owns one primary scrolling
content region while important local navigation may remain sticky. It must
adapt when Conversation, Event context, or a file opens beside it. On narrow
screens its sections become suitable full-screen views or compact navigation.
Normal Surface navigation never requires a popup window.

The iframe is a security boundary, not a visual boundary. A Surface should
inherit Aloy's design tokens, typography, themes, spacing, motion, and
accessibility behavior; fill its pane without foreign-looking borders; and
resize without losing its selected section, filters, local input, or scroll.

#### When a Surface is warranted

Aloy looks for a **Surface opportunity**, not a keyword. A Surface becomes
valuable when several of these conditions are true:

- the outcome will continue for days, weeks, or months;
- several changing records or dependencies must be tracked;
- the user needs comparisons, schedules, maps, timelines, documents, or
  progress at a glance;
- repeated decisions or recurring work are expected;
- current truth is difficult to recover from Conversation alone;
- the user repeatedly asks where things stand or what needs attention.

A simple explanation or one-off answer should remain in Conversation. A
semester, international trip, job campaign, relocation, thesis, renovation, or
business launch usually benefits from a Surface. The user may request one
explicitly, and Aloy may propose one when the opportunity is clear; the user
does not need to know the internal term before it is useful.

Before generating code, Aloy produces a structured Surface brief containing:

- the Event's primary user job and current phase;
- the questions the Surface must answer immediately;
- its views and information hierarchy;
- canonical entities, evidence posture, and data requirements;
- local interactions, durable intents, reasoning requests, and protected
  actions;
- required trusted widgets and host services;
- responsive, accessibility, loading, empty, stale, and failure states.

This brief is the product contract for the candidate. It prevents an attractive
but unhelpful dashboard from passing merely because it compiles.

The ordinary Event model makes this judgment during the same Conversation turn;
there is no keyword trigger, regex patch, or second intent-classifier call. When
the request would be better served by a durable interactive experience, that
model submits the structured brief through a host-controlled Surface-request
tool. The tool records the decision and queues a separate purpose-scoped Surface
Builder Run. The conversation model never receives source, build, preview, or
publication tools, and a queued request is never presented as a finished
Surface.

The dedicated Builder receives the accepted brief, a bounded read-only
projection of Event truth and relevant text artifacts, the current Surface
draft, and the exact Surface Builder skill. It receives **no model-visible
tools**. It returns one schema-validated complete source candidate through the
provider's structured-output contract. A Surface is **ready** only when the
trusted host persists, validates, builds, inspects, publishes, and verifies that
this exact Run owns the current live publication receipt. A generated candidate
or a model claim is insufficient. Failed attempts retain the last-good
published Surface.

Surface generation is a product-owned specialist role. Aloy's developers and
operators choose the Builder model, its versioned skill, budgets, and the
independent Critic model; end users do not select these components. The
Conversation model may be optimized for latency and dialogue while the Builder
is allocated to a model proven to author and repair constrained React projects.
Changing either model is an operator configuration change, not University,
travel, or other domain logic inside Aloy. A model is promoted into the Builder
role only after it passes the Surface structured-generation, build, repair,
interaction, and quality evaluation suite.

Surface structure is not Task structure. Timetable entries, itinerary rows,
dashboard sections, navigation items, map markers, and similar display records
remain Event/Surface data. Aloy creates a canonical Task only for genuine
actionable work with a useful definition of done; it must not manufacture Tasks
as a way to assemble UI.

#### Construction and authority

The Surface is real React and CSS authored by the model within a constrained
project. Aloy may create components, layouts, forms, filters, local state, and
Event-specific interactions. Requests such as “add a grade calculator” change
the Surface code. Updates such as “the exam is on Friday” change Event data.
Code revision, data revision, and local presentation state remain separate.

V1 has one **App Surface** runtime. A simple timetable, summary, checklist, or
dashboard is a small React Surface; a sophisticated workspace uses more React
components and trusted SDK widgets. Aloy does not maintain a parallel raw-HTML
or "Surface Lite" runtime. Two runtimes would duplicate the SDK, sandbox,
security, accessibility, quality, and migration contracts before measurements
show that React compilation is a material bottleneck. If production telemetry
later disproves this decision, a declarative rendering mode may be reconsidered
without changing the Event or publication model.

Most Surface interactions do not call a model:

- filtering, sorting, opening, and changing local tabs stay local;
- durable selections are validated and persisted once;
- requests for reasoning enter the canonical Event Conversation as structured
  turns and start a Run;
- booking, sending, paying, publishing, or deleting stages a Proposal;
- requests for new UI capability start the source/build/quality/publish loop.

#### Canonical Surface state and command routing

The model designs the experience; Aloy's host owns state, commands,
persistence, authority, and verification. Generated React never makes its DOM,
component state, or iframe storage the source of truth. It renders a scoped
projection of canonical Event state and sends typed commands through the
host-owned SDK:

```text
generated React
-> typed host SDK command
-> schema, permission, revision, and idempotency validation
-> host-owned command executor
-> canonical Event state plus append-only interaction evidence
-> reactive Surface projection and Event Conversation context
```

The versioned Surface contract declares entities, payload schemas, and explicit
write semantics such as `create`, `replace`, `merge`, and `delete`. Generated
code may choose a button's presentation and placement, but it does not invent
persistence behavior, reducers, retries, completion claims, or approval
authority. Host-owned command hooks expose pending, committed, failed,
conflicted, and retryable states and reconcile the interface from canonical
data. V1 contracts remain readable while the stricter command contract is
introduced and existing Surfaces are migrated deliberately.

Aloy accesses Surface changes through the Event, not by inspecting the iframe.
Every committed command advances `data_revision`, records actor, provenance,
and semantic Trail evidence, and invalidates the old Event context snapshot.
The next Event Run receives a bounded Surface-state projection automatically;
larger or detailed state remains available through a tenant- and Event-scoped
read tool. Surface and Conversation therefore answer from the same snapshot,
while prompt context stays bounded and cacheable.

Every meaningful control declares one host-validated effect and wake policy:

- `local`: tabs, filters, sorting, disclosure, and temporary form input remain
  inside the Surface and produce no durable interaction;
- `state`: save, select, move, annotate, or mark changes canonical Event state
  without starting a model Run;
- `reasoning`: review, compare, explain, plan, or prepare starts an immediate
  Run in the Event's canonical Conversation with selected entity references and
  the exact committed `data_revision`;
- `external_action`: send, book, pay, publish, or delete stages a Proposal and
  follows approval, execution, receipt, and reconciliation rails;
- `automation`: a user-enabled schedule or approved incoming-data rule creates
  a durable background wake rather than an implicit model call on every edit;
- `source_change`: requests for a new view or capability queue the separate
  Surface Builder lifecycle.

Generated code may propose which declared command a control needs, but the host
determines whether that command is allowed, whether it may wake Aloy, and which
approval policy applies. A normal card move never starts a model. A deliberate
**Review my pipeline** action does. A payment receipt may trigger a configured
follow-up, but only through an explicit Event automation. Immediate and
background wakes carry a trusted host-rendered envelope containing Event,
command, selected entity references, state revision, and snapshot fingerprint;
Surface payloads remain structured data and cannot inject hidden system
instructions.

Trigger execution is idempotent, rate-limited, depth-limited, and deduplicated.
An agent-originated state change cannot recursively wake itself without an
explicit workflow edge. Runtime and publication gates are generated from the
host contract across success, rejection, stale revision, reconnect, empty,
populated, partial, and permission-denied states; model-authored UX checks may
add coverage but are never the sole authority for correctness.

Generated code is untrusted. It executes in an opaque-origin sandboxed iframe,
has no host credentials or direct authenticated API access, and has no arbitrary
network, package, storage, navigation, or device authority. A capability-scoped
`@aloy/surface` SDK supplies validated Event reads, reactive updates,
structured intents, and host-owned privileged widgets such as maps, approvals,
file viewers, and credential collection.

Generated code chooses how trusted widgets are composed into the Event
experience. The host retains credentials, provider calls, permanent files,
authentication, authorization, payments, approvals, receipts, and device
access. This allows a model-authored map, calendar, document workflow, or
payment-preparation flow to feel native without granting generated code the
authority behind it.

The Surface may present canonical records, but it cannot redefine them. A
selected hotel is not a booking; “I paid” is a user report; a provider action
is pending until a receipt exists; a crash-window uncertainty is
`indeterminate`, never confidently committed.

#### Build, quality, and publication lifecycle

Every Surface follows a controlled lifecycle:

```text
opportunity or request
→ Surface brief
→ isolated candidate build
→ deterministic checks
→ viewport and state renders
→ independent critique
→ primary user-job simulation
→ bounded repair
→ publish
→ monitor and improve
```

The Builder submits one complete candidate revision. It does not orchestrate
the mechanical lifecycle by calling persist, build, preview, publish, and
answer tools in sequence. The trusted host owns that sequence atomically,
returns structured diagnostics when repair is required, and grants at most a
bounded number of candidate resubmissions. This keeps model behavior focused on
authoring while publication correctness remains deterministic.

The Builder path is therefore not an agent tool loop. It is a bounded
structured-generation loop: candidate, trusted host diagnostics, and at most
one complete repair candidate in V1. Aloy records each candidate fingerprint
and stage receipt; the model cannot directly mutate a draft, launch a compiler,
or advance the live publication pointer.

Structured generation may take minutes and does not stream partial source.
That must never look like inactivity. The Builder writes a durable heartbeat
while waiting for the complete candidate, and every host-owned transition
updates the Run: queued, generating, validating, compiling, inspecting,
repairing, publishing, ready, failed, or overdue. Conversation shows a compact
activity card after the assistant queues the work; opening it reveals the same
state in the Surface Workbench with elapsed time and bounded retry count. The
user may keep talking while this background work continues.

A provider-level success is not a valid candidate until schema validation
succeeds. Rejected output remains diagnosable: Aloy records the exact parser
error, token usage, response length and hash, whether React-like source was
present, and a bounded head/tail excerpt. Retry attempts preserve earlier
rejection receipts. Raw failure evidence is bounded and owner-scoped; it is not
silently discarded or mistaken for a successful build.

Provider compatibility belongs below Aloy in the Pori model layer. A
structured-output policy declares the provider's supported JSON Schema
dialect, whether the schema must also be present in the prompt, strictness, and
model-family request controls. Product schemas remain complete Pydantic
contracts and host validation remains unchanged; only the request schema is
adapted to the selected provider. The immutable Builder assignment also freezes
a short per-generation deadline. That deadline is separate from the durable
Run's wider pipeline budget, so one opaque model call cannot consume the full
build lifecycle. Invalid structured output and generation timeout do not cause
blind identical retries; failover requires an explicitly configured and
qualified alternate Builder assignment.

Provider shape validation and host authority validation are separate gates.
The provider must return a complete candidate envelope with summary, primary
jobs, and files. Aloy then validates paths, file types, sizes, manifests, SDK
authority, and compiler ownership. A valid envelope containing a forbidden
file such as `index.html` becomes trusted repair feedback and may consume the
single bounded repair submission; the file is never accepted or published.

Compilation happens only when source changes, never when an Event or published
Surface is reopened. The fixed toolchain contains React, the Surface SDK,
approved design assets, compiler, preview browser, and quality tools; generated
projects never install packages. Warm isolated builders, content-addressed
builds, immutable runtime documents, and private authenticated caching target
sub-second reopening and a small non-model build budget. Aloy records routing,
model generation, sandbox acquisition, validation, compilation, preview,
quality, storage, and publication timings independently so optimization follows
evidence rather than assuming React is slow.

Deterministic gates validate the manifest, types, imports, build, bundle
limits, capability declarations, and forbidden authority. Visual gates render
the candidate at required desktop, split-pane, tablet, mobile, light, and dark
states and check overflow, hierarchy, contrast, responsiveness, and failure
behavior. Interaction gates simulate the Event's real jobs: selecting,
filtering, updating durable data, asking Aloy, opening files, requesting an
action, approving or rejecting it, receiving failure, and restoring after a
reload.

Surface quality is engineered rather than requested with the word “beautiful.”
Every publish candidate is built in isolation, rendered at required viewports
and states, checked deterministically, critiqued independently, exercised
against the Event's primary user jobs, and repaired within bounded limits. A
failed or weak candidate never replaces the last-good revision. The Surface
Builder skill teaches the model how to work, but schemas, tools, sandboxing,
CSP, host bridges, and the publish service enforce the boundary.

Publishing is versioned and risk-aware. A safe visual repair or read-only view
may publish automatically after passing the gate. A major navigation change
should be explained or previewed. A newly declared external, financial,
destructive, or permission-changing capability always remains subject to the
host's policy and approval boundary. The user can inspect revisions, request a
redesign, prefer an earlier layout, or roll back. A failed build, runtime crash,
or exhausted repair attempt leaves the last-good revision in place.

#### Persistence and evolution

An Event may live for years; a browser process, model Run, and iframe do not.
The server persists Event truth, Surface data, published revision, build
history, interactions, and per-user workspace preferences. Reopening the Event
loads the last-good Surface over current Event state and restores its useful
local context.

Aloy improves a Surface only when there is evidence that the current
application no longer serves the Event well, for example:

- the Event entered a new phase;
- new canonical data no longer fits the current information architecture;
- the user repeatedly asks for buried information or manual workarounds;
- an important view is unused or a primary job repeatedly fails;
- a trusted capability such as a map or calendar becomes relevant;
- the user explicitly requests a new view, calculator, workflow, or redesign;
- quality monitoring detects a runtime, responsive, accessibility, or stale-
  data failure.

Aloy prefers a focused patch when the application remains sound and a larger
redesign when the Event's phase or user jobs have materially changed. It does
not randomly rearrange a familiar workspace. A University Surface may evolve
from timetable and course capture, to assignment management, to exam revision,
to results and next-semester planning while preserving one Event, one
continuous Conversation, and the same durable academic truth.

The complete contract is defined in
[`aloy-surface-spec.md`](./aloy-surface-spec.md).

### 3.7 Trail — the durable narrative

The Trail explains what happened, why Aloy acted, what changed, and what
evidence exists. It belongs to the Event and continues across Conversations,
Runs, workers, app restarts, and Surface revisions.

Trail records semantic activity:

- Event lifecycle, phase, trigger, and important memory changes;
- Task creation, execution milestones, blockers, completion, and failure;
- Run start and terminal outcome;
- artifact creation and promotion;
- meaningful Surface interactions and published revisions;
- Proposal, decision, execution, receipt, and reconciliation transitions.

Conversation messages are not duplicated into Trail. Low-level model and tool
events remain in Run Replay. Presentation-only actions such as opening a pane,
sorting a table, or resizing the workspace create no Trail noise.

Trail is append-only. Corrections append new entries. Each Task execution forms
an expandable narrative linked to its Conversation message, Run Replay,
artifacts, Proposal, receipt, and evidence.

### 3.8 Proposal — consent before consequence

Agents may freely research, compare, simulate, draft, and stage within an
Event. Crossing into external or shared reality requires a durable Proposal:

```text
staged intent → Proposal → routing → decision → executor → receipt
```

Always-Ask V1 actions include sending messages, spending money, booking,
publishing, deleting important external data, and changing permissions.
Approval never executes unvalidated prose. The Proposal stores the normalized
tool and arguments; a non-agent executor rechecks authorization, policy,
credentials, schema, Event lifecycle, and idempotency before calling the
provider.

A Proposal is not committed until evidence exists. If a provider may have
accepted an action before Aloy stored the receipt, the outcome becomes
`indeterminate` and is reconciled rather than blindly retried.

Future Auto and Notify behavior uses this same mechanism with explicit
pre-granted consent. Autonomy never bypasses the Proposal rail or fixed safety
policy.

### 3.9 Files, artifacts, and memory

Durable bytes belong to Life or a dedicated Event—not to a transient Run and
not merely to a Conversation. Aloy presents three product categories:

- **Uploads:** material supplied by the user or a trusted connection;
- **Working files:** evolving drafts and intermediate durable state;
- **Outputs/artifacts:** promoted results with Run, Task, and evidence
  provenance.

These categories are product semantics, not fragile physical folders. A
sandbox scratch file becomes an artifact only when the finalizer stores it
durably, records provenance, and adds the semantic Trail entry. Files can link
to Tasks, Runs, Proposals, receipts, and Surface records.

Memory is a curated index over durable truth, not a copy of every message and
not a substitute for canonical state. Aloy has four distinct context layers:

1. **Global user memory:** stable identity, preferences, and constraints that
   may be useful across Life and dedicated Events;
2. **Event memory:** accepted facts, decisions, routines, learned terminology,
   important episodes, unresolved questions, and Event-specific preferences;
3. **Canonical Event state:** Tasks, Proposals, receipts, files, connections,
   Trail, Surface data, and lifecycle truth loaded directly rather than
   remembered approximately;
4. **Transcript history:** durable messages and summaries retrieved on demand,
   not automatically promoted into accepted memory.

Event memory is isolated by tenant, user, and Event before ranking or semantic
retrieval. A Run inside one Event may receive permitted global memory plus the
owning Event's memory; it never retrieves another Event's memory. Life follows
the same rule and cannot silently read dedicated-Event memory. Cross-Event
retrieval remains prohibited until a future explicit sharing mechanism proves
zero-leakage isolation.

Memory has two independent scope dimensions:

```text
ownership:    personal > team > organisation
situation:    Event > global
```

Within an Event, Event-specific accepted memory wins over conflicting global
memory. Personal knowledge wins over team or organisation knowledge at the
same situational scope. Promotion from Event memory into global memory is an
explicit, inspectable policy decision; useful facts do not leak merely because
the model observed them inside an Event.

Every memory record carries kind (`semantic`, `episodic`, or `procedural`),
scope, confidence, sensitivity, retention, provenance, and optional conflict
identity. Corrections supersede earlier records without destroying their
history. Source evidence, committed provider state, receipts, and current
canonical Event rows always outrank model memory.

On each Run, Aloy assembles a bounded context from global memory, relevant
Event memory, current canonical Event state, recent Conversation turns,
retrieved older Event history, and compaction summaries. The permanent Event
Conversation may therefore continue for months without placing its complete
transcript in every model prompt.

The host materializes the stable portion as an immutable, content-addressed
`EventContextSnapshot`. It contains Event identity, canonical state, context
readiness, evidence references, and the active typed Event Brief; raw evidence
bodies remain available through scoped retrieval rather than being copied into
every prompt. The snapshot is injected as trusted reference data before
transcript history, while the current user task remains the final instruction.
An unchanged fingerprint may reuse a provider prompt-cache prefix; any
canonical change creates a new snapshot and naturally invalidates that prefix.
Confidential or restricted evidence disables application-controlled message
prefix caching. Prompt caching is only a latency and cost optimization: it does
not determine durability, memory acceptance, authority, or truth.

Memory consolidation runs only after meaningful evidence: a confirmed user
statement, accepted correction, important decision, completed phase, durable
result, or explicit **remember this** request. It does not turn casual model
inference into fact. The user can inspect what Aloy remembers for an Event,
see its source and scope, correct it, forget it, or explicitly promote it.
Archiving freezes active consolidation and triggers while retaining memory for
reopening; deletion follows the Event's retention and deletion policy.

### 3.10 Triggers — explicit reasons Aloy wakes

An Event may remain alive for months while no model process is running. Aloy
wakes only because a durable reason exists:

1. the user sends a Conversation message;
2. the user presses **Work on this**;
3. a meaningful Surface interaction requests reasoning or action;
4. an explicitly configured schedule fires;
5. approved incoming data or an external-state change arrives;
6. clarification, approval, or reconciliation resumes blocked work;
7. a user requests a Surface capability or design change.

Every proactive wake is Event-scoped, budgeted, idempotent, and written to
Trail. Dormant Events have no active triggers. Users can pause proactive work
globally or per Event.

### 3.11 Today — the attention lens

Today is not another Event or Surface. It is the explainable cross-Event view of
what needs the user now:

1. decisions waiting for approval;
2. overdue or time-critical work;
3. blocked work requiring input;
4. user-pinned priorities;
5. stale Tasks;
6. recent meaningful changes.

Deterministic facts—due dates, blockers, approvals, and pins—outrank model
judgment. Ordinary progress remains ambient. V1 uses badges and Today rather
than unsolicited push; interruption budgets and proactive reach-out come later.

Today greets the user by their profile name and ranks dedicated Events by what
needs attention: decisions and blockers first, then stale or urgent work,
active background work, upcoming work, and finally quiet Events. Events with
nothing relevant collapse to lightweight rows instead of consuming equal card
space. **New conversation** and the distinct premium **New Event** action remain
available from the header.

Notifications are a host-owned inbox of meaningful change, not model-authored
prose and not a replacement for Today's priority ordering. They are derived
from durable Proposals, Task state, receipts, and semantic Trail, retain their
originating Life/Event references, and can be reviewed without losing context.
Read state belongs to the user's persisted preferences. Today answers “what
needs me now?”; notifications answer “what changed?”

## 4. The constitution

These invariants override feature convenience and model preference:

1. **No agent consequence without a durable Proposal.**
2. **No committed claim without evidence.** A checkmark means a receipt or
   authoritative record exists.
3. **No invisible work.** Every Run and meaningful state transition is durable
   and explainable through Task, Trail, evidence, and Run Replay.
4. **No accidental execution.** Creating a Task is not permission to start it.
5. **Dedicated Events have one canonical Conversation; Life may have many.**
6. **Tasks are executable contracts.** Completion follows definition of done
   and resulting evidence, not model prose.
7. **A persistent Event is not a permanently running model.** Work starts only
   from a durable, visible wake reason.
8. **Context is a cache, not a home.** Durable truth lives in Event state,
   artifacts, memory, Trail, and receipts.
9. **Events own context; shared reality is referenced, not copied.**
10. **Scope before retrieval.** Tenant and Event boundaries are applied before
    search, ranking, or model context assembly.
11. **Bounded agency.** Every Run has explicit step, time, tool, cost, and
    concurrency limits.
12. **Dormant means quiet.** Dormant Events do not run triggers or agents.
13. **Generated code is always untrusted.** Skills guide behavior; host
    enforcement defines authority.
14. **Presentation cannot manufacture truth.** UI state never upgrades a
    selection, report, estimate, or pending action into a committed fact.
15. **The user can stop and recover.** Work is cancellable, retryable, and
    reconcilable without duplicating consequences.

## 5. Product experience

Desktop is Aloy's primary experience; the web app remains a complete fallback.
The workspace takes inspiration from flexible desktop tools, but its
information architecture is specific to persistent agent work.

### 5.1 The shell

```text
global Aloy sidebar | Conversation | Workbench | Event context
```

- **Global Aloy sidebar:** Life Conversations, dedicated Events, Today, and app
  navigation. It is independent of the open Event and can auto-hide or reveal
  from the left edge without permanently consuming workspace width.
- **Conversation:** the continuous Event Session—the place to instruct, think,
  clarify, review progress, and receive compact outcomes.
- **Workbench:** a first-class flexible pane containing the Event Surface,
  opened files and artifacts, and Run Replay.
- **Event context:** trusted Aloy chrome for Tasks, Trail, approvals, receipts,
  and relevant file navigation. It collapses to a compact rail when the main
  workspace needs more room.

The global sidebar and Event context are different things. The sidebar
navigates the whole product; Event context explains and controls the currently
open Event. Generated Surface code owns neither.

Creation controls must express the product model clearly. Aloy exposes one
**New conversation** action rather than competing “Chat” and “New chat” primary
destinations. **New Event** is a distinct, deliberate premium workspace action,
not a second button styled exactly like conversation creation. Its identity
uses Aloy's own Event language and iconography rather than a generic AI sparkle
or “intelligence” symbol.

### 5.2 Event creation and setup

**New Event** opens a dedicated creation page; it does not immediately open a
chat or create a partially configured Event. The page teaches the first-use
mental model in one short sentence:

> **Create an Event.** Give something important its own ongoing space. Aloy
> remembers its context, work, and decisions until you finish it.

Creation has two deliberate modes.

#### Start simple — the default

The default page is a small host-owned setup, not an AI interview. The user:

1. provides an Event name;
2. may give Aloy a head start through one flexible context composer by typing
   background, dropping files, pasting links, or granting narrowly scoped
   access to an existing connection;
3. explicitly selects **Create Event**.

A simple creation atomically creates the dedicated Event and its lifetime
Conversation, transfers accepted draft context, and opens a minimal Workbench.
Ingestion, a first Surface, Tasks, richer structure, and the generated cover
may arrive later; none of them block creation.

The Event name field contains one quiet trailing assistance action using
Aloy's own mark:

> **Ask Aloy**

Selecting **Ask Aloy for help** changes the existing setup draft into assisted
mode. It preserves the name and all supplied context and never creates the
Event merely because the user entered the assistant flow. The creation page is
about useful grounding, not asking the user to design a cover or application.

#### Ask Aloy for help — assisted setup

Assisted setup begins with **What should Aloy help you make happen?** The user
may describe a clear outcome, describe a situation without knowing the right
structure, attach useful context, or ask Aloy to help them work it out. Aloy
asks only necessary clarifying questions and proposes:

- the Event's name, purpose, and time horizon;
- missing or useful context sources;
- a conservative starting Surface brief;
- initial work and missing context;
- proposed autonomy and approval boundaries.

The proposal remains editable through the setup conversation. The user can
return to **Set up myself** without losing the draft. Only the explicit,
host-owned **Create Event** action promotes the draft into the Event, its
lifetime Conversation, its accepted context grants, and any accepted initial
Tasks. The assisted setup transcript becomes the beginning of that lifetime
Conversation. Surface and cover generation remain asynchronous system work,
not design choices required from the user.

An Event setup draft is not an active Event and is not the deferred
**Emerging** lifecycle state. It is a resumable host-owned creation record with
no Event Triggers, autonomous work, or external-action authority. Drafting may
use bounded model Runs, but those Runs cannot act as the future Event.

Life provides a contextual entry into the same assisted setup. Aloy may say
that a long-lived subject could benefit from its own Event, but the user must
accept the suggestion, review the draft, choose what context to bring, and
explicitly create it. V1 performs no automatic Conversation-to-Event transfer.

#### Setup context and Event bootstrap

An `EventSetupDraft` owns temporary `ContextItem` records before the Event
exists. A context item may be a user note, uploaded file, pasted link, selected
template seed, or Event-scoped connection grant. Each item has explicit
`pending`, `ingesting`, `ready`, or `failed` state and retains source,
freshness, sensitivity, and provenance. A connection grant references a
globally managed credential plus the narrow calendar, folder, page,
repository, account, or other source the Event may access; generated Surface
code receives neither credentials nor unrestricted connection access.

Explicit creation atomically creates the Event and canonical Conversation and
transfers the draft context. Source ingestion continues asynchronously. When
enough evidence is ready, an idempotent Event-bootstrap Run produces a typed,
versioned `EventBrief` containing purpose, outcomes, time horizon, important
entities and dates, recurring work, available sources, unknowns, and the user
jobs a first Surface should support. Every important assertion links to
evidence; inaccessible sources and unknown facts remain visible rather than
being invented.

A host-owned readiness gate controls expensive bootstrap work:

- **name only:** create the Event, keep Conversation primary, and ask at most
  one useful question; do not generate a speculative Surface;
- **little context:** identify unknowns and wait for or request only context
  that materially changes the first Surface;
- **sufficient context:** generate the Event Brief, starting work, cover brief,
  and a deliberately small first Surface asynchronously;
- **rich context:** build from the same general pipeline with stronger evidence,
  never through domain-specific University, travel, or career conditionals.

Before a Surface exists, the trusted Workbench remains fully usable through
Conversation, Tasks, Files, Trail, and context status. It may say **Your
Surface will take shape as Aloy understands this Event**. A published first
Surface contains only evidence-supported views and actions and passes the same
build, capability, Critic, and last-good publication gates as every later
revision.

New sources update the Event Brief incrementally. They do not rebuild the
Surface after every message. A new revision is considered only after a
meaningful source, goal, phase, user request, or demonstrated usability gap.
Pinned regions and user preferences survive accepted redesigns.

#### Event cover identity

The cover is a durable host-managed Event media asset, separate from generated
Surface code. A dedicated cover-generation profile and skill turn the accepted
Event understanding into a constrained visual brief; a developer-selected
image model renders the choices. The Surface Builder does not generate or own
the cover.

The cover is not a required creation decision. The Event starts with a quiet
host-owned placeholder. Once sufficient permitted context exists, Aloy creates
a sanitized visual brief from the Event Brief and queues cover generation in
the background. Raw sensitive files and unrestricted memory are never sent to
the image model. The user may later upload their own cover, request another,
or continue with the placeholder. Image generation may take longer than Event
creation and must never delay, halt, roll back, or make the Event unusable.

The placeholder reserves the final aspect ratio so completion causes no layout
jump. Cover state is explicit (`queued`, `generating`, `ready`, or `failed`),
and the Event's live transport updates Home/Today and the open Workbench when a
quality-gated image becomes ready. The host fades the cover into the reserved
region without stealing focus. Failure leaves the Event and placeholder intact
and offers a bounded retry.

Generated artwork contains no baked-in Event title, invented personal
likeness, or fake institution/provider logo. Host UI places readable text and
status over or beside the image. The stored asset includes an original, alt
text, focal-point metadata, palette hints, and responsive crops for Home/Today
thumbnails and the Event Workbench header. Publication uses compare-and-set
against the Event's current cover version: a late background result never
overwrites a cover the user uploaded, removed, or regenerated while the job was
running. The accepted identity remains stable; Aloy may stage a new candidate
after a meaningful Event phase change, but it never churns the visible cover
after ordinary messages.

### 5.3 Flexible workspace behavior

Conversation and Workbench are composable peers. The user may choose:

1. **Conversation focus** — the Workbench is hidden or minimized;
2. **Split** — Conversation and the active Workbench item are visible and
   resizable;
3. **Workbench focus** — Surface, file, artifact, or replay receives the main
   canvas.

On wide screens Aloy may show Conversation + Surface + an opened document. On
medium screens the active Workbench tab replaces the less important pane. On
narrow screens Conversation, Surface, and documents become explicit
full-screen tabs or sheets.

Pane sizes, open items, selected tabs, and layout mode persist per user and
Event. Live updates never steal focus, reset local interaction, move scroll, or
silently rearrange the workspace.

Opening a PDF, report, image, code file, spreadsheet, or Run artifact creates a
host-owned Workbench tab—not a cramped overlay drawer. Multiple files become
tabs inside the viewer. **Ask Aloy about this file** reveals Conversation beside
the document and attaches a trusted file reference to the next turn. Generated
Surface code never receives unrestricted file access.

### 5.4 Conversation-to-Surface handoff

When Aloy produces a new successful Surface revision while the Surface is not
visible, the trusted Conversation host places a compact **Surface ready** card
after Aloy's completed response.

The card:

- is driven by successful build metadata, not assistant prose;
- is host-owned and cannot be imitated by generated code outside its iframe;
- appears once for each unseen revision;
- opens Split on a suitable wide screen or Surface focus on a narrow screen;
- starts no model Run and creates no Trail entry merely because it was opened.

This makes the Surface discoverable without forcing it open or interrupting the
Conversation.

### 5.5 Entry loops

The first-use loops stay intentionally small:

```text
New conversation
→ talk with Aloy in Life
→ optionally create a loose Task or deliberately create an Event

New Event
→ understand what an Event is
→ start simple or deliberately ask Aloy for help
→ review the setup draft
→ explicitly create the Event
→ enter a deliberate premium workspace
→ continue one lifetime Conversation
→ create a Task
→ Work on this
→ watch durable progress
→ review an artifact
→ approve a consequence
→ inspect the receipt
```

### 5.6 Speed and continuity

Aloy should feel immediate even when meaningful work is long-running:

- Conversation begins streaming as soon as a Run produces output;
- creating or starting durable work acknowledges quickly and continues through
  the worker;
- local Surface presentation interactions remain local;
- the last-good Surface opens immediately while a new revision builds;
- live Event transport updates the correct Conversation, Task, Trail, and
  Surface without full-page polling;
- reopening an Event restores its Conversation and workspace rather than
  starting over.

### 5.7 Authenticated browser work

Aloy may eventually use a user's authenticated web accounts to perform Event
work that an API, Search, or Fetch cannot complete: checking a private flight
booking, reading an account, monitoring a portal, completing a form, or staging
an external action. This is a post-V1 capability, not part of the current R5
Surface runtime.

The Event remains durable while browser Sessions are disposable leases. A
globally managed browser connection owns one user's identity for one site and
login; an Event receives a narrower revocable grant over that connection.
Login, CAPTCHA, two-factor authentication, and credential entry occur through
a trusted embedded browser takeover. Credentials, provider Contexts, raw CDP,
and vendor Live View capabilities never enter Conversation, model input,
generated Surface code, or general Event memory.

Aloy remains the agent and authority. Browser infrastructure may provide
Sessions, persistent browser profiles, proxies, observability, and adaptive DOM
operations, but Pori/Aloy owns the Task and Run loop, budgets, permission
checks, Proposal boundary, success verification, recovery, Trail, and Receipt.
Browser work follows the least-powerful ladder: official connection/API,
Search, Fetch, deterministic Playwright recipe, bounded adaptive DOM operation,
visual fallback, then human takeover.

Consequential browser actions are staged and approved before execution. Aloy
freezes the account, target, material values, page evidence, and action
fingerprint in a Proposal, revalidates them immediately before the one approved
submission, and commits success only from a typed external Receipt. A timeout
or disconnect after submission becomes an indeterminate outcome that must be
reconciled read-only; it is never blindly retried.

The authenticated browser is distinct from both the Surface publication
browser and the sandbox that compiles generated Surface code. Its complete
provider-neutral architecture, security boundary, recovery protocol, UX, and
delivery gates live in
[`aloy-browser-agent-spec.md`](./aloy-browser-agent-spec.md).

## 6. Product proofs

### 6.1 Career OS — the V1 end-to-end proof

The primary V1 flow begins with a real Event and a stale-but-executable Task:

> Career OS → “Research US companies for startup jobs” → **Work on this**

The proof is complete only when:

```text
open Career OS
→ resume its continuous Conversation
→ start the research Task
→ Aloy plans and performs sourced web research
→ progress streams into Conversation and Trail
→ a cited report becomes an Event artifact
→ the Task satisfies its definition of done
→ Aloy stages an email-summary Proposal
→ the user approves it
→ the executor sends through Gmail
→ a provider receipt commits the consequence
→ Conversation, Surface, Today, and Trail update live
```

The crash-window drill kills the process after provider acceptance but before
database commit. Reconciliation must find or expose the provider outcome and
must not send a duplicate.

A static Task list, attractive Surface, or convincing chat response is not
enough. The durable loop must actually work.

### 6.2 University — the continuity and usefulness proof

A University Event should know the student's timetable, courses, upcoming
tests and exams, study materials, and their provenance. Its Surface should let
the student find the next class, inspect upcoming work, request study help,
report a submission without claiming LMS confirmation, and later ask Aloy to
add a grade calculator without losing Event data.

The same continuous Conversation remains beside the evolving Surface for the
duration of university life. Tasks can continue in the background; results,
files, and evidence remain part of the Event.

### 6.3 Madrid El Clásico — the rich interaction and trust proof

For a user travelling from South Africa, a Madrid Event may combine a map,
flights in ZAR, visa readiness, match-ticket evidence, hotels, budget, and an
itinerary. The user can shortlist options and ask Aloy to compare them without
turning every click into a model call.

The Surface must distinguish official from unconfirmed tickets, estimates from
current provider data, a chosen hotel from a booking, and user-reported payment
from receipt-backed payment. Booking or payment must pass through Proposal,
decision, executor, and receipt.

University and Madrid are Aloy's first polished **showcase templates**. A new
user can open either seeded Event to understand the product through a complete,
useful experience instead of an empty onboarding tour. They are also live
marketing demonstrations: University shows long-lived context and study work;
Madrid shows rich planning, evidence, choice, and protected action.

A showcase template is portable Event content, Surface source, manifest,
sample data, and guided jobs instantiated through Aloy's normal Event and
Surface pipelines. It is not University or travel logic compiled into the app,
backend, SDK, or host shell. The same runtime must build, publish, execute,
modify, and remove these templates exactly as it would any model-authored
Surface. Demo facts are visibly identified as sample data, and creating from a
template produces an independent Event the user can replace or evolve with
Aloy.

## 7. V1 scope

V1 includes:

- one permanent Life Event with multiple user-started Conversations;
- explicitly created dedicated Events, through simple or Aloy-assisted setup,
  with one continuous Conversation each;
- executable Tasks with explicit initiation, durable worker execution,
  Stop/Retry/Resume, and bounded budgets;
- sandboxed model-authored Event Surfaces with revisioned source, isolated
  builds, live Event data, validated interactions, and last-good recovery;
- a semantic Trail plus detailed Run Replay;
- Event uploads, working files, outputs/artifacts, and scoped memory;
- provider-neutral sourced web research and evidence-bearing artifacts;
- Ask-routed Proposals and receipt-backed Gmail execution for the Career OS
  proof;
- Today for approvals, blockers, stale work, priorities, and meaningful change;
- crash recovery, idempotency/reconciliation, context longevity, responsive
  behavior, accessibility, and isolation tests.

V1 deliberately excludes:

- automatic or emergent Event detection;
- automatic transfer of Life Conversations, Tasks, or files into Events;
- unrestricted cross-Event retrieval and Life-wide coordination;
- learned Auto/Notify autonomy and unsolicited push notifications;
- Reality Objects for Calendar, Money, or People;
- arbitrary Surface npm dependencies, generated backend services, direct
  generated network/provider access, or unsandboxed model-generated code;
- user-installed Surface plugins and unreviewed privileged widgets;
- shared cross-user Events and multi-agent negotiation;
- unrestricted concurrent Runs inside one Event or account;
- local-folder desktop integration and full native mobile clients.

## 8. Product and platform boundary

Pori is the product-agnostic agent kernel. It provides bounded execution,
planning, tools, memory contracts, teams/sub-agents, evaluation, checkpointing,
streaming, and sandbox abstractions. Pori does not know about Aloy Events,
Surfaces, Today, or Proposals.

Aloy is the product harness around Pori. It supplies:

- Event-aware prompts, context assembly, policy, and run profiles;
- explicit skills such as Surface Builder;
- tenant and Event-scoped tools and virtual filesystems;
- durable workers, product Triggers, and Proposal execution;
- REST + SSE product APIs and the trusted desktop/web host;
- isolated build and runtime boundaries for generated Surfaces;
- product-specific evaluation and quality gates.

Skills improve model behavior but never define security. Authorization,
tenancy, schemas, idempotency, execution limits, sandboxing, CSP, receipts, and
publish gates remain enforced outside model instructions.

Architecture remains one-way:

```text
products → extensions → pori
```

Aloy's TypeScript surfaces reach the backend only over REST + SSE. Product
state and policy remain in `products/aloy`; the kernel remains independently
usable and extractable.

The target architecture is provider-neutral. Trusted app/API/worker services
may run on an ordinary cloud platform, while untrusted Surface builds and
agent execution that require isolation run in managed remote sandboxes. V1
must exercise the remote path so latency, recovery, and cost are understood;
it must never fall back to executing generated Surface builds as a host-local
subprocess.

### 8.1 Sandbox roles and durable workspace identity

A sandbox is a disposable execution unit, not the Event itself and not a
source of durable truth. An Event can remain active for months or years while
its physical sandboxes are created, leased, paused, replaced, or reconstructed
as work requires. Event memory, canonical records, files, artifacts, Trail,
receipts, Run state, and environment provenance remain in Aloy's durable data
plane outside the sandbox.

Aloy has two intentionally separate sandbox classes:

| Sandbox class | Purpose | Lifetime and authority |
| --- | --- | --- |
| **Surface Build Sandbox** | Compile and inspect one model-authored React candidate with the pinned Aloy SDK and browser gate. | Short-lived and least-privileged. The host uploads validated source, invokes one fixed build/inspect command, retrieves immutable outputs and receipts, then destroys the sandbox. It receives no credentials, provider tokens, arbitrary model-owned shell command, dependency installation, direct production data access, or public runtime URL. |
| **Event Execution Workspace** | Future isolated execution for Tasks that genuinely need a filesystem, repository, services, data tools, or longer-running processes. | Leased on demand under an Event-scoped logical workspace identity. Its physical machine may pause or be reconstructed from a declared environment plus durable inputs. Capabilities, network policy, budgets, approvals, and retention are assigned per Run; they are never inherited from the Surface Builder. |

This separation prevents the fast Surface path from growing into a general
remote computer and prevents richer agent execution from weakening Surface
isolation. A published Surface still runs as an opaque-origin iframe inside
the trusted Aloy host; it is never served to the user from a sandbox preview
URL.

The workspace infrastructure follows four planes:

1. **Control plane:** schedules leases, selects a versioned environment,
   reports readiness, enforces deadlines, and may maintain measured warm pools.
2. **Execution plane:** provides the exact filesystem, compiler, browser, and
   other declared tools for that sandbox class.
3. **Security and network plane:** starts secretless, denies network access by
   default, and routes allowed external capabilities through host-owned
   gateways where credentials can be injected outside the sandbox and governed
   by permissions, Proposals, idempotency, and receipts.
4. **Data plane:** supplies a reproducible starting state and persists canonical
   outputs, execution traces, and health evidence outside the disposable VM.

Every lease records provider, template or image version, resource limits,
input and output hashes, readiness and stage timings, termination reason, and
the Run or Build it served. Execution traces stream to Aloy's durable Trail and
Run evidence rather than existing only in sandbox logs. Warm pools and
intent-based prewarming are latency optimizations introduced only when
telemetry justifies their cost; correctness and recovery must work from a cold,
fresh sandbox.

Pori exposes the provider-neutral lease, lifecycle, policy, artifact, and trace
contracts. Aloy selects the product-specific sandbox class and capabilities.
E2B is the first remote Surface Build provider because its versioned templates,
isolated Linux VMs, lifecycle controls, and per-second usage fit the fixed-build
flow; Daytona or another provider must remain replaceable behind the same
contract. The design draws on the [E2B sandbox and template
model](https://e2b.dev/docs) and NeoSigma's [control, execution, security, and
data-plane workspace architecture](https://neosigma.ai/blog/agent-workspaces),
without importing a general autonomous workspace into the least-privileged
Surface Builder.

## 9. Delivery and document map

The active delivery sequence and acceptance gates live in
[`aloy-v1-plan.md`](./aloy-v1-plan.md). Each phase uses its own branch and PR
into the `aloy-v1` integration branch. `main` remains protected until the
Career OS loop, Surface proofs, visual QA, reliability drill, and required
verification gates pass. Branch names do not use agent or tool prefixes.

This document is the parent product source of truth. Child documents may add
detail but must not redefine its primitives or invariants:

- [`aloy-v1-plan.md`](./aloy-v1-plan.md) — active delivery sequence and gates;
- [`aloy-surface-spec.md`](./aloy-surface-spec.md) — generated Surface project,
  SDK, isolation, interaction, quality, and publication contract;
- [`aloy-browser-agent-spec.md`](./aloy-browser-agent-spec.md) — authenticated
  browser identity, execution ladder, permission, recovery, and provider
  contract;
- [`aloy-wedge-spec.md`](./aloy-wedge-spec.md) — implemented Event, Proposal,
  file, Trail, and initial workspace foundation;
- [`Aloy.md`](./Aloy.md) — monorepo and product architecture;
- [`engineering-excellence-spec.md`](./engineering-excellence-spec.md) —
  engineering quality bar;
- [`../products/aloy/BOOT.md`](../products/aloy/BOOT.md) — local boot guide;
- [`../.agent/progress/current.md`](../.agent/progress/current.md) — current
  branch, shipped slices, verification, and immediate next work.

When a child document or implementation conflicts with this vision, the
conflict must be resolved explicitly. It must not be hidden by adding another
parallel concept or a second source of truth.
