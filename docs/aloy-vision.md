# Aloy — product vision

_Canonical product definition, version 3.1, revised 2026-07-16. This document
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

New Event → dedicated Event + its lifetime Conversation
```

A conversation does not automatically become a dedicated Event. An Event is a
deliberate commitment to durable context and an outcome. Aloy may suggest that
a Life Conversation deserves an Event later, but it never silently moves or
reclassifies the user's work.

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

#### Construction and authority

The Surface is real React and CSS authored by the model within a constrained
project. Aloy may create components, layouts, forms, filters, local state, and
Event-specific interactions. Requests such as “add a grade calculator” change
the Surface code. Updates such as “the exam is on Friday” change Event data.
Code revision, data revision, and local presentation state remain separate.

Most Surface interactions do not call a model:

- filtering, sorting, opening, and changing local tabs stay local;
- durable selections are validated and persisted once;
- requests for reasoning enter the canonical Event Conversation as structured
  turns and start a Run;
- booking, sending, paying, publishing, or deleting stages a Proposal;
- requests for new UI capability start the source/build/quality/publish loop.

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

Memory is an index over durable truth:

- **Global memory:** identity and stable preferences;
- **Life/Event memory:** accepted facts, decisions, summaries, results, and
  file pointers scoped to the owner;
- **Transcript history:** durable messages retrieved on demand, not treated as
  automatically accepted facts.

Receipts and provider evidence outrank model memory. Tenant and Event scope is
applied before ranking or semantic retrieval. Cross-Event retrieval remains
deferred until zero-leakage isolation is proven.

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

### 5.2 Flexible workspace behavior

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

### 5.3 Conversation-to-Surface handoff

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

### 5.4 Entry loops

The first-use loops stay intentionally small:

```text
New conversation
→ talk with Aloy in Life
→ optionally create a loose Task or deliberately create an Event

New Event
→ enter a deliberate premium workspace
→ continue one lifetime Conversation
→ create a Task
→ Work on this
→ watch durable progress
→ review an artifact
→ approve a consequence
→ inspect the receipt
```

### 5.5 Speed and continuity

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

University and Madrid are behavioral north stars, not templates hardcoded into
the Aloy app. Their complete acceptance jobs live in the Surface specification.

## 7. V1 scope

V1 includes:

- one permanent Life Event with multiple user-started Conversations;
- manually created dedicated Events with one continuous Conversation each;
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
