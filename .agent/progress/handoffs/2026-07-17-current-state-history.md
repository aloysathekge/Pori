# Current State

_Last updated: 2026-07-17 (R5.5b durable Event context ingestion)._

## NEW: R5.5b durable Event context ingestion (2026-07-17)

The active `aloy-v1-event-context-ingestion` branch turns promoted setup files
and public links into durable, model-independent ingestion jobs. Every source
has a visible `pending -> ingesting -> ready|failed` lifecycle, bounded attempts,
worker lease and lease-expiry recovery, automatic backoff for transient
failures, and a tenant-scoped manual retry path. Event creation and its
permanent Conversation still complete immediately; neither extraction nor an
unavailable source can block or roll back creation.

The existing worker now extracts text, HTML, JSON/CSV/XML/YAML, PDF, DOCX, and
XLSX sources into Event-scoped `KnowledgeEntry` rows. Entries retain source and
content checksums, retrieval time, content type, source URL/freshness headers,
sensitivity, Event-lifecycle retention, and exact draft/context provenance.
Public-link ingestion rejects non-HTTP schemes, local/private network targets,
unsafe redirects, oversized responses, and unsupported content. Completion and
failure create semantic Trail entries, which drive live Workbench refresh.

The Event Surface response now includes trusted context-source state, and the
Workbench Files panel shows queued, ingesting, ready, and failed sources with
errors and retry controls. Internal storage and knowledge identifiers remain
server-owned. Focused ingestion/setup tests pass `10`; the complete Aloy backend
suite passes `296`; backend mypy is clean across `101` source files; and app
ESLint plus its production TypeScript/Vite build pass.

Remaining R5.5 work is the host-owned readiness policy, typed/versioned
evidence-linked EventBrief and bootstrap Run, exact memory precedence and user
controls, connection-source synchronization/revocation, and the first
evidence-grounded Surface and sanitized cover brief.

## NEW: R5.5a durable Event setup context (2026-07-17)

The active `aloy-v1-event-setup` branch now implements the model-independent
first slice of the revised Event-creation contract. Setup is a tenant- and
user-scoped server draft instead of browser-local state. It autosaves the Event
name, description, and simple/assisted mode and owns typed context items for
files, links, template seeds, and grants to existing account connections.

The setup page is now a compact context composer. **Ask Aloy** remains embedded
inside the name field, but neither mode requires model access. The user may
describe the Event, drop or select files, add a link, or grant an already
connected account to the Event. Cover selection is no longer part of default
setup: the host queues later cover generation from richer accepted context.

Explicit **Create Event** promotes the draft once and is idempotent. It creates
the dedicated Event and permanent Conversation, transfers staged files into
the Event library, records description/link/template context as Event-scoped
knowledge with provenance, persists narrow connection grants, adds semantic
Trail, and leaves cover/bootstrap work queued. Pending or failed ingestion
cannot block or roll back Event creation. Sequential retry returns the same
Event, and cross-user draft reads are rejected.

Focused verification: 4 Event-setup backend tests and the 57-test related
Event/connection/upload/file regression set pass, including resumability,
tenant isolation, file/link/connection transfer, idempotent promotion,
migration round-trip, and failed-ingestion tolerance. All 4 app tests pass;
backend Ruff is clean; Aloy app ESLint and the production TypeScript/Vite build
pass. Its previously open asynchronous file/link ingestion work is superseded
by the R5.5b section above.

## NEW: Event bootstrap and scoped-memory contract (2026-07-17)

The parent vision and active delivery plan now make setup context—not cover
design—the useful optional work during Event creation. A resumable
`EventSetupDraft` may own notes, files, links, template seeds, and narrowly
scoped connection grants. Explicit creation stays available with only a name;
accepted context transfers into the Event while ingestion continues without
blocking the permanent Conversation or Workbench.

After a host-owned readiness gate finds sufficient evidence, an idempotent
bootstrap Run produces a typed, evidence-linked `EventBrief`. The same general
brief drives a deliberately small first Surface through the existing quality
and publication pipeline and a sanitized cover brief through the background
image path. Name-only and little-context Events remain Conversation-first and
must not receive speculative generated UI.

Memory is now explicitly specified as global user memory plus isolated Event
memory, alongside canonical Event state and retrieved transcript history.
Event memory is evidence-backed, typed, provenance-bearing, confidence- and
sensitivity-aware, supersedable, inspectable, correctable, forgettable, and
promoted to global scope only through explicit policy. Required precedence is
Event over global and personal over team over organisation. The current kernel
and backend already carry `event_id` through memory scope and load only global
plus owning-Event knowledge; the remaining work is consolidation, exact
precedence, user controls, bootstrap ingestion, retention behavior, and
zero-leakage adversarial evaluation.

## Earlier Event creation foundation (superseded by R5.5a, 2026-07-17)

The active `aloy-v1-event-setup` branch now documents and implements the first
model-independent Event creation slice. **New Event** opens a dedicated setup page whose default
is a small host-owned **Start simple** flow: name, optional cover, and explicit
creation. Only **Ask Aloy for help** transitions that draft into the assisted
experience, where Aloy may clarify the intent and propose the cover, Surface,
initial work, and boundaries before the user confirms.

The setup draft is not yet an active Event and has no Event Triggers or action
authority. Simple and assisted paths both require an explicit host-owned create
action. Assisted setup becomes the beginning of the lifetime Conversation only
after confirmation. Generated covers are stable host-managed Event media made
through a dedicated cover profile and image model, not Surface Builder code;
the same asset supplies responsive Home/Today and Workbench crops. If creation
commits without a cover, the Event opens immediately with a reserved
placeholder while a durable background image job runs. Its quality-gated
result appears through live Event updates and cannot overwrite a cover the user
chose while generation was in flight; cover latency or failure never blocks or
rolls back Event creation.

The app now persists an unfinished setup draft locally, creates through an
explicit confirmation, supports validated PNG/JPEG/WebP cover upload, and
shows host-owned cover placeholders/thumbnails on Today and in the Workbench.
The backend exposes tenant-scoped cover upload/read routes, stores cover media
through the existing object-store seam, records semantic Trail, and marks a
no-cover Event as queued for automatic generation without delaying creation.
The assisted setup panel is intentionally provider-empty while model access is
unavailable; it states what will be proposed instead of faking intelligence.
The next slice is the durable cover worker/provider and assisted-setup model
profile, followed by live fade-in and responsive full cover treatments.

## NEW: Today and Life product experience (2026-07-16)

The `aloy-v1-today-life-reimagination` branch replaces equal-size Event cards
with a user-centred attention workspace: a profile-name greeting, distinct New
conversation and New Event actions, a permanent Life band, deterministic
priority groups, active work, upcoming work, quiet Event rows, and a host-owned
notification rail. Notifications are derived from durable Proposal, Task, and
Trail state; the user's read cursor lives in persisted profile preferences.

Life is now named explicitly in global navigation instead of being hidden
behind “Chat.” It remains the permanent personal Event with many clean
Conversation threads and durable shared state, while dedicated Events retain
their single lifetime Conversation. This phase does not yet create a fixed Life
Surface or proactive push-notification system.

## NEW: Atomic Surface publication and last-good rollback (2026-07-16)

The active `aloy-v1-r5-surface-publication-recovery` branch separates a
successful draft build from the live Event Surface. `surface_projects` now
points to both the published immutable revision and the exact published build;
an append-only, idempotent publication ledger records publish and rollback
transitions with their previous pointers, actor, Run, and semantic Trail.

Publication accepts only the current draft's successful deterministic build,
reopens the retained artifact, verifies its checksum and runtime bundle, then
advances both live pointers with compare-and-set protection. Artifact failure,
a stale writer, or a failed draft leaves the current live Surface untouched.
Rollback can target only a build that was previously published and never
rewrites canonical Event data or revision history.

The Workbench now resolves only the explicit published build rather than the
newest successful draft, labels it Live, and provides trusted version history
with Restore controls. Runtime context and meaningful interactions are granted
only to the current published build. The complete Aloy backend suite passes
with 285 tests; focused publication/SDK tests, backend formatting and typing,
the Surface SDK build, and app tests/lint/build are also green.

## NEW: University and Madrid are showcase templates (2026-07-16)

Product direction is now explicit: University ships first and Madrid second as
polished first-run templates that teach new users how Aloy works and serve as
live marketing demonstrations. University covers navigation, timetable,
courses, exams/tests, and study actions. Madrid follows the dedicated reviewed-
widget phase and covers map, flights, hotels, visa, budget, itinerary, and
protected payment intent.

These are portable seed packages, not domain behavior hardcoded into Aloy.
They must instantiate normal tenant-owned Events, data, source revisions,
builds, SDK interactions, Proposals, receipts, and Trail through the exact same
paths as every other Surface. The host runtime and backend must contain no
University/travel switch. This supersedes earlier wording that called the two
experiences “not templates”; they are templates at the product/catalog layer,
while remaining proofs of the general platform.

## NEW: Surface runtime resilience (2026-07-16)

The active `aloy-v1-r5-surface-runtime-resilience` branch hardens the bound
Surface runtime without granting generated code any new authority. A bridge is
healthy only after the iframe SDK acknowledges a fresh session id; all bridge
messages are session-bound, and a host heartbeat detects a stalled generated
runtime. Context and interaction requests have bounded abortable timeouts.

`@aloy/surface` now exposes runtime state, rejects messages from stale bridge
sessions, answers heartbeats, bounds interaction Promises, and safely replays
one retryable request using the exact original idempotency key. The Workbench
keeps the last-good iframe mounted through bounded 1/2/4-second reconnects and
shows trusted reconnect/degraded UI with a manual recovery action instead of
replacing valid generated UI with a fatal error screen.

Focused host tests cover exact-session acknowledgement, stale-session
rejection, and idempotent interaction routing. Aloy app tests and ESLint pass;
the app and standalone SDK production builds pass. Remaining R5 runtime work:
finish runtime/publication correctness, create the University showcase first,
then handle reviewed widgets in a dedicated phase before the Madrid showcase;
publication/last-good selection, render/resource watchdogs, and Critic remain
open.

## NEW: Surface product lifecycle clarified in parent vision (2026-07-16)

`docs/aloy-vision.md` is now version 3.1 and explicitly defines an Event
Surface as a rich, navigable application rather than a single screen or fixed
dashboard. Surface-owned tabs, sub-navigation, routes, scrolling, responsive
states, and custom Event-specific UI remain distinct from Aloy's trusted
workspace navigation and host-owned privileged capabilities.

The parent vision now defines the complete opportunity → brief → isolated
build → deterministic/visual/interaction evaluation → independent Critic →
bounded repair → risk-aware publish → monitor/improve lifecycle. Durable Event
truth remains independent of replaceable Surface code; last-good publication,
user rollback, phase-aware evolution, and evidence-based redesign prevent weak
builds or arbitrary layout churn from disrupting a long-lived Event. This is a
product-contract clarification only; it does not claim the remaining R5
publication, Critic, widget, or proof implementation is complete.

## NEW: Surface Run and Proposal lifecycle reconciliation (2026-07-16)

The active `aloy-v1-r5-surface-sdk-bridge` branch now closes the trusted
interaction loop. Surface reasoning requests move `queued → running →
completed|failed|cancelled` from the durable worker, and external actions move
`waiting_approval → approved → executing → committed|rejected|failed|
indeterminate` only from Proposal decision/executor truth. Run, Proposal,
SurfaceInteraction, semantic Trail, and host-owned Conversation messages are
updated in the same transaction at each trusted boundary. Request/outcome
message ids make lifecycle cards exactly-once across retries and recovery.

Runtime context now carries the Event Surface's recent interaction ledger and
`@aloy/surface` exports `useInteractions`, allowing generated UI to render live
status and receipt-backed outcomes without inventing completion from local
state. Proposal Trail SSE frames resolve the Proposal's origin Conversation so
the open permanent Session refreshes immediately. The app renders protected
action request/outcome messages as trusted lifecycle cards.

Verification is green: all 282 Aloy backend tests pass; a focused 31-test
Surface/Proposal/worker/live-routing set passes; backend Black/isort are clean
across 179 files; backend mypy is clean across 98 source files; the standalone
Surface SDK compiles; and the Aloy app passes ESLint and its production build.

## NEW: Surface SDK and interaction bridge foundation (2026-07-16)

The active branch is `aloy-v1-r5-surface-sdk-bridge`, based on the merged Event
Workbench PR #178. The first bound-SDK slice is implemented. A new private
`@aloy/surface` workspace package provides reactive Event/data hooks plus
`dispatch`, `askAloy`, and `requestAction`. Generated code receives its context
only through a host-created `MessageChannel`; the opaque-origin iframe still
receives no bearer token, REST access, cookies, storage, direct network, or
Electron authority.

`surface.json` is now a versioned, fail-closed manifest that declares read
capabilities, typed intents, namespaced data writes, host action tools, and
reviewed widgets. The backend persists independent Surface data revisions,
provenance/posture-bearing data records, and exactly-once revision/build-bound
interactions. Capability-scoped context reads expose only declared Event data.
Durable selection dispatches validate the declared schema and atomically update
one namespaced record plus semantic Trail; `askAloy` appends one structured turn
to the Event's canonical Conversation and queues one durable Run; external
actions validate the exact live tool schema and stage a Proposal without
execution. User-originated Surface writes can claim only `user_reported`, never
committed or receipt-backed posture.

Event SSE invalidation now refreshes the bound context without remounting the
iframe when the successful build is unchanged, preserving Surface-local UI
state. A newly successful build still replaces the runtime document. The
Surface Builder skill includes the exact SDK/manifest contract.

The initial foundation verification remains superseded by the lifecycle result
above. Runtime handshake, heartbeat, timeouts, and recovery continue in the
runtime-resilience section above; the remaining R5 gates are tracked there.

## NEW: Event Workbench shell (2026-07-16)

The active branch is `aloy-v1-r5-event-workbench`, based on the merged iframe
host. The Event screen now treats generated Surface UI, message artifacts,
Event files, and Run replay as first-class persistent Workbench tabs rather
than modal drawers. Conversation / Split / Workbench modes and both pane ratios
persist per Event; wide layouts may keep Surface beside another active resource.
Event context is grouped into Tasks, Approvals, Receipts, Files, and Trail and
collapses to a compact trusted rail. The global Aloy sidebar can remain open or
auto-hide and reveal from the left edge.

Host-owned artifact and stored-file viewers support safe preview, copy/download,
and **Ask Aloy**, which attaches the existing durable file reference to the
Event composer without re-uploading it. Run replay supports an embedded pane
while preserving its existing modal use in ordinary chat.

App ESLint and the production TypeScript/Vite build pass. Automated visual QA
is blocked because the in-app browser connection cannot initialize; complete a
signed-in manual pass at the running local Event URL before merging. This shell
does not add the Surface capability SDK, independent Critic, or publish gate.

## NEW: Model-authored Event Surface architecture (2026-07-16)

The University and Madrid product walkthroughs superseded the uncommitted
typed-block R5 direction. The canonical contract in
`docs/aloy-surface-spec.md` now defines a Surface as a model-authored React
application for one Event: generated source is built in isolation, published
as immutable last-good revisions, executed in a sandboxed iframe, and connected
to tenant/Event data and the permanent Session only through a capability-scoped
SDK and validated interaction bridge.

The two north-star images are preserved in `docs/assets/`. They define the
behavior and quality expected from the University and Madrid showcase
templates: University demonstrates timetable/course/assessment truth plus a
study-help interaction; Madrid demonstrates map/travel choices, visa readiness,
estimates, provenance, uncertainty, and safe action intent. Both must use the
same general runtime without hardcoding either domain into it. Tasks,
Proposals, receipts, files, evidence, and Trail remain canonical whether or not
generated code displays them.

R4 is merged into `aloy-v1` as PR #172. The first authoring-harness foundation
was merged into `aloy-v1` as PR #173 at merge commit
`81572b7a16743ad2bf18c5ceda85a3d67483be48`. Pori now has immutable,
fingerprinted run profiles that bind prompts, tools, explicit skills, and model
capabilities with fail-fast validation; Aloy forwards configured prompts and
provider capabilities and bundles an explicit-only Surface Builder skill.

The filesystem foundation was merged into `aloy-v1` as PR #174 at merge commit
`f5e9831ff02081ea4ee222b15bc3cd662efac1d1`. It provides typed, deny-by-default
virtual mounts without adopting host-filesystem or local-shell backends.

Surface persistence was merged into `aloy-v1` as PR #175 at merge commit
`6aac2d78068e4dd5713b5c41ef9901258e8b949c`. Aloy now has one tenant-
scoped `surface_project` per Event and immutable source/manifest revisions with
parent lineage, exact checksums, creator Run provenance, optimistic draft
claims, idempotent mutation keys, file/source limits, and user lock state. The
explicit Surface Builder profile alone receives `surface_read_project` and
`surface_write_files`, plus read/write/list/edit tools over a read-only `/event`
projection and run-local `/workspace` seeded from the durable draft. Public
clients can read project/revision metadata but not source contents; there is no
public source-mutation route. Every committed draft adds a semantic Trail row.

R5 build/preview was merged as PR #176 at merge commit
`a63384188e5f9bc13563bc36fb9b69d976d282b2`. Each
immutable revision can produce an idempotent durable `surface_build` with
deterministic source checks, retained diagnostics/logs/metrics, a content-
addressed bundle pointer, preview artifact metadata, and a semantic Trail row.
Generated source runs only through a fixed Aloy toolchain contract in a non-
local sandbox provider; absent isolation returns `blocked` and never falls back
to a host subprocess. Public tenant-scoped reads expose safe metadata but never
the object-store key, source, build log, or executable bundle. `surface_preview`
inspects metadata only.

The next security boundary is implemented on
`aloy-v1-r5-surface-iframe-host`. A tenant-authenticated endpoint reads a
successful immutable bundle without exposing its object-store key, validates
the exact `surface.js` plus optional `surface.css` ZIP contract, neutralizes
raw-text end-tag injection, and creates Aloy-owned HTML under a nonce-bound CSP
with direct network, objects, frames, workers, forms, and base URLs denied. The
app turns that authenticated document into a revocable Blob URL and runs it in
an opaque-origin `sandbox="allow-scripts"` iframe. Event workspaces now have
Conversation, resizable Split, and Surface focus modes while the trusted Event
context rail remains outside generated code. After an assistant response, a
host-owned Surface-ready card reveals each unseen successful build once when
the Surface is not already visible; it opens Split on wide screens and Surface
focus on narrow screens without starting a model turn or adding Trail noise.
This is explicitly a preview; publication/last-good selection and the
capability SDK/interaction bridge are not yet implied.

Verification for build/preview: all 274 Aloy backend tests pass; the focused
16-test Surface suite passes; backend mypy is clean across 94 source files; and
backend Black/isort checks pass.

Verification for the iframe host: all 275 Aloy backend tests pass; backend
mypy is clean across 95 source files; backend Black/isort checks pass; and the
Aloy app passes ESLint, TypeScript, and its production build. Signed-in visual
QA remains the final local acceptance step.

Next complete signed-in visual QA for the iframe host, then implement the
bound SDK + interaction transport, the independent Critic,
and the two proofs. §13 of the Surface spec also locks the design brief,
design system, screenshot states, deterministic audit, independent Critic,
user-job simulation, bounded repair, user control, scorecard, and last-good
publish gate. Do not add R6 research providers or R7 Gmail behavior.

Verification for the authoring-harness foundation: 605 kernel tests passed
(1 skipped), 259 Aloy backend tests passed, kernel mypy is clean across 107
source files, backend mypy is clean across 87 source files, the Surface Builder
skill validator passed, and a built Aloy wheel contains the bundled skill.

Verification for the active filesystem harvest: 612 kernel tests passed
(1 skipped), 259 Aloy backend tests passed, kernel mypy is clean across 108
source files, backend mypy is clean across 87 source files, and kernel/backend
Black and isort checks pass. The boundary script could not run because the
optional `import-linter` executable is not installed in the local environment;
no dependency was changed merely to satisfy that local check.

Verification for active Surface persistence: 268 Aloy backend tests pass; the
15-test focused persistence/profile/migration/workspace suite passes; backend
mypy is clean across 91 source files; and backend Black/isort checks pass.

## NEW: Aloy V1 R4 - live Surface and semantic Trail (2026-07-15)

R3 is merged into `aloy-v1` as PR #171 at squash merge
`e2587e64e67c8857c1025f48786f0d6ffd96f46f`. R4 is implemented on
`aloy-v1-r4-live-surface-trail`; do not implement the R5 composition runtime
or R6 research providers on this branch.

Each Event now exposes a tenant-scoped, database-backed SSE stream. The API
replays Trail rows after an opaque reconnect cursor and follows worker writes
across process boundaries, including the Task or Run's origin Conversation.
The Event workspace no longer polls every two seconds: it reports live,
reconnecting, stale, or offline state; invalidates the trusted Surface on
durable changes; and refreshes transcript state only when the live frame names
the currently displayed Conversation. A regression uses two sibling Life
Conversations to prove terminal Task replay returns to the origin rather than
the sibling.

Event Trail and Conversation messages use stable `(created_at, id)` keyset
pagination. Initial reads are bounded to 50 Trail entries and 100 messages;
older pages load explicitly without duplicates, while Conversation export
still returns the complete transcript. Recent Task Runs appear as expandable
execution narratives linked to their origin Conversation, Run Replay,
artifacts, Proposals, and committed receipts. Today calls out blocked Tasks and
non-blocked active Tasks unchanged for more than 24 hours.

Automated verification is green: all backend tests, the focused R4 reconnect/
pagination/provenance suite, backend Black/isort/mypy, and app lint + production
build. The local API, web app, and worker boot successfully. Complete a short
signed-in interaction pass for the live indicator, Task execution refresh,
reconnect, Trail expansion, and load-older controls before merging R4.

## NEW: Aloy V1 R3 - durable Task execution (2026-07-15)

R2 is merged into `aloy-v1` as PR #170 at squash merge
`b47f0fd41a027c3a51024be0c329b1ab74d11428`. R3 is implemented on
`aloy-v1-r3-task-execution`; do not add R4 Event SSE or R5 research providers
to this branch.

Event Tasks now have explicit **Work on this**, Stop, Retry, and Resume command
routes. Work creates one idempotent durable Run, assembles the bounded work
order from the Event and Task, and reports to the Task's selected Conversation.
Blocked Resume reuses the same checkpointed Run; Retry creates one fresh Run.
Compact queued, started, blocked, approval, failed, stopped, and result messages
return to the selected Conversation.

The worker atomically claims the queued Task, records per-step Trail milestones,
survives process/app closure through the existing lease + checkpoint system, and
synchronizes clarification, Proposal, cancellation, success, and failure back to
Task state. Admission permits one active Task Run per Event, one active Run per
Conversation, and a bounded account-wide number of running Runs while leaving
additional Task work queued. The Event UI explains each state and uses bounded
refresh until R4 replaces it with Event SSE.

Automated verification is green: all `253` backend tests, the focused `60`-test
R3/worker integration set, backend mypy across `85 source files`, and app lint +
production build. Complete local running-stack product QA, then commit, push,
open the R3 draft PR, and let CI gate the merge.

## NEW: Aloy V1 R2 - executable Task model (2026-07-15)

R1 is merged into `aloy-v1` as PR #169 at squash merge
`7889bc2e12560b2b9c36434a1065367cf0199af5`. R2 is implemented on
`aloy-v1-r2-task-model` in draft PR #170. All seven PR checks are green; R2 is
ready to mark for review and merge into `aloy-v1`.

Task now carries the durable execution contract: eight legal statuses,
instructions, definition of done, priority, due date, manual execution mode,
assigned agent, origin Conversation, current Run, result summary, blocker, and
bounded budget policy. Run has a nullable Task link, inherited by child Runs.
Life Tasks may originate from any Conversation in Life; a dedicated Event Task
may reference only its owning Event's Conversation.

`task_state.py` is the single transition/provenance/claim boundary for user and
agent mutations. Illegal transitions roll back without a Trail entry. A queued
Task claim is a compare-and-set update tied to a matching Run; concurrent
claimers yield one winner and one semantic Trail entry. R3 will add **Work on
this** and durable queue execution; R2 deliberately does not start work.

Migration `w9a0b1c2d3e4` adds the Task/Run fields, preserves existing
`open|done` rows, and backfills a valid origin from the Event's resume target
when available. A URL-free backend `alembic.example.ini` restores the documented
local migration workflow. Verification is green: `246` backend tests, Black across
`156` files, mypy across `84 source files`, app ESLint/build, focused R2 tests,
and a clean-database upgrade through every migration to the new head.

## NEW: Aloy V1 R1 - Life conversations and dedicated Event sessions (2026-07-15)

R0 is merged into `aloy-v1` as PR #168 at squash merge
`069f173ec59dad02b0f9bbb26cf3598b51c10c47`. R1 is implemented on
`aloy-v1-r1-life-conversations` in draft PR #169. All seven PR checks are
green; signed-in manual product QA is the remaining merge gate.

The default Chat API and switcher now contain only the signed-in user's Life
Conversations. **New chat** creates a fresh Life thread; **New event** creates a
dedicated Event with its canonical continuous Conversation. The Event rail
excludes Life. A Life Conversation can explicitly seed a new Event while
retaining inspectable origin provenance and leaving the source thread intact.
The entry actions now communicate that difference visually: Chat is one
navigation row with an integrated create control, while New Event uses a
restrained workspace treatment and a product-specific bounded-trail glyph
across navigation, Today, and Chat.

Life's newest Conversation is its resume target. Deleting that thread safely
retargets Life to the most recent remaining Conversation or an empty state;
dedicated Event canonical Conversation deletion remains `409` protected.
Reading an empty Life Event does not manufacture a blank chat. Runtime context
hydrates only the current transcript while indexing sibling Event/Life
Conversations for explicit scoped history retrieval.

Verification is green: a complete backend run passed `228` tests; the exact
final Event/context regression set passed `14` tests; Black is clean across
`153` backend files; backend mypy is clean across `83 source files`; app ESLint
and the production build pass. The existing Vite bundle-size warning remains
non-blocking. Before merging R1, complete signed-in manual QA for New chat, the
Life-only switcher, New event, canonical Event reopen, Life deletion, and
Create Event from conversation.

## NEW: Aloy vision v2 + V1 reset plan (2026-07-15)

The product model was re-audited after the Phase 5 workspace exposed a core
gap: the shipped Task is durable checklist state, but it does not itself prompt
Aloy to perform work. `docs/aloy-vision.md` is now version 2.1 and makes the
operating loop explicit: intention → Event → executable Task → bounded Run →
trusted state, with Proposal → decision → receipt for protected consequences.
It also separates Conversation, Event, and bounded Run; defines the trusted
Surface and semantic Trail; and selects Career OS research as the V1 hero flow.
The follow-up product clarification is now explicit: Life is the permanent
user–Aloy personal Event and may contain many fresh Conversations, while each
dedicated Event retains one canonical continuous Conversation.

`docs/aloy-v1-plan.md` is the active implementation plan and supersedes the old
remaining phase order in `docs/aloy-wedge-spec.md`. The immediate phase is R0:
close the current `aloy-v1-phase-5-surfaces` branch by applying its formatting-
only migration correction, completing signed-in visual/streaming/reopen QA,
making CI green, and merging it into `aloy-v1`. Only then create
`aloy-v1-r1-life-conversations`. R1 establishes New Conversation versus New
Event semantics, Life-only chat history, transcript isolation, safe Life chat
deletion, and explicit Event creation from a Life chat. R2–R7 then add the
executable Task state model, durable **Work on this** execution, live
Surface/Trail updates, sourced research, the Gmail decision/receipt hero loop,
and reliability/context release gates.
Do not jump directly to the old "Building Aloy" hero-flow phase.

R0 automated close-out update: commits `1cbd7a5` (migration formatting) and
`98d1710` (vision v2.1 + R0–R7 plan) are pushed to PR #168. All seven GitHub
checks are green. Local verification is also green: Black (`153 files`), all
`225` backend tests using a workspace-local pytest temp root, backend mypy
(`83 source files`), app ESLint/build, API `/v1/health`, Vite HTTP 200, and the
worker process chain. The founder then completed the requested signed-in local
workspace pass and confirmed it is working. `design-qa.md` now records manual
acceptance, with automated captured comparison explicitly deferred to R7.
R0 is clear to mark PR #168 ready and merge into `aloy-v1`; after merge, the
next branch is `aloy-v1-r1-life-conversations` from the updated integration
branch.

## NEW: Aloy V1 Phase 5 — Tasks + Project Surface + Today (2026-07-15)

Phase 4 is merged into `aloy-v1` as #167. Phase 5 is implemented on
`aloy-v1-phase-5-surfaces`: manual Project Event creation; one canonical,
continuous working Session per Event (legacy rows retained as provenance);
direct user and agent Task mutations with atomic Trail
entries; a trusted templated Event Surface recomputed from Event/Task/Proposal/
Trail/StoredFile rows; and a Life-first Today lens over pending decisions,
recent committed/evidenced changes, recent Trail activity, and open Tasks.
Today is now the signed-in app landing page. The Event view is a flexible
workspace with the lifetime conversation at center and Tasks, decisions,
files, and Trail in a collapsible context pane. Pending Proposals can be approved
or rejected from both Today and their Event workspace through the Phase 4 commit
rail. `task_create`/`task_update` are Aloy product tools; the kernel gained a
small generic async-tool execution seam so database-backed product tools do not
block the agent loop. Verification: 601 kernel tests passed (1 skipped), all
225 backend tests passed, kernel mypy is clean across 106 files, backend mypy
is clean across 83 files, and the Aloy app build + lint are green. The active
next step is R0 in `docs/aloy-v1-plan.md`; the former Phase 6 sequence is
superseded. Do not add Reality Objects, learned routing, cross-Event retrieval,
or free-form model-composed Surfaces.

## NEW: Aloy V1 Phase 4 — Proposal executor + commit rail (2026-07-15)

Phase 3 is merged into `aloy-v1` as #166. Phase 4 is implemented on
`aloy-v1-phase-4-execution`: tenant-scoped approve/reject/edit decisions,
same-tool edit validation, compare-and-set decision and execution claims, a
standalone non-agent executor with current membership/policy/Event/tool/schema/
credential checks, receipt-backed commit Trail entries, expiry reject defaults,
and stale/crash-window `indeterminate` reconciliation without blind retries.
The existing `/conversations/approve/{id}` UI route is a compatibility alias
over durable Proposals; the worker also processes approved Proposals. Verification:
219 full backend tests passed, the final 35 proposal/approval/worker tests passed,
and mypy is clean across 80 backend source files. Phase 5 is Tasks + Project
Surface + Today; do not add those surfaces to Phase 4.

## NEW: Backend package renamed pori_cloud -> aloy_backend (2026-07-07)

The Aloy backend's Python package was `pori_cloud` (a leftover from when the
hosted product was "Pori Cloud", before it was named Aloy). Renamed to
`aloy_backend` for identity consistency: `git mv` the package + the deploy
service file; scoped text replace `pori_cloud`->`aloy_backend` and
`pori-cloud`->`aloy-backend` across the backend + tools/ci (51 files); entry
points now `aloy-backend`/`aloy-backend-worker`/`aloy-backend-gateway`;
`uv lock` regenerated; importlinter boundary updated (still KEPT). 77 backend
tests pass. Note: the archived DONOR REPOS keep their historical names
(pori_cloud/pori_cloud_client/pori_website) in provenance docs — only the
in-repo package was renamed. Deploy Dockerfile/compose still assume the
pre-monorepo sibling layout (known deploy-pass debt, unchanged here).

## NEW: Read-only run replay viewer (2026-07-07)

Aloy-only (kernel untouched — the boundary paid off). The kernel already
EMITS the full PoriEvent stream via on_event; Aloy taps the existing SSE
`push` sink and persists a coalesced log. Backend: `RunEventLog` table (one
row per run, migration m9c0d1e2f3a4), `aloy_backend/event_log.py`
EventLogCollector (coalesces consecutive text/thinking deltas into blocks,
keeps structural events verbatim, caps at MAX_EVENTS), recorded on the
serving-loop consume side in streaming.py (no thread race), persisted in the
same txn as the Run in conversations.py send_message, `run_id` added to
assistant Message metadata, `GET /v1/runs/{id}/events` (RUN_READ, tenant-
scoped). App: `api/runEvents.ts`, `RunReplay` modal (read-only timeline with
play/pause + scrubber), a "Replay" button on assistant MessageBubbles that
carry a run_id. Backend 77 tests (8 new: collector coalescing/caps + endpoint
happy/404/cross-org). NOTE: capture is on the STREAMING (interactive chat)
path only; durable-worker/cron/gateway runs don't yet capture a log — clean
follow-up (pass a collector on_event in background.py). This is the cheap-80%
of the OpenHands event-stream idea; NOT event-sourcing (state still resumes
from checkpoints, not replay).

## SESSION SUMMARY — everything below is MERGED to main, CI green (2026-07-07)

PRs #92–#106 all merged; zero open PRs; the branches are deleted. Do NOT
re-open or re-implement any of it — read this file, then AGENTS.md, before
starting. What shipped this stretch, newest first:

- **Sandbox / execution isolation (#106).** Kernel: `pori/sandbox/env_safety.py`
  strips host secrets (API keys/tokens/DB URLs/venv markers) from every
  agent-run subprocess — the LocalSandbox previously leaked the full env; this
  is the cheap universal win, live now. New `E2BSandboxProvider`
  (`pori/sandbox/e2b.py`, modeled on Hermes's Daytona backend): optional cloud
  microVM, resume-or-create via a thread→sandbox-id JSON ledger (a resumed run
  reconnects the SAME sandbox, files intact). Gated: needs `E2B_API_KEY` +
  `pori[sandbox-e2b]` extra; `create_sandbox_provider(backend)` factory +
  `config.sandbox.backend`. Aloy: worker sets the provider at startup from
  `settings.sandbox_enabled`/`sandbox_backend` (falls back to local on error);
  `GET /v1/system/execution` + a read-only "Secure execution" card in app
  Settings REFLECTS it (Aloy-managed model — user configures nothing).
  **Open decision:** bring-your-own-key (per-tenant E2B keys) NOT built — left
  for the user (managed vs BYO-key is a real product fork).
- **App design system (#105).** Aloy identity: self-hosted variable fonts
  (Inter/Bricolage Grotesque/JetBrains Mono), a hand-drawn signal-dot icon
  family (`src/components/icons.tsx`) replacing lucide in nav, the AloyMark
  everywhere the pori-* logos were (all deleted). Palette matches the LANDING
  PAGE (warm off-white light theme) via one remap of the zinc ramp in
  `@theme` (src/index.css) — components authored dark-mode, ramp inverted, so
  one block restyles all screens. Closes the "visual rebrand of the app"
  roadmap item.
- **Tier-1 Hermes gap (#102/#103/#104)** — details in the older section below.

**Verification reality unchanged:** all of this is green in CI and
verified-by-construction, but the stack has STILL never run end-to-end. That
remains milestone #1 (see below) and is the highest-value next step.

## NEW: Tier 1 of the Hermes gap IMPLEMENTED (2026-07-07, PRs #102/#103/#104)

All three Tier-1 items from docs/hermes-gap-2026-07.md, as independent PRs:
(1) #102 multimodal message content — TextBlock/ImageBlock in llm/messages.py,
mapped in all provider adapters (OpenRouter/Fireworks inherit via ChatOpenAI);
str content stays valid everywhere. (2) #103 cross-provider failover —
FailoverChatModel chain consuming the existing error classifier; llm.fallbacks
config; sticky switch; overflow/content-policy deliberately NOT triggers;
credential POOLING still open. (3) #104 Telegram gateway slice —
aloy_backend/gateway/ (BasePlatformAdapter ABC, TelegramAdapter over raw Bot API
via httpx, registry, DeliveryRouter), pairing codes (POST /v1/gateway/pair →
send code to bot), GatewayLink+migration l8b9c0d1e2f3, inbound messages become
durable Runs in a per-chat Conversation (get resume/salvage free), results
delivered back on completion; pori-cloud-gateway entrypoint + compose service
(profile 'gateway'). NEXT gateway steps: voice-note STT, images-in (needs
#102), group semantics, Slack adapter, cron-delivery via DeliveryRouter.

## NEW: Second Hermes mining pass — gap analysis (2026-07-07)

Source-level sweep of references/hermes-agent for everything NOT yet
harvested/tracked. **Canonical output: `docs/hermes-gap-2026-07.md`** (ranked
by Aloy leverage). Headline: loop-quality is now at parity+; the gap is
SURFACE BREADTH. Tier 1: (1) multimodal message plumbing — messages.py is
str-only, blocks vision/screenshots/photos; (2) messaging gateway — Hermes has
~20 platform adapters + DeliveryRouter + relay, Pori has zero (GW-7 now
UNBLOCKED, Telegram first); (3) provider failover chains + credential pool
(classifier exists, no cross-provider switch). Cheap wins identified:
docx/xlsx/ipynb extraction folded into read_file (stdlib), large tool-result
spill-to-file, `pori doctor`, blueprints (skills x cron frontmatter — both
halves now exist). ALIGNMENT.md updated: SK-5 DONE-with-deviation (cron landed
product-layer), GW-7 unblocked.

## NEW: Marathon Phases 1–3 IMPLEMENTED (2026-07-06, stacked PRs #95/#96/#97)

All three phases of `docs/long-running.md` landed as stacked PRs (merge in
order; user merges). Phase 1 (kernel): write-ahead tool journal
(`status='dispatched'` persisted before side effects; `pending_dispatches()`
after crash), per-step TaskState checkpoint (n_steps/plan/activity),
`Agent(resume_task_id=…)` resume, salvage summary on dead runs
(`result['partial_result']`), compress_context default ON, sqlite config
default. Phase 2: wall-clock budget (`BudgetLedger.start_clock`), orchestrator
resume passthrough, Aloy worker resumes re-claimed runs from `runs.progress`
(new column, migration j6e7f8a9b0c1) — the per-step checkpoint callback IS the
lease heartbeat; docker-compose worker service added (audit blocker closed).
Phase 3: cron engine (`aloy_backend/cron.py`, CronJob table k7a8b9c0d1e2,
croniter dep, /v1/cron CRUD, tick piggybacked on worker loop,
advance-before-enqueue at-most-once); delivery = cron job's conversation_id →
assistant Message on completion. NOT done: delegate_task(background)→run-queue
bridge (API/worker support exists), team checkpointing. Kernel 540 tests,
backend 57 tests, all green. Fixed pre-existing broken
test_streaming_plan (stale poll_interval kwarg).

## NEW: Legacy donor repos retired (2026-07-06)

`pori_cloud`, `pori_cloud_client`, `pori_website` — local folders deleted
(verified fully pushed first; zero unpushed commits) and their GitHub repos
**archived** (read-only, reversible via unarchive). `pori_docs` also deleted at
the user's direction (was not a git repo; held only historical design notes —
Letta memory research, old implementation plans — all superseded by shipped
code and docs/). The monorepo is now the single source of truth; the workspace
holds only `Pori/` (monorepo) + `references/` (Hermes deep-dives). The
"pori_docs will be merged in" note in docs/README.md is now stale — remove it
next time docs/ is touched.

## NEW: Kernel/product separation is now ENFORCED (2026-07-06)

The Pori-vs-Aloy boundary went from designed-on-paper to CI-enforced:
- `tools/ci/importlinter.ini` rewritten for the REAL layout (`pori` at root,
  `aloy_backend` under `products/aloy/backend`) — was inert, referenced a
  never-built `packages/pori` layout. `check-boundaries.sh` activated (handles
  Git Bash/Windows paths via cygpath). Verified both ways: clean tree → 2
  contracts KEPT; injected `import aloy_backend` into kernel → BROKEN, exit 1.
- New `boundaries` CI job in `ci.yml` runs it on every push/PR.
- Kernel wheel verified self-contained: only `pori` + prompts, zero product
  leakage (`uv build` + zip inspection).
- **Discovery: `pori` 1.4.0 IS on PyPI already** (2026-04-23, by Aloy) — the
  kernel is separately consumable today, just 219 commits stale. Next release
  (1.5.0 + changelog) closes the gap. `publish.yml` stale DISABLED header fixed;
  it uses Trusted Publishing — user must configure the Trusted Publisher on
  pypi.org before the next release, or wire the token path.
- README now has a "One kernel, many products" section + `pip install pori`;
  MONOREPO.md and tools/ci/README.md updated from "staged" to enforced.
- User's decision pending (asked, was AFK): enforced monorepo (recommended,
  what was implemented — extraction stays a 2-line swap) vs splitting repos now.
  Nothing done blocks a later split.

## NEW: Full professionalism audit (2026-07-06)

A 4-agent audit of every surface (kernel, backend, app+client, website, desktop, repo
hygiene) produced a prioritized findings list with file:line refs and a suggested order
of attack. **Read `.agent/progress/audits/2026-07-06-professionalism-audit.md` before
starting improvement work.** Headline blockers: LICENSE file missing (MIT advertised in
3 places), release story broken (no PyPI, CHANGELOG 219 commits stale), zero TS tests/CI,
backend durable worker never started in docker-compose, clarify/rate-limit break under
`--workers 2`, ~20 silent catches + no Error Boundary in the app, landing-page CTAs all
`href="#"`, tracked `debug.log`/`config.yaml` at root.

## What Pori/Aloy is

**Pori** is an eval-native, memory-native agent **kernel**. **Aloy** is the first
**product** built on it — a personal + org OS agent (Hermes-class and beyond).
Many products can sit on the same kernel; the repo is structured so any product
can later be lifted into its own repo.

## Actual repo layout (this is real, on `main` — not a plan)

```
pori/                     KERNEL (Python). import pori. Product-agnostic.
  agent/                  the agent as a PACKAGE (split this session, was agent.py 2521 loc):
    core.py (1701)        the Plan→Act→Reflect→Evaluate loop + lifecycle
    prompting.py          system-prompt / message-window / context rendering
    planning.py           optional plan/reflect phases + gating heuristics
    artifacts.py          execution-receipt / tool-artifact tracking
    authorization.py      tool side-effect authorization + HITL resolution
    schemas.py            the pydantic models
    __init__.py           re-exports the public API (unchanged: from pori.agent import Agent, …)
  memory.py, metrics.py, llm/, tools/, orchestrator/, team/, eval/, sandbox/, …
extensions/               reusable pori-* libs (promote-on-second-use; mostly empty)
packages/
  pori-client/            @pori/client — shared TS REST+SSE (PoriEvent) client (was apps/shared)
products/
  aloy/
    backend/              FastAPI — composes the kernel; tenancy/auth/persistence
    app/                  the web SPA (Vite+React) — @aloy/app (was apps/web; renamed for clarity)
    desktop/              Electron shell wrapping app/ (STUB — README only)
    website/              the marketing landing page (self-contained static; bun run dev)
    BOOT.md               how to boot the whole stack locally
package.json + bun.lock   root TS workspace (packages/* + products/*/{app,desktop,website})
MONOREPO.md               ← canonical layout + one-way-dep rule + EXTRACTION PLAYBOOK
docs/aloy-vision.md       ← canonical Aloy product definition
docs/aloy-v1-plan.md      ← active delivery sequence
```

**Naming trap that already bit us:** `products/aloy/app` = the product SPA (needs
Supabase env + backend); `products/aloy/website` = the static landing. To preview
the landing: `cd products/aloy/website && bun run dev`.

## What's BUILT and on `main`

- **Kernel delegation** (`pori/subagents.py`, the `delegate_task` tool): single /
  parallel-batch / background children, leaf-vs-orchestrator depth, curated
  specialists (`.pori/agents/*.md`), provider-agnostic model tiers.
- **Aloy surfaces**, all unified on the kernel's **`PoriEvent`** stream:
  `@pori/client` (typed REST+SSE), the web **app** (live streaming, tool chips,
  delegation, clarify buttons, a Skills screen), and the **backend**
  (`products/aloy/backend`, harvested from `pori_cloud`) — multi-tenant, Supabase
  JWT auth, clarify via a worker-thread `ClarifyBridge`.
- **The moat** (`products/aloy/backend/aloy_backend/scope_resolver.py`): layered
  org→team→personal knowledge; most-specific wins on a `conflict_key`. Personal
  layer populated; org/team slot in with no resolver change.
- **The landing page** (`products/aloy/website/index.html`): calm modern-SaaS
  identity (warm off-white + teal `#0F8571`), self-contained static.
- **Monorepo restructure**: platform (kernel + `packages/*`) vs products; each
  product self-contained + extractable (see `MONOREPO.md` extraction playbook).
- **agent.py → package** (see layout above); public API unchanged; 524 tests +
  mypy green.

## NOT done — next-session targets (roughly in priority, refreshed 2026-07-07)

1. **BOOT THE STACK — still milestone #1, still blocked on the user.** Everything
   through #106 is CI-green + verified-by-construction but has **never run
   end-to-end.** Follow `products/aloy/BOOT.md` (needs a free Supabase project +
   an LLM key; optionally `E2B_API_KEY` to exercise the sandbox and a Telegram
   bot token for the gateway). A real run surfaces the first genuine bugs — the
   marathon/resume, cron, gateway, and sandbox paths are all unexercised live.
2. **Cheap-win batch from `docs/hermes-gap-2026-07.md`** (each small, high-ROI):
   docx/xlsx/ipynb extraction folded into `read_file` (stdlib, no new tool);
   large tool-result spill-to-file; `pori doctor` (expose the existing
   `diagnose_provider()`); blueprints (skill-with-cron-frontmatter — Pori now
   has both halves).
3. **Gateway follow-ups (Tier-1 continues):** voice-note STT (the #1 chat input),
   image input (plumbing exists via #102 — needs an upload path + vision tool),
   group-chat semantics (per-user session lanes, mention gating), a Slack
   adapter, and wiring cron/background completions through the `DeliveryRouter`.
4. **Sandbox end-to-end:** reap idle E2B sandboxes off the worker lease; store
   the sandbox id in `runs.progress` so Aloy resume reconnects it; port the env
   blocklist to the Aloy worker's own subprocess surface. **Open product fork:**
   bring-your-own-key (per-tenant E2B keys) vs Aloy-managed (built) — user decides.
5. **Credential pooling** (multiple keys per provider + cooldown) — the noted
   follow-up to the #103 failover chain.
6. **Finish the moat**: user↔team membership + tagging knowledge as org/team
   (today everything defaults personal) + a Profiles/scope UI.
7. **`main.py` (~1700 loc, the CLI)** — package-split like agent.py, if still
   paying down god-files.
8. **PyPI release:** published `pori` is 1.4.0 (~220+ commits stale now). Cut
   1.5.0 + refresh CHANGELOG; Trusted Publisher must be configured on pypi.org.
9. Other control screens needing backend: MCP (**explicitly parked** — don't
   plan unless asked), Channels/Webhooks, Files/Logs. Desktop (Electron) still a
   README-only stub.

**DONE this stretch (do not re-do):** visual rebrand of the app (#105) + landing
match; cron/Schedules (#97/#100); LICENSE + audit quick wins (#99); kernel/product
boundary in CI (#92); multimodal (#102); failover (#103); Telegram gateway (#104);
sandbox + secrets blocklist + execution-status UI (#106).

## Constraints / process notes to carry forward (do not relearn these)

- **No Claude/AI attribution on commits OR PRs** (harness default injects a
  `🤖 Generated with Claude Code` line — actively strip it; it slipped into #89's
  body once and had to be removed).
- **Architectural, not patches**; **no costly verification gates**; **surfaces are
  copy-then-rebrand from Hermes (MIT), the kernel stays pattern-harvest**
  (never paste). Standing rule: **always reference/harvest Hermes first.**
- Scope `git add` on structural commits — a broad `git add -A` once captured a
  stray `untitled.txt`.
- Background **forks can silently no-op** (return a plan description with 0 tool
  calls) — verify a fork actually executed (check the branch/PR/files) before
  trusting it.
- CI gotcha (fixed): filesystem-tool tests need the pytest tmp base made an
  allowed dir — handled by an autouse fixture in `tests/conftest.py`.

## Canonical docs (read these, don't re-derive)

`MONOREPO.md` (layout + extraction), `docs/aloy-vision.md` (product parent),
`docs/aloy-v1-plan.md` (active delivery),
`products/aloy/BOOT.md` (run it), `docs/Pori.md` (kernel PRD),
`docs/ALIGNMENT.md` (Hermes-alignment tracker), `HARVEST.md` (donor provenance).
