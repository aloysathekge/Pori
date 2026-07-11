# Aloy — the vision

_The canonical product definition. Version 1.0, settled 2026-07-11 after the
founder's ideation sessions and design discussion; all six original question
clusters are closed. **Specs, code, and agents build from THIS document.**
Changes go through discussion with the founder, then land here._

**How to read this if you are an agent working on Aloy:**
§1–2 tell you what the product is. §3 defines every primitive — these are
the nouns of the system; use their names exactly. §4 lists the invariants —
never write code that violates one. §5–6 give the design stance and the
experience targets. §7 maps each primitive to the existing codebase. §8 is
what's deliberately deferred. Deeper codebase context: `CLAUDE.md`
(kernel map), `docs/architecture-primer.md` (hosted-agent fundamentals),
`docs/engineering-excellence-spec.md` (the quality bar).

---

## 1. Thesis

**Aloy is an operating system for persistent events, where agents act on
reality and humans interact through surfaces and conversations.**

Current AI: `conversation → response`. The chat is the container, and
everything the AI does dissolves into scrollback.

Aloy: `reality → Event → Surface + Chat → agent actions → updated reality`.

> **Chat is no longer the container. Reality is the container.**

Aloy is an external brain that doesn't just respond — it continuously
understands, remembers, and acts. It turns moments in a person's life into
dynamic, interactive systems — each with its own UI, logic, and evolving
state — rather than static pages, notes, or threads.

**The philosophy in one split:** the user operates life at the level of
**intentions and decisions**; the system handles everything else.

| | manages |
|---|---|
| **User** | intentions · decisions · review |
| **Aloy** | planning · tracking · execution · adaptation · coordination |

One-line roles: **Life** = intake + coordination layer · **Events** =
execution systems · **Aloy** = operator + intelligence layer · **User** =
decision-maker. The transformation: from *"I need to organize and do
everything"* to *"I define what I want and guide decisions while the system
handles the rest."* Interaction is **decision-only** by design: the user
engages for approvals, tradeoffs, and commitments — everything else is
ambient.

## 2. The core model

```
                      REALITY GRAPH
              (shared objects: calendar, money,
               people, documents, accounts…)
                           ↑ references
                        EVENT GRAPH
                 (Events: the sources of truth)
                            |
        ------------------------------------------------
        |                   |                          |
     SURFACE              CHAT                  AGENT SYSTEM
    "What is?"       "Help me think"           "Make progress"
        |                                              |
   Today / views                                  AGENT TRAIL
  (lenses over the graph)                      "What happened?"
                                                       |
                                                  PROPOSALS
                                            "Should reality change?"
                                                       |
                                            Trust + Routing Engine
                                                       |
                                            Receipts + Validators
                                                       |
                                              Committed Reality
```

An **Event** is the underlying reality. **Surface**, **Chat**, and the
**Agent Trail** are lenses — different ways a human understands and
interacts with that reality. Agents make progress against Events and change
reality only through **Proposals**.

## 3. The primitives

### 3.1 Event — the canonical object

An Event represents something meaningful in the user's life or work: *Trip
to San Francisco. Building Aloy Studio. Applying for a job. Weekly review.
Health journey.* It owns: identity, goals, state, timeline, tasks, projects,
documents, memory references, connected services, agent context. An Event is
not a page, a conversation, or a task list — it is the reality Aloy manages.
Events are permanent; **Sessions** (interaction periods) are temporary.

**Event lifecycle (Machine 2 — deliberately thin, substrate-owned):**

```
Emerging → Active → Dormant → Concluded → Archived
```

- **Emerging** — a noticed pattern, pre-acceptance ("mentioned Japan 4×,
  searched flights twice"). Lives in Life; becomes an Event via Proposal.
- **Dormant** — exists, retains memory, **agents paused, demands no
  attention**. The anti-noise state (the shelved Japan trip). Load-bearing.
- **Phases within Active are model-defined, never substrate-defined.** The
  substrate provides `lifecycle` + a free `phase` value + Trail-recorded
  transitions + triggers on transitions. The *vocabulary* (trip: research →
  booking → travel → reflection; job hunt: discovery → application →
  interview → offer) is composed by the model per event. Hard-coding domain
  phases builds yesterday's assumptions into tomorrow's models.

**The Life Event is the default root — the intake and coordination layer.**
Its five responsibilities:

1. **Capture** — all inputs land here first (thoughts, requests, ideas,
   external signals).
2. **Routing** — does this belong to an existing Event? Should it become a
   new one (via Proposal)? Or stay unstructured?
3. **Undefined-things storage** — "I want to travel someday", "maybe learn
   guitar" live here indefinitely without pressure.
4. **Global awareness — cross-event coordination.** *Events are not
   isolated; Life coordinates them.* Life is itself an Event whose agents'
   scope is the graph: they Observe across Events, detect conflicts (exam
   tomorrow + gym scheduled → suggest adjusting) and priority collisions,
   and raise them as Proposals or attention items. Coordination is
   Observe/Compute (free, Trail-logged); *resolution* routes to the user
   (V1 arbitration). Life is the honest broker — event agents never
   negotiate through a back channel.
5. **The Today view** — what needs attention, what changed, what decisions
   are pending (see §3.3).

Unstructured reality earns structure:
`unstructured reality → Life → Proposal → dedicated Event`. Event creation
is itself a Proposal (Aloy notices, proposes; the user decides).

**The cold start: Aloy is useful before any structure exists.** With zero
Events, Life is a smart evolving stream — you capture anything, get
time-based reminders, ask "what do I have this week?", and see a basic
Today view. Three stages, never forced:

1. **No Events** — Life = stream, Aloy = observer, the user just talks.
2. **Emerging structure** — Aloy forms *soft clusters* internally
   (school-ish, health-ish, social). Soft clusters are **Life's working
   state**: invisible, agent-owned, Trail-logged — the same two-state model
   as every Event, graduating into a visible Proposal only when confident.
3. **Events created** — systems form; automation increases.

> Events are not required to start — they are the *result* of your life
> having structure. Life is your intelligent inbox until patterns are
> strong enough to become systems.

(Wedge consequence: Life + Today ship first; the cinematic dedicated event
comes second. The system must be useful the moment someone says "don't
forget to call mom.")

**Event-worthiness is a model judgment, not a formula** (per invariant #6 —
no mention-count thresholds we'd tune for today's models). The rubric the
model weighs, as qualities:

- **It has a future** — multiple steps over time ("call mom" = a Life task;
  "plan mom's 60th" = an Event).
- **Context accumulates** — history would improve future action.
- **It coordinates** — naturally touches tasks + documents + calendar +
  people together.
- **An agent could act on it** — there's ongoing work a background agent
  could do.

One line: *something deserves an Event when structure would let agents make
progress on it without you.*

**The threshold self-tunes because the errors are asymmetric.** Proposing
too early costs one dismissed proposal — mild, and the dismissal is signal
(learned routing stops re-proposing). Proposing too late costs a slightly
messy Life — recoverable, nothing lost. The genuinely bad behaviors are
already outlawed by the invariants: structure without consent (impossible —
creation is a Proposal) and nagging after dismissal (dismissals demote).
Event-creation proposals ride the attention ladder at the **digest tier by
default** — the morning brief, never a push. Smart-vs-annoying is decided
by cheap, consent-gated, dismissal-learning proposals — not by a cleverer
threshold.

**Recurring events are ONE Event with occurrences** (Weekly Review owns
July 5, July 12… as occurrences inside it). The Event accumulates patterns,
preferences, trust, history; an occurrence is a moment inside it. Splitting
each occurrence into its own event would destroy accumulated context.

### 3.2 Reality Objects — shared truth, referenced not copied

Calendar, Money, People, Documents, Locations, Accounts, Preferences live in
a shared **Reality Graph**. Events hold their own contextual state but
**reference** shared objects (the Trip *uses* the travel-budget object; it
never owns a copy). This prevents duplicated truth, conflicting state,
fragmented memory. In V1, proposals touching shared objects **default to
user arbitration** — agents never negotiate over shared reality silently.
(Object guardians with their own policies are a later evolution.)

### 3.3 Surface — the new primitive

The living visual representation of an Event. Not a dashboard: a dashboard
displays information; a Surface **represents an evolving reality** — it
reflects current state, offers actions, updates as the Event changes, and
adapts to lifecycle and phase (before-trip: planning/packing; during:
real-time context).

Surfaces are **schema-composed from a trusted component vocabulary** (typed
JSON rendered by our components — the model composes, we render; the
vocabulary is a floor, not a ceiling). Branding comes from DESIGN.md-style
design files (Google Labs' open spec), scope-resolved org → team → personal.

The Surface renders: committed truth, relevant working state, and proposals
needing attention — **never unsupported claims** (see §4, Verifiable
Reality).

**The home screen is the Today lens** — a view across the Event graph
(needs-attention, pending proposals, agent activity overnight), NOT an
Event's surface. Life's inbox is one tap away. Home is a lens over the
graph.

### 3.4 Chat — a lens, not the center

The conversational channel: questions, reasoning, brainstorming,
instruction. Chat helps the user interact with an Event; it does not
represent the Event. Chat remains first-class — you can always just talk to
Aloy — but it is no longer the center of the experience.

### 3.5 Agent Trail — trust made visible

The per-Event history of agent activity: what happened, why, what changed.
The Trail is the **narrative** (all agent activity, including working-state
progress); Proposals are the **consent points**. Both are complete by
construction — see the invariants.

### 3.6 Proposal — the agency primitive

Agents do not directly change important reality. They emit **Proposals** —
`action + reason + impact + risk` — and a **routing engine** decides the
latency of consent:

```
Proposal → (risk + context + trust) → Auto | Notify | Ask
```

- Routing inputs: **fixed policies** (always ask: spending money, external
  messages, deleting important data), **user trust grants** ("Aloy may
  manage my calendar"), **learned behaviour** ("you always approve flight
  recommendations").
- **Auto is not a bypass** — it is a proposal where policy already granted
  consent. Notify is do-then-tell. Ask waits.
- **Every Ask carries an expiry and a safe default** ("book the refundable
  option if no answer by Friday" / "withdraw if unanswered"). No agent
  blocks forever; no user is nagged forever.
- Trajectory: from *ask everything* toward **earned autonomy** — routing
  learns; the invariant (no reality change without a Proposal) never moves.

### 3.7 State — Working vs Committed (Machine 1: the fact lifecycle)

Every meaningful piece of state inside an Event is one of two kinds:

- **Working State** — agent-owned scratch (research progress, option
  rankings, drafts). No consent required; still writes Trail entries.
- **Committed State** — reality-backed facts. Reached only through the
  universal fact lifecycle:

```
Working → Proposed → Routed → Executing → Committed + Receipt
                                   ├→ Failed
                                   └→ Withdrawn
```

An Event is a **container of many facts in different states** — never
itself "proposed" or "committed."

### 3.8 The agent boundary — consequence, not capability

Agents freely **Observe** (read anything scoped to their event), **Compute**
(research, compare, simulate, reason), and **Stage** (draft emails, prepare
itineraries, assemble documents) — everything inside the workspace,
consequence-free and reversible. Crossing the **reality boundary** — any
mutation of a Reality Object or the external world (send, book, spend,
publish, delete) — requires a Proposal. Examples: draft email **no** / send
email **yes**; search flights **no** / book flight **yes**.

### 3.9 Attention — how Aloy interrupts a human

Interruption is a routing problem, same shape as Proposals. **The attention
ladder**, least intrusive first:

1. **Ambient** — Surfaces and Today are simply up to date when you look.
   Where most agent activity belongs; the Trail absorbs it.
2. **Digest** — the batched brief (morning/evening/user-scheduled): nothing
   lost, nothing pings.
3. **Inbox/badge** — pending Asks accumulate visibly but silently.
4. **Push** — true interruption: time-critical AND consequential only.
5. **Reach-out** — off-app channels (Telegram gateway, email) past the
   highest bar.

**The attention budget**: a user-set hard cap on interruptions; overflow
degrades down the ladder (if it can wait for the digest, it waits), never
lost. Learned like proposal routing: acted-on interruptions earn their
tier, dismissed ones get demoted. Composition rule: **the Event determines
relevance; urgency picks the rung; the budget caps the volume.**

> Aloy defaults to being findable, not heard. Interruption is a scarce
> resource the user allocates; Aloy spends it like money — budgeted,
> receipted, and learned.

### 3.10 What wakes Aloy — event-scoped triggers

Aloy is event-driven: **time** (Sunday → weekly review wakes), **incoming
information** (email → job-application event updates), **state changes**
(flight cancelled → trip reacts), **patterns** (repeated mentions → propose
an Emerging event). Subscriptions belong to Events, not to a global
firehose: a finance event monitors markets; a trip event monitors flights;
a dormant event monitors nothing.

## 4. The invariants (the constitution — never violate)

1. **No reality change without a Proposal.** Auto/Notify/Ask are latencies
   of consent on one mechanism, not different mechanisms. Protected forever.
2. **Verifiable Reality.** Aloy never represents agent belief as reality.
   The Surface renders only state backed by evidence; committed changes
   require receipts; the checkmark means "this state has a receipt," not
   "an agent said so." (This is the Pori kernel moat — receipts +
   validators — surfaced as product truth: **verifiable agency**.)
3. **The Trail is complete by construction.** All agent activity — working
   state included — writes Trail entries. No invisible magic.
4. **Events reference shared reality; they never copy it.** One truth per
   object in the Reality Graph.
5. **The agent boundary is consequence.** Observe/Compute/Stage are free;
   crossing into shared or external reality requires a Proposal.
6. **Thin substrate, model-defined meaning.** Lifecycle is substrate; phase
   vocabularies, surface compositions, and event semantics belong to the
   model. Build no heuristic a more capable model would obsolete.
7. **Attention is budgeted.** Interruptions degrade down the ladder under
   the user's cap; they are never silently dropped.

## 5. Design stance

**Build for the future where models are very capable.** Engineering effort
goes to what models never bring themselves: durable state, tenancy,
permissions, receipts, event history, trust rails. Model capability fills in
the intelligence — and gets smarter for free; substrate doesn't. Memory
stays an index over durable things (pointers in memory, bytes in storage,
work in the sandbox).

**The context architecture: context is a cache, not a home.** The goal is
never to fit a life into a context window (attention degrades, cost and
latency scale with every carried token) — it is the operating-system answer:
load the **working set**. CoreMemory = registers (identity, always loaded);
the Event's committed + relevant working state = RAM (kilobytes per run);
the Event graph, Reality Objects, captures, and files = disk (queried,
paged in by tools on demand); receipts = the journal. Three consequences:

1. **Scoping is compression — and scoped ≠ sealed.** An agent working the
   trip loads *the trip plus everything the trip references*: its Reality
   Objects (the calendar — which carries the exam; the budget; the people),
   CoreMemory, and task-relevant recall — but never the irrelevant bulk of
   other events' working state. Four layers keep scoped agents from missing
   context: **references** carry cross-cutting facts; **queries** fetch
   known-unknowns on demand (the page fault); **Life's coordinator** watches
   across events for the collisions no scoped agent would look for; and
   **Proposals** gate whatever slips through before it commits. Relevance
   beats volume: the Event decomposition IS the context solution.
2. **State lives as state, not as transcript.** Truth is receipt-backed
   committed state, readable in one fact — never buried in conversation
   scrollback. Chat is disposable; the Event state machine is the memory.
3. **Memory gets denser, not bigger.** Curation distills captures into
   compact knowledge with provenance pointing back to evidence; old signals
   age into structure; forgetting is a feature (retention/supersession).

This is robust to small context windows and scales with big ones — larger
contexts mean bigger working sets and fewer page-ins, never a dependency.
When memory contradicts a receipt, **the receipt wins**.

## 6. The hero flow and the wedge

**Hero flow — the Event loop** (the 60-second demo):

> "Plan my San Francisco trip."

```
Create Event → Surface appears → agent works → Trail explains
→ Proposal appears → user decides → reality changes → Surface evolves
```

The magic moment: watching Aloy transform an intention into a living
system.

**What it feels like across lives** (the same primitives, different
intelligence): a *student's* University event knows exams, deadlines, weak
areas — builds study plans, surfaces urgency ("Math exam in 5 days; you're
behind on Chemistry"). A *developer's* project event analyzes bugs,
monitors deployments, filters interruptions. A *gym* event plans workouts
and tracks consistency. A *household* event coordinates family logistics
and remembers everything. And Life coordinates across them — the exam
reschedules the workout.

**The wedge (V1):** the Event loop end-to-end on a thin slice — Life Event +
one dedicated event type with a Surface, the Agent Trail, Proposals with
routing (Auto/Notify/Ask + expiry defaults), and the Today lens. Full
canvas, deep integrations, guardians, and the Reality Graph's long tail come
after the loop proves itself. **Sequencing:** engineering-excellence phases
0–3 land first (founder directive), then the wedge spec cites this document.

## 7. Vision → substrate map (for agents: where things land)

| Vision primitive | Existing substrate | Status |
|---|---|---|
| Agent system | Pori kernel: loop, sub-agents, teams, streaming, stop/resume | Built |
| Agent Trail | Receipts, traces, run event logs (replayable) | Built (backend); needs per-Event surfacing |
| Proposal routing | HITL config + clarify bridge + policy engine | Seeds built; Proposal object + router are new |
| Verifiable Reality | `ToolExecutionReceipt`, artifact receipts, validators | Kernel moat built; commit-gate wiring is new |
| Reality Objects: Documents | Durable object storage + file library | Built |
| Reality Objects: Accounts | Connections (Gmail/Calendar/MCP, encrypted tokens) | Built |
| Reality Objects: Preferences | CoreMemory (persona/human blocks) | Built |
| Reality Objects: Calendar/Money/People | — | New |
| Event / lifecycle / occurrences | — (conversation is today's aggregate root) | New; **known migration: workspace/memory/files re-key from conversation → Event** |
| Surface schema + renderer | Artifact drawer + schema-driven plan (DESIGN.md adopted) | New (design decided) |
| Triggers: time | Durable worker + cron | Built |
| Triggers: incoming data / state / patterns | — | New (lands on the worker chassis) |
| Attention: digest | Scheduled event on cron | Chassis built |
| Attention: inbox | Query over pending Proposals | Falls out of Proposal model |
| Attention: reach-out | Telegram gateway | Built |
| Attention: push | — | New |
| Memory as index | Knowledge entries + library pointers + scope resolver | Built |
| Event-scoped memory | MemoryRecord scoping contract (org/user/agent/session) | Schema evolution — event dimension rides the conversation→Event migration |
| Life capture stream + soft clusters | Cron chassis + kernel curator concept + typed memory contract | New: captures/signals table + a scheduled Life curation agent (pattern detection = model judgment over the store, per invariant #6) |
| Learned trust (routing) | — | Comes free with the Proposal table (structured decision history — deliberately NOT prose memory) |
| Semantic recall at Life scale | Kernel Chroma archival seam | Hosted wiring (pgvector or similar) when scale demands; seam exists |
| Tenancy (personal + org) | Orgs, RBAC, policies, org→team→personal scoping | Built |

## 8. Deferred (deliberately, not forgotten)

- Reality Object **guardians** (object-owned policies) — after V1
  user-arbitration proves the shape.
- **Freeform code-generated artifacts** (sandboxed mini-apps) — the schema
  DSL is the floor; revisit only if the vocabulary genuinely constrains.
- **Multi-agent negotiation** over shared objects — user arbitrates in V1.
- Cross-user shared Events / shared Surfaces (the org plane's version).
- RAG over file contents — agentic retrieval + memory pointers until scale
  demands embeddings (kernel already has the Chroma seam).

## 9. Process & related documents

Founder ideates (scratchpad `products/aloy.txt` — gitignored, raw, read
critically); we discuss; settled decisions land HERE; specs cite this
document. Related: `docs/engineering-excellence-spec.md` (codebase bar,
phases 0–3 precede the wedge) · `docs/aloy-object-storage-sandbox-spec.md`
(storage substrate) · `docs/architecture-primer.md` (hosted fundamentals) ·
`CLAUDE.md` (kernel map) · `.agent/progress/current.md` (live session
state).
