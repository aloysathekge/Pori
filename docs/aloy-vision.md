# Aloy — the vision, refined

_Living document (started 2026-07-11, major update same day after the
founder's cluster decisions). Part 1 is decided truth — specs build from it.
Part 2 holds what's still open. History: the six original question clusters
were answered via founder ideation (`products/aloy.txt` scratchpad) +
discussion; settled answers live here now._

---

## Part 1 — What Aloy is (decided)

**One sentence:** Aloy is an operating system for persistent events, where
agents act on reality and humans interact through surfaces and
conversations.

**The thesis: chat is no longer the container — reality is the container.**
Current AI: `conversation → response`. Aloy:
`reality → Event → Surface + Chat → agent actions → updated reality`.

### The core model

```
                        EVENT  (source of truth)
                          |
        ----------------------------------------
        |                 |                    |
     SURFACE            CHAT             AGENT SYSTEM
   "What is?"      "Help me think"      "Make progress"
                          |
                     AGENT TRAIL
                   "What happened?"
                          |
                      PROPOSALS
                "Should reality change?"
```

### Event — the canonical object

An Event represents something meaningful in the user's life: Trip to San
Francisco, Building Aloy Studio, Applying for a job, Weekly review, Health
journey. It owns: identity, goals, state, timeline, tasks, projects,
documents, memory references, connected services, agent context. Events are
permanent (Sessions — interaction periods — are temporary). An Event is not
a page, conversation, or task list; it is the reality Aloy manages.

**The Life Event is the default root.** Reality enters unstructured —
thoughts, loose tasks, questions, observations, incoming information — and
lives in Life until it earns structure. Promotion is itself a Proposal:
Aloy notices repeated interest ("mentioned Japan 4×, searched flights
twice") and proposes `Create Event: Japan Trip`; the user accepts.
Lifecycle: `unstructured reality → Life Event → Proposal → dedicated Event`.

**Events contain contextual state but REFERENCE shared Reality Objects.**
Calendar, Money, People, Documents, Locations, Accounts, Preferences are
shared objects in a Reality Graph; the Trip references the travel-budget
object rather than owning a copy. This prevents duplicated truth,
conflicting state, fragmented memory. (Engineering note: several Reality
Objects already have substrate — Documents = the durable file store;
Accounts = connections; Preferences = core memory.)

### Surface — the new primitive

A Surface is the living visual representation of an Event — not a dashboard
(dashboards display information; a Surface represents an evolving reality).
It reflects current state, provides actions, updates as the Event changes,
adapts to context (before-trip: planning/packing; during: real-time), and is
composed from **trusted components via a typed schema** — the model composes,
our components render. Branded via DESIGN.md-style scope-resolved design
files (Google Labs' open spec).

### Chat — a lens, not the center

The conversational channel for questions, reasoning, brainstorming,
instruction. Chat helps the user interact with the Event; it does not
represent it. Chat remains first-class — you can always just talk to Aloy.

### Agent Trail — trust made visible

The history of agent activity per Event: what happened, why, what changed.
(Engineering note: this is the kernel's receipts/traces/run-event-log moat
surfaced as product.)

### Proposal — the agency primitive

Agents do not directly change important reality; they emit Proposals
(action + reason + impact + risk). **Routing is a core primitive** with
three outcomes:

```
Proposal → Risk + Context + Trust → Auto | Notify | Ask
```

Routing considers: **fixed policies** (always ask: spending money, external
messages, deleting important data), **user trust grants** ("Aloy may manage
my calendar"), and **learned behaviour** ("you always approve flight
recommendations"). Approval is just one outcome; Notify is do-then-tell.

### What wakes Aloy — event-scoped triggers

Aloy is event-driven: time (Sunday → weekly review wakes), incoming
information (email → job-application event updates), state changes (flight
cancelled → trip reacts), patterns (repeated Japan mentions → propose an
event). **The anti-noise rule: the Event determines relevance** — a finance
event monitors markets, a trip event monitors flights; subscriptions belong
to events, not to a global firehose.

### The hero flow — the Event loop

> "Plan my San Francisco trip."

`Create Event → Surface appears → agent works → Agent Trail explains →
Proposal appears → user decides → reality changes → Surface evolves.`
The magic moment: watching Aloy transform an intention into a living system.

### Design stance

**Build for the future where models are very capable.** Durable substrate
(events, state, bindings, rendering contracts, proposals/routing, trust
rails) — model capability fills in the intelligence. The component
vocabulary is a floor, not a ceiling. No heuristics that better models will
obsolete. Memory stays an index over durable things.

### The wedge (V1)

The Event loop end-to-end on a thin slice: Life Event + one dedicated event
type with a Surface, the Agent Trail, and Proposals with routing. Full
canvas, deep integrations, and the Reality Graph's long tail come after the
loop proves itself.

### Substrate already built (as of 2026-07-11)

Pori kernel (loop, sub-agents, teams, streaming, stop/resume, receipts);
multi-tenant backend (orgs, RBAC, policy); memory (core + knowledge +
library pointers); durable object storage + sandboxes + provisioning;
connections (Gmail/Calendar/MCP); durable background runs + cron; live
re-attach/stop/continue; premium chat shell; CI gates on every surface.

### Known architectural implication (accepted, not yet executed)

Today the *conversation* is the aggregate root (memory, workspace, files
keyed per-conversation). The model demotes conversation to a lens and
promotes Event to the root — workspaces, files, and context follow the
EVENT across chats. A real but tractable migration; sequence it with the
wedge.

---

## Part 2 — Still open

1. **The Tuesday screen** (last piece of Cluster 1): what do you see when
   you open Aloy — the Life surface? Active-event cards? A "today" braid
   across events? **ANSWER:**

2. **The agent boundary** (Cluster 3, sharpened by Proposals): what may
   Aloy do WITHOUT creating a Proposal, and what always requires one?
   Under active discussion — current working position: the
   workspace/reality line (agents freely observe, compute, and stage inside
   their workspace; ANY mutation of Reality Objects or the external world is
   a Proposal, and routing decides its latency). Plus: do event-internal
   state tweaks (phase updates, checking off the agent's own research task)
   ride the auto tier by default? **ANSWER:**

3. **Event hierarchy**: can Events nest (projects inside "Building Aloy
   Studio")? Do sub-events get their own Surfaces? **ANSWER:**

4. **How Aloy reaches you** (Cluster 3 remainder): the Notify tier's
   channels — in-app inbox, push, daily brief, gateway (Telegram)?
   Frequency discipline? **ANSWER:**

---

## Part 3 — Process

Founder ideates (scratchpad: `products/aloy.txt`, gitignored — read
critically, it's raw); we discuss; settled answers graduate into Part 1;
specs cite this document. Related: `docs/engineering-excellence-spec.md`
(the codebase bar — phases 0–3 precede the wedge),
`docs/aloy-object-storage-sandbox-spec.md`, `docs/architecture-primer.md`.
