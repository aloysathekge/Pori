# Aloy — the vision, refined

_Living document (started 2026-07-11). This is where "my version of Aloy"
gets pinned down. Part 1 is what's already decided across the ideation
sessions (`products/aloy.txt` iterations + conversations). Part 2 is the six
question clusters that determine the product's shape — Aloy (the user)
answers inline under each **ANSWER:** marker; we then discuss, and settled
answers move up into Part 1. Specs build from THIS document, not from vibes._

---

## Part 1 — What Aloy is (decided so far)

**One sentence:** Aloy is an agentic operating system for your life and work
— an external brain that doesn't just respond, but continuously understands,
remembers, and **acts**.

**The organizing primitive is the EVENT, not the chat.** Each important part
of life — "Trip to San Francisco", "Weekly review", "Building Pori this
week" — gets its own dedicated, branded, evolving interface: a living space,
not a page. Interfaces are **state-aware** (you check into the hotel, the
space reflects it) and **time-aware** (before-trip shows planning and packing;
during-trip shows real-time context; afterwards, progress). Chat remains one
surface among several — you can always just talk to Aloy.

**The three-way loop is the edge:** User ↔ Interface ↔ Agent. You interact
with an event's interface; it updates the system instantly; agents react —
suggest, replan, modify the interface. Nobody's chat app closes this loop.

**Interfaces are schema-driven, not generated code.** The model composes an
interface from a trusted component vocabulary (typed JSON schema rendered by
our components) — safe, versionable, diffable, brandable. Branding comes from
DESIGN.md-style design files (Google Labs' open spec), scope-resolved like
knowledge: org brand → team → personal.

**Design stance: build for the future where models are very capable.** We
build durable substrate — events, state, bindings, rendering contracts,
permissions, trust rails — and let model capability fill in the intelligence.
The component vocabulary is a floor, not a ceiling. No hand-crafted
heuristics that better models will obsolete. (Models get smarter for free;
substrate doesn't.)

**Memory is an index over durable things.** Aloy's memory holds pointers —
to files, events, facts — not copies. The bytes live in durable storage; the
agent fetches what a task needs into its workspace. (Built and proven: the
file library / "my CV" flow.)

**The wedge (V1): the Agentic Task + Life Manager** — tasks, events,
documents, one or two background agents — with the event-based interface as
its display layer. Full automation, deep integrations, and the complete
canvas come after the wedge proves the loop.

**Already true today (the substrate so far):** the Pori kernel (agent loop,
sub-agents, teams, streaming, stop/resume, receipts); multi-tenant backend
(orgs, RBAC, policies); memory (CoreMemory + knowledge + library); durable
object storage + per-conversation sandboxes; connections (Gmail, Calendar,
MCP); durable background runs + cron; live re-attach/stop/continue; the
premium chat shell.

---

## Part 2 — The six question clusters (answer inline)

### Cluster 1 — The event primitive ⭐ (everything bends around this one)

"Trip to SF" is bounded and phased. "Weekly review" recurs. "Building Pori"
is an ongoing project. "Health" never ends. Are these all ONE kind of thing
(events), or two (events vs. areas of life)? Does everything in Aloy live
inside some event, or do free-floating tasks/notes exist? Who creates events
— only you, explicitly, or can Aloy notice ("you've mentioned SF flights
three times…") and propose one? What ends an event, and what happens to its
space afterwards (archive? searchable history? feeds memory?)

Also: when you open Aloy on a random Tuesday, what do you SEE first — a list
of current events? A "today" space drawing from all of them? Describe the
first screen as you picture it.

**ANSWER:**

### Cluster 2 — The interfaces

Is an event's interface generated once and then *evolved*, or re-generated
at each phase change? If you manually rearrange or edit an interface, does
your layout win over the agent's next update (and how do the two coexist)?
What belongs in the smallest component vocabulary that makes the trip AND the
weekly review feel alive (checklist, timeline, card, form, button, metric,
map…)? Should interfaces be shareable (send someone your trip space, read-
only) — now or someday?

**ANSWER:**

### Cluster 3 — Agency & trust

What may an agent do with nobody watching? Draft an email vs SEND it;
suggest a booking vs MAKE it; reprioritize your tasks vs propose the change.
Where's your line today, and where should the line be able to move per-user
(a trust dial)? And in reverse: how does Aloy reach YOU — a morning brief?
push notifications? or silently-up-to-date spaces you visit when you want?
How does it ask when it's unsure mid-task and you're not there?

**ANSWER:**

### Cluster 4 — What wakes the system

Time (schedules), incoming data (email/calendar events), state changes (task
completed), observed patterns (you always plan Sundays)? Which of these
should trigger agents in V1? And the taste question: what separates
"proactive" from "annoying" for you personally — frequency caps? importance
thresholds? everything visible-but-quiet until you look?

**ANSWER:**

### Cluster 5 — The data spine

How do events, tasks, files, memory, and chat relate? Is chat per-event, one
global chat that routes to events, or both? When an agent asks "what is the
state of Aloy's (the user's) life right now?" — what is the ONE structure it
consults? What's canonical vs derived?

**ANSWER:**

### Cluster 6 — The hero flow

Which single experience, done end-to-end, makes someone FEEL the whole
vision in 60 seconds? Candidates: (a) the **weekly review** — recurring, so
it compounds; exercises tasks + interface + agent reaction; (b) the **trip**
— cinematic, phased, time-aware; (c) something else you have in mind. Which
one is THE demo, and what exactly happens in those 60 seconds?

**ANSWER:**

---

## Part 3 — Process

1. Aloy answers the clusters above (rough is fine — voice-note-style text
   welcome).
2. We discuss each cluster; disagreements get argued, decisions get made.
3. Settled answers move into Part 1 as decided truth; this file is the
   canonical product definition.
4. The wedge spec (and everything after) cites this document.

Related: `docs/aloy-object-storage-sandbox-spec.md` (storage substrate),
`docs/engineering-excellence-spec.md` (the bar the codebase meets first),
`docs/architecture-primer.md` (hosted-agent fundamentals), aloy-vision memory
notes (session-level context).
