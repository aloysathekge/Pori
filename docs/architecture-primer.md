# Where does the agent live? — an architecture primer

_A teaching doc, written 2026-07-10. It uses three systems you already know —
**Hermes** (a local CLI agent), **Pori** (our kernel), and **Aloy** (our hosted
product) — to teach the general software-engineering concepts underneath the
decisions we keep making: statelessness, tenancy, isolation, storage tiers,
compute models, and how files reach an LLM. Read it once slowly; afterwards,
every "should X go in the kernel / backend / sandbox / desktop?" debate should
feel obvious._

---

## 1. The core question

Every AI-agent product must answer one question before any other:

> **Where does the agent's code actually run, and whose machine is it?**

Everything else — what the agent can touch, what persists, what's safe, what it
costs — falls out of that answer. There are three basic answers:

| Model | Runs on | Example |
|---|---|---|
| **Local** | the user's own machine | Hermes, the Pori CLI, Claude Code |
| **Hosted** | your servers, shared by all users | Aloy's backend, ChatGPT, claude.ai |
| **Hybrid** | hosted brain + some local or isolated compute | Aloy + E2B sandbox; future Aloy Desktop |

---

## 2. What "a machine" gives you for free

When a program runs on the user's own computer, it silently inherits four
enormous privileges. They feel so natural that you only notice them when
they're gone:

1. **A persistent disk.** Save `~/.hermes/cache/report.pdf` today; it's still
   there next week. No extra infrastructure — the home directory *is* the
   database, the file store, and the cache.
2. **A shell.** The program can run `pip install pymupdf`, `python extract.py`,
   `git clone …`. Arbitrary compute, no permission system needed beyond the
   OS's, because…
3. **One identity.** Everything on the machine belongs to the one human running
   it. There is no "other user" to protect. Reading files, running commands,
   storing secrets in plain env files — all fine, it's *their* machine.
4. **The user's actual stuff.** Their real Downloads folder, their installed
   apps, their clipboard. An agent here can "organize my desktop."

Hermes is built entirely on these four privileges. That's why its design looks
the way it does: files are cached to disk and the model is told *"the file is
at this path — extract it yourself with the terminal."* Cheap, lazy, perfect —
**on a machine you own.**

The price: the machine sleeps when the laptop lid closes (no always-on), there
is exactly one user (no teams/orgs), and the user has to install and update it.

---

## 3. What a hosted product actually is

A hosted backend (Aloy's FastAPI server) looks superficially like "a machine
that's always on." It is not. It's better modeled as:

> **A stateless request processor in front of shared, durable stores.**

### 3.1 The request lifecycle

Each chat message is an HTTP request: it arrives, a handler runs, a response
streams back, and *everything about that request evaporates*. The next message
may be served by a different process, a different container, even a different
physical machine (horizontal scaling), and any of them may be replaced at any
moment by a deploy.

This is the single most important mental model shift:

- **Local:** "my process runs for weeks; memory and disk accumulate state."
- **Hosted:** "my process may die between any two requests; anything I want to
  keep MUST be written to a durable store before the request ends."

We felt this rule bite three times in one week: the streaming path "forgot" to
persist memory, then traces, then usage — because state lived in the request
and nothing wrote it down. The fix (`persist_run_outcome`, one finalizer both
paths share) is this rule turned into architecture.

### 3.2 Why the local disk lies to you

The server has a filesystem, and in dev (one process, one machine) it behaves
like a laptop's. In production it lies:

- Another **replica** serves the next request → the file "disappears."
- A **redeploy** replaces the container → the file is gone.
- Every **tenant shares it** → org A can read what org B wrote (we found
  exactly this in review: artifacts written to a shared working directory).

Rule of thumb: on a hosted backend, the local filesystem is a **scratchpad for
the duration of one request**, never a store.

### 3.3 The storage tiers that replace it

| Tier | Lifetime | Use for | In Aloy |
|---|---|---|---|
| Request memory | one request | in-flight work | Python variables |
| **Database** | forever, structured | messages, runs, tokens, config | SQLite dev / Postgres prod via SQLModel |
| **Object storage** (S3-style) | forever, blobs | uploaded files, big artifacts | *future slice* |
| Cache (Redis…) | seconds–hours | rate limits, sessions, queues | *future — today in-process* |

Note the review finding that maps here: `CLARIFY_BRIDGES` and the rate limiter
live in **process memory**, which works with one process and silently breaks
with two ("the answer POST landed on a worker that doesn't hold the bridge").
In-process state on a hosted backend is a bet that you'll never scale.

---

## 4. Multi-tenancy: the moment two users share a process

The instant a second user exists, four things stop being free:

1. **Isolation.** Every query must be scoped (`organization_id`, `user_id`).
   One missing WHERE clause is a data breach. This is why Aloy's every table
   and resolver carries tenant columns, and why "personal stays private even
   from admins" needed an explicit test.
2. **Secrets.** A local app can keep an API key in an env file — it's the
   user's own key on their own machine. A hosted product holds *everyone's*
   tokens, so they must be encrypted at rest, scoped per tenant, and never
   placed anywhere shared (our connect-engine: Fernet-encrypted
   `OAuthConnection`, per-user or per-org `scope`).
3. **Compute.** You cannot hand users a shell on the host — that's every
   tenant running commands on the box holding everyone else's data. (This is
   THE reason hosted agents don't naturally have terminals.)
4. **Blast radius.** A crash, a poison-pill job, a full-table scan hits every
   customer, not one laptop. (Worker crash-loop and missing `runs` indexes —
   both review findings — are this category.)

**Sharing** is the flip side — the reward for the pain: only a hosted product
can offer org-shared Gmail, shared MCP servers, member roles, one bill. A
local app has nothing to share through.

---

## 5. The compute ladder for hosted agents

"Users can't have a shell on the host" doesn't mean hosted agents can't
compute. There's a ladder, each rung buying more capability for more cost:

```
Rung 0: in-process tools        — fast, free; runs IN the API process.
        (Aloy's default: file tools, HTTP tools, memory tools)
        Danger: shared cwd, shared process. Fine for trusted, bounded ops.

Rung 1: worker processes        — same trust level, but off the request path.
        (aloy-backend-worker: durable runs, cron, retries, leases)
        Solves ALWAYS-ON, not isolation.

Rung 2: sandboxes (E2B)         — a fresh isolated micro-VM per session.
        REAL terminal + filesystem, but throwaway and metered.
        Solves ISOLATION + "agent needs a machine," hosted.

Rung 3: the user's own machine  — a desktop app bridging local actions.
        Solves "act on MY files/apps." Nothing else can.
```

The key insight (from our discussion): **the sandbox gives every hosted user a
Hermes-style machine on demand** — terminal, filesystem, `pip install` — with
none of the multi-tenant danger, because it's *their own* disposable VM. What
it does *not* give is persistence (VMs are ephemeral and billed by the second),
which is why rung 2 pairs with object storage: **durable files live in
storage; compute is provisioned per run.** Separating storage from compute is
one of the oldest and most reusable ideas in systems design (it's also how
Snowflake, BigQuery, and lambdas work).

---

## 6. Always-on: why laptops can't cron

A local agent dies when the lid closes. Anything that must happen *while the
user is away* — scheduled runs, long marathons, a Telegram bot answering at
3am — requires a process that never sleeps, which means a server:

- Aloy's **worker** claims queued runs with DB leases (so a crashed worker's
  run gets re-claimed), heartbeats progress, and survives redeploys by
  checkpointing to the DB — statelessness applied to long jobs.
- **Cron** piggybacks on the same loop.
- The **gateway** (Telegram) is a subscriber process.

This is the second thing (after orgs) a local architecture can never offer,
and together they're why "just make Aloy local like Hermes" would be a
downgrade, not a simplification.

---

## 7. Case study: getting a user's file to the LLM

This one feature crosses every concept above, which is why it made a good
fight. There are exactly four ways a file can reach a model:

1. **Inline text.** Read the file, paste its text into the prompt.
   Works everywhere, costs context tokens. Right for small text/code files.
   (Aloy: ≤200KB text files inlined as `<attached-file>` blocks; Hermes does
   the same at ≤100KB.)

2. **Native content blocks.** Send the raw bytes to the provider API — models
   can *see* images and *read* PDFs directly now (Anthropic `document` blocks,
   Gemini inline data, OpenAI file parts). Zero agent steps, no deps, works in
   one stateless request. Right default for a hosted product.
   (Pori: `ImageBlock` / `DocumentBlock` mapped by all three adapters.)

3. **Server-side extraction.** Formats no provider ingests (DOCX/XLSX) get
   converted to text on the backend, then path 1. Keep the converter
   dependency-free if you can. (Aloy: Hermes's stdlib OOXML parser, harvested
   — `zipfile` + `ElementTree`, no python-docx.)

4. **Retrieval-based (the Hermes way).** Store the file somewhere the agent
   can reach, tell the model *where it is*, and let it extract what it needs
   with tools. Most flexible, most steps; needs a persistent place + a
   terminal. On a laptop that's the home dir; hosted, it's **object storage +
   sandbox** — the right path for the 200MB CSV the model should analyze with
   pandas, not read as prose.

Why Hermes picked (4) for everything: when it was designed, providers couldn't
read PDFs, and it had a free persistent machine. Why Aloy picks (2) as the
default: providers now can, and a stateless request has no free machine. Same
problem, different constraints, different right answers — **architecture is
context, not doctrine.**

---

## 8. The desktop-app trilemma

"Ship a desktop app" means one of three very different things:

1. **Thin shell** — Electron around the hosted web app. UX sugar (tray,
   notifications). Solves nothing architectural.
2. **Shell + local bridge** — the hosted brain, plus a connection to the
   user's machine: the agent can request local file reads/writes, command
   runs, screenshots — *with consent*, exposed as gated tools that only exist
   while the desktop is connected. Solves the ONE gap nothing hosted can:
   acting on the user's actual machine. Keeps orgs, always-on, mobile.
3. **Fully local** — bundle the kernel + backend on the machine (Hermes
   model). Gains privacy + free persistence; loses orgs, always-on, mobile.

For a personal+org product, (2) is the target: **local hands, hosted brain.**
Note how naturally it fits Pori's existing seams — desktop tools are just
another gated capability group with a `check_fn` ("is this user's desktop
connected?"), exactly like Gmail tools gate on a connection.

---

## 9. How Pori/Aloy embodies all of this

The kernel/product split is what lets one codebase serve every model above:

- **Pori (kernel)** is deployment-blind: the agent loop, tools, memory, LLM
  adapters. It runs identically inside a CLI on a laptop (local model), a
  FastAPI worker (hosted), or — someday — a desktop app. It never knows about
  tenants; it receives *resolved* things (tokens, server lists, attachments)
  per run.
- **Aloy (product)** owns everything tenancy: who you are, what you may use,
  where state persists, which compute rung a run gets. It resolves per-user
  context and hands the kernel a fully-specified run.

That's why adding capability rarely means re-architecting: images/PDFs were a
new *block type* through existing seams; MCP was a per-run *server list*;
desktop will be a *tool provider*. The seams absorb the change.

---

## 10. Rules of thumb (the checklist)

- **State:** if it must survive this request, write it to a durable store
  *inside* the request. If two paths persist the same thing, unify them into
  one finalizer before they drift.
- **Filesystem (hosted):** scratchpad, not storage. Blobs → object storage.
- **Process memory (hosted):** assume N replicas. Registries, limiters,
  queues → shared store, or document the single-process constraint loudly.
- **Every query:** scoped by tenant. Every secret: encrypted, scoped.
- **Shells:** never on the shared host. Sandbox per user-session.
- **Always-on work:** a leased, checkpointed worker — never a request handler.
- **Files → LLM:** small text inline; images/PDF native; office → extract;
  huge/data → storage + sandbox.
- **Local vs hosted is not a war:** hosted+sandbox+storage reaches almost
  everything a local machine offers; the user's own machine is the one
  exception, and a bridge — not a rewrite — is how you get it.
- **When two designs disagree** (Hermes vs Aloy), ask what *constraint*
  each was solving. Copy code where constraints match; change course where
  they don't.

---

*Further reading in this repo: `docs/aloy-send-message-refactor-spec.md` (the
statelessness lesson), `docs/aloy-multitenant-connections-mcp-spec.md`
(tenancy + scope), `docs/pori-mcp-spec.md` (session-scoping vs process-global),
`references/hermes-agent-deep-dives/` (the source studies).*
