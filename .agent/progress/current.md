# Current State

_Last updated: 2026-07-06 (marathon implementation session)._

## NEW: Tier 1 of the Hermes gap IMPLEMENTED (2026-07-07, PRs #102/#103/#104)

All three Tier-1 items from docs/hermes-gap-2026-07.md, as independent PRs:
(1) #102 multimodal message content — TextBlock/ImageBlock in llm/messages.py,
mapped in all provider adapters (OpenRouter/Fireworks inherit via ChatOpenAI);
str content stays valid everywhere. (2) #103 cross-provider failover —
FailoverChatModel chain consuming the existing error classifier; llm.fallbacks
config; sticky switch; overflow/content-policy deliberately NOT triggers;
credential POOLING still open. (3) #104 Telegram gateway slice —
pori_cloud/gateway/ (BasePlatformAdapter ABC, TelegramAdapter over raw Bot API
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
Phase 3: cron engine (`pori_cloud/cron.py`, CronJob table k7a8b9c0d1e2,
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
  `pori_cloud` under `products/aloy/backend`) — was inert, referenced a
  never-built `packages/pori` layout. `check-boundaries.sh` activated (handles
  Git Bash/Windows paths via cygpath). Verified both ways: clean tree → 2
  contracts KEPT; injected `import pori_cloud` into kernel → BROKEN, exit 1.
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
docs/Aloy.md              ← the Aloy product plan (surfaces, moat, streaming)
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
- **The moat** (`products/aloy/backend/pori_cloud/scope_resolver.py`): layered
  org→team→personal knowledge; most-specific wins on a `conflict_key`. Personal
  layer populated; org/team slot in with no resolver change.
- **The landing page** (`products/aloy/website/index.html`): calm modern-SaaS
  identity (warm off-white + teal `#0F8571`), self-contained static.
- **Monorepo restructure**: platform (kernel + `packages/*`) vs products; each
  product self-contained + extractable (see `MONOREPO.md` extraction playbook).
- **agent.py → package** (see layout above); public API unchanged; 524 tests +
  mypy green.

## NOT done — next-session targets (roughly in priority)

1. **BOOT THE STACK.** Everything is verified-by-construction but has **never run
   end-to-end.** Follow `products/aloy/BOOT.md` (SQLite default, needs a free
   Supabase project + an LLM key). This is *the* milestone; a real run will surface
   the first genuine bugs. Blocked on the user's Supabase + key.
2. **Finish the moat**: user↔team membership + a way to tag knowledge as org/team
   (today everything defaults personal) + a Profiles/scope UI.
3. **Visual rebrand of the app**: swap `pori-*` logo/assets in
   `products/aloy/app/public`, tune the palette to Aloy's.
4. **`main.py` (1699 loc, the CLI)** — same package-split treatment as agent.py, if
   we keep paying down god-files.
5. Other Hermes control screens needing backend: Cron, MCP, Channels/Webhooks,
   Files/Logs. **MCP is explicitly parked** (do not plan it unless asked).
6. Desktop (Electron) is a stub.

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

`MONOREPO.md` (layout + extraction), `docs/Aloy.md` (product plan),
`products/aloy/BOOT.md` (run it), `docs/Pori.md` (kernel PRD),
`docs/ALIGNMENT.md` (Hermes-alignment tracker), `HARVEST.md` (donor provenance).
