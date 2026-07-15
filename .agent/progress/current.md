# Current State

_Last updated: 2026-07-15 (Aloy V1 Phase 5 Tasks + Surfaces)._

## NEW: Aloy V1 Phase 5 — Tasks + Project Surface + Today (2026-07-15)

Phase 4 is merged into `aloy-v1` as #167. Phase 5 is implemented on
`aloy-v1-phase-5-surfaces`: manual Project Event creation; two-or-more Sessions
sharing one Event; direct user and agent Task mutations with atomic Trail
entries; a trusted templated Event Surface recomputed from Event/Task/Proposal/
Trail/StoredFile rows; and a Life-first Today lens over pending decisions,
recent committed/evidenced changes, recent Trail activity, and open Tasks.
Today is now the signed-in app landing page. Pending Proposals can be approved
or rejected from both Today and their Event Surface through the Phase 4 commit
rail. `task_create`/`task_update` are Aloy product tools; the kernel gained a
small generic async-tool execution seam so database-backed product tools do not
block the agent loop. Verification: 601 kernel tests passed (1 skipped), all
224 backend tests passed, kernel mypy is clean across 106 files, backend mypy
is clean across 83 files, and the Aloy app build + lint are green. Phase 6 is
the founder-account 60-second hero flow and deliberate crash-window drill; do
not add Reality Objects, learned routing, cross-Event retrieval, or free-form
model-composed Surfaces.

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

`MONOREPO.md` (layout + extraction), `docs/Aloy.md` (product plan),
`products/aloy/BOOT.md` (run it), `docs/Pori.md` (kernel PRD),
`docs/ALIGNMENT.md` (Hermes-alignment tracker), `HARVEST.md` (donor provenance).
