# Current State

_Last updated: 2026-07-04 (end of a large multi-part session)._

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
