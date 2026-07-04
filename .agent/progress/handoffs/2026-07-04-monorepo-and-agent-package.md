# Handoff — 2026-07-04 — Aloy surfaces, monorepo restructure, agent package

A large session. It took Aloy from "how does Claude Code do subagents" to a
merged, CI-green, multi-product platform, and paid down the kernel's biggest
structural debt.

## What landed on `main` (in rough order)

1. **Kernel sub-agent delegation** — Hermes-style `delegate_task` (goal+context+role,
   leaf/orchestrator depth), parallel + background delegation, provider-agnostic
   model tiers, and an optional Claude-Code-style specialist layer
   (`.pori/agents/*.md`).
2. **Aloy surfaces, unified on `PoriEvent`** — `@pori/client` (typed REST+SSE),
   the web app (live `text_delta` streaming, tool chips, delegation, clarify
   buttons, Skills screen), and the backend (adopted from `pori_cloud`) with a
   worker-thread `ClarifyBridge` for blocking `ask_user`.
3. **The moat** — `scope_resolver.py`: layered org→team→personal knowledge,
   most-specific wins on `conflict_key`; wired into `conversation_runtime`.
4. **Boot kit** — `products/aloy/BOOT.md` (SQLite default, no Postgres).
5. **Landing page** — `products/aloy/website/index.html`, calm modern-SaaS (teal),
   self-contained static, `bun run dev`.
6. **Monorepo restructure (PR #86)** — dissolved root `apps/`; surfaces moved under
   `products/aloy/`; shared client → `packages/pori-client` (`@aloy/shared` →
   `@pori/client`); root bun workspace; `MONOREPO.md` rewritten with an
   **extraction playbook** (products liftable via two dep-source swaps).
7. **`web` → `app` rename (PR #88)** — `products/aloy/web` → `products/aloy/app` to
   stop the web/website confusion.
8. **`agent.py` → `pori/agent/` package (PRs #89, #90)** — 2521-line god-file split
   into core (loop, 1701) + prompting + planning + artifacts + authorization +
   schemas. Behavior byte-identical (methods bound onto the class), public API
   unchanged.
9. **CI fixes (PRs #82, #83)** — filesystem-tool tests failed only on Linux CI
   because the pytest tmp base is outside `$HOME`/cwd; fixed with an autouse
   conftest fixture. Also merged the dependabot dep/action bumps; the 3
   workflow-file bumps (#45–47) were merged by the user (OAuth `workflow` scope).

## Verification

- Full suite **524 passed / 1 skipped**; **mypy clean**; the web app builds
  (`tsc -b && vite build`, 2423 modules); `bun install` links the workspace.
- Every PR was CI-green before merge. Zero open PRs at session end.

## The one thing NOT done

**The stack has never actually run.** It's all verified-by-construction. The next
real milestone is booting it (`products/aloy/BOOT.md`) — needs the user's Supabase
project + an LLM key. Expect the first genuine runtime bugs there.

## Process notes (also in current.md)

- Strip the harness's `🤖 Generated with Claude Code` line from commits AND PR
  bodies (user's standing rule; it slipped into #89 once).
- A background fork **silently no-op'd** (returned a plan, 0 tool calls) — the
  agent.py split was done directly instead. Verify forks actually executed.
- Scope `git add` on structural commits (a stray `untitled.txt` got captured once).
