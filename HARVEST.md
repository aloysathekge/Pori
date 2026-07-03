# HARVEST — Provenance Ledger

**Status:** v0.1 · 2026-07-02
**Purpose:** record every pattern Pori/Aloy adopts from an OSS harness — *pattern → source → license → destination → why*. This is "receipts for the codebase": every borrowed idea is traceable and license-clean.

---

## Donor location

Donor OSS repos live **outside this git repo**, at `../references/` (i.e. `Pori/references/`). They are reference-only — **never a runtime dependency**, cloned as external checkouts or submodules.

**Present donors** (as of this pass):

| Donor | Path | License | How we harvest |
|---|---|---|---|
| Hermes Agent (NousResearch) | `../references/hermes-agent` | MIT | adapt + attribute |
| Claude Code | `../references/claude-code` | non-permissive | **design/behavior only (clean-room)** |
| Agno | `../references/agno` | (verify) | verify license before adapting |
| agent-oss | `../references/agent-oss` | (verify) | verify license before adapting |
| Hermes deep-dives / study | `../references/hermes-agent-deep-dives`, `../references/hermes-agent-study-report.md` | — | our own analysis |

> Additional donors to add when needed (see PRD §11): OpenHands (event stream → receipts), Aider (repo map / git-diff), LangGraph (checkpointing/HITL), Letta/MemGPT (memory blocks), Inspect/promptfoo/DSPy (validators/eval), LiteLLM (provider normalization), MCP + AG-UI/ACP (interop/event protocol).

---

## Rules

1. **Harvest patterns, not paste.** Re-express every idea through Pori's contracts (`Receipt`, `Validator`, `MemoryStore`, `ToolBackend`, …). If it can't be expressed as a kernel contract, it doesn't belong in the kernel.
2. **License hygiene per source.** Permissive (MIT/Apache) → adapt with attribution. Non-permissive / closed (e.g. Claude Code) → **clean-room from observable behavior only; never copy source.** Verify a donor's license before adapting its code.
3. **Log every adoption** in the ledger below.

---

## Source → strength map

| Concern | Primary donor(s) |
|---|---|
| Tool ergonomics, permission/hooks, subagent/Task, plan/todo, steering UX | Claude Code *(design only)* |
| Prompt caching, context compression, transport, sandbox, gateway, supply-chain | Hermes |
| Event stream / replay (→ receipts) | OpenHands |
| Repo map, git-diff editing | Aider |
| Checkpointing / HITL interrupts | LangGraph |
| Memory blocks / self-editing memory | Letta/MemGPT |
| Validator / eval methodology | Inspect, promptfoo, DSPy |
| Provider normalization | LiteLLM |
| Tool interop / event protocol | MCP, AG-UI / ACP |
| Multi-agent role/team patterns | Agno, AutoGen, CrewAI |

---

## Ledger

> Add one row per adopted pattern. Keep it honest — an entry is a claim that the pattern was re-expressed, not copied (unless license permits and it's noted).

| Date | Pattern | Source (file if known) | License | Landed in | Why |
|---|---|---|---|---|---|
| _template_ | _e.g. hash-chained receipts_ | _OpenHands event stream / Hermes verification_evidence.py_ | _MIT_ | _packages/pori/receipts_ | _tamper-evident, replayable audit substrate_ |
| 2026-07-03 | Shared TS transport package (typed client + package/tsconfig layout) | Hermes `apps/shared` (`json-rpc-gateway.ts`, `websocket-url.ts`, `package.json`, `tsconfig.json`) | MIT | `apps/shared` (`@aloy/shared`) | one REST+SSE client for web+desktop — **PTY/JSON-RPC-over-WebSocket bridge stripped**, retargeted to Pori's `PoriEvent` SSE contract, rebranded `@hermes/shared`→`@aloy/shared` (surface copy-then-rebrand, per docs/Aloy.md §3a) |
| 2026-07-03 | Web SPA build scaffold + chat structure (Vite+React+Tailwind) | Hermes `web/` (`vite.config.ts`, `package.json` stack, `ChatPage`/`Markdown` patterns) | MIT | `apps/web` (`@aloy/web`) | Aloy chat SPA — harvested the Vite/React/Tailwind scaffold + chat/streaming/clarify structure; **did NOT copy** `@nous-research/ui` (private DS), `@xterm`/PTY, three/leva/gsap, dashboard-token plugin, or the Hermes-only pages; wired to `@aloy/shared`, own theme + branding |
