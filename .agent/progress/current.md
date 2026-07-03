# Current State

## Active Task — Pori → Aloy kernel/product architecture

Established the north-star architecture and began scaffolding it. Pori is being
evolved into an **eval-native, receipt-first, memory-native agent KERNEL**, with
**Aloy** (a personal + org OS agent, Hermes-class and beyond) as the first
product built on it. Multiple future agent products can sit on the same kernel.

Reference OSS lives at `../references/` (hermes-agent, claude-code, agno,
agent-oss) and is mined for best-of-breed patterns (harvest, not paste;
license-clean; logged in `HARVEST.md`).

### Decisions (formalized in docs)

- **Kernel moat = one loop:** work → **receipts** → **validators** judge them →
  verdict is a receipt → continue/halt. Memory writes ride the same rails
  (write → receipt → validate → commit). Receipts + validators + the memory
  **engine** all live in the kernel; tenancy/scope/policy live above it.
- **Three-band monorepo:** `packages/pori` (kernel, publishable) → `packages/ext/pori-*`
  (reusable, promote-on-second-use) → `products/aloy` (product #1). CI-enforced
  one-way deps: `products → ext → pori`, never upward.
- **Kernel keeps Pori's identity:** Plan→Act→Reflect→**Evaluate** loop, unified
  eval/guardrail (→ validators), Trace/Span (→ receipts), CoreMemory Blocks, Team.
- **Roadmap:** Phase 0 (tenancy-aware fixes) → Phase 1 (NormalizedResponse →
  prompt caching → context compression → error classifier) → receipts/validators
  retrofit → memory engine + learning loop → package migration → surfaces/org plane.

### Artifacts created this session (all additive; no existing file modified)

- `docs/Pori.md` — PRD for Pori as a standalone kernel product.
- `docs/Pori_Implementation_Plan.md` — phased implementation plan (M0…M7) with per-workstream
  current-state → target → steps → tests → donor → risks, and 12 flagged open questions.
- `MONOREPO.md` — three-band structure + one-way dependency rule + migration staging.
- `HARVEST.md` — provenance ledger + donor map + license rules.
- `packages/`, `products/`, `tools/ci/` skeleton (READMEs + staged import-linter boundary contract).

## Constraints carried forward

- **No costly verification gates** (a receipt-backed verification V1 was reverted
  earlier for cost). The kernel's receipts are cheap inline appends and validators
  are **deterministic-tier-first, LLM-optional** — consistent with this rule.
- Local sandbox still runs shell with `shell=True`; Phase 0 adds a non-bypassable
  hardline command floor (checked before HITL).

## Open questions (do not assume — see docs/Pori_Implementation_Plan.md §12 / MONOREPO.md)

Kernel-thinness ratification; receipt storage + hash algorithm; event protocol
(AG-UI/ACP vs native); MCP & Team placement; provenance ContextVar placement;
auth default; Python floor; lint stack; naming; pre-1.0 policy; first donors to
clone (OpenHands + Inspect recommended); **repo topology** — sibling projects
`Pori/pori_cloud`, `pori_website`, `pori_docs` live outside this git repo, and
`pori/api` vs the standalone `pori_cloud` need reconciling.

## Alignment is driven by docs/ALIGNMENT.md

Main goal: align Pori with the Hermes deep-dives. `docs/ALIGNMENT.md` is the
consolidated, prioritized tracker (37 recommendations; IDs AC- agent-core,
SK- skills/plugins/learning, GW- gateway, CLI- cli/tui, INF- infra/security;
the tools deep-dive has no Part B). Work items reference their ID.

**Structure — flat, intent-named (like Hermes; no `packages/` wrapper, no
name/name nesting):** kernel `pori/` at the repo root (single root
`pyproject.toml`, `where=["."]` + isort `known_first_party=["pori"]`). Bands:
`pori/` · `extensions/` · `products/aloy/{backend}` · `apps/{web,desktop}` ·
`website/` · `docs/` · `tools/ci/`. Tests/black/isort/mypy green.
`pori/api` still inside the kernel — extract to `products/aloy/backend/` next
(declare fastapi/starlette; fix the `RequestResponseFunction` import).
uv-workspace split (per-package pyprojects) deferred.

**Done:**
- AC-1 (prompt caching) — **DONE**. AC-1a: cache the stable **tools+system**
  prefix (`pori/llm/prompt_caching.py` `cached_system`; `anthropic.py`
  `ainvoke`/`ainvoke_tools` send `system` as a cache-marked block). AC-1b:
  `_build_messages` now puts the volatile per-step context second-to-last so
  system+history+frozen+task is a stable prefix, and `mark_last_messages` marks
  the last 3 (CURRENT TASK stays last — fenced-below-task invariant preserved).
  344 passed. Cache-token metrics were already wired.
- AC-4 (normalized usage) — `Usage` + `normalize_usage()` in `pori/llm/messages.py`
  own the provider token-key knowledge; `agent.py` reads normalized fields instead
  of branching on Anthropic vs OpenAI/Google keys.
- AC-3 (context compression) — opt-in aux-LLM summary of dropped context
  (`pori/compression.py` `compress_context`, gated by
  `AgentSettings.compress_context`, default off); reference-only framed,
  anti-thrash via the summary cache, fail-open. Replaces the role-count stub;
  memory window-split extracted (`_select_window`). Prerequisite for AC-2's
  overflow→compress. 352 passed.
- AC-2 (error classification) — `pori/llm/error_classifier.py` (`classify_error`
  + `FailoverReason` + retryable/should_compress/should_fail_fast hints);
  `retry.py` `is_transient_error` delegates to it; `get_next_action` recovers:
  context-overflow → compress+retry once, auth/billing → `FatalAgentError` halts
  the run (no burning `max_failures` on a hopeless call). 360 passed.
- AC-5 (loop guardrail) — `pori/tool_guardrails.py` `ToolCallGuardrailController`
  (cross-step exact-failure / same-tool / idempotent-no-progress counters,
  warn-then-halt); wired around the tool-execution site; on by default
  (`AgentSettings.tool_loop_guardrail`). 365 passed.
- AC-6 (verify-nudge + budget refund) — **DEFERRED**: verify-on-stop conflicts
  with the no-costly-verification-gates rule + the reverted receipt-verification
  V1; budget-refund isn't needed until Pori adds nested/programmatic tool
  calling. **All Agent-Core (AC-1..AC-6) items are now DONE or intentionally
  deferred** — see `docs/ALIGNMENT.md`.
- **Post-AC follow-ups:** exposed `compress_context`/`tool_loop_guardrail` in
  `config.agent`; verbose run-metrics line now prints `cache_read`/`cache_write`
  (AC-1 visibility); **model-aware context sizing** — `pori/llm/model_context`
  `get_model_context_length` sizes the history budget to the model's real context
  (Claude 200K, GPT-4.1/Gemini 1M–2M, default 128K) via
  `AgentSettings.context_window_auto` (default on), so large-context models use
  their capacity and AC-3 compression is the overflow safety net. 370 passed.
- INF-1 (sandbox hardline floor) — `pori/sandbox/command_safety.py`
  (`normalize` NFKC/de-obfuscation + `hardline_violation`) wired into
  `LocalSandbox.execute_command` *before* HITL so irrecoverable commands
  (`rm -rf /`|`~`, `--no-preserve-root`, `mkfs`, `dd` to raw device, redirect to
  raw device, fork bomb, shutdown/reboot, `kill -1`) are refused unconditionally.
  Tiny/no-recovery-only to avoid false positives; 399 passed.
- INF-2 (supply-chain hardening) — bounded dependency ranges
  (`>=floor,<next_major`) on all core deps + extras in `pyproject.toml`
  (uv.lock unchanged); `tools/ci/check_dep_bounds.py` + a `dep-bounds` CI job
  fail any unbounded `>=`; `.github/workflows/osv-scanner.yml` (detection-only,
  weekly) and `.github/dependabot.yml` (pip + github-actions). 400 passed.
  Follow-up: SHA-pin the GitHub Actions (dependabot manages them meanwhile).
- INF-3/4/5/6 (security hardening) — `${VAR}` config expansion + secrets-only
  `.env` (INF-3); symlink-safe sandbox `_safe_join` via resolve()+relative_to()
  (INF-4); deterministic prompt-injection/exfil scanner `pori/threat_patterns.py`
  — warn on web results, block on memory writes (INF-5); sensitive-write gate on
  config.yaml/.env/.pori in filesystem tools (INF-6). INF-8 satisfied by the
  behavior-contract tests throughout. 428 passed. **INF cluster complete.**
- CLI-1/2/3 (CLI cluster) — `pori/cli_commands.py` central `CommandDef` registry
  driving `/help` + the unknown-command hint (killed the stale hardcoded list);
  `pori/cli_prompt.py` slash-completion + history via optional `prompt_toolkit`
  (`cli` extra, falls back to `input()`); `pori/bootstrap.py` Windows UTF-8
  bootstrap. CLI-4 (async Ctrl-C) + CLI-5 (main.py split) DEFERRED. Also
  committed `uv.lock` (was gitignored) so INF-2 OSV scanning works. 436 passed.
- GW-3/5 + GW-2 (kernel gateway) — `Orchestrator.execute_task` duplicate-run
  guard (`session_key`/`on_busy`, `ConversationBusy`, slot-claim-before-await)
  (GW-3); per-turn identity contextvars `use_identity`/`current_identity` in
  `pori/utils/context.py`, bound per run (GW-5); `build_session_key` lane
  primitive in `pori/sessions.py` (GW-2, CLI resume/branch wiring deferred).
  GW-4 (SSE) DEFERRED — `pori/api` can't import (fastapi undeclared); GW-6
  DEFERRED (premature). 445 passed.
- SK-2/6/7 (skills cluster, small items) — `pori/skill_provenance.py` write-origin
  ContextVar + agent-created ledger (SK-2, safety prereq for SK-1); per-tool
  `check_fn` gating in `ToolRegistry.snapshot` + Footprint Ladder in CLAUDE.md
  (SK-6); `pori/skills_ast_audit.py` opt-in AST hint scanner (SK-7). 452 passed.
  **Remaining SK (larger features, not started):** SK-1 learning loop (flagship),
  SK-3 plugin manifest, SK-4 declarative provider factory, SK-5 cron — each
  warrants a focused session.
- **SK-1 COMPLETE (flagship learning loop)** — layer 1: `/learn` +
  `build_learn_prompt` + `write_skill` (user-triggered authoring). Layer 2:
  `Orchestrator._spawn_background_review` — cheap, non-blocking, isolated review
  agent that mines a finished session and authors a skill (opt-in
  `config.skills.background_review`, `background_review` origin → agent-created via
  SK-2). Layer 3: `pori/curator.py` deterministic curator (active→stale 30d→
  archived 90d, 7d grace, agent-created-only, archive = move to `.archive/`,
  never delete), triggered inactivity-style at CLI startup; selected skills
  recorded as used. Pori now authors, grows, and maintains its own skills. 470 passed.
- Clarify buttons (full loop) — a streamed run that calls `ask_user` with options
  emits a `clarification_request` SSE frame and pauses; `POST /v1/clarify/{id}`
  resumes it with the tapped answer. `clarify.ask_sync` (threading.Event) blocks
  the run (which executes on its own loop in a worker thread, so `ask_user` can't
  deadlock the serving loop); `Agent`/`Orchestrator` thread a `tool_context_extra`
  (the bridge's `clarify_handler`). Completes CLI-menu → gateway-buttons. 490 passed.
- GW-4 SSE — `POST /v1/tasks/stream` streams normalized `PoriEvent`s as
  Server-Sent Events over an `asyncio.Queue` (`on_event` →
  `call_soon_threadsafe`), keepalive on idle, closes on `RUN_END`; client
  disconnect cancels the run. `tests/test_api_sse.py`. This is also the transport
  the clarify `ClarifyBridge` (#58) needs. 487 passed.
- API repair — `pori/api` now imports, starts up, and serves. Bounded `api`
  extra (fastapi/uvicorn/httpx); `middleware.py` uses the modern
  `RequestResponseEndpoint`. `tests/test_api_smoke.py` (TestClient /v1/health +
  lifespan state); the GW-1 isolation test now actually runs. This unblocks GW-4
  (SSE) and the clarify-button last-mile. 486 passed; mypy clean (91 files).
- GW-1 — per-request `AgentMemory` isolation (`pori/api/deps.py`
`get_request_memory` + `Orchestrator.execute_task(memory=...)` override +
`tests/test_api_memory_isolation.py`; 338 passed, 1 fastapi-guarded skip;
black/isort/mypy clean).

**Blocker parked:** `pori.api` can't import in a clean env — `fastapi` is
undeclared and `pori/api/middleware.py` imports `RequestResponseFunction`,
removed from the current starlette. Only blocks API-specific items
(GW-2/3/4/8); fold the dependency-declaration fix into INF-2.

## Next Session Should Start With

Execution order (from `docs/ALIGNMENT.md`): **AC-1** (Anthropic prompt caching —
single highest-leverage change) → INF-1 (sandbox hardline floor) → INF-2
(dep pins + supply-chain CI) → AC-3 (context compression) → AC-2 (error
classifier) → CLI-1 → SK-1/SK-2. **Start AC-1 next.**
