# Pori ↔ Hermes Deep-Dive Alignment Tracker

This is the driving checklist for aligning Pori with the Hermes deep-dives. The
source of truth is `references/hermes-agent-deep-dives/` — every row below is
extracted verbatim (in intent and priority) from the "Part B — What Pori can
adopt" section of the corresponding deep-dive. Do not re-prioritize here; if a
recommendation and priority disagree with the doc, the doc wins.

**Legend for Status:** TODO / IN PROGRESS / DONE / DEFERRED

IDs are prefixed per area and numbered in each doc's own priority order:
`AC-` agent-core, `TL-` tools, `SK-` skills/plugins/learning, `GW-` gateway,
`CLI-` cli/tui, `INF-` infrastructure/security.

---

## Agent Core (`agent-core.md`)

Part B is "Ordered by leverage." The single highest-leverage change (prompt
caching) is #1.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| AC-1 | Add Anthropic prompt caching; make the message build cache-friendly (port `prompt_caching.py`; cache system + last-3 + tool schema; move volatile per-step context into one trailing message; date-only timestamps) | `pori/llm/anthropic.py`, `pori/agent.py` (`_build_messages`, `_setup_system_message`), new `pori/llm/prompt_caching.py` | #1 (HIGHEST LEVERAGE) | Low (½–1 day) / Very high (~75% input-token savings) | DONE (2026-07-02) |
| AC-2 | Classify API errors instead of blind retry (trimmed `FailoverReason` + `retryable`/`should_compress` hints; context-overflow → shrink/compress, billing/auth → fail fast) | `pori/llm/retry.py`, new `pori/llm/error_classifier.py`, `pori/agent.py` (`step`/`get_next_action`) | #2 | Medium (1–2 days) / High | DONE (2026-07-02) |
| AC-3 | Replace token-trimming with real context compression (`CompressingContextEngine`: free tool-output pruning pre-pass, head/tail token protection, aux-model structured summary, anti-thrashing) | `pori/context.py`, `pori/memory.py` | #3 | Medium-High (2–4 days; ½ day for pruning pre-pass) / High | DONE — opt-in aux-LLM summary (2026-07-02) |
| AC-4 | Add a `NormalizedResponse`/`Usage` shape in `llm/base.py` to kill provider-key leakage (normalized `cached_tokens`/`finish_reason`; `provider_data` quarantine) | `pori/llm/base.py`, provider wrappers, `pori/agent.py` (remove branching at `676-694`) | #4 | Low-Medium / Medium | DONE (2026-07-02) |
| AC-5 | Loop / no-progress guardrail beyond same-step dedupe (port `tool_guardrails.py`: canonical arg hashing, exact-failure / same-tool / idempotent-no-progress counters, warn-then-halt) | new port of `tool_guardrails.py`, `pori/agent.py` (around `execute_tool`) | #5 | Low-Medium / Medium-High | DONE (2026-07-02) |
| AC-6 | (Optional) Edit→verify-before-finish nudge (port `build_verify_on_stop_nudge`, OFF by default) **and** iteration-budget refund (`IterationBudget.refund`) | `pori/agent.py` (answer/done gate, `BudgetLedger`) | #6 | Low each / Medium (verify-on-stop), Low-until-needed (refund) | DEFERRED (see note) |

> **AC-6 deferred (2026-07-02).** *Verify-on-stop* conflicts with the project's
> "no costly verification gates" rule and the receipt-backed verification V1 that
> was reverted earlier for cost, so it is intentionally not built. *Budget refund*
> is not needed until Pori adds programmatic/nested tool calling (there is no
> per-step budget to protect yet). Revisit if either premise changes.

---

## Tools System (`tools-system.md`)

This deep-dive is a pure technical walkthrough of Hermes' tools directory
(registry/dispatch, Footprint Ladder, built-in tools, security defense-in-depth,
the six terminal backends, MCP integration). It ends at a "Summary of notable
design decisions" and contains **no "Part B — What Pori can adopt" section and
no Pori-file-mapped recommendations**. Nothing is extracted here to avoid
inventing recommendations the doc does not make.

Note: the tools-adjacent adoption items that *do* exist live in the
skills/plugins deep-dive — see **SK-3** (plugin `register_tool` facade) and
**SK-6** (Footprint Ladder discipline + per-tool `check_fn`), and the
tools-security items that surface as adoption recs live in the infrastructure
deep-dive — see **INF-1**, **INF-4**, **INF-5**.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| — | (No Part B recommendations in this deep-dive) | — | — | — | — |

---

## Skills, Plugins & the Learning Loop (`skills-plugins-learning.md`)

Part B is "prioritized, concrete… Highest-leverage first." IDs preserve the
doc's `B.n` numbering.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| SK-1 | ★ FLAGSHIP: Close the learning loop on Pori's existing substrate — (1) `build_learn_prompt` + `/learn` CLI, (2) background review as an Orchestrator post-run hook, (3) deterministic curator (`active→stale→archived`, archive-only) | new `pori/skills_learn.py`, `pori/orchestrator/core.py`, new `pori/curator.py`, `pori/main.py`, `.pori/skill_usage.json` | B.1 | Medium / ★ Very high | DONE — learning loop complete: /learn + background review + curator (2026-07-02) |
| SK-2 | Provenance-gated autonomy (write-origin `ContextVar`, `mark_agent_created`, curator touches only agent-created skills, ceiling = archive) — safety prerequisite for SK-1 layer 3 | new port of `skill_provenance.py`, `pori/orchestrator/core.py`, skill-write tool, `.pori/skill_usage.json` | B.2 | Low / High | DONE (2026-07-02) |
| SK-3 | Real plugin manifest + `PluginContext` facade + fail-*visible* loading (manifest, `register(ctx)`, directory + entry-point discovery, `PORI_SAFE_MODE`, surface load errors) | new `pori/plugins.py`, `pori/tools/standard/__init__.py`, `pori/memory.py` | B.3 | Medium / High | TODO |
| SK-4 | Declarative provider factory (`api_mode` dispatch: anthropic / openai_compatible / google; profiles own `base_url`/`env_vars`; optional `OMIT_TEMPERATURE`, `fetch_models`) | `pori/config.py` (`create_llm`), `providers` module | B.4 | Medium (refactor) / Medium-High | TODO |
| SK-5 | Cron / scheduled automation with typed blueprints — enter at Footprint-Ladder rung 2 (`pori cron` CLI + skill + JSON job store), not a model tool | new (greenfield) `pori cron` CLI + blueprint catalog | B.5 | High / Medium (product-dependent) | DONE (deviation) — landed at the product layer instead of a kernel CLI: Aloy backend cron engine (`pori_cloud/cron.py`, `/v1/cron`, worker tick, at-most-once) + app Schedules screen (2026-07-06/07). Blueprints layer still open — see `docs/hermes-gap-2026-07.md` §8 |
| SK-6 | Adopt the Footprint Ladder as explicit discipline (doc) + finish the `check_fn` rung (per-tool `check_fn` honored in `ToolRegistry.snapshot`) | `CLAUDE.md`/AGENTS adapter, `pori/tools/registry.py` | B.6 | Very low (doc) / Low (code) — Medium impact | DONE — check_fn + Footprint Ladder doc (2026-07-02) |
| SK-7 | Deeper skill security: opt-in AST audit (`pori skills audit --deep`) + honest guard framing (scans are hints, not boundaries; sandbox is the boundary) | new port of `skills_ast_audit.py`, docs | B.7 | Low / Low-Medium | DONE — audit module (CLI subcommand pending) (2026-07-02) |

---

## Gateway & Messaging (`gateway-messaging.md`)

Part B is split into Tier (i) — applies to Pori's existing api/CLI now — and
Tier (ii) — bigger bets gated on Pori becoming a gateway. IDs preserve the doc's
`Bn` numbering; the recommended sequence is B1→B2→B3→B4→B5/B6, then B7–B10 gated.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| GW-1 | Bind session/identity per request; kill the shared-memory bleed (per-request `AgentMemory` scoped to `org:user:agent:session`; share only the namespace-keyed `MemoryStore`) | `pori/api/deps.py`, `pori/api/models.py` | B1 — 🔴 highest leverage (Tier i) | S–M / Very high (fixes multi-tenant bug; precondition for GW-2/GW-3) | DONE (2026-07-02) |
| GW-2 | Split stable `session_key` (lane) from `session_id` (instance); wire the orphaned `pori/sessions.py` into CLI `/new`/`/resume`/`/branch` | `pori/sessions.py`, `pori/memory.py`, `pori/main.py` | B2 (Tier i) | M (mostly wiring existing code) / High | PARTIAL — session_key primitive done; CLI /resume /branch wiring deferred (2026-07-02) |
| GW-3 | Slot-claim-before-await to prevent duplicate concurrent runs (in-process dict keyed by `session_key`, reject/coalesce loser, release in `finally`; 409 on busy) | `pori/orchestrator/core.py`, `pori/api/routers/agents.py` | B3 (Tier i) | S / High | DONE (2026-07-02) |
| GW-4 | Add SSE streaming to the API over the existing `PoriEvent` contract (`asyncio.Queue` + `call_soon_threadsafe`, keepalive timeout, client-disconnect → interrupt) | `pori/api/` (new stream endpoint), `pori/observability/events.py` | B4 (Tier i) | M / High | DONE (2026-07-03) — POST /v1/tasks/stream over PoriEvent; closes on RUN_END |
| GW-5 | Per-turn identity via contextvars, not constructor-threaded globals (`session_id_var`/`user_id_var`/`org_id_var`, set per turn, clear in `finally`, `current_session()` accessor) | `pori/context.py`, `pori/api/` endpoint/deps, `pori/main.py` | B5 (Tier i) | S / Medium now, high later (prereq for GW-6) | DONE (2026-07-02) |
| GW-6 | (Optional now) sync→async bridge helper for blocking tool work (`run_blocking` = `copy_context()` + Pori-owned `ThreadPoolExecutor`); route shell/sandbox through it | new util, `pori/sandbox/` tool execution path | B6 (Tier i) | S–M / Medium (sequence after B1–B5) | DEFERRED — deep-dive marks premature until the API serves concurrent blocking traffic |
| GW-7 | Thin platform-adapter ABC + capability flags + entry-point plugins (`connect`/`disconnect`/`send`/`chat_info`; `pori.surfaces` entry point) | new adapter package (gated) | B7 (Tier ii) | L / High *iff* multi-surface is a real goal | UNBLOCKED (2026-07-07) — the gate condition is met: Aloy's product plan names Telegram (personal) + Slack (org) as surfaces. See `docs/hermes-gap-2026-07.md` Tier 1 §2 for the harvest shape (adapter ABC + registry + DeliveryRouter, Telegram first) |
| GW-8 | Code-based pairing instead of a flat API-key list; immediate partial win: make empty-key case **fail closed** and store hashed keys | `pori/api/security.py` | B8 (Tier ii) | S (fail-closed) / L (full pairing) — Medium impact | TODO |
| GW-9 | Drain / graceful-shutdown for the API (`draining` flag → 503 on `/v1/tasks`, `gather` in-flight with timeout in lifespan) | `pori/api/__init__.py`, `pori/api/background.py` | B9 (Tier ii) | S–M / Medium (hosted product; N/A for a library) | DEFERRED |
| GW-10 | Slash-command access tiers, scale-to-zero, delivery routing | — | B10 (Tier ii) | — / **Skip** unless/until Pori *is* a chat gateway | DEFERRED |

---

## CLI / TUI (`cli-tui.md`)

Part B is "Ordered highest-leverage first." IDs preserve the doc's `Bn`
numbering.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| CLI-1 | Replace ad-hoc slash dispatch with a `CommandDef` registry driving dispatch + help (attach handler to `CommandDef`; kills the stale-help bug; foundation for CLI-2) | new `pori/cli_commands.py`, `pori/main.py` | B1 — ⭐ do first | Medium / High | DONE (2026-07-02) |
| CLI-2 | Add slash autocomplete + history via `prompt_toolkit` (`PromptSession.prompt_async`, `PoriCompleter`, `FileHistory`, `AutoSuggestFromHistory`) | `pori/main.py` | B2 (after B1) | Medium / Medium | DONE — optional prompt_toolkit extra (2026-07-02) |
| CLI-3 | Windows UTF-8 + import-path hardening bootstrap (port `apply_windows_utf8_bootstrap`; import first in cli/`__main__`; lets `_console_safe_text` be dropped) | new `pori/_bootstrap.py`, `pori/cli.py`, `pori/__main__.py` | B3 — ⭐ cheap win | Low (an afternoon) / Medium | DONE — UTF-8 bootstrap (import-path hardening N/A: pori is namespaced) (2026-07-02) |
| CLI-4 | Cooperative Ctrl-C cancel (asyncio, not threads): run turn as task, SIGINT → `run.cancel()`, double-press-within-2s exit; agent loop needs a between-steps cancellation check. Scope to cancel-only (skip full mid-run steering + thread/queue machine) | `pori/main.py`, `pori/agent.py` | B4 — scope carefully | High / Medium | DEFERRED — high-effort async cancellation; revisit with steering |
| CLI-5 | Split `pori/main.py` into CLI modules (`cli_commands.py`, `cli_render.py`, `cli_skills.py`; plain modules, not mixins) — fold into B1; don't over-do | `pori/main.py`, new `pori/cli_render.py`, `pori/cli_skills.py` | B5 — later / fold into B1 | Medium / Medium | DEFERRED — optional refactor; cli_commands/cli_prompt already extracted |
| CLI-6 | Managed uv / dep-ensure | — | B6 — **Skip** (overkill for a pip-installed library) | High / Low | DEFERRED |

---

## Infrastructure, Config & Security (`infrastructure-security.md`)

Part B is "Ordered highest-leverage first." IDs preserve the doc's `Bn`
numbering.

| ID | Recommendation (short) | Target Pori file(s) | Priority | Effort/Impact | Status |
|----|------------------------|---------------------|----------|---------------|--------|
| INF-1 | Hardline command floor + obfuscation-normalized detection in the sandbox (tiny no-recovery-only `HARDLINE_PATTERNS`; `normalize()` NFKC + strip null/`\`/`''`; called *before* HITL so it can't be approved away) | new `pori/sandbox/command_safety.py`, `pori/sandbox/local.py` (`execute_command`) | B1 | Medium (~150 lines + tests) / High | DONE (2026-07-02) |
| INF-2 | Exact/bounded dependency pins + supply-chain CI (bounded `>=x,<next_major` core / `==` for test+all; `osv-scanner.yml`; `dep-bounds` gate; `dependabot.yml`; SHA-pin GitHub Actions) | `pyproject.toml`, `.github/workflows/` (new `osv-scanner.yml`, `ci.yml`, `dependabot.yml`) | B2 | Low–Medium (an afternoon) / High | DONE — SHA-pin of actions deferred (2026-07-02) |
| INF-3 | Secrets-only `.env` + `${VAR}` config expansion (port `_expand_env_vars`; document `.env` as secrets-only; optional startup warning on behavioral keys) | `pori/config.py` (`load_config`), `config.example.yaml`, `CLAUDE.md` | B3 | Low (~30 lines + doc) / Medium-High | DONE (2026-07-02) |
| INF-4 | Harden sandbox path resolution to `resolve()+relative_to()` (symlink-safe; replace string `normpath`+`startswith`; share one resolver with file tools) | `pori/sandbox/path_resolution.py` (`_safe_join`), `pori/tools/standard/filesystem_tools.py` | B4 | Low (~40 lines + tests) / Medium | DONE (2026-07-02) |
| INF-5 | Deterministic prompt-injection / exfil scanner for context + tool results (trimmed `threat_patterns.py`: `all` + `INVISIBLE_CHARS`; warn on tool results, block on memory writes) | new port of `threat_patterns.py`, `pori/memory.py`, internet/web tools | B5 | Medium (~1 day + wiring 2 sites) / Medium | DONE (2026-07-02) |
| INF-6 | Freeze security-gating config at read-time (snapshot `HITLConfig`/`SandboxConfig` at construction; gate `config.yaml`/`.env`/`.pori/` as sensitive write/delete targets) | `pori/agent.py`/`pori/orchestrator/core.py`, `pori/tools/standard/filesystem_tools.py`, INF-1 floor | B6 | Low (~½ day) / Medium | DONE — sensitive-write gate; config already snapshotted at construction (2026-07-02) |
| INF-7 | (Only if adding convenience-install) allowlist-gated lazy deps with spec sanitation; never accept a spec from config/LLM; no `--index-url`/`git+`/paths | new `ensure(feature)` util | B7 — partial adopt | Medium (if adopted) / Low for a library | DEFERRED |
| INF-8 | Testing philosophy: behavior-contracts (assert invariants, not frozen values), no change-detector tests, a real E2E path per security/resolution boundary | `CLAUDE.md`, `tests/` (esp. for INF-1/INF-4/INF-5) | B8 | Low (mostly discipline) / Low-Medium | DONE by practice — behavior-contract tests across INF-1/2/4/5/6 (2026-07-02) |

---

## Suggested execution order

Highest-leverage-first across all areas, respecting the dependencies the docs
state. Prompt caching is the single highest-leverage agent-core change; the
error classifier's context-overflow recovery depends on real compression; the
per-request memory-isolation fix (GW-1) is the prerequisite for all other API
work and is already **DONE**.

1. **AC-1 — Anthropic prompt caching + cache-friendly message build.** The
   single highest-leverage change in the whole tracker (~75% input-token
   savings, ½–1 day). Do first.
2. **GW-1 — Per-request `AgentMemory` isolation (multi-tenant memory-bleed fix).**
   Prerequisite for GW-2/GW-3/GW-4 and a real correctness/privacy bug.
   *Already implemented — DONE (2026-07-02).*
3. **INF-1 — Hardline command floor + obfuscation-normalized detection.** The
   single biggest security gap: Pori ships a shell tool with zero floor. Closes
   the "agent wipes the host" hole no current layer covers.
4. **INF-2 — Bounded/exact pins + OSV scan + dep-bounds + dependabot + SHA-pin
   actions.** High impact, an afternoon; turns "we got lucky" into policy.
5. **AC-3 — Real context compression (`CompressingContextEngine`).** Start with
   the free tool-output pruning pre-pass; it is also the prerequisite for AC-2's
   context-overflow recovery path.
6. **AC-2 — Error classification.** Depends on AC-3 for the `should_compress` /
   context-overflow recovery branch; fail-fast on billing/auth.
7. **CLI-1 — `CommandDef` registry (dispatch + help).** Do-first CLI change;
   kills the stale-help bug and is the foundation for CLI-2 (autocomplete). Pair
   with the cheap CLI-3 UTF-8/import bootstrap win.
8. **SK-1 + SK-2 — Close the learning loop, provenance-gated.** The flagship
   self-improvement differentiator on substrate Pori already paid for; ship
   SK-1 layer 1 (`/learn`) + SK-2 (provenance) first as the safety rail.

Then, in descending leverage: **GW-2 → GW-3 → GW-4** (session lane/instance split,
duplicate-run guard, SSE streaming — the coupled correctness+visibility wins),
**INF-3/INF-4/INF-5** (config secrets, symlink-safe paths, deterministic threat
scan), **AC-4/AC-5** (normalized response, loop guardrail), **SK-3/SK-4**
(plugin facade, declarative provider factory), **CLI-2/CLI-4** (autocomplete,
cooperative Ctrl-C), **GW-5/GW-6** and **INF-6/INF-8** (contextvar identity,
executor bridge, config freeze, test discipline). Treat **SK-5** (cron),
**GW-7/GW-8/GW-9/GW-10**, **CLI-5/CLI-6**, **INF-7**, and **AC-6** as gated on an
explicit product decision or "only if needed" per their docs.
