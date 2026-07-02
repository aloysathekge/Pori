# Pori — Implementation Plan

**Status:** Draft v0.1
**Date:** 2026-07-02
**Derived from:** [`Pori.md`](./Pori.md) (PRD). This plan formalizes decided work; it does not add new architecture. Undecided points are marked **⚠ OPEN QUESTION** and consolidated in §12.

> Ground-truth code references below are to the **current** `pori/` tree (single-package layout, before the monorepo migration). Line numbers are as observed during architecture review and should be re-verified at implementation time.

---

## 0. Approach & sequencing philosophy

- **Kernel-first, incremental, in-place then migrate.** Stand up the workspace skeleton + CI boundary check early; keep code in today's `pori/` while doing Phase 0/1; land each change in its *target* band; migrate modules into packages incrementally; **promote reusable code into the extension band on second use**, not on spec.
- **Every workstream is specified as:** *current state (file refs) → target → steps → tests → donor to harvest → risks.*
- **Testing discipline (from Hermes AGENTS.md):** behavior contracts over snapshots; E2E over green mocks for resolution/config/security/IO; runnable with empty API keys; subprocess-per-file test sharding.
- **No big-bang refactor.** The kernel/ext/product split is realized by *moving code that already works*, behind interfaces, after it is hardened.

---

## 1. Milestone M0 — Workspace & tooling skeleton

**Goal:** make the three-band structure and the one-way dependency rule real before code moves.

**Steps**
1. Convert the repo to a **uv workspace** with:
   - Flat, intent-named bands: `pori/` (kernel), `extensions/`, `products/aloy/`, `apps/`, `website/`, `references/`, `tools/ci/`.
   - **Single root `pyproject.toml`** for now (`packages.find where=["."]`); the per-package uv-workspace split is deferred until `pori` is published standalone or `extensions`/`products` add Python packages.
2. Add the **dependency-boundary check** (e.g. `import-linter` contract or a small script in `tools/ci/`) enforcing `products → ext → pori`, never upward. Wire into CI as a required gate.
3. Add **supply-chain gates** to CI: OSV scan (lockfile), a dependency-bounds check (reject unbounded `>=`), and SHA-pinned actions. *(Donor: Hermes `osv-scanner.yml`, `supply-chain-audit.yml`.)*
4. Create `references/HARVEST.md` (ledger scaffold: pattern → source → license → destination → why) and add donor repos to `references/` (git-ignored or submodules).
5. Keep the existing `pori/` importable during migration (a shim or path alias) so nothing breaks mid-move.

**Definition of Done:** CI runs the boundary check + tests + supply-chain gates; a trivial `pori` build succeeds standalone; `HARVEST.md` exists.

**⚠ OPEN QUESTION:** Python floor (3.10 vs 3.11), lint stack (black/isort/mypy vs ruff), package/namespace naming, repo root name — resolve before finalizing `pyproject`s.

---

## 2. Milestone M1 — Kernel contracts (define before retrofitting)

**Goal:** lock the two foundation types and the core interfaces; everything else hangs off them.

**Artifacts to define (sketches — exact fields finalized here, subject to the open questions):**

- **`Receipt`** — typed record with: `id`, `kind` (llm/tool/memory/validator/subagent/step), `actor`/`tenant` tags, `inputs`/`outputs` (hashed, with content refs), `parent_id`, `prev_hash` (hash-chain), `evidence_refs`, `cost`/`tokens`, `timing`. *(Seed: `observability/` `Trace`/`Span`, `events.py:16-39` `PoriEvent`; donor: OpenHands event stream.)*
- **`Verdict`** — `{ outcome: pass|warn|block|score, reason, evidence_refs, severity }`.
- **`Validator` ABC** — `run(receipts) -> Verdict`, plus scope + lifecycle metadata (pre-gate / post-check / eval), tier (deterministic | llm). *(Seed: `eval/` `BaseEval` + `guardrails`; donor: Inspect graders, DSPy.)*
- **Event/streaming protocol** — formalize atop today's `PoriEvent`. **⚠ OPEN QUESTION:** adopt AG-UI/ACP vs Pori-native.
- **Interfaces (ABCs):** `MemoryProvider`, `MemoryStore`, `SkillProvider`, `ToolBackend`, `ContextEngine` (the latter already exists at `context.py:31-42`).
- **`NormalizedResponse` + `Usage`** — flat response type (content, tool_calls, finish_reason, usage incl. cache tokens) with property shims to avoid churn at call sites.

**Definition of Done:** the types/ABCs compile as the kernel's public surface; a receipt can be emitted, hash-chained, and replayed in a unit test; a trivial deterministic validator returns a Verdict recorded as a receipt.

**⚠ OPEN QUESTIONS:** receipt storage backend + hash algorithm; event-protocol standard; final kernel boundary ratification.

---

## 3. Phase 0 (Milestone M2) — Tenancy-aware fixes

Three self-contained fixes; each is also the first brick of an Aloy pillar. Read each file first to confirm current shape.

### 0.1 — API per-request memory isolation *(Tenancy Pillar v0)*
- **Current:** `pori/api/deps.py:18-53` constructs one `AgentMemory` at startup and hands the same instance to every request; memory is namespaced per `org:user:agent:session` (`pori/memory.py:355-361`). Concurrent callers share one transcript.
- **Target:** per-request `AgentMemory` bound to the request's full tenant path, sharing only the namespace-keyed `MemoryStore`.
- **Steps:** move memory construction into a request-scoped dependency; key on the full tenant path; the store remains the shared, namespaced backend.
- **Tests:** two concurrent requests with distinct tenant paths never observe each other's messages (E2E against a temp store).
- **Risk:** ensure no other code caches the startup memory instance.

### 0.2 — Fail-closed auth
- **Current:** `pori/api/security.py:26-30` allows all requests (with a `print`) when no key is configured.
- **Target:** fail closed — require a key. **⚠ OPEN QUESTION / assumed default:** allow an explicit `PORI_ALLOW_NO_AUTH=1` local-dev opt-in.
- **Tests:** no key + no opt-in → 401; opt-in flag → allowed with a loud warning.

### 0.3 — Sandbox hardline command floor
- **Current:** `pori/sandbox/local.py:41-64` runs `subprocess.run(shell=True)` with no dangerous-command detection; only opt-in HITL gates it.
- **Target:** a small `HARDLINE_PATTERNS` set + a `normalize()` pass (NFKC, strip null bytes / backslash-escapes), checked **before** HITL so it cannot be approved away. This is the base layer of the future validator safety floor. *(Donor: Hermes `approval.py` hardline + normalization.)*
- **Tests:** representative destructive commands (and obfuscated variants) are blocked regardless of HITL/auto-approve; benign commands pass.

**Definition of Done:** all three fixed with tests; no regression to existing agent runs.

---

## 4. Phase 1 (Milestone M3) — Cost/robustness core

Build in dependency order to minimize rework. Preserves the Evaluator loop, CoreMemory, and tracing.

### 1.1 — `NormalizedResponse` (foundation)
- **Current:** provider-specific handling and branching (e.g. `agent.py:676-694`); Anthropic path sends system prompt as a plain string (`llm/anthropic.py:177-199`).
- **Target:** one `NormalizedResponse`/`Usage` in `llm/base.py`; adapters map each provider; property shims keep existing call sites working; the loop stops branching on provider.
- **Tests:** golden per-provider adapter tests mapping raw → normalized; the loop consumes only the normalized type.
- **Donor:** Hermes transports (`agent/transports/*`), LiteLLM (normalization idea).

### 1.2 — Prompt caching (the highest-leverage change)
- **Current:** no `cache_control` emitted anywhere; volatile content (runtime facts, recent actions, current task) is appended at the *tail* each step (`agent.py:_build_messages:880-953`), and the system prompt is a plain string — so the full prefix is re-billed every step (default `max_steps` high). Cache-token fields already exist to be read (`agent.py:682-687`, `anthropic.py:203-211`).
- **Target:** 3-tier prompt — **stable** (identity/tool guidance) / **context** (skills index, context files) / **volatile** (memory snapshot + runtime facts in *one* trailing message, **date-only** timestamp) — built once per run; emit `cache_control` on system + last-N messages + tool schema. CoreMemory Blocks map onto the volatile tier (identity preserved).
- **Tests:** prompt is byte-stable across steps except the volatile trailing block; cache-read tokens become non-zero on step ≥2 (integration).
- **Donor:** Hermes `prompt_caching.py`, `system_prompt.py`.

### 1.3 — `CompressingContextEngine`
- **Current:** context is trimmed to a small window (`agent.py:115`, ~3000 tokens) and the "summary" is a non-LLM stub listing role counts/snippets (`memory.py:550-565`) — long tasks lose information. The seam exists (`context.py:31-42` `ContextEngine`).
- **Target:** a `CompressingContextEngine` implementing the existing seam: free tool-output pruning pre-pass, then an aux-model structured summary with anti-thrashing (skip if recent passes saved little), protecting head/tail and never splitting tool-call/result pairs.
- **Tests:** compression reduces tokens beyond a threshold without dropping the latest user turn or splitting tool pairs; anti-thrash guard prevents repeated no-progress passes.
- **Donor:** Hermes `context_compressor.py`.

### 1.4 — Error classifier
- **Current:** `llm/retry.py` matches exception names/status codes only; context-overflow 400s are re-raised into `consecutive_failures`; hopeless billing/auth errors may be retried.
- **Target:** a `FailoverReason` taxonomy with precomputed hints (`retryable`, `should_compress`, `should_fallback`) feeding the retry loop: overflow → compress (needs 1.3), auth/billing → fail-fast.
- **Tests:** representative provider errors classify correctly and route to the right recovery; overflow triggers a compression pass then retry.
- **Donor:** Hermes `error_classifier.py`.

**Definition of Done:** all four landed with tests; measurable cache hit + token reduction on a multi-step run; overflow recovers via compression instead of failing.

---

## 5. Phase 2 (Milestone M4) — Receipts + validators retrofit; sessions; CLI

### 2.1 — Receipts v1 (make the trace a receipt chain)
- **Current:** `observability/` `Trace`/`Span` + `PoriEvent` (`events.py:16-39`).
- **Target:** add hash-chain + evidence links + actor/tenant tags to the receipt type; wire the loop so every LLM/tool/memory/subagent step emits a receipt; a completed run is replayable from its chain.
- **⚠ OPEN QUESTION:** receipt store backend + hash algorithm (M1 decision).
- **Donor:** OpenHands event stream / trajectory; Hermes `verification_evidence.py`.

### 2.2 — Validators v1 (tiered registry)
- **Current:** `eval/` `BaseEval` (Accuracy/Reliability/Performance/AgentJudge) + `guardrails` (ContentPolicy/Factuality/Topic).
- **Target:** unify under the `Validator → Verdict` contract; a tiered registry (deterministic-first, LLM optional); run validators as pre-gate and post-check in the loop; port concrete validators (threat-regex, hardline floor from Phase 0.3, verify-on-stop) as the starter library; verdicts emitted as receipts.
- **Donor:** Inspect / DSPy (methodology); Hermes `threat_patterns.py`, `approval.py`, `verification_stop.py`.

### 2.3 — Memory writes on the rails
- **Target:** route memory writes through **write → receipt → validate → commit** (using 2.1 + 2.2), closing the loop described in the PRD.

### 2.4 — Sessions: key/id split + duplicate-run guard
- **Current:** `pori/sessions.py` has full lineage/branch support but is **orphaned** (unused by api/CLI); `Orchestrator.execute_task` has no dedup (`orchestrator/core.py:66-101`).
- **Target:** wire `sessions.py` in; adopt a stable `session_key` (lane) vs `session_id` (instance) split; add slot-claim-before-await keyed on `session_key` to prevent duplicate concurrent runs; enable resume/branch as id-swaps on the same key.
- **Donor:** Hermes `gateway/session.py`, `run.py` slot-claim.

### 2.5 — CLI command registry
- **Current:** ad-hoc `if/elif` dispatch in the real CLI (`pori/main.py`); hand-written help already out of sync.
- **Target:** a single `CommandDef` registry as source of truth driving dispatch + help + autocomplete (attach handlers to `CommandDef` to eliminate the dispatch ladder).
- **Donor:** Hermes `hermes_cli/commands.py`.

**Definition of Done:** every step produces a receipt; a run replays deterministically; validators gate and check in the loop; memory writes are receipted+validated; sessions support resume/branch with no duplicate runs; CLI help is registry-derived.

---

## 6. Phase 3 (Milestone M5) — Memory engine + learning loop

- **Kernel:** formalize the memory engine (block model, recall→inject cache-safely, write lifecycle, `MemoryStore` interface) in `pori/memory/`.
- **Extension/product:** the **org→team→personal scope resolver** (layered inheritance; populate the personal layer first, org/team resolve to empty but the resolution path exists), RBAC, concrete stores, and the **autonomous learning loop** (learn / background-review / curator) — gated by write-origin **provenance** so autonomy only touches agent-created artifacts, archive-not-delete.
  - **Current substrate to reuse:** `pori/skills.py`, `skills_tools.py` (progressive-disclosure skills), `pori/evolution.py` (eval-gated self-evolution), the experiences store.
- **⚠ OPEN QUESTIONS:** provenance ContextVar placement (kernel receipts vs ext); how much of memory-scope is `ext/pori-tenancy` vs `products/aloy`.
- **Donor:** Letta/MemGPT (memory), Hermes `learn_prompt.py`, `background_review.py`, `curator.py`, `skill_provenance.py`.

**Definition of Done:** memory engine is kernel-clean behind its interface; a personal-layer scope resolver works with the org/team path present-but-empty; an autonomous post-run review can propose a skill/evolution gated by provenance.

---

## 7. Phase 4 (Milestone M6) — Package migration & publish

- **Move** current modules into their bands per the boundary:
  - **`pori`:** runtime (`orchestrator/`, loop skeleton, `**⚠ Team placement OPEN**`), `llm/`, tools engine (`tools/registry.py`, executor), `observability/` (receipts + event protocol), `context.py`, `sandbox/`, validation (Validator interface + safety floor), memory engine (`memory/`), interfaces.
  - **`extensions/pori-*`:** memory-scope/tenancy, skills, learning, providers, gateway (later), cli-kit.
  - **`products/aloy`:** the backend (**evolve `pori/api`** into `aloy-api`), aloy-cli, org policy, tenancy shape.
- **Enforce** the dependency-boundary check throughout; **publish** `pori` standalone (pre-1.0; breaking changes allowed during migration — **⚠ confirm**).
- **Concrete/standard tools** move to ext or product per the footprint discipline.

**Definition of Done:** `pori` builds/publishes standalone with no `ext`/`product` imports; boundary check green; Aloy composes on top.

---

## 8. Phase 5 (Milestone M7+) — Surfaces & org plane (Aloy product)

- **SSE on the backend:** `TaskCreateRequest.stream` is currently a no-op (`api/models.py:11`); implement SSE by bridging the existing `PoriEvent` stream onto an async queue (keepalive, disconnect→interrupt). *(Donor: Hermes `api_server.py` pull-adapter.)*
- **Surfaces (harvest, don't clone):** adapt Hermes `web/` (SPA) and `apps/desktop` (Electron shell + shared transport), **strip the PTY/JSON-RPC bridge**, retarget to REST + SSE.
- **Gateway:** port the *architecture* (thin adapter ABC, typed streaming, session lanes) onto Pori's events; start Slack (org) + Telegram (personal) — not the monolithic runner.
- **Org plane:** tenancy/RBAC models, the org **policy engine** (as scoped validators), audit + cost attribution (from receipts), and an admin/control plane (seed: Hermes managed-scope).

---

## 9. Testing & CI strategy

- **Behavior contracts over snapshots**; **E2E over mocks** for resolution/config/security/IO; **empty-key** runs; **subprocess-per-file** sharding.
- **Dependency-boundary check** as a required gate.
- **Supply-chain:** bounded/pinned deps, OSV scan, dependency-bounds check, SHA-pinned actions.
- **Kernel-standalone test:** `pori` passes its suite with no product installed.

---

## 10. Migration & compatibility

- **Pre-1.0:** breaking changes allowed during migration (**⚠ confirm**).
- **Incremental behind interfaces:** move a subsystem only after it is hardened and expressed against a kernel interface; keep the old import path working (shim) until callers migrate.
- **Feature-flag** risky swaps (e.g. compression engine, caching) so they can be toggled during rollout.

---

## 11. Workstream dependency graph

```
M0 skeleton ─▶ M1 contracts ─▶ Phase 0 (bugs) ─▶ Phase 1 (cost core)
                    │                                   │
                    ▼                                   ▼
              Phase 2 (receipts/validators/sessions/CLI) ── needs M1 + Phase 1
                    │
                    ▼
              Phase 3 (memory engine + learning) ── needs Phase 2 receipts+validators
                    │
                    ▼
              Phase 4 (migration + publish) ── needs kernel subsystems hardened
                    │
                    ▼
              Phase 5 (surfaces + org plane, Aloy) ── needs SSE + published kernel
```
Notes: Phase 0 is independent of M1 (pure fixes) and can start immediately after M0; Phase 1.4 (error classifier overflow→compress) depends on Phase 1.3 (compression); Phase 2.3 depends on 2.1 + 2.2.

---

## 12. Consolidated open questions / decisions needed

| # | Question | Blocks |
|---|---|---|
| 1 | Kernel thinness ratification (LLM/tools/context/sandbox in kernel) | M1, Phase 4 |
| 2 | Receipt storage backend + hash-chain algorithm | M1, Phase 2.1 |
| 3 | Event/streaming protocol: AG-UI/ACP vs Pori-native | M1 |
| 4 | MCP in kernel vs extension | Phase 4 |
| 5 | `Team` (ROUTER/BROADCAST/DELEGATE) kernel vs extension | Phase 4 |
| 6 | Provenance ContextVar placement (kernel vs ext) | Phase 3 |
| 7 | Auth default: require-key + `PORI_ALLOW_NO_AUTH` opt-in | Phase 0.2 |
| 8 | Python floor (3.10 vs 3.11) | M0 |
| 9 | Lint stack (black/isort/mypy vs ruff) | M0 |
| 10 | `ext/pori-*` naming + repo root name | M0 |
| 11 | Pre-1.0 breaking changes allowed | Phase 4/10 |
| 12 | First donors to clone (recommended: OpenHands + Inspect) | M1 |

---

## 13. Definition of Done per milestone

- **M0:** workspace + boundary check + supply-chain gates + HARVEST.md; standalone `pori` build.
- **M1:** Receipt/Verdict/Validator/interfaces/NormalizedResponse defined; emit+chain+replay a receipt; a validator returns a Verdict recorded as a receipt.
- **Phase 0/M2:** three fixes with tests; tenant isolation, fail-closed auth, non-bypassable hardline floor.
- **Phase 1/M3:** NormalizedResponse + caching + compression + classifier; measurable token reduction; overflow recovers.
- **Phase 2/M4:** receipts on every step + replay; tiered validators in the loop; memory writes receipted/validated; sessions resume/branch + dedup; registry-derived CLI.
- **Phase 3/M5:** kernel memory engine; personal-layer scope resolver with org/team path present; provenance-gated autonomous review.
- **Phase 4/M6:** `pori` published standalone; boundary green; Aloy composes on top.
- **Phase 5/M7+:** SSE; harvested surfaces on REST+SSE; Slack+Telegram gateway; org policy/audit/admin.
