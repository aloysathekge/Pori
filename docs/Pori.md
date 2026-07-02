# Pori — Product Requirements Document (PRD)

**Status:** Draft v0.1
**Date:** 2026-07-02
**Scope:** Pori as a standalone, publishable agent **kernel**. Aloy and future products are *consumers* of Pori and appear here only as context.
**Companion:** [`Pori_Implementation_Plan.md`](./Pori_Implementation_Plan.md) — the engineering plan derived from this PRD.

> This PRD formalizes decisions already made in the architecture discussion. It does not introduce new architectural direction. Genuinely undecided points are marked **⚠ OPEN QUESTION** and collected in §14.

---

## 1. Summary

**Pori is an eval-native, receipt-first, memory-native agent kernel.** It is the small, opinion-light substrate on which agent *products* are built — starting with **Aloy** (a personal + org OS agent), and any number of future agent products, each with its own niche.

Pori's defining property is a single, closed, auditable loop:

> **work runs on a manager/worker runtime → emits *receipts* → *validators* judge the receipts → each verdict is itself a receipt → the loop continues or halts.**
> Memory writes ride the same rails: **write → receipt → validate → commit.**

Where other frameworks treat verification, evaluation, and memory management as add-ons, Pori makes them the **kernel contract**. That contract is the moat, and it is exactly what org/enterprise agent products need (audit, governance, policy, cost attribution) — so Pori's differentiator and its flagship consumer's requirements are the same thing.

Pori is **independently publishable** (e.g. to PyPI) and usable on its own. "The open framework" is **Pori (kernel) + a reusable extension band**; products (Aloy and others) compose on top.

---

## 2. Problem & Motivation

Building a serious agent today forces teams to re-solve the same hard substrate — the reasoning loop, provider differences, context/cost control, memory, and *some* notion of safety/eval — every single time, and usually to bolt verification on at the end.

The existing landscape leaves a specific gap:

- **Orchestration-first frameworks** (e.g. LangGraph) model control flow well, but verification and evaluation are the developer's problem, layered on top.
- **Product-hardened personal agents** (e.g. Hermes) are excellent at cost, portability, and breadth, but are single-tenant by design, concentrate logic in very large files, and optimize for cost/portability rather than *provable* work.
- **Glue frameworks** (e.g. LangChain-style) compose components but impose no correctness contract.

None of them make **correctness (verifiability), evaluation, and memory management the core kernel contract.** And none are structured so that *multiple* differentiated agent products can be spun up on one clean, reusable kernel.

For an org/enterprise agent OS, the missing piece is even more acute: **audit, governance, policy enforcement, and cost attribution** are non-negotiable — and they fall out naturally from a kernel whose primitives are receipts and validators.

---

## 3. Vision, Positioning & Niche

### 3.1 Niche
Pori is **the eval-native, receipt-first, memory-native agent kernel** — verifiability, evaluation, and memory management are the *contract*, not features.

### 3.2 Positioning vs. alternatives

| Framework | Optimizes for | Pori's difference |
|---|---|---|
| LangGraph | graph/state orchestration, checkpointing | Pori is a *loop* kernel with verification/eval as the contract, not a graph DSL |
| Hermes | cost, portability, breadth (personal product) | Pori is product-agnostic and eval/receipt-native; Aloy is Pori's Hermes-class consumer |
| LangChain-style | component composition | Pori imposes a correctness contract (receipts + validators) |
| Letta/MemGPT | memory architecture | Pori adopts the memory-management thesis *and* wraps every write in receipts + validators |

### 3.3 The moat — one loop, three cores
Receipts, validators, and the memory engine are **not three features — they are one loop**, and that loop *is* the kernel:

```
work runs ─▶ RECEIPTS ─▶ VALIDATORS judge receipts ─▶ verdict is a RECEIPT ─▶ continue / halt
                  ▲                                                      │
            memory writes: write ─▶ receipt ─▶ validate ─▶ commit ◀──────┘
```

### 3.4 Multi-product vision
Pori is built to carry **many** agent products. A product = **kernel + chosen extensions + a niche composition** (persona, tool selection, validators/policies, memory config, surfaces). Aloy (personal + org OS) is product #1; future products (e.g. a research agent, a coding agent, a support agent) are thin compositions on the same kernel. See §8.

---

## 4. Goals & Non-Goals

### Goals
- **G1 — Standalone kernel.** Product-agnostic, publishable, usable on its own.
- **G2 — The one loop.** First-class receipts, validators, and memory engine wired into a single auditable loop.
- **G3 — Provider-agnostic.** One `NormalizedResponse` across providers; the loop never branches on provider.
- **G4 — Cost discipline by construction.** Prompt caching and context compression are built into the kernel's prompt/context path.
- **G5 — Cheap verification first.** Deterministic validators run always; LLM-judge validators are optional (no costly verification gates).
- **G6 — Multi-product enablement.** Three-band monorepo (kernel → extension band → products) with a CI-enforced one-way dependency rule.
- **G7 — Preserve Pori's identity.** Keep the explicit **Plan → Act → Reflect → Evaluate** loop, the unified eval/guardrail model (→ validators), span tracing (→ receipts), CoreMemory Blocks, the `Team` model, async-everywhere APIs, and entry-point extensibility.
- **G8 — Best-of-breed by harvest.** Adopt the strongest patterns from multiple OSS harnesses, re-expressed through Pori's contracts, with license hygiene and provenance.

### Non-Goals
- **NG1 — Not a product.** No personal/org OS features, tenancy, RBAC, gateway, or surfaces in the kernel — those live in the extension band or in products.
- **NG2 — Not a graph DSL.** Pori is a loop kernel, not a workflow-graph framework.
- **NG3 — Not provider- or deployment-locked.**
- **NG4 — No speculative extension infrastructure.** Reusable extensions are created only when a real second consumer exists (promote on second use).

---

## 5. Users & Personas

- **Product teams** building agent products on Pori (primary: the Aloy team).
- **Framework users / OSS adopters** who want an eval-native kernel rather than a glue framework.
- **Researchers** who need receipts/replay and a validator/eval contract for measurable agent behavior.

---

## 6. Principles & Tenets

1. **Dependency inversion.** Kernel owns *contracts + mechanism*; implementations and policy live above it. The kernel loop depends only on interfaces.
2. **One-way dependencies.** `products → ext → pori`, never upward — CI-enforced. This is the single safeguard against god-files.
3. **Harvest patterns, not paste.** Every borrowed idea is re-expressed through Pori's contracts; provenance and license are recorded.
4. **Cheap verification first.** Deterministic checks precede any LLM-judge; the kernel never burns a model call when a cheaper check suffices.
5. **Architectural, not patches.** Fix structure and contracts, not symptoms.
6. **Promote on second use.** Reusable machinery graduates into the extension band when a second product needs it — not on spec.
7. **Cost-safe by construction.** Prompt caching and compression are part of the kernel path, not an afterthought.

---

## 7. Core Model & Concepts (the kernel contract)

### 7.1 The one loop
The runtime executes the **Plan → Act → Reflect → Evaluate** cycle. Every unit of work (LLM call, tool call, memory write, subagent) emits a receipt; validators judge receipts; verdicts are receipts; the loop uses verdicts to continue or halt.

### 7.2 Receipts
A **receipt** is a typed, verifiable record of one unit of work — not a log line. Decided properties:

1. **Typed & tree-structured** — inputs, outputs, actor, cost/tokens, timing, parent link. *(Seed: Pori `Trace`/`Span`.)*
2. **Content-addressed & hash-chained** — inputs/outputs hashed; each receipt references the prior's hash → tamper-evident sequence.
3. **Evidence-linked** — a claim receipt links to the evidence receipts that justify it. *(Generalizes Hermes verification-evidence.)*
4. **Replayable** — captured inputs make a receipt chain a deterministic replay/debug log and free eval/training data.
5. **Attributable** — every receipt carries actor/tenant tags → per-tenant audit + cost attribution.
6. **Cheap to emit** — inline append; never an extra LLM call.

### 7.3 Validators
A **validator** is a pluggable check over receipts, with one contract and three lifecycles:

- **Contract:** `Validator: (receipts) → Verdict { pass | warn | block | score, reason, evidence_refs }`.
- **Three lifecycles from one contract:** pre-action **gate** (guardrail/approval), post-action **check** (success/regression), offline **eval** (score a trajectory). *(Seed: Pori `BaseEval` = eval + guardrail.)*
- **Tiered:** deterministic validators (schema, hardline command floor, threat-regex, exit-code, invariants, loop detection) run always; LLM-judge validators run only when enabled.
- **Consume + emit receipts:** a validator reads the chain (incl. evidence) and its verdict is itself a receipt → auditable and composable.
- **Scoped:** the kernel owns a minimal **safety floor**; org/role policy is a set of validators registered by the extension band/product. "A policy" = "the validators applied to this scope."
- **Severity-driven & evidence-grounded:** fail-closed for consent/safety, warn/score for quality; a verify-on-stop validator refuses a "done" claim without a fresh evidence receipt. *(Seed: Hermes `threat_patterns`, `approval` hardline, `verification_stop`.)*

### 7.4 Memory engine
The **memory engine** is a kernel moat. It owns:
- the **block model** (working/core memory, e.g. CoreMemory Blocks),
- **recall → inject** (bringing memory into the prompt cache-safely),
- the **write lifecycle** on the loop's rails (write → receipt → validate → commit),
- the **`MemoryStore` interface**.

Memory **tenancy** (org → team → personal scope resolution), RBAC, concrete stores, and the learning-loop curator are **not** kernel — they live in the extension band / product.

### 7.5 Runtime (manager/worker)
The kernel runtime owns the manager/worker execution model, the turn lifecycle, the iteration budget, and the **Evaluator** step (Pori's differentiator that Hermes lacks).
**⚠ OPEN QUESTION:** whether the multi-agent `Team` modes (ROUTER / BROADCAST / DELEGATE) are kernel runtime or an extension (see §14).

### 7.6 LLM & transport
The kernel exposes a provider-agnostic transport with a single normalized response type (`NormalizedResponse`, incl. usage/cache tokens). Provider quirks are quarantined in adapters. *(Seed: Pori `llm/base.py` + `anthropic/openai/google`; donors: Hermes transports, LiteLLM.)*

### 7.7 Tools
The kernel owns the tool **registry + executor engine** and the `ToolBackend` interface, including service-gating (a `check_fn`-style predicate deciding per-turn visibility). Concrete/standard tools live in the extension band or product.
**⚠ OPEN QUESTION:** whether the kernel speaks **MCP** natively or via an extension (§14).

### 7.8 Context
The kernel owns the `ContextEngine` interface plus the compression *mechanism* and the **prompt-caching** discipline: a 3-tier prompt — **stable** (identity/tool guidance), **context** (skills index, context files), **volatile** (memory snapshot in one trailing message, date-only timestamp) — built once per run, with cache breakpoints on system + last-N messages + tool schema. *(Donor: Hermes `prompt_caching`, `context_compressor`.)*

### 7.9 Sandbox
The kernel owns execution backends, path security, and a minimal **hardline safety floor** (dangerous-command detection with input normalization) that runs *before* any human-in-the-loop approval and cannot be approved away. *(Donor: Hermes `environments`, `approval` hardline, `path_security`.)*

### 7.10 Interfaces (ABCs)
The kernel defines the contracts implemented above it: `MemoryProvider`, `SkillProvider`, `ToolBackend`, `Validator` (and the receipt/event types). Dependency inversion: the loop imports interfaces; extensions/products supply implementations.

---

## 8. Architecture — the bands & kernel layout

### 8.1 The bands (flat, intent-named)
```
repo root (single distribution for now; see MONOREPO.md)
├─ pori/          KERNEL — product-agnostic
│                 runtime · protocol · receipts · validation · llm · tools · context ·
│                 sandbox · memory engine · interfaces
├─ extensions/    reusable pori-* libs — opt-in (memory-scope/tenancy · skills · learning ·
│                 gateway · providers · cli-kit); created on promotion, not on spec
├─ products/
│  ├─ aloy/       FLAGSHIP PRODUCT #1 (personal + org OS): backend · cli · gateway
│  │              + Aloy-specific org policy, tenancy shape, branding
│  └─ <future>/   additional products: kernel + chosen extensions + niche glue
├─ apps/          frontend surfaces (web, desktop) — REST + SSE to a product backend
├─ website/       public marketing site
├─ docs/          PRD, plan, ALIGNMENT, design docs
├─ tools/ci/      dependency-boundary enforcement, lint, tests
└─ (donors)       external OSS at ../references/ (Hermes, OpenHands, Aider, Letta, …)
```

### 8.2 Dependency rule (CI-enforced, staged)
`products / apps → extensions → pori`, **never upward.** A product may use the kernel directly and skip an extension. Enforced by a dependency-boundary check (`import-linter` or a small script) so any commit where `pori` imports `extensions`/`products` fails the build.

### 8.3 Publishability & "the open framework"
Today `pori` is the sole distribution (single root `pyproject.toml`). When split into a uv workspace, `pori` becomes independently publishable with no `extensions`/`product` imports. **"The open framework" = `pori` + the `extensions` band.** Products are branded/private compositions.

### 8.4 Anti-speculation rule
The kernel is product-agnostic from day one. An `extensions/pori-*` package is created only when the capability is *obviously* generic; otherwise it is built in `products/aloy/` and **promoted on second use**.

---

## 9. Kernel Public Surface (stable contracts)

The kernel's public API centers on these types/ABCs (exact signatures designed in Implementation Plan M1):

- **Types:** `Receipt`, `Verdict`, `NormalizedResponse`, `Usage`, the streaming/event type(s).
- **ABCs:** `Validator`, `MemoryProvider`, `MemoryStore`, `SkillProvider`, `ToolBackend`, `ContextEngine`.
- **Runtime entry:** the manager/worker run interface (async), returning a result plus its receipt chain.

---

## 10. Requirements

### 10.1 Functional (kernel)
- **FR-1 (Runtime):** Execute Plan→Act→Reflect→Evaluate with an iteration budget and a first-class Evaluator step.
- **FR-2 (Receipts):** Every LLM/tool/memory/subagent unit emits a typed receipt; receipts form a hash-chained, evidence-linked, replayable tree carrying actor/tenant tags.
- **FR-3 (Validators):** A single `Validator → Verdict` contract runnable as pre-gate, post-check, and offline eval; tiered deterministic-first; verdicts are receipts; validators are scoped and severity-driven; a minimal safety floor is always active and non-bypassable.
- **FR-4 (Memory engine):** Block model, recall→inject (cache-safe), and a write lifecycle gated by receipts + validators; `MemoryStore` interface.
- **FR-5 (LLM):** Provider-agnostic transport with a single `NormalizedResponse`; provider quirks isolated in adapters.
- **FR-6 (Context/cost):** 3-tier prompt built once per run; cache breakpoints; a compression mechanism behind the `ContextEngine` seam.
- **FR-7 (Tools):** Registry + executor engine with per-turn service-gating and a `ToolBackend` interface.
- **FR-8 (Sandbox):** Pluggable execution backends, path security, and a non-bypassable hardline command floor with input normalization, checked before HITL.
- **FR-9 (Errors):** Structured error classification driving recovery (e.g. overflow→compress, auth/billing→fail-fast).

### 10.2 Non-functional
- **NFR-Cost:** Prompt caching yields a large input-token reduction on multi-step runs (target on the order of ~75% on cache-eligible prefixes); no verification path adds an LLM call when a deterministic check suffices.
- **NFR-Observability:** 100% of steps produce receipts; a run is replayable from its receipt chain.
- **NFR-Security:** Hardline floor + path security non-bypassable; validators fail-closed for consent/safety.
- **NFR-Testability:** Behavior-contract tests over snapshots; E2E over mocks for resolution/security/IO paths; runnable with empty API keys.
- **NFR-Portability:** Kernel has no product/tenant assumptions and no hard dependency on a specific provider or deployment.
- **NFR-Supply-chain:** Bounded/pinned dependencies, OSV scanning, dependency-bound checks, SHA-pinned CI actions.
- **NFR-Licensing:** All harvested material is license-clean with recorded provenance.

---

## 11. Harvest & Provenance Strategy

Pori assembles best-of-breed patterns from multiple OSS harnesses.

| Concern | Primary donor(s) | License handling |
|---|---|---|
| Tool ergonomics, permission model, hooks, subagent/Task, plan/todo, steering UX | **Claude Code** | design/behavior only (clean-room) |
| Prompt caching, context compression, transport, sandbox, gateway, supply-chain | **Hermes** | MIT — adapt + attribute |
| Event stream / replay substrate (→ receipts) | **OpenHands** | MIT — adapt |
| Repo map, git-diff editing, code context | **Aider** | Apache — adapt |
| Checkpointing/persistence, HITL interrupts | **LangGraph** | MIT — adapt |
| Memory blocks / self-editing memory | **Letta/MemGPT** | Apache — adapt |
| Validator/eval methodology | **Inspect, promptfoo, DSPy** | permissive — adapt |
| Provider normalization | **LiteLLM** | MIT — adapt / idea |
| Tool interop, event protocol | **MCP, AG-UI / ACP** | open specs — implement |
| Multi-agent role/team patterns | **AutoGen, CrewAI** | permissive — ideas |

**Rules:** (1) harvest *patterns re-expressed through Pori's contracts*, never paste; (2) license hygiene per source (permissive → adapt with attribution; closed/non-permissive → clean-room from observable behavior only); (3) every adoption logged in `references/HARVEST.md` as *pattern → source → license → destination → why* ("receipts for the codebase").

---

## 12. Success Metrics — what "world-class" means, measurably

- **Standalone:** `pip install pori` runs the loop with a trivial stub product (no Aloy required).
- **Cost:** measurable prompt-cache hit rate and large input-token reduction on multi-step runs.
- **Verifiability:** every step yields a receipt; a completed run is deterministically replayable from its chain; tamper-evidence verifiable.
- **Validation:** a starter library of deterministic validators ships; LLM-judge is strictly opt-in; the safety floor is provably non-bypassable.
- **Boundary integrity:** the dependency-boundary check is green (no upward imports) on every commit.
- **Multi-product proof:** Aloy (product #1) plus at least one additional sample product composed with minimal glue.
- **Provider-agnostic:** the loop contains no provider `if/elif`; adding a provider is an adapter.

---

## 13. Roadmap (PRD altitude)

Detailed steps live in the Implementation Plan. At PRD altitude:

- **M0 — Workspace & CI skeleton** (three bands, dependency-boundary check, HARVEST.md).
- **M1 — Kernel contracts** (Receipt, Verdict/Validator, event protocol, interfaces, NormalizedResponse).
- **Phase 0 (M2) — Tenancy-aware fixes** (API per-request memory isolation, fail-closed auth, sandbox hardline floor).
- **Phase 1 (M3) — Cost/robustness core** (NormalizedResponse → prompt caching → compression → error classifier).
- **Phase 2 (M4) — Receipts + validators retrofit; sessions key/id; CLI command registry.**
- **Phase 3 (M5) — Memory engine formalization + autonomous learning loop (ext/product).**
- **Phase 4 (M6) — Package migration & publish `pori`.**
- **Phase 5 (M7+) — Surfaces & org plane** (Aloy: SSE, web/desktop, gateway, tenancy/RBAC/policy/admin).

---

## 14. Open Questions & Risks

**Open questions (genuinely undecided — do not guess):**

1. **Kernel thinness ratification.** The recommended "mechanism kernel" places LLM/tools/context/sandbox in `pori`; this was directionally agreed but never formally ratified against the "thin substrate" alternative.
2. **Receipt storage & hashing.** Storage backend (append-only log / SQLite / content-addressed store) and the exact hash-chain algorithm are undecided.
3. **Event/streaming protocol.** Adopt an existing standard (AG-UI / ACP) vs. a Pori-native contract atop today's `PoriEvent` — undecided.
4. **MCP placement.** Native MCP in the kernel vs. an extension — undecided.
5. **`Team` placement.** Multi-agent modes (ROUTER/BROADCAST/DELEGATE) as kernel runtime vs. an extension — undecided.
6. **Provenance ContextVar placement.** Write-origin provenance (for gated autonomy) as a kernel receipts concern vs. an extension — undecided.
7. **Auth default (Aloy backend).** Proposed default: require-key + `PORI_ALLOW_NO_AUTH=1` dev opt-in — proposed, not confirmed.
8. **Python floor.** Current Pori targets 3.10; donors (Hermes) target 3.11+. Kernel floor undecided.
9. **Lint/tooling.** Keep black/isort/mypy vs. adopt ruff (as Hermes does) — undecided.
10. **Package/namespace naming.** `ext/pori-*` naming and repo root name — proposed, not ratified.
11. **Pre-1.0 stability.** Assumed: breaking changes allowed pre-1.0 during migration — confirm.
12. **First donors to clone.** Recommended OpenHands + Inspect first (they most shape receipts/validators) — pending.

**Risks:**
- **Frankenstein risk** from multi-source harvest — mitigated by the contract discipline + one-way deps + HARVEST.md.
- **Speculative-generality risk** in the extension band — mitigated by promote-on-second-use.
- **Migration risk** moving current `pori/` modules into bands — mitigated by incremental migration behind interfaces (see Implementation Plan §10).
- **Identity-dilution risk** — mitigated by explicitly preserving the Evaluator loop, unified eval/guardrail, tracing, CoreMemory, and Team (G7).

---

## 15. Glossary

- **Receipt** — typed, hash-chained, evidence-linked, replayable record of one unit of work.
- **Validator** — a check over receipts returning a `Verdict`; runnable as gate, post-check, or eval.
- **Verdict** — `{ pass | warn | block | score, reason, evidence_refs }`; itself recorded as a receipt.
- **Memory engine** — kernel-owned block model + recall/inject + write lifecycle + `MemoryStore` interface.
- **The one loop** — work → receipts → validators → verdict(receipt) → continue/halt; memory writes ride the same rails.
- **Kernel / extension band / product** — the three monorepo bands; dependency direction is `products → ext → pori`.
- **Harvest** — adopting an OSS pattern re-expressed through Pori's contracts, with recorded provenance and license.
- **Promote on second use** — graduate reusable code into the extension band only when a second product needs it.
