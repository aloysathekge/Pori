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

## Next Session Should Start With

1. (Carried over) Open a PR for `fix/sandbox-working-dir-and-artifact-tracking`
   (base `main`) — the sandbox `working_dir` + artifact-tracking fixes are
   committed and green (337 tests) but unmerged.
2. Resolve the M0/M1-blocking open questions (Python floor, lint stack, naming,
   event protocol, receipt storage, first donors) OR start **Phase 0** (the three
   tenancy-aware fixes) in the current `pori/` tree.
