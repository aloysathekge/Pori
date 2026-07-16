# docs — design & architecture

Design and architecture documentation for the Pori kernel and the Aloy
product. Complete index — if you add a doc, add it here.

## Start here

- [`aloy-vision.md`](./aloy-vision.md) — **the canonical Aloy product
  definition**. Its active children are
  [`aloy-v1-plan.md`](./aloy-v1-plan.md) for delivery and
  [`aloy-surface-spec.md`](./aloy-surface-spec.md) for the model-authored
  Surface, including its quality gate. Start there; older Aloy records do not
  override this hierarchy.
- [`architecture-primer.md`](./architecture-primer.md) — hosted-agent
  fundamentals: tenancy, storage tiers, the compute ladder, file→LLM paths.
  Read for any "where does X belong" question.
- [`engineering-excellence-spec.md`](./engineering-excellence-spec.md) — the
  quality bar (standards + phased plan) the codebase is being held to.

## Decisions

- [`adr/`](./adr/README.md) — **Architecture Decision Records**: dated
  load-bearing decisions (no-LangChain, loop-stays-whole, fail-open,
  footprint ladder, memory-as-index, single-finalizer, front-door-only,
  single-API-worker, and recorded supersessions). If something looks odd,
  check here before changing it.

## Kernel (Pori)

- [`Pori.md`](./Pori.md) — PRD for Pori as a standalone kernel.
- [`Pori_Implementation_Plan.md`](./Pori_Implementation_Plan.md) — the phased
  implementation plan.
- [`memory-architecture.md`](./memory-architecture.md) — the memory model.
- [`runtime-trust-model.md`](./runtime-trust-model.md) — receipts, validators,
  trust rails.
- [`session-continuity.md`](./session-continuity.md) — sessions & resume.
- [`long-running.md`](./long-running.md) — durable runs, checkpoints, leases.
- [`pori-mcp-spec.md`](./pori-mcp-spec.md) — the session-scoped MCP client.

## Aloy product specs (built arcs; kept as design records)

- [`Aloy.md`](./Aloy.md) — **historical** original Aloy plan; retained for
  architecture provenance and superseded by `aloy-vision.md`.
- [`aloy-wedge-spec.md`](./aloy-wedge-spec.md) — implemented Event, Session,
  Proposal, and initial workspace foundation; newer Surface work is superseded
  by `aloy-surface-spec.md`.
- [`aloy-connections-spec.md`](./aloy-connections-spec.md) — connect-engine +
  Gmail (OAuth token custody).
- [`aloy-multitenant-connections-mcp-spec.md`](./aloy-multitenant-connections-mcp-spec.md)
  — user|org scoped connections + MCP servers.
- [`aloy-send-message-refactor-spec.md`](./aloy-send-message-refactor-spec.md)
  — the single-finalizer persistence rule.
- [`aloy-object-storage-sandbox-spec.md`](./aloy-object-storage-sandbox-spec.md)
  — durable files, sandbox provisioning, the user file library.

## Reviews & analyses (point-in-time; check dates)

- [`codebase-review-2026-07-09.md`](./codebase-review-2026-07-09.md) —
  verified findings tracker (most items since fixed).
- [`hermes-gap-2026-07.md`](./hermes-gap-2026-07.md) — Hermes capability gap
  analysis.
- [`deepagent-migration-plan.md`](./deepagent-migration-plan.md) — historical
  migration plan.
- [`ALIGNMENT.md`](./ALIGNMENT.md) — earlier alignment notes.

## Related, outside this folder

- [`../MONOREPO.md`](../MONOREPO.md) — repo structure, the one-way dependency
  rule, product-extraction playbook.
- [`../HARVEST.md`](../HARVEST.md) — OSS harvest provenance ledger.
- [`../CLAUDE.md`](../CLAUDE.md) — the kernel codebase map for coding agents.
- `../products/aloy/backend/deploy/` — RUNBOOK + OAuth verification guide.
- `../references/hermes-agent-deep-dives/` — Hermes subsystem studies.
