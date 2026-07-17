# Current State - 2026-07-17

## Active Task

Finish and review the active `aloy-v1-event-context-pack` slice. It implements
canonical context snapshots, readiness, Event Brief persistence, memory
precedence, and safe prompt-cache boundaries without requiring model credits.

## Decisions Made

- Keep detailed session history in handoff files; keep this file as the next-session briefing.
- Event creation must remain available even when context ingestion is pending or fails.
- Prompt caching is a latency/cost optimization, never durable memory or truth.
- Confidential/restricted Event evidence disables application-owned message-prefix caching.

## Important Discoveries

- The R5.5b slice added durable Event context-source ingestion, status, retry, provenance, and Workbench visibility.
- R5.5c adds immutable content-addressed Event context snapshots, deterministic
  readiness, typed evidence-linked Event Brief persistence, trusted prompt
  placement, and Event-over-global conflict precedence.
- Verification: kernel `615 passed, 1 skipped`; backend `301 passed`; kernel
  mypy `108` files; backend mypy `102` files; changed-file Ruff; Aloy app lint,
  production build, and `4` Surface bridge tests.
- The full previous work journal is preserved in `.agent/progress/handoffs/2026-07-17-current-state-history.md`.

## Blockers

- A live end-to-end stack run remains user-blocked because it needs Supabase and LLM credentials; see `products/aloy/BOOT.md`.

## Next Session Should Start With

Read `AGENTS.md`, the archived handoff, and the current diff. Then wire the
configured bootstrap model profile into an idempotent Run that fills the frozen
snapshot's Event Brief contract. After that, implement memory
inspect/correct/forget/promote controls, connection refresh/revocation, and the
first evidence-grounded bootstrap Surface plus sanitized cover brief.
