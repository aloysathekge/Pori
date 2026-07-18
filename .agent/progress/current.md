# Current State - 2026-07-17

## Active Task

Finish and review `aloy-v1-event-bootstrap`: the purpose-scoped background Run
that turns a frozen, sufficient Event context snapshot into an evidence-linked
Event Brief without requiring live model credits during development.

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
- R5.5d adds an idempotent no-tool bootstrap Run, frozen bounded evidence,
  structured generation, stale-snapshot replacement, safe retry, and visible
  Workbench status/manual retry. Event promotion and completed ingestion trigger
  it automatically.
- Verification on the active branch: the backend full suite passed in three
  groups (`126 + 76 + 105 = 307`); after adding two final dispatch/policy tests,
  the affected `15` tests pass. Backend mypy passes across `103` source files;
  Aloy app ESLint and production build pass. The build retains the pre-existing
  large-chunk warning.
- Live provider smoke: Fireworks Kimi K2.6 completed the real queued
  `event_bootstrap` worker path against an isolated SQLite database, published
  Event Brief v1, and passed evidence-reference validation. The test exposed
  and fixed Fireworks/Kimi structured-output compatibility by including the
  schema in the prompt and disabling Kimi reasoning for schema-bound calls.
- Post-fix verification: kernel `617 passed, 1 skipped`; kernel mypy passes
  across `108` source files; the focused Fireworks/config suite is included in
  that total.
- Verification: kernel `615 passed, 1 skipped`; backend `301 passed`; kernel
  mypy `108` files; backend mypy `102` files; changed-file Ruff; Aloy app lint,
  production build, and `4` Surface bridge tests.
- The full previous work journal is preserved in `.agent/progress/handoffs/2026-07-17-current-state-history.md`.

## Blockers

- The isolated live Kimi bootstrap path is verified. A full authenticated UI
  smoke still depends on the local Supabase setup described in
  `products/aloy/BOOT.md`.

## Next Session Should Start With

Read `AGENTS.md`, the archived handoff, and the current diff. Finish/merge the
Event bootstrap slice. Then implement memory inspect/correct/forget/promote
controls, connection refresh/revocation, and the first evidence-grounded
bootstrap Surface plus sanitized cover brief.
