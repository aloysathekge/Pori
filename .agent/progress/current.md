# Current State - 2026-07-20

## Active Task

R8 is active on `aloy-v1-r8-release-gate`. Four reliability slices are now
implemented: Run/Task watchdogs, durable Run budgets, bounded Conversation
context compaction, and read-only provider reconciliation. The next gate is
responsive and accessibility QA for the Event Workbench and generated Surface
runtime at the required desktop, tablet, and mobile viewports.

## Decisions Made

- Keep detailed session history in handoff files; keep this file as the next-session briefing.
- Provider execution and reconciliation are separate rails. Unknown provider
  outcomes never authorize a repeat consequence.
- Gmail new sends use a deterministic RFC822 Message-ID; Calendar creates use a
  deterministic provider event ID. `gmail_send_draft` remains explicitly
  indeterminate when its outcome cannot be proven.
- Automatic provider lookups are leased, exponentially backed off, and bounded
  to eight attempts before uncertainty remains for review.

## Important Discoveries

- Provider crash-window verification passes with one send and a recovered
  receipt after simulated database commit loss. Complete kernel verification is
  `627 passed, 1 skipped`; affected backend suites and kernel/backend mypy pass.
- Full historical context is archived at
  `.agent/progress/handoffs/2026-07-20-progress-history.md`.

## Blockers

- Live University, Madrid, and Career model acceptance remains deferred while
  provider credits are unavailable.
- Remote Surface builds still require the pinned E2B toolchain template.

## Next Session Should Start With

Audit and repair Event Workbench and generated-Surface responsiveness, keyboard
operation, focus behavior, semantics, contrast, and reduced-motion behavior.
Keep `source_change` and `automation` fail-closed.
