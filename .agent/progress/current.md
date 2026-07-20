# Current State - 2026-07-20

## Active Task

R8 is active on `aloy-v1-r8-release-gate`. Five engineering slices are now in
the branch working tree: Run/Task watchdogs, durable Run budgets, bounded
Conversation compaction, read-only provider reconciliation, and release
readiness. The latest release-readiness changes are not committed yet.

## Decisions Made

- Responsive/accessibility acceptance is a manual user gate while Codex browser
  control is unavailable; static inspection must not be recorded as passing.
- Existing parent-linked docs now carry boot, operator, architecture, and demo
  guidance; no parallel Aloy document was added.
- Surface inspection waits for the actual runtime document before transferring
  the secure bridge, with one bounded deadline per interaction check.
- Aloy product code uses only Pori's public package front door.

## Important Discoveries

- The old inspector raced Chrome navigation and could miss the Surface SDK
  listener, producing false `runtime_bridge_failed` diagnostics for valid forms.
- Final automated gates pass: kernel `627 passed, 1 skipped`; backend `401
  passed`; app bridge `7 passed`; app lint/build; kernel/backend mypy; and all
  three import-linter contracts.
- The boot guide now includes no-credit operation, recovery rules, Windows-safe
  pytest temp roots, model roles, and the 60-second Career OS proof.
- Detailed work is in
  `.agent/progress/handoffs/2026-07-20-r8-release-readiness.md`.

## Blockers

- Manual desktop/tablet/mobile responsiveness, keyboard, focus, semantics,
  contrast, reduced-motion, and generated-Surface checks remain unaccepted.
- Live University, Madrid, and Career model acceptance is deferred without
  provider credits.
- Remote Surface acceptance still needs the pinned E2B toolchain template.

## Next Session Should Start With

Review and commit the release-readiness working tree. Have the user run the
manual viewport/accessibility checklist in `products/aloy/BOOT.md`; repair any
reported defect. With provider access, run the three real-domain proofs and
Career OS demo before opening the final R8 merge PR.
