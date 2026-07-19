# Event memory controls — 2026-07-19

## Product outcome

Each Event now exposes the accepted memory Aloy may carry into future work.
The Event's own settings, opened from the settings icon directly after Trail in
the right Event-context dock, separate mutable Event memory from read-only
inherited global memory. A user can correct, forget, or explicitly make an Event memory
available across Aloy without editing canonical Tasks, files, Trail, Surface
data, receipts, or transcript history.

## Architecture and invariants

- `aloy_backend.event_memory` owns Event-memory mutation invariants; the route
  module owns HTTP authorization and presentation.
- Reads require tenant, user, and Event ownership and never return another
  Event's memory.
- Corrections create a new `user_correction` record, supersede the prior row,
  retain provenance, retire a derived Event Brief, refresh the Event context
  snapshot, reuse the existing bootstrap readiness gate, and write Trail.
- Forgetting is idempotent soft deletion, respects legal hold, refreshes the
  Event context snapshot through the same Brief-retirement path, and writes
  content-free Trail evidence.
- Promotion uses a deterministic global record identity. Repeating an active
  promotion is idempotent; promoting again after the global copy was forgotten
  safely restores it. The source Event record retains the global record link.
- Inherited global memory is visible but read-only in Event settings.
- Mutation Trail payloads contain type/sensitivity and record references, not
  the memory content itself.

## Verification

- Focused Event memory/context tests: `11 passed`.
- Changed backend Ruff and focused mypy: passed.
- Aloy app ESLint and production build: passed; only the existing large-chunk
  warning remains.
- Full backend suite: `362 passed`, with two unchanged local headless Surface
  bridge handshake tests failing because the runtime did not acknowledge the
  secure bridge. Both exact tests remained failing on isolated rerun; no
  Surface code changed in this phase. Full backend mypy passed across `116`
  source files and full Black passed across `217` files.
- Automated browser visual smoke was attempted and blocked before navigation by
  the existing desktop browser bootstrap error `Cannot redefine property:
  process`.

## Follow-up

Run a manual visual smoke of the Event settings panel, then implement the
separate evidence-gated memory consolidation phase. Do not promote casual
model inferences automatically.
