# Current State - 2026-07-20

## Active Task

R9 is active on `aloy-v1-r9-surface-quality`, branched from the PR #198 merge
on `aloy-v1`. Exact-build quality receipts plus the first deterministic
viewport/accessibility evidence slice are complete. The next slice is the
host-owned state contract used by both real Surfaces and trusted inspection.

## Decisions Made

- `aloy-v1` remains the integration branch; R9 is not a path directly to
  `main`.
- Quality evidence belongs to the trusted host and exact retained build. The
  Builder, Critic, and generated application never receive publication power.
- The first policy covers deterministic validation, runtime, declared
  interactions, five responsive viewports, capture integrity, and basic DOM
  accessibility. State variants, focus indicators, contrast, Critic, and
  primary-job evidence remain explicit planned extensions.
- A stricter new-publication gate must not remove rollback to a previously
  published legacy last-good build.

## Important Discoveries

- Normal Builder flow inspected before publishing, but the publication service
  itself previously verified only validation state and bundle integrity. A
  direct internal publication call could therefore omit runtime inspection.
- Trusted preview can durably store the receipt in existing build resource
  metrics, avoiding a schema migration while binding build, revision, source
  checksum, bundle hash, checks, diagnostics, policy, and fingerprint.
- All Surface-focused backend verification passes: `82 passed`, including
  build, pipeline, publication, SDK, Event Surface, and live compatibility.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- Non-populated state captures, focus-indicator and contrast audits, remote
  capture transport, independent Critic, reviewed widget registry, and
  primary-job simulation remain R9 work.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

Add the host-owned state-fixture contract for loading, empty, stale, error, and
pending/indeterminate views, then add deterministic focus-indicator and contrast
evidence. Add the independent Critic only after those facts are durable.
