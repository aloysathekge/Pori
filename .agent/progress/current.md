# Current State - 2026-07-20

## Active Task

R9 is active on `aloy-v1-r9-surface-quality`, branched from the PR #198 merge
on `aloy-v1`. Exact-build quality receipts plus the first deterministic
viewport/accessibility evidence slice are complete. The next slice is the
host-owned state contract used by both real Surfaces and trusted inspection;
that slice is complete on the R9 branch. Focus and contrast are next.

## Decisions Made

- `aloy-v1` remains the integration branch; R9 is not a path directly to
  `main`.
- Quality evidence belongs to the trusted host and exact retained build. The
  Builder, Critic, and generated application never receive publication power.
- The second policy covers deterministic validation, runtime, declared
  interactions, five responsive viewports, seven public resource states at
  desktop/mobile sizes, capture integrity, and basic DOM accessibility. Focus
  indicators, contrast, Critic, and primary-job evidence remain explicit
  planned extensions.
- Resource state is real product context, not a publication-test flag. Generated
  code consumes it through `useSurfaceResourceState` and binds its visible
  primary region using SDK-owned feedback props.
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
- The resource-state slice passes its 47 focused backend tests, 126-file mypy
  gate, SDK/app TypeScript builds, app lint, and all 7 bridge tests.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- Focus-indicator and contrast audits, long-content and approval fixtures,
  remote capture transport, independent Critic, reviewed widget registry, and
  primary-job simulation remain R9 work.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

Add deterministic focus-indicator and contrast evidence, then long-content and
approval fixtures. Add the independent Critic only after those facts are durable.
