# Current State - 2026-07-20

## Active Task

R9 is active on `aloy-v1-r9-surface-quality`, branched from the PR #198 merge
on `aloy-v1`. Exact-build quality receipts plus the first deterministic
viewport/accessibility evidence slice are complete. The next slice is the
host-owned state contract used by both real Surfaces and trusted inspection;
that slice is complete on the R9 branch. Deterministic focus and contrast
evidence plus streamlined retained artifacts and stage timings are implemented
in the current working tree; long-content and approval fixtures are next.

## Decisions Made

- `aloy-v1` remains the integration branch; R9 is not a path directly to
  `main`.
- Quality evidence belongs to the trusted host and exact retained build. The
  Builder, generated application, and any optional Critic never receive
  publication power.
- The third policy covers deterministic validation, runtime, declared
  interactions, five responsive viewports, seven public resource states at
  desktop/mobile sizes, keyboard traversal, visible focus, deterministic text
  contrast, capture integrity, and basic DOM accessibility. Primary-job
  evidence remains an explicit planned extension.
- A visual Critic is deferred and optional. It is not a V1 publication gate;
  deterministic inspection plus Builder rules carry the required quality path.
- All 19 required compositions remain deterministically inspected. Only the
  five baseline viewports retain PNGs; the 14 state compositions retain signed
  observations and fingerprints, avoiding unnecessary image/model latency.
- Focus Visible is blocking at the WCAG 2.2 AA level. Stronger 2px/3:1 focus
  appearance is recorded as evidence without claiming the optional AAA level.
- Text contrast blocks below 4.5:1 for normal text or 3:1 for large text. Text
  over an unresolvable image/gradient backdrop fails closed instead of guessing.
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
- The resource-state slice passes 47 focused backend tests, SDK/app builds,
  app lint, and all 7 bridge tests. Focus/contrast passes 28 build/publication
  tests, both final browser proofs, and the 126-file mypy gate.
- The streamlined receipt passes 30 focused build/publication/quality tests and
  now records reconciled bootstrap, viewport, state, interaction, and total
  inspection timings.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- Long-content and approval fixtures, remote evidence transport, reviewed
  widget registry, and primary-job simulation remain R9 work.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

Add long-content and approval fixtures, then primary-job simulation over those
host-owned facts.
