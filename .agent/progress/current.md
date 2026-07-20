# Current State - 2026-07-20

## Active Task

R9 is active on `aloy-v1-r9-surface-quality`, branched from PR #198 on
`aloy-v1`. Exact-build receipts, responsive/resource-state evidence, focus,
contrast, lean retained artifacts, stage timings, dense-content fixtures, and
approval-required fixtures are complete. The latest scenario work is in the
current R9 branch; primary-job simulation is next.

## Decisions Made

- `aloy-v1` remains the integration branch; R9 is not a path directly to
  `main`.
- Quality evidence belongs to the trusted host and exact retained build. The
  Builder, generated application, and any optional Critic never receive
  publication power.
- Quality policy `@4` covers validation, runtime, declared interactions, five
  responsive viewports, seven resource lifecycles plus dense-content and
  approval scenarios at desktop/mobile sizes, keyboard focus, text contrast,
  capture integrity, and basic DOM accessibility. Primary-job evidence remains.
- A visual Critic is deferred and optional. It is not a V1 publication gate;
  deterministic inspection plus Builder rules carry the required quality path.
- All 23 required compositions are deterministically inspected. Only five
  baseline viewports retain PNGs; 18 state/scenario compositions retain signed
  observations and fingerprints, avoiding unnecessary image/model latency.
- Focus Visible is blocking at the WCAG 2.2 AA level. Stronger 2px/3:1 focus
  appearance is recorded as evidence without claiming the optional AAA level.
- Text contrast blocks below 4.5:1 for normal text or 3:1 for large text. Text
  over an unresolvable image/gradient backdrop fails closed instead of guessing.
- Resource state is real product context, not a publication-test flag. Generated
  code consumes it through `useSurfaceResourceState` and binds its visible
  primary region using SDK-owned feedback props.
- Long content populates ordinary capability-scoped Event shapes. Approval uses
  pending Proposal and `waiting_approval` Interaction truth;
  `useSurfaceApprovalState` binds a summary while controls remain host-owned.
- A stricter new-publication gate must not remove rollback to a previously
  published legacy last-good build.

## Important Discoveries

- All Surface-focused backend verification passes: `82 passed`, including
  build, pipeline, publication, SDK, Event Surface, and live compatibility.
- The scenario slice passes 47 focused backend tests, the SDK TypeScript build,
  Ruff, and mypy across 126 backend files. Browser proofs accept the SDK-bound
  approval summary and reject both missing approval binding and dense overflow.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- Remote evidence transport, reviewed widget registry, and primary-job
  simulation remain R9 work.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

Implement primary-job simulation over host-owned Event facts and typed Surface
interactions.
