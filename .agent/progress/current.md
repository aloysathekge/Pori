# Current State - 2026-07-21

## Active Task

R9 remote inspector provider is active on
`aloy-v1-r9-remote-inspector-provider`, branched from `aloy-v1` after PR #202.
The isolated adapter is implemented: it sends one sealed request to the fixed
Surface toolchain and receives one typed bounded result. The next acceptance
work is an actual remote provider proof with acquisition/inspection/recovery
timings, without granting the sandbox storage, credentials, Event truth, or
publication authority.

## Decisions Made

- `aloy-v1` remains the integration branch; R9 is not a path directly to
  `main`.
- Quality evidence belongs to the trusted host and exact retained build. The
  Builder, generated application, and any optional Critic never receive
  publication power.
- Quality policy `@5` covers validation, runtime, declared interactions, five
  responsive viewports, seven resource lifecycles plus dense-content and
  approval scenarios at desktop/mobile sizes, keyboard focus, text contrast,
  capture integrity, basic DOM accessibility, and primary-job evidence.
- A Surface request freezes its requested jobs before Builder execution as a
  fingerprinted receipt. Candidate metadata and `surface.json` must preserve
  those exact ids, descriptions, and order; mismatch fails before persistence.
- Primary jobs use bounded accessible click/fill/select steps. Assertions can
  prove exact named UI, exactly one schema-valid SDK request, committed
  capability-scoped Surface state, or host-owned approval state. The host
  browser resets Event context and binds each result and timing to the build.
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

- The current primary-job slice passes `44` focused backend tests across
  requests, pipeline, exact-build quality, and real browser build inspection;
  Ruff passes on all changed Python files.
- The scenario slice passes 47 focused backend tests, the SDK TypeScript build,
  Ruff, and mypy across 126 backend files. Browser proofs accept the SDK-bound
  approval summary and reject both missing approval binding and dense overflow.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- The transport, immutable inspection receipt, and evidence artifact foundation
  are merged as PR #202. The next task is the actual remote inspector adapter
  and its provider acceptance proof; the reviewed Map/widget adapter follows.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

Implement the remote inspector adapter against the existing fixed sandbox
toolchain. It must return only the typed inspection result and bounded evidence
to the host, then prove cold/warm timing, timeout, and recovery behavior before
moving to the reviewed Map/widget adapter.
