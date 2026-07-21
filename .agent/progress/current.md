# Current State - 2026-07-21

## Active Task

R9 internal Surface health is active on `aloy-v1-r9-surface-health`, branched
from `aloy-v1` after PR #208 merged trusted live-Surface reinspection and PR
#209 merged the trusted universal file viewer. The exact published build now has
one operator-only health contract for ready, checking, regression, and
inspector-outage states. Aloy's user interface contains no health badge or
manual check control. Accepted regressions still enter the ordinary Builder and
last-good publication path.

The merged viewer resolves uploads and binary Run artifacts through one
host-owned presentation contract. PDF, image, audio, video, Markdown,
text/code, DOCX, XLSX, and PPTX have fixed trusted renderers; unknown, corrupt,
encrypted, or oversized previews retain honest download and Ask Aloy fallbacks.

Exact-build receipt reuse is now the first inspection-planning optimization:
reopening a build and ordinary SDK data/state changes do not rerun the remote
gate. New executable content still fails closed through the complete gate until
trusted compiler impact provenance can justify narrower evidence reuse.

The isolated remote inspector adapter and exact-build receipt reuse are already
merged. A real provider acceptance proof remains deferred until provider access
is available; it does not block implementing and testing host-owned evolution
semantics locally.

## Decisions Made

- File presentation is trusted-host functionality. Generated Surfaces never
  receive original object URLs, arbitrary file rendering authority, or an
  executable document DOM.
- Original bytes remain immutable. Office previews are bounded inert JSON and
  binary artifacts reuse their durable `StoredFile` instead of passing through
  the text-artifact endpoint.
- Hosted media uses short-lived presigned object URLs for native seeking. Local
  development keeps authenticated byte-range delivery but currently falls back
  to a Blob in the web app because media elements cannot attach the Bearer
  header directly.
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
- Inferred Surface evolution never queues work on its own. The host stores and
  aggregates signals; only explicit acceptance can queue Builder work.
- Proposals are bound to the published revision/build. Acceptance fails closed
  if that Surface changed, and dismissal suppresses the same proposal for 14
  days without erasing its evidence history.
- Generated code cannot submit evolution evidence or decide proposals. Feedback
  lives in host chrome, command failures come from host receipts, and Event phase
  changes come from the canonical Event update boundary.
- Failed requested candidates do not generate another proposal. Trusted quality
  signals are reserved for background reinspection of the live build to avoid a
  self-retrying Builder loop.
- Reinspection never replaces the publication receipt. Inspector outages retry
  as infrastructure failures and never become Surface-quality evidence.
- Automatic reinspection is cost-gated and disabled by default. A bounded
  worker planner can be enabled by an operator; manual checks use the same Run.
- Surface health is operator-only infrastructure. Neither generated code nor
  Aloy's user interface can claim health, request inspection, or access raw
  inspection artifacts.
- Inspector failure is `unavailable`, never `needs_improvement`; only trusted
  evidence bound to the still-live build can create the improvement state.

## Important Discoveries

- The Surface health slice passes focused backend tests, including the
  credit-free regression -> proposal -> accepted Builder queue flow while the
  old build remains published and ordinary members are denied operator health
  controls. App typecheck/build and ESLint pass with no health UI.
- The live-reinspection slice passes 24 focused backend tests across queueing,
  worker dispatch, throttling, fresh exact-build evidence, quality proposals,
  and infrastructure-failure isolation. Ruff and focused mypy pass.
- The viewer slice passes 21 focused backend tests, all 10 app tests, the app
  production build and ESLint, Ruff, and focused mypy.
- PDF failure came from a fragile sandboxed iframe. The viewer now uses the
  browser's trusted PDF object path; audio and video use native controls.
- The existing DOCX/XLSX extraction seam was sufficient for safe structured
  previews and now also exposes DOCX blocks, XLSX sheets, and PPTX slide text.
- The scenario slice passes 47 focused backend tests, the SDK TypeScript build,
  Ruff, and mypy across 126 backend files. Browser proofs accept the SDK-bound
  approval summary and reject both missing approval binding and dense overflow.
- The full backend suite exceeded a bounded six-minute command deadline without
  reporting a failure; it is not recorded as passing or failing.

## Blockers

- Manual Workbench QA with real large PDF, Office, image, audio, and video files
  remains before merge.
- Layout-faithful Office rendering, PDF thumbnails/search, media captions and
  waveforms, and Desktop `Open in default application` remain enrichments on
  the same presentation contract, not hidden claims of this slice.
- A real remote inspector provider acceptance proof still needs provider-backed
  acquisition, inspection, timeout, and recovery evidence.
- Live University, Madrid, and Career provider proofs and pinned remote E2B
  acceptance remain deferred gates.

## Next Session Should Start With

After PR #210, bootstrap the private `aloy-internal` repository and its typed
operator client. First prove operator-only health read and manual reinspection
against a locally published fixed Surface, then exercise a trusted failed test
inspection and accepted improvement while the old Surface stays live.

Also exercise one real file from each viewer renderer class in the target
Chromium/Electron runtime, including mobile overflow, download, Ask Aloy,
corrupt Office fallback, and hosted video seeking.
