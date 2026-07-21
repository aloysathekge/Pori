# Current State - 2026-07-22

## Active Task

R11 durable Work Story hardening passed all seven GitHub checks and merged into
`aloy-v1` through PR #216. R12 is active on
`aloy-v1-r12-event-template-catalog`. The first model-free slice now defines
versioned catalog/release/asset/compatibility/seed/guided-job/installation
records and a generic idempotent installation transaction. A fake Career OS
release proves the contract in tests without adding Career behavior to Aloy.
The second slice adds the protected global authoring boundary: Aloy Internal
can stage, review, and publish exact releases through versioned APIs with
deployment-scoped authority, organization RBAC, idempotent intent, and immutable
audit receipts rather than direct database access.

The trusted universal file viewer, Surface quality/inspection contracts,
operator-only control APIs, and the separate private `aloy-internal` control
plane are established. R10 document ingestion/OCR is fully specified but not
implemented. Real provider acceptance remains deferred until credentials are
available and does not block model-free template and ingestion contracts.

## Decisions Made

- Work Story sequencing cannot depend on a `Run` row because inline
  Conversations emit milestones before their terminal Run is committed. A
  dedicated per-Run cursor atomically allocates sequence numbers in the same
  transaction as the event.
- Replay identity is derived only from the bounded public projection. Raw tool
  arguments and private model output never enter idempotency keys.
- Same-process live readers wake immediately through a replaceable notifier;
  sequence-cursor database replay remains authoritative, so a missed or
  cross-process notification cannot lose data. A hosted broker or Postgres
  notification adapter can replace this seam without changing the API.
- Work Stories retain at most 50,000 bounded semantic milestones per Run and
  64 KiB per public payload. Normal execution budgets remain far below that
  safety ceiling.
- Event templates are opt-in, versioned catalog data rather than domain logic in
  Aloy. Installing one creates an independent ordinary Event pinned to that
  release; updates never silently overwrite user data.
- Student, Individual, Professional, Team, and Business are discovery taxonomy.
  Subscription packages separately control limits/capabilities and do not gate
  template availability by matching label.
- Career OS is the first permanent starting template, University follows, and
  Madrid waits for the trusted Map/widget phase. Sparse context creates visible
  setup gaps rather than invented facts.
- A template release may seed Surface source and canonical data, but it cannot
  publish itself. Installation leaves the project at a template-source-ready
  build boundary; the ordinary host build, inspection, and publication gate
  must make it live.
- Template sample data and setup gaps use distinct canonical postures and retain
  release provenance. Neither is represented as user-confirmed truth.
- Global template publication requires both ordinary operator RBAC and an
  explicit deployment-owned subject allowlist. The allowlist is empty and
  authoring is disabled by default; being an owner of an ordinary organization
  does not grant global catalog authority.
- Import only stages a checksum-bound draft. Publication requires the exact
  reviewed checksum, revalidates stored content and real asset bytes, then
  advances the catalog pointer only if its reviewed prior value still matches,
  and records an immutable receipt.
- File ingestion is separate from presentation. Upload returns after durable
  original storage; OCR, normalization, indexing, and enrichment are resumable
  host-owned background work.
- Extracted content is Event-scoped evidence, not accepted memory. Citations
  remain bound to original hash, extraction version, page, block, and region.
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

- Inline Conversation execution writes Work Story milestones before the Run
  finalizer creates its terminal history row; a Run-column counter would have
  broken this valid path.
- A terminal Run may have `status="completed"` with `success=false` after a
  bounded stop. Terminal reconciliation therefore uses both fields and never
  renders a failed budget outcome as successful work.
- Focused hardening tests cover replay idempotency, two concurrent writers,
  terminal crash repair, cancellation, payload drift, event limits, cursor
  pagination beyond 500 entries, notification wakeup, and organization
  isolation.
- Seven template tests prove published discovery without subscription coupling,
  ordinary Event/context/Task/Surface materialization, replay idempotency,
  intentional repeat installs, catalog-withdrawal independence, corrupt-release
  rejection, and tenant isolation. Twenty-three existing Event/context/Surface
  regressions pass; the complete backend suite passes 506 tests; and an empty
  SQLite database upgrades to template migration `s1d2e3f4a5b6` at Alembic
  head.
- Eight authoring proofs cover fail-closed deployment authority, organization
  RBAC, draft visibility, idempotent imports/publications, exact-checksum draft
  replacement, published-release immutability, post-review tamper rejection,
  stored-asset integrity, and installed-Event independence. Together with the
  seven install/catalog proofs, all 15 pass; an empty SQLite database upgrades
  to authoring migration `t2d3e4f5a6b7` at Alembic head, and the complete
  backend suite passes all 514 tests.
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
- The product database and protected API now have the generic catalog authoring
  contract but no production Career OS release yet. Aloy Internal still needs
  its typed client/review UI over this API, and template source still needs a
  generic host-build kickoff before a seeded Surface can become live.

## Next Session Should Start With

Add the typed catalog client and release-review workflow in `aloy-internal`,
using only the protected import/list/detail/publish APIs, then load the first
real Career OS catalog release as database data. Connect a successful
installation's template-source-ready record to the normal host
build/inspection/publication pipeline before adding template discovery and
install UI. Add a protected asset-upload boundary before shipping releases with
binary assets; do not grant Aloy Internal direct object-store access, create a
parallel template build path, or let a release publish itself.

In parallel planning, keep R10 document ingestion behind the provider-neutral
`DocumentProcessor` and `DocumentGraph` contracts. Also exercise one real file
from each existing viewer renderer class in Chromium/Electron when manual QA is
available.
