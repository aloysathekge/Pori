# Current State - 2026-07-24

## Live Revision Runs — Convergence Proven, Cap Is the Boundary (2026-07-24 evening)

Two live packing-checklist revision runs on the merged stack. Run 1 died on
the 40-step ceiling mid-repair -> fixed by PR #234 (request the designed
3x20+10 shape; default policy narrows to 50). Run 2 (50 steps) delivered
the decisive datapoint: sub 1 gate in 86s -> visibility rejection; sub 2
repaired -> gate; **sub 3 FIXED the state-region/visibility rule itself**
(the "unsolvable" rule is self-repairable given room) and then failed on a
NEW shallower layer: self-authored primary-job proofs referencing items
("Test item", "Passport") that don't exist in an empty checklist — proofs
must create their data (fill -> add -> then assert). 74-second
candidate->gate cycles, 91% cached, ~$0.10/run. Updated next-session
priorities: (1) skill: job proofs must be self-sufficient + an eval case
for it; (2) the 3-submission cap is the wrong boundary — the model was
converging one layer per submission; M2's budget-bound session (no
submission count) would likely have published on attempt 4; (3) the
visibility lever demoted from "designed fix required" to "headroom or
primitive". Stack stopped after this; no runs pending.

## Round 3 — Static Rule Works, Visibility Half Remains (2026-07-24, PR #233)

The unwired-capability rule moved into the free static tier
(`validate_surface_source`: declared tasks/files/proposals/data:*/records:*
capability must appear in a `useSurfaceResourceState` call), the baseline
gained a `ResourceSection` primitive (hook + feedbackProps + status banner
built in), and the skill states the invariant. Round-3 eval: kimi now WIRES
the hook (behavior change confirmed), `revision-small` still publishes
($0.06), but `revision-jobtracker` fails on the rule's second half — the
wired region sits behind navigation with no `resource_views` click path.
That half is statically undecidable (root-visible regions cannot declare
resource_views; steps require >=1 click), and kimi ignored the gate's
explicit remediation across three repairs. Next levers to design fresh:
richer repair guidance for this specific diagnostic, an App-shell pattern
where adding a nav view auto-suggests the resource_views entry, or a
workspace-tier warning when a data capability is wired but absent from
resource_views AND the nav has multiple buttons. Eval r3 JSON in session
scratchpad. After that: M2.

## First Measured Eval Cycle — Loop Convergence (2026-07-24, PR #232)

The eval harness ran its first full measure→fix→measure cycle. Round 1:
kimi-k2p6 and glm-5p2 both 0/2 (`did not finish within 20 turns`);
transcripts showed sessions reaching green checks then reading files until
the cap, and glm's only finish attempt dying on summary string validation.
Three deterministic loop fixes (PR #232): per-turn host-owned countdown
(cache-safe append-only), finish summary normalization (display metadata
never kills a candidate), and transcript-carrying SurfaceBuilderLoopError
with per-action counts in the eval report. Round 2: **both models
published `revision-small` on the first gate attempt** — kimi: 16 turns,
$0.056, 96s, the first model-authored Surface ever to clear the full real
gate. Both fail `revision-jobtracker` on the identical rule
(`state_region_missing` on new views, 3 attempts each) — a
model-independent teachability finding: the next lever is baseline
primitives that wire `useSurfaceResourceState` into new views by
construction (or a deliberate review of the rule for local-state-only
views). Head-to-head verdict: **kimi-k2p6 confirmed as Builder** (equal
pass/convergence at ~3x lower cost than glm-5p2). Eval JSONs in session
scratchpad; next session should fold the state-region lever, then M2.

# (2026-07-23 log follows)

## Local Environment Recovery (2026-07-23)

The R13 Surface workspace merged into `aloy-v1` through PR #224; this checkout
now tracks `origin/aloy-v1`. Surfaces did not build locally for three stacked
reasons, all corrected:

1. The `@aloy/surface` SDK `dist/` was never built in this checkout, so
   `LocalDevelopmentSurfaceBuildRunner.available` was false and every build
   returned `blocked: local_toolchain_unavailable`. Fixed with `bun install`
   at the workspace root plus `bun run build` in `packages/aloy-surface`.
2. The local `aloy.db` was stamped at Alembic `q3a4b5c6d7e8` — 30 migrations
   behind head `t2d3e4f5a6b7` — with `stored_files` created out-of-band by
   dev `create_all`. Recovery: backup to `aloy.db.bak-2026-07-23`, stamp to
   `s5c6d7e8f9a0` (skipping the two already-materialized stored_files
   migrations), then `alembic upgrade head`. All 30 migrations applied; the
   Life event was created and all 24 conversations / 80 runs backfilled onto
   it. A metadata-vs-DB diff shows zero missing tables or columns.
3. A real Windows defect in `surface_runtime_inspection.py`: Chrome's
   crashpad handler outlives the terminated browser and keeps
   `chrome-profile/CrashpadMetrics-active.pma` memory-mapped, so
   `TemporaryDirectory` cleanup raised `PermissionError` and destroyed
   otherwise-complete inspection results (12 browser-gate tests failed; live
   publication would fail the same way). Fixed by launching Chrome with
   `--disable-breakpad --disable-crash-reporter` and using
   `ignore_cleanup_errors=True` on the inspection temp directory. This code
   change is uncommitted on `aloy-v1`.

Verification: a direct `LocalDevelopmentSurfaceBuildRunner` smoke build
succeeds (typecheck → instrumentation → Vite → 62 KB bundle). All 193
Surface-related backend tests pass (builder/loop/builds/workspace/pipeline/
publication/requests/materialization 93, remaining Surface files 100). SDK
tests (5) and Aloy app tests (30) pass; the app production build passes.
Focused mypy on the Surface modules is clean; black is clean on the edited
file. Ruff reports 7 pre-existing unused-import errors in old migrations and
`memory_store.py`, untouched.

## Surface Update Pipeline Repair (2026-07-23)

User-reported: updating an existing Surface always failed. Two structural
defects in the workspace Builder path were found and corrected (uncommitted on
`aloy-v1`, together with the crashpad fix above):

1. **Repair-base desync** (`surface_builder.py`). The workspace invoker
   advanced its diff base to the finished candidate after every submission,
   but the executor only advanced `base_files` when a revision persisted. Any
   pre-persistence rejection (typically the primary-job contract gate) made
   the next edit envelope — a diff against candidate N — get materialized onto
   the ORIGINAL source: the first submission's changes silently vanished, the
   model's workspace and the host's candidate diverged permanently, and the
   repair budget burned down to guaranteed failure. `base_files` now advances
   to the exact rejected candidate source after every materialized edit-mode
   submission, restoring the documented "repairs use the exact rejected
   source" invariant for both the workspace invoker and the legacy repair
   prompt.
2. **Primary-job contract blindness in the workspace** (`surface_pipeline.py`,
   `surface_builder_loop.py`). Every request — including a revision — freezes
   its own job contract (ids hash the new descriptions), but the existing
   published `surface.json` declares the PREVIOUS request's jobs, and the
   workspace's trusted check never saw the contract: `finish_candidate` went
   green and the paid host gate then deterministically rejected
   `primary_job_manifest_mismatch` (bind's count-equal rebind fallback only
   rescued same-count updates — Career OS's 4 old jobs vs 1 new job always
   failed). The contract check is now a single shared function
   (`surface_primary_job_contract_diagnostics`) used by the host pipeline and
   evaluated (after the same bind) at `finish_candidate`, so a stale manifest
   is refused in-workspace for free instead of burning a paid submission.

Supporting changes: the Builder skill now states that `surface.json`
`primary_jobs` must declare exactly this request's contract, replacing any
previously declared set; `SurfaceRequestParams.jobs` now instructs the Event
agent that a revision must restate the complete job set (retained + new), not
only the delta. The stale-proof hazard is also closed:
`bind_surface_manifest_primary_jobs`' positional fallback now requires the
declared ids to exactly match the host-issued ids (paraphrased-description
rescue only); a same-count manifest still declaring a previous request's
foreign job ids fails the gate instead of silently carrying its old browser
proofs onto new job identities.

Verification: new regression tests
`test_workspace_repair_keeps_prior_changes_after_pre_persistence_rejection`
(surface_builder),
`test_finish_refuses_stale_primary_job_contract_before_paid_submission`
(surface_builder_loop), and
`test_host_never_rebinds_stale_same_count_jobs_from_a_previous_request`
(surface_pipeline) pass; the affected Surface slice passes 83 tests (two
browser-gate tests flaked under load and pass in isolation); black, isort,
focused mypy, and Ruff are clean on the changed modules.


## Active Task

R12 Event-resource work merged into `aloy-v1` through PR #223. The active R13
branch, `aloy-v1-r13-surface-workspace`, replaces the default one-shot Surface
Builder path with a provider-neutral iterative development workspace. The local
provider uses an ephemeral source-only Git repository; the Builder receives
bounded list/read/search/edit/check/diagnostic/finish tools, works across
multiple turns, and no longer requires provider `structured_output`. A fresh
trusted check must match the exact finished source fingerprint before the
existing immutable revision, browser-quality, and atomic publication pipeline
can make it live. Native tool calls are preferred and a validated JSON-action
fallback supports text-only providers.

R13 also adds a host-owned Surface element selector. The fixed local build
toolchain overwrites intrinsic JSX with real source attribution, the isolated
SDK captures bounded semantic element context without firing the underlying
control, the host binds it to the authenticated build/revision, and the backend
rejects stale selections. **Ask Aloy about this** and **Change this** route plain
language plus hidden advisory selection context into the permanent Event
Conversation. The integrated 78-test Surface suite passes against the rebuilt
SDK, along with 32 kernel tool-call/skill tests, all 30 Aloy app tests, SDK and
app production builds, app lint, full kernel/backend mypy, focused Ruff, Black,
and isort. CI remains before merge.

The live provider acceptance exposed and the branch now corrects a Builder
control-flow flaw before any further paid testing: runtime inspection used to
stop after each quality class, allowing successive model submissions to reveal
viewport, state, and job failures one paid cycle at a time. The trusted browser
now completes every independent post-boot inspection and returns one compact
diagnostic bundle. V1 permits one initial generation plus at most two focused
repairs under a 200,000 aggregate-token cap; duplicate/no-op repair terminates
immediately. Existing revisions support safe
single-match `replace_text` transactions in addition to file writes/deletes, and
tabbed/routed SDK state uses explicit manifest `resource_views` rather than
guessing from arbitrary job controls. The Surface Builder run profile is v3.
Focused Pori tests, the full 73-test Surface slice, Ruff, full kernel/backend
mypy, and Aloy app tests/lint/build are green without a provider call. A second
spend-boundary defect found during local recovery is also corrected: Event
reads no longer enqueue the one-time Brief bootstrap, and the bootstrap now
keys only on its versioned setup/evidence projection. Operational
Surface/Task/Trail churn may refresh general context but cannot supersede the
Brief or trigger another paid Run. Regression coverage proves both boundaries.
Both accidental local bootstrap Runs were cancelled before claim; the corrected
API stayed at zero Runs under live UI polling and the restarted worker stayed
idle. A controlled Career OS Builder acceptance then stopped at the two-call
cap and exposed an over-strict source rule: both otherwise structured outputs
contained ordered edits to one path. The transaction now supports sequential
same-file edits atomically, with exact-match checks on every step. Its
prompt projection also removes duplicate published source and compacts internal
Trail/evidence context, reducing the live fixture from 213,346 to 65,506
serialized characters. A second bounded acceptance honored the two-call cap
(57,686 total tokens, no conversation-model call) and exposed a final host
semantic error: an already-satisfied `surface.json` write caused the entire
otherwise meaningful transaction to be rejected. The host now normalizes such
idempotent operations, rejects only a net-unchanged transaction, and never
reports a nonexistent third submission. Focused checks and the expanded
78-test Surface slice are green. A subsequent real Career OS request exposed
that a source-contract rebase consumed the only repair before a missing
TypeScript binding could be corrected. Repair prompts now use the exact rejected
source without unrelated Event context. That request also exposed a browser-test
isolation flaw: an empty-files job was asserted against current populated data,
then its fixture leaked into the following interaction suite. Primary jobs can
now declare trusted state fixtures and reset independently; the ordered
interaction suite resets once to canonical data and then preserves accepted
state between its checks. The retained candidate was recovered
with zero model tokens and published as Career OS revision 14 after compilation,
fixture-aware browser inspection, interaction proofs, and primary-job proofs all
passed. The expanded 82-test Surface/request slice, focused Ruff and mypy, and
the Aloy app's 29 tests, lint, typecheck, and production build are green.

The Event-actions UX slice passed all seven GitHub checks and merged into
`aloy-v1` through PR #222. The follow-on trusted Event-resource boundary also
passed all checks and merged through PR #223: generated Surfaces receive typed
file/artifact metadata, a host-owned `openResource(fileId)` Workbench intent
with no Run or storage URL, and resource-aware `askAloy` turns whose explicit
file IDs are revalidated by both the browser host and backend before entering
the permanent Event Conversation.

R11 durable Work Story hardening passed all seven GitHub checks and merged into
`aloy-v1` through PR #216. R12's governed Event-template catalog passed all
seven checks and merged through PR #217. The first model-free slice defines
versioned catalog/release/asset/compatibility/seed/guided-job/installation
records and a generic idempotent installation transaction. A fake Career OS
release proves the contract in tests without adding Career behavior to Aloy.
The second slice adds the protected global authoring boundary: Aloy Internal
can stage, review, and publish exact releases through versioned APIs with
deployment-scoped authority, organization RBAC, idempotent intent, and immutable
audit receipts rather than direct database access.
The separate `aloy-internal` phase-4 work consumes that boundary through
a typed client, bounded loopback bridge, and fixture/live-local Governance
review screen. It displays the exact checksum and observed catalog pointer and
returns Pori's receipt unchanged.
Phase 4 passed clean-checkout CI and merged through internal PR #4. The
phase-5 branch now holds the first real Career OS v1 release and a
generic release kit. Its exact React source passes Pori's manifest, build,
runtime, interaction, and four-primary-job gates, plus a local protected
stage/publish/install proof. It creates one honest setup Task, separates sample
and setup-gap posture, and routes research to the permanent Event conversation.
The next R12 slice passed all seven checks and merged into `aloy-v1` through PR
#218: installing reviewed source atomically
queues a durable model-free Surface materialization Run. The worker sends the
frozen revision through the same build, inspection, quality, and publication
authority used after model generation, while leaving the Event conversation
available to the user.

The trusted universal file viewer, Surface quality/inspection contracts,
operator-only control APIs, and the separate private `aloy-internal` control
plane are established. R10 document ingestion/OCR is fully specified but not
implemented. Real provider acceptance remains deferred until credentials are
available and does not block model-free template and ingestion contracts.

The user-facing discovery/install slice is implemented on
`aloy-v1-r12-template-discovery`. Every New Event entry point now opens a
responsive chooser backed only by the published catalog API. Users may inspect
a release's guided starting jobs, rename the Event, and install the exact
listed release with a stable retry key, or continue into the existing custom
setup flow. Catalog failure never blocks custom Event creation.

The Event lifecycle slice is implemented on `aloy-v1-r12-event-lifecycle`.
Event settings now archive a workspace through the canonical lifecycle
boundary; normal navigation lists only active Events, while an Archived Events
library restores or permanently deletes an exact-name-confirmed Event. Life is
never archivable or deletable, active Runs are asked to stop on archive, and
permanent deletion refuses until they are terminal. Purge removes the complete
tenant-scoped Event aggregate and its owned object-storage keys without touching
shared template releases or other Events.

The live Surface form boundary is corrected on the same branch. Generated
React forms now receive `allow-forms` inside the opaque-origin iframe while the
host runtime retains `form-action 'none'`, no same-origin access, and no network
access. Form submits therefore reach the typed MessageChannel bridge without
granting generated code direct transmission authority. The SDK also generates
transport/idempotency identifiers without assuming secure-context
`crypto.randomUUID`, and the browser publication gate exercises that fallback.
The running Career OS proof saved `Aloy Studio / AI Engineer / Remote`, showed
trusted success feedback, and preserved the row after a full runtime reload.
The command contract now also has an explicit `upsert` operation for singleton
settings and preferences: it creates on first save and replaces on later saves,
while strict `replace`, `merge`, and `delete` still expose missing-record
conflicts. Career OS Direction was upgraded through an immutable revision and
the ordinary build, browser-inspection, and publication pipeline. The signed-in
live proof saved Direction twice and preserved it after a Surface reload.
Surface reasoning no longer leaks command ids or worker mechanics into the
Event conversation. Reviewed intent labels drive compact Surface-request
lifecycle cards; legacy leaked messages render through a safe compatibility
projection, zero-action failures use ordinary product language, raw failures
remain in Trail/operator evidence, and reasoning controls stay busy for the
full accepted interaction lifecycle. Career OS revision 4 was published with
the `Research matching roles` label, and the signed-in UI proof contains no
command identifier, retry-exhaustion copy, action counter, or technical-details
control on those request cards.

## Builder Eval Harness (2026-07-24 — PR #231)

`aloy_backend/builder_eval.py`: offline model qualification by measurement.
CLI (`python -m aloy_backend.builder_eval --provider fireworks --model …`)
replays fixture revision requests (Notes view; the realistic job tracker)
through the real workspace loop + real contract/compile/browser gate
against the bundled baseline, up to 3 gate attempts with repair feedback
and identical-fingerprint no-progress detection. Reports gate pass rate,
turns, calls, fresh/cached/output tokens, estimated cost (price
overrides), wall time, failing diagnostics; JSON output for cross-model
comparison. 3 scripted-model tests (no spend) including a golden-path
proof through the real gate. First real use: run kimi-k2p6 vs candidates
(glm-5p2, deepseek) on both cases to settle the turn-convergence
question with data. Next after merge: M2 `builder_session.py`.

## Cache Economics Proven (2026-07-23, night — PR #228)

First edit-mode Builder run against the baseline burned 810k tokens (33
uncached calls of a ~24k prompt that embedded the source twice). Three
fixes, measured live on the next run: session affinity (run id as the
OpenAI `user` field — Fireworks serverless caching is automatic but
per-replica, which is why hits were 0%), cached-token accounting
(`prompt_tokens_details.cached_tokens` → `cache_read_tokens`), and a
source-free workspace prompt (paths only; the workspace owns file bodies).
Result: **89% cache hit rate** (437,708 of 489,650 input tokens cached),
~$0.18 real cost vs ~$0.84. kimi-k2p6 IS Kimi K2.6 with $0.16/M cached
pricing. Remaining Builder gap is model convergence — kimi burned the
20-turn loop cap without finishing (plus a local DNS outage killed the
retry); that is eval-harness territory, not infrastructure. The
conversation agent also queues duplicate surface requests when the user
retries — three duplicates were DB-cancelled twice today; worth a product
fix. PRs today: #225 (hardening+v2 direction), #226 (M1 delivery), #227
(promote-route delivery), #228 (cache economics).

## M1: Baseline Delivery Shipped (2026-07-23, late evening)

`aloy_backend/baseline_delivery.py` + the `create_event` route now persist
the bundled baseline draft (checksum-bound, fingerprint =
`baseline_surface_fingerprint()`) and queue the model-free materialization
Run on every custom Event creation. Life, archived Events, Events with
existing Surface state, and replays are exempt/idempotent; a delivery
failure logs and never blocks creation. Gated by
`settings.surface_baseline_enabled` (default True; the test conftest
disables it by default so pre-baseline suites keep their own fixtures, and
`tests/test_baseline_delivery.py` re-enables it for its four delivery
proofs including worker dispatch to publication via the injectable
pipeline). Spec §4 updated (v1.1 note): delivery consumes the bundled asset
directly; catalog-release publication stays optional hosted hardening.
38 event/surface regression tests pass alongside the 7 baseline tests.
Next: the Builder eval harness, then M2 (`builder_session.py`).

## Builder v2 Redesign (2026-07-23, evening)

After the full live acceptance day (three bounded runs on Fireworks kimi; the
final one reached contract-pass → persist → compile → browser gate → final
repair before dying on the 40-step ceiling at 765k uncached tokens), the
decision is a staged rewrite of the generation layer, specified in
`docs/aloy-surface-builder-v2-spec.md`: one BuilderSession replaces
Run-submissions (no envelopes/shim — the workspace IS the candidate), checks
become tiered with a deterministic autofix tier, the full gate runs in place
at finish with revisions persisted only on gate-pass, five ceilings collapse
into one cache-aware cost budget, provider physics move into declared
profiles, and change requests carry jobs_added/jobs_removed with the host
computing the frozen contract. Trust layer (revisions, gates, publication,
SDK/iframe) explicitly unchanged. Migration M1–M4; M1 = baseline delivery
(S2/S3), whose S1 template is already built and gate-proven
(`aloy_backend/product_surfaces/baseline/`, `tests/test_baseline_surface.py`).
Also landed today, uncommitted: per-call provider timeout + retry in the
loop, `reasoning_effort` applied to kernel tool-calls (the actual root cause
of the Fireworks dead-air hangs), and the k2p6 model switch. No more paid
from-scratch acceptance runs — each just re-proves the same conclusion.

## Baseline Surface Direction (2026-07-23)

Live acceptance exposed that from-scratch Surface creation is the fragile
path (turn-count × uncached-context token burn, provider stall exposure,
boilerplate gate rejections). Decision, specified in
`docs/aloy-baseline-surface-spec.md` (v1.0): every custom Event receives a
published v1 baseline Surface at creation via the existing model-free R12
materialization pipeline (zero model tokens); the Builder then always runs in
edit mode against real source — creation-from-nothing disappears as a
generation mode. Industry research backing this (v0/Lovable/Bolt/Replit/Manus
patterns, verified claims, and five concrete spec adjustments — protected
primitives, deterministic autofixers, cache discipline, verbatim error
context, small-model repair tier) is in
`docs/research-ai-app-builder-templates.md`. The Event description personalizes v1 as runtime data
through the SDK, not as generated code. Implementation slices S1–S4 are in
the spec; nothing is implemented yet. Additional session fixes pending
commit: stall watchdog honors per-call generation timeout for non-streaming
workspace turns (with regression test), Surface Builder token ceiling is now
`SURFACE_BUILDER_MAX_TOTAL_TOKENS` config (default 200k, local 800k),
backend/.env now deterministically wins over the repo-root .env
(aloy_backend/__init__.py), and the Builder model was being switched to
kimi-k2p6 after kimi-k2p7-code showed repeated dead-air provider hangs
(stack restart still pending user go-ahead).

## Decisions Made

- Surface authoring uses a provider-neutral, source-only development workspace.
  Local Git supplies execution checkpoints and diffs; immutable database
  revisions remain canonical, and only the shared host pipeline can publish.
- The Builder does not require provider structured output. Native tool calls or
  a validated text action envelope drive the same bounded workspace tools.
- Surface element selection is host-owned, source-attributed, and bound to the
  current publication twice: by the authenticated iframe bridge and by the
  message endpoint. It is advisory context, never action authority.
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
- Persisted template source does not invoke the Builder model. A dedicated
  model-free Run is bound to the exact revision checksum and uses the shared
  revision pipeline; it has no source-write, network, model, or conversation
  authority.
- Materialization publishes only after the normal host quality receipt passes.
  Crash replay reuses the same stage idempotency keys; a classified transient
  host failure advances a durable pipeline-attempt counter, while invalid,
  tampered, or superseded source fails terminally without changing live state.
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
- The materialization slice passes 34 focused proofs across generic revision
  lifecycle, template installation/authoring, user-visible activity, tamper
  rejection, worker dispatch, transient retry, and real local build → browser
  inspection → publication. Focused Ruff and mypy pass; the Aloy app production
  build and ESLint pass.
- Template discovery passes the Aloy production build, ESLint, and all 14 app
  tests. Signed-in desktop and 390x844 mobile browser checks prove the route,
  responsive fallback, and custom-creation escape path against the currently
  running pre-catalog backend.
- Aloy Internal phase 5 passes `bun run check` after merging the clean-checkout
  package-resolution fix: all 18 tests, typechecks, lint, and production builds
  pass. Career OS v1 merged into the private repo's `main` through PR #5; its
  exact offline review fingerprint is
  `a914fb5865531441e12bc3cffab53a88220447ac049b10216d679b54e190c0ae`.
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
- The product database, protected API, Aloy Internal review workflow, automatic
  host-build kickoff, and discovery/install UI now exist. Career OS v1 cannot
  be staged into the developer's live local catalog until Pori has an explicit
  catalog-operator subject and Aloy Internal has its loopback organization and
  operator-token configuration. Do not bypass this boundary with direct DB
  writes or committed credentials.

## Next Session Should Start With

Cancel or reconcile the interrupted local `event_bootstrap` Run before starting
the worker. Then run exactly one explicitly approved provider acceptance request
against the Builder v2 ceiling and inspect its retained generation, exhaustive
diagnostic, exact-edit, build, and publication receipts. Do not restore three
automatic submissions or stage-by-stage paid repair.

Review and merge the Event lifecycle slice, then continue the protected
template asset-upload boundary before shipping releases with binary assets. Do
not grant Aloy Internal direct object-store access, create a parallel template
build path, or let a release publish itself.

In parallel planning, keep R10 document ingestion behind the provider-neutral
`DocumentProcessor` and `DocumentGraph` contracts. Also exercise one real file
from each existing viewer renderer class in Chromium/Electron when manual QA is
available.
