# Current State - 2026-07-19

## Active Task

Complete and review `aloy-v1-mobile-ux`: make Aloy's shared shell, Event
workspace, creation flows, utility pages, dialogs, and controls deliberately
responsive while preserving the full durable product model on mobile web.

## Decisions Made

- Keep detailed session history in handoff files; keep this file as the next-session briefing.
- Aloy is one user-facing assistant; specialist roles and legacy AgentConfig
  infrastructure are operator-owned and absent from customer navigation.
- Event creation must remain available even when context ingestion is pending or fails.
- Prompt caching is a latency/cost optimization, never durable memory or truth.
- Confidential/restricted Event evidence disables application-owned message-prefix caching.
- Surface opportunity detection belongs to the ordinary Event model's product
  judgment, not a keyword trigger or separate classifier call.
- Ordinary Event Runs may request a Surface but never receive authoring,
  build, preview, or publication tools.
- Schedule/display/navigation records remain Surface data; they are not Tasks
  unless they represent genuine actionable work with a definition of done.
- V1 has one React App Surface runtime. Simple Surfaces remain small React
  projects; no parallel HTML/Surface Lite runtime is planned without telemetry
  proving the fixed React compiler is the bottleneck.
- The Builder will submit one complete candidate; the host will own persistence,
  validation, build, preview, quality, publication, and repair diagnostics.
- The Builder receives no lifecycle or filesystem tools. It returns a complete
  candidate through provider structured output; Aloy grants one bounded full
  repair submission from trusted diagnostics.
- Event memory controls are host-owned and Event-local. Corrections supersede
  immutable history, forgetting soft-deletes, and global promotion always
  creates a separate provenance-linked record through an explicit user action.
- Event memory is governed from the Event's own settings, opened by the settings
  icon directly after Trail in the right Event-context dock; it is not a peer
  operational tab.
- Canonical Event state is never edited through memory controls. Tasks, files,
  receipts, Trail, Surface state, and transcript history retain their own
  authoritative stores.
- Mobile Aloy preserves the same durable state and authority model as desktop,
  but reorganizes it around bottom navigation, action sheets, one primary Event
  region at a time, and a full-screen Event-context panel. Mobile is not a
  compressed desktop split view.
- The planned Expo application is a first-class trusted Aloy host. Host UI uses
  native controls; generated Surfaces remain retained sandboxed React bundles
  in an embedded WebView behind a transport adapter for the same versioned SDK
  protocol. Device capabilities are typed host intents, never direct generated-
  code authority.

## Important Discoveries

- The active `aloy-v1-mobile-ux` branch adds a route-aware mobile shell with
  persistent Today/Life/New/Events/More navigation, mobile Event switching and
  creation sheets, dynamic viewport and safe-area handling, 44-pixel touch
  targets, mobile-safe form inputs, bottom-sheet dialogs, single-region Event
  modes, full-screen Event context, scrollable Workbench tabs, and responsive
  setup/utility pages. App lint, production build, and all `7` Surface bridge
  tests pass. Automated browser screenshots remain blocked by the known desktop
  bootstrap error `Cannot redefine property: process`.

- The `aloy-v1-event-memory-settings` branch adds a dedicated Event-memory
  domain module and `/events/{event_id}/memory` controls, plus an Event settings
  tab directly after Trail in the right context dock. The UI separates mutable Event memory from
  read-only inherited global memory, shows provenance/scope, surfaces failures,
  and confirms destructive or cross-Event-scope actions. Backend isolation and
  context regressions pass (`11 passed`); app lint and production build pass.
  Automated visual browser inspection remains blocked by the known desktop
  bootstrap error `Cannot redefine property: process`. The full backend suite
  reached `362 passed`; two unchanged local headless Surface handshake tests
  failed and remained failing in isolation. Full backend mypy passes across
  `116` source files and full Black passes across `217` files.

- The customer Agents page, raw provider/model/prompt/tool controls, and its
  frontend API/types are removed. `/agents` safely redirects to Today. Legacy
  AgentConfig routes and explicit Conversation assignment now require
  `policy:manage`; members can still create ordinary Conversations without a
  config. Existing config-backed Conversations remain compatible. Focused RBAC
  tests pass (`10 passed`), as do app lint/build, changed-file Ruff/Black, and
  focused backend mypy. Detailed handoff:
  `.agent/progress/handoffs/2026-07-19-user-facing-agent-config-retirement.md`.

- The active `aloy-v1-today-focus-stream` branch now contains an uncommitted
  Event-owned Schedule slice alongside the Today focus-stream and Connections
  navigation work. New Schedules require one active dedicated Event, interpret
  cron recurrences in an IANA timezone, freeze `report_only` or `organize`
  authority into each Run, withhold MCP and specialist Surface creation from
  unattended work, retain protected provider actions behind Proposals, and
  persist `cron_job_id` occurrence lineage. Schedule creation, edits, pauses,
  wakes, terminal outcomes, and deletion are Trail-backed; active occurrences
  appear in Today, failures notify, and the Schedule screen exposes ordinary
  recurrence controls plus bounded Run history. Deleted Schedules are retained
  as soft-deleted history anchors rather than breaking Run receipts.
- Schedule verification: focused Schedule/Today tests pass (`15 passed`), the
  migration round-trip passes, backend mypy is clean across `114` source files,
  changed-file Ruff/Black/isort pass, and Aloy app lint and production build
  pass with only the existing large-chunk warning. The complete backend suite
  reached `358 passed` with the documented headless Surface bridge race as its
  only failure; that exact unchanged test passed immediately on rerun.
- The local SQLite schema is at `h0e1f2a3b4c5`, API health is `ok`, the worker
  is running, and `/schedules` serves through the existing Vite app. The first
  pre-migration backup command used a wrong working-directory-relative path and
  did not create a backup; the migration itself succeeded and a post-migration
  copy exists under `.agent/runtime/db-backups/`. Browser visual smoke remains
  blocked by the existing browser-control bootstrap error `Cannot redefine
  property: process`.

- The R5.5b slice added durable Event context-source ingestion, status, retry, provenance, and Workbench visibility.
- R5.5c adds immutable content-addressed Event context snapshots, deterministic
  readiness, typed evidence-linked Event Brief persistence, trusted prompt
  placement, and Event-over-global conflict precedence.
- R5.5d adds an idempotent no-tool bootstrap Run, frozen bounded evidence,
  structured generation, stale-snapshot replacement, safe retry, and visible
  Workbench status/manual retry. Event promotion and completed ingestion trigger
  it automatically.
- Verification on the active branch: the backend full suite passed in three
  groups (`126 + 76 + 105 = 307`); after adding two final dispatch/policy tests,
  the affected `15` tests pass. Backend mypy passes across `103` source files;
  Aloy app ESLint and production build pass. The build retains the pre-existing
  large-chunk warning.
- Live provider smoke: Fireworks Kimi K2.6 completed the real queued
  `event_bootstrap` worker path against an isolated SQLite database, published
  Event Brief v1, and passed evidence-reference validation. The test exposed
  and fixed Fireworks/Kimi structured-output compatibility by including the
  schema in the prompt and disabling Kimi reasoning for schema-bound calls.
- Post-fix verification: kernel `617 passed, 1 skipped`; kernel mypy passes
  across `108` source files; the focused Fireworks/config suite is included in
  that total.
- Verification: kernel `615 passed, 1 skipped`; backend `301 passed`; kernel
  mypy `108` files; backend mypy `102` files; changed-file Ruff; Aloy app lint,
  production build, and `4` Surface bridge tests.
- The full previous work journal is preserved in `.agent/progress/handoffs/2026-07-17-current-state-history.md`.
- Event bootstrap merged through PR #186 at `92e3d0a`; the mistaken University
  Run's 25 Tasks were removed locally after an exact scoped selection, with a
  full database backup, JSON export, and corrective Trail records retained.
- The current branch adds `request_event_surface`, idempotent
  `surface_builder` Runs, strict profile fingerprint validation, scoped
  `/event` and `/workspace` mounts, bounded text-artifact projection, builder
  lifecycle Trail, verified publication receipts, last-good failure behavior,
  and a published-runtime-only Conversation card.
- A host-owned Event routing prompt now tells the ordinary model to choose from
  meaning and durable product value; no keyword list, regex, or second model
  classifier is involved. The builder's `answer` tool is completion-gated and
  cannot terminate the Run until that exact Run owns the live publication.
- Live University smoke on Fireworks Kimi K2.6 proved that the ordinary Event
  model can request an appropriate durable interactive view without the user
  saying "Surface" and without manufacturing Tasks. It also proved Kimi K2.6
  is not currently qualified for the specialist Builder role: it drafted a
  coherent React application in reasoning but attempted to answer eight times
  instead of invoking `surface_write_files` and the build/publish tools. The
  host rejected every false completion and the smoke Run was cancelled with no
  Surface publication and zero Event Tasks.
- Verification on the current branch: focused Surface/backend tests `15
  passed`; backend full suite reached `312 passed` with one stale assertion in
  the new test, which has been corrected; backend mypy passes across `105`
  source files; changed-file Ruff passes; Aloy app lint and production build
  pass with the existing large-chunk warning.
- The R5 Builder control plane separates developer-owned Surface Builder and
  Critic roles from ordinary Conversation AgentConfig. A Surface request now
  resolves a qualified role, freezes its credential-free assignment and
  fingerprint on the Run, and the worker validates that assignment before
  constructing the specialist model. Provider/model/skill provenance,
  assignment resolution time, total elapsed time, tokens, and cost flow into
  Run metrics and execution receipts without persisting credentials.
- Verification for the Builder control-plane branch: backend full suite `318
  passed`; focused Surface regressions `49 passed`; backend mypy passes across
  `107` source files; changed-file Ruff, full Black/isort checks, lockfile
  validation, and migration round-trip tests pass.
- Builder control plane merged into `aloy-v1` through PR #188 at `7c84bd2`.
- The active host-pipeline branch now dispatches `surface_builder` Runs away
  from the general agent loop. The dedicated executor supplies the exact skill
  and bounded Event/current-source context to a structured-output model, while
  `SurfaceHostPipeline` alone replaces the draft, builds, inspects, and
  publishes. Candidate fingerprints, per-stage timings, token usage, zero tool
  calls, repair diagnostics, and publication receipts remain durable.
- Phase 2 verification passes `52` focused Surface regressions and the complete
  backend suite (`324 passed`); backend mypy passes across `109` source files.
  Changed-file Ruff, Black/isort, and diff checks pass.
- A live Fireworks Kimi K2.6 structured-output smoke returned one schema-valid
  University candidate with six complete files, five primary jobs, `0` tool
  calls, and `5,721` tokens in `47.6s`. This directly resolves the prior
  lifecycle-tool failure mode; an Event-driven end-to-end host build and
  publication remains the final acceptance smoke.
- The local ordinary Conversation and Surface Builder assignments have now
  moved to Fireworks GLM 5.2 after Kimi conversation failures. GLM passed a
  live normal-response smoke and a provider-enforced structured-output smoke;
  both backend processes were restarted with the new configuration.
- The local schema is at migration head and the API, durable worker, and Vite
  app have been restarted from the active Phase 2 branch.
- Phase 2 now exposes pre-build Surface activity instead of waiting for the
  first build row. Five-second generation heartbeats and host-stage updates
  feed a tenant-scoped status route; Conversation shows a compact progress
  card and the Workbench shows Generate, Validate, Build, and Publish with
  elapsed time, retries, terminal failure, and overdue state. Trail fallback
  keeps coarse status visible during a rolling local backend update.
- Live GLM 5.2 Career OS acceptance exposed a qualification failure: three
  worker attempts each completed the Fireworks call but returned no
  schema-valid `SurfaceCandidate`. The Run failed safely after about 21 minutes
  with zero project, build, or publication rows. This confirms the progress UX
  and retry safety, but GLM is not yet qualified for the full Builder schema.
- Verification after the progress work: focused Surface regressions `21
  passed`; complete backend suite `326 passed`; backend mypy passes across
  `109` source files; changed-file Ruff/Black/isort pass; Aloy app ESLint and
  production build pass with the pre-existing large-chunk warning.
- Rejected structured output now retains an owner-scoped bounded diagnostic
  receipt across worker retries: exact parse error, usage, response length,
  SHA-256, truncation flag, React-source signal, and raw head/tail. The generic
  failure no longer erases the evidence or reports zero tokens merely because
  parsing failed. Post-fix focused tests pass (`9` Builder/status/pipeline plus
  `2` kernel structured-output tests); kernel and backend mypy remain clean.
- The GLM failure was traced to a provider-contract mismatch rather than weak
  React authoring: Pydantic emitted schema constraints Fireworks does not
  support, and GLM reasoning was not disabled for schema-bound calls. Pori now
  owns a reusable structured-output policy that adapts the provider request
  dialect while leaving full host validation intact. Builder assignments also
  freeze a per-generation deadline; malformed or timed-out responses fail
  closed instead of triggering three identical expensive worker attempts.
- A live smoke using the exact production `SurfaceCandidate` schema and the
  local Fireworks `accounts/fireworks/routers/glm-5p2-fast` Builder assignment
  returned a valid two-file React candidate in `6.98s` (`1,501` total tokens,
  no tools). The local assignment now allows up to `100,000` output tokens with
  a `300s` frozen generation deadline; these are ceilings, not targets. API and
  worker were restarted healthy with that assignment.
- Post-change verification: kernel structured/native tests `27 passed`; focused
  Aloy model-role/Builder/pipeline/request/status tests `16 passed`; kernel
  mypy `109` files and backend mypy `109` files pass; changed-file Ruff and
  diff checks pass. Concurrent full kernel/backend suites produced no reported
  failure but hit the `300s` orchestration timeout, so they are not claimed as
  passing in this entry.
- Candidate parsing now separates the provider-facing envelope from Aloy's
  authoritative source contract. Host-invalid files such as model-authored
  `index.html` produce bounded deterministic diagnostics and use the existing
  one-repair submission instead of terminating before the repair loop. The
  exact regression passes alongside Builder/pipeline/status tests (`10
  passed`), and backend mypy remains clean across `109` files.
- An explicit `SURFACE_BUILD_BACKEND=local_dev` path now compiles generated
  source with the repository-pinned Vite, React, and Surface SDK in an
  ephemeral directory using one host-owned command/config. It emits only the
  existing `surface.js`/optional `surface.css` ZIP contract; generated HTML,
  packages, plugins, config, dependencies, and commands remain forbidden.
  Production still defaults to isolated/fail-closed. A live local smoke built
  and runtime-validated a React/CSS bundle in `1.64s`; focused Surface tests
  pass (`19 passed`) and backend mypy remains clean across `109` files.
- The first full GLM Career OS acceptance produced valid React source twice and
  the fixed local compiler completed both bundles in about `1.7s`; publication
  failed only because the canonical local-storage destination was `279`
  characters and Windows rejected the atomic rename under legacy `MAX_PATH`.
  `LocalDiskObjectStore` now uses extended-length syscall paths without
  changing logical object keys. The exact retained revision compiles in
  `1.59s`, and its `182,540`-byte bundle round-trips through the real
  `288`-character storage path.
- Surface pipeline outcomes now distinguish candidate-repairable failures from
  `host_failed` infrastructure failures. Storage, sandbox availability, and
  runner-contract faults stop immediately and retain diagnostics instead of
  consuming the model's bounded repair submission. Focused storage/build/
  pipeline/Builder/status verification passes (`29 passed`); backend mypy is
  clean across `109` files and changed-file Ruff passes.
- The recovered bundle exposed a second host-toolchain defect in the iframe:
  React's CommonJS entrypoint retained `process.env.NODE_ENV`, but browser
  Surfaces have no Node.js `process` global. The fixed local Vite contract now
  defines production mode at compile time, and deterministic source validation
  rejects model-authored Node globals. The retained Career OS revision was
  rebuilt in `604ms` to a `69,459`-byte browser-safe bundle and atomically
  republished; the stored script contains no unresolved `process.env.NODE_ENV`.
  The local build/source regression suite passes (`10 passed`).
- The next live load exposed a generated-SDK shadowing failure: GLM created
  local `sdk.ts`/`sdk-runtime.ts` files and a fake `window.postMessage`
  protocol instead of importing `@aloy/surface`, so the host bridge correctly
  refused to acknowledge it and showed reconnect attempts. The fixed compiler
  now always bootstraps the trusted SDK, source validation rejects direct fake
  bridge code, and interactive manifests require a real `@aloy/surface`
  import. The retained Career OS source was corrected through a new immutable
  revision and republished as build `sbuild_2ec0d0b73a9b4ac7bb5a6604ba09f756`;
  bundle inspection confirms the secure bridge is present, the fake protocol
  is absent, and browser-unsafe Node environment access is absent. The focused
  build suite passes (`11 passed`).
- Live interaction then exposed a render-time data contract bug: the Surface
  treated a durable `career.stage_changed` patch as a complete Application and
  dereferenced missing presentation metadata. Static checks and compilation
  cannot catch that class of defect. Local preview now mounts the exact built
  runtime in headless Chrome, injects the exact current Event context through
  the production MessageChannel bridge, and rejects uncaught exceptions,
  missing bridge acknowledgement, or an empty React root before publication.
  Trusted preview context may inspect an unpublished succeeded build; public
  runtime context remains restricted to the atomic published pointer. The
  Career OS source now merges partial durable records into complete seeded
  entities, passed the browser gate with its real current Event data, and is
  live as revision `srev_72a4355497414ebda8ed5214b58f53d7` / build
  `sbuild_006bfc22d8284296a75c8617876f2538`. Focused build/pipeline/SDK tests
  pass (`23 passed`); backend mypy is clean across `110` files.
  Browser exceptions and empty/failed mounts now return bounded candidate
  diagnostics for the Builder's one repair submission, while unavailable or
  broken inspection infrastructure returns `host_failed` and never wastes a
  model repair call.
  Publication also fails closed when an isolated builder provides no trusted
  browser-inspection receipt, so the future E2B toolchain must run the same
  gate before it can publish. Every compiled runtime is additionally wrapped
  in a host-owned React error boundary: unforeseen future-state failures show
  a safe repair view instead of a blank crash and expose a deterministic fault
  marker to inspection. The hardened Career OS build
  `sbuild_e10594eb9a454b30bf18994f73b58ba7` passed the real-context browser gate
  and is the current live publication.
- The next Career OS acceptance exposed a behavior-quality gap rather than a
  bridge outage: the generated Add form could emit its SDK request, but kept
  the form open, hid the resulting row below the visible pane, claimed local
  success before persistence, and swallowed every SDK rejection. Surface
  manifests now declare accessible `interaction_checks` for each durable
  intent. Local preview executes those user paths in the real browser runtime,
  verifies the expected SDK method/name and schema-valid payload, and rejects
  missing, disabled, throwing, or unwired controls. Static validation also
  rejects uncovered intents and swallowed Promise failures. Builder guidance
  requires awaited writes, visible pending/error states, and canonical
  reconciliation. Career OS was repaired to await Save, preserve input on
  failure, close only after host confirmation, and replace complete records on
  status changes; both Add/Save and status-change checks passed before revision
  `srev_365098d7779f491dbbc7cc70c99cf65a` / build
  `sbuild_b4c1f82ebaaf469ba96f34d4788b5e2a` became live. Focused Builder,
  build, pipeline, and SDK verification passes (`33 passed`); backend mypy is
  clean across `110` files and changed-file Ruff/Black checks pass.
- The founder's E2B Hobby allowance (`20` concurrent, one-hour sessions, up to
  `8` vCPU/`8GB`) is sufficient for development Surface builds. It was not the
  cause of the local failure. Remote builds remain intentionally disabled
  until the optional E2B SDK is installed and a pinned E2B template provides
  `/opt/aloy-surface-toolchain/bin/build-surface`; the default E2B image does
  not currently contain Aloy's host-owned compiler contract.
- PR #189 merged the verified host-owned build pipeline into `aloy-v1`. The
  command-runtime branch now introduces command contract v1, strict
  `create`/`replace`/`merge`/`delete` entity semantics, a legacy `dispatch`
  compatibility seam, fixed wake-policy classification, a bounded canonical
  Surface-state Event-context projection, and an Event-scoped detailed-state
  read tool. Reasoning commands create a host-rendered trigger containing the
  exact Event, data revision, and context snapshot fingerprint; arbitrary
  Surface payload is retained as data but is not interpolated into the Run
  instruction.
- Command-runtime slice verification: complete Aloy backend suite `346 passed`;
  backend mypy clean across `113` source files; backend Black/isort clean; all
  three import-boundary contracts kept; focused command/state regressions `9
  passed`; Surface bridge tests `4 passed`; Aloy app lint and production build
  pass with only the existing large-chunk warning.
- The command runtime now supplies `useSurfaceCommand()` as the standard React
  lifecycle for generated controls. It suppresses duplicate submissions,
  preserves an immutable JSON snapshot and idempotency key for exact-action
  retry, exposes typed
  `pending`/`committed`/`accepted`/`conflict`/`failed` states and structured
  errors, and provides accessible feedback metadata. The host now sends
  refreshed canonical context before its success acknowledgement. The browser
  gate simulates that ordered commit/context flow and rejects a command whose
  acknowledged outcome is not visibly rendered. Legacy `dispatch()` remains
  compatible, and the existing immutable Career OS bundle has not been patched.
  Verification: `23` focused Surface build/SDK backend tests and `5` host-bridge
  tests pass; backend mypy is clean across `113` files; SDK TypeScript, app
  lint, and the app production build pass with only the existing chunk warning.
- Surface commands now have an append-only attempt ledger beside the accepted
  interaction ledger. Every accepted command records its initial host outcome;
  stale revisions, idempotency conflicts, validation failures, and policy
  denials survive the request rollback with safe error code, retryability,
  requested/observed data revisions, and a durable attempt ID. The iframe
  context exposes bounded attempt receipts without rejected payloads, while
  `SurfaceRequestError` and `useCommandAttempts()` let generated React match
  immediate errors to their durable record. The host refreshes canonical
  context before returning either an accepted command or a durable rejection.
  Host tests cover structured conflict, non-retryable permission denial, and a
  commit whose context refresh fails and recovers only after reconnect. Focused
  backend command/migration tests pass; the combined Surface SDK/build set
  passed `24` tests with one known headless-browser handshake race, whose exact
  unchanged assertion passed immediately on rerun. Backend mypy is clean across
  `113` files; `7` bridge tests, SDK TypeScript, app lint, and production build
  pass with only the existing chunk warning.

- Live Career OS migration exposed three host-boundary reliability gaps. Safe
  `/src` and `/surface.json` candidate shorthand and overlong descriptive
  summaries are now canonicalized before authoritative validation instead of
  consuming the model's single semantic repair. The browser interaction gate
  now applies declared `create`/`replace`/`merge`/`delete` semantics to its
  canonical smoke context, so a committed Add can expose the exact row used by
  subsequent Status/Edit/Delete checks. A real two-command headless-browser
  regression covers this sequential reconciliation.
- The retained Career OS source then proved that Vite transpilation alone is
  insufficient: GLM destructured `useSurfaceData()` as an object, passed an
  argument to `useCommandAttempts()`, read a nonexistent runtime `connected`
  property, and treated DOM-only `feedbackProps` as controller state. The
  host-owned Surface toolchain v2 now runs strict TypeScript contract checking
  before Vite, returns bounded file/line `typescript_contract_error`
  diagnostics, records separate typecheck/compile timings, and requires the
  isolated toolchain command to run the same check. The Builder skill now
  includes exact SDK signatures and a persistently mounted feedback pattern.
  The bad retained revision is rejected in `1.86s` with `13` exact diagnostics;
  focused pipeline/build/browser verification passes (`25 passed`).
- The final normal Builder smoke produced a much closer candidate and the new
  gate returned five precise assignability errors, but Fireworks rejected the
  reserved repair with HTTP `412`: the account is suspended or spend-limited.
  The last live Career OS pointer remains revision 7 / build
  `sbuild_55b34dc8a23f4b5191a608c5b20a7807`, and canonical data revision remains
  `6`; no generated application was patched and no Event data was reset.

## Blockers

- Fireworks currently returns HTTP `412` for the local GLM Builder assignment
  because the account is suspended or spend-limited. Restore provider access
  (or configure another qualified Builder) before rerunning the retained
  Career OS type-diagnostic repair and live publication acceptance.
- GLM 5.2 Fast remains temporarily allowed only in the local gitignored
  model-role file. The exact production-schema smoke now passes, but production
  qualification still requires the governed Builder evaluation and a complete
  Event-driven build/preview/publication acceptance.
- E2B account capacity is available, but the repository still needs the pinned
  Surface toolchain template and template-aware provider configuration before
  `SURFACE_BUILD_BACKEND=isolated` can be accepted end to end.

## Next Session Should Start With

Review and commit the Event-memory controls branch, then visually smoke the
Event settings and Memory dialog when the desktop browser bootstrap is available. After that, add
evidence-gated automatic consolidation so only confirmed statements,
corrections, decisions, and durable outcomes become accepted Event memory.
Then restore qualified Builder provider access and rebuild the retained Career
OS draft through the normal Builder and verify the typecheck, sequential browser
checks, Save feedback, canonical data preservation, and atomic publication.
After that, enable the separately governed `source_change` and `automation` routes.
Do not patch individual generated applications or continue showcase/widget work
until the command runtime owns their persistence and wake behavior.
