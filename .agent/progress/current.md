# Current State - 2026-07-18

## Active Task

Implement `aloy-v1-r5-builder-control-plane`: developer-owned specialist model
allocation, qualification, provenance, and diagnostics for Surface Builder and
future Surface Critic Runs.

## Decisions Made

- Keep detailed session history in handoff files; keep this file as the next-session briefing.
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

## Important Discoveries

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

## Blockers

- No specialist model is yet configured and qualified for the Surface Builder
  role. The Phase 1 control plane deliberately fails closed until the developer
  supplies that assignment; qualification against the complete live pipeline
  follows in the University proof phase.

## Next Session Should Start With

Finish the Builder control-plane branch and merge it into `aloy-v1`. Then start
`aloy-v1-r5-host-build-pipeline`, replacing the fragile multi-tool Builder
workflow with one candidate submission and a deterministic host-owned pipeline.
