# Aloy Baseline Surface — warmed Event workspace specification

_Version 1.0, 2026-07-23. Extends
[`aloy-surface-spec.md`](./aloy-surface-spec.md) (v2.1). It changes how a
Surface begins, not what a Surface is._

## 1. Decision

**The Surface Builder never sees an empty workspace.** Every Event receives a
published version-1 Surface at creation — a curated, host-reviewed React
baseline materialized through the existing model-free template pipeline with
zero model tokens. From then on, every Builder run is an *edit* of a real,
working project (`candidate_mode="edit"`); creation-from-nothing disappears as
a generation mode.

This is the same structural insight behind Bolt and Lovable: the model never
scaffolds a project, it only modifies a template that already works. Aloy's
version is stronger because the host owns the toolchain (no `package.json`,
`index.html`, or config in model space) and because the R12 template
catalog + model-free materialization pipeline already exists to deliver
reviewed source without a model call.

## 2. Motivation (evidence)

Live acceptance on 2026-07-23 showed the from-scratch path is the fragile
half of the system:

- A from-scratch build needs 10–20 workspace turns; each turn re-sends the
  full ~16–20k-token projection. On a provider without prompt caching this
  exhausted the 200k aggregate budget at turn ~9 with no finished candidate.
- Every additional model call is another exposure to provider stalls; two of
  three live attempts died to dead-air provider hangs mid-loop.
- Most Builder quality-gate rejections are boilerplate failures (landmark,
  focus, contrast, resource-state wiring) that a curated baseline embodies
  once, permanently.

The update path, by contrast, is the hardened one: bounded edits, exact
rejected-source repair, the shared primary-job contract gate, and the
immutable revision/publication pipeline. Making creation *be* an update
routes all generation through the strong path.

## 3. The baseline template

A bundled product asset (versioned like `product_skills/`), small enough to
read in one workspace turn: ~7 files, ≈15–20 KB total, well under
`MAX_SURFACE_FILES` / source-size limits.

```
/surface.json                  manifest: capabilities, intents, jobs, resource_views
/src/App.tsx                   SurfaceRoot shell, header, view switching
/src/views/Overview.tsx        brief, phase, quick stats
/src/views/Tasks.tsx           task list rendering every resource state
/src/views/Files.tsx           workspace files + openResource intent
/src/components/primitives.tsx Section, EmptyState, Stat, Button
/src/theme.css                 spacing/color tokens (≥4.5:1), visible focus ring
```

### 3.1 Contracts the template must embody

The baseline is the living style guide the model extends. It must ship —
already passing the full `@5` browser gate — with:

- exactly one `main` landmark (SurfaceRoot's; no nested `main`/`role="main"`);
- visible keyboard focus on every control (`theme.css` focus ring; no bare
  `outline: none`);
- deterministic solid backdrops meeting 4.5:1 normal / 3:1 large text;
- responsive recomposition at 1440/768/640/390/360 px with no page-level
  horizontal overflow, driven by the primitives, not per-view media-query
  heroics;
- `useSurfaceResourceState` + spread `feedbackProps` on each data-driven
  region, with honest loading / empty / stale / error / permission-denied /
  pending / indeterminate views rendered through `EmptyState`;
- declared `resource_views` navigation for any region behind a tab;
- only allowed imports (`@aloy/surface`, `react`, `react-dom/client`,
  `react/jsx-runtime`, local files); none of the forbidden patterns
  (`fetch`, storage, `dangerouslySetInnerHTML`, host-frame access, …).

### 3.2 Baseline manifest

`/surface.json` declares read-oriented capabilities (`tasks`, `files`,
`ask_aloy`), an `aloy.ask` reasoning intent, an `event.resource.open`
interaction, `resource_views` for its tabs, and the **baseline release's own
reviewed primary jobs** (e.g. "See this Event's current work at a glance",
"Open a workspace file") with browser proofs written against declared
fixtures — a fresh Event is mostly empty states, and proving them is itself
a demonstration of the pattern for the model.

## 4. Delivery: Event creation → live v1

Reuses the R12 revision/materialization boundary end-to-end; no new
authority.

1. The baseline is a **bundled, fingerprinted product asset**
   (`aloy_backend/product_surfaces/baseline/`, identity
   `aloy-baseline-surface@1`), reviewed through ordinary source review and
   proven against the full gate in CI. _Implementation note (v1.1): delivery
   consumes the bundled asset directly rather than requiring a catalog
   release, because the protected catalog flow exists for tenant/global
   authoring and would leave fresh deployments with no baseline until an
   operator stages one. Publishing the same asset as a catalog release
   remains an optional hardening step for hosted deployments._
2. **On Event creation** the host persists the baseline draft revision
   (checksum-bound, `request_fingerprint` = the bundled asset fingerprint)
   and queues the existing model-free Surface materialization Run. It runs in
   the background; Event creation stays instant.
3. The materialization Run goes through the ordinary compile → browser
   inspection → atomic publication pipeline (same idempotency and crash-replay
   semantics it already has). Seconds of compute, zero model tokens.
4. When the user first opens the Event, v1 is live. Failure to materialize
   never blocks Event creation or conversation; the Surface pane shows its
   ordinary "not yet available" state and the Run retries under existing
   worker rules.

Life is exempt (no auto-Surface for the Life Event) unless a later decision
says otherwise. Template-installed Events (Career OS, University releases)
keep their richer template source — the baseline applies only to custom
Events that today start empty.

## 5. How the Event description is used

**The description is data, not code.** The baseline renders the Event's
actual title, brief/summary, phase, tasks, and files through the SDK context
at runtime, so one reviewed source is personalized for every Event with no
model involvement. A richer description improves what v1 shows purely through
the existing Brief/context projection.

Deliberately deferred (optimizations, not v1):

- description-driven selection among a small set of baseline *variants*
  (workspace / planner / tracker arrangements) via a cheap classification;
- an optional background Builder run at creation that personalizes layout
  from the description — this is an ordinary paid revision and must never be
  implicit; V1 keeps the zero-token guarantee at creation.

## 6. Builder interaction

- With v1 always published, `resolve_surface_authoring_runtime` always finds
  non-empty draft files → `candidate_mode="edit"` → the R13 workspace loop is
  the only generation path. The `complete` mode remains code-supported but
  unreachable in normal operation.
- The Builder skill gains one instruction: *extend the existing views and
  primitives; prefer `replace_text` against them; create new files only for
  genuinely new views or components.* Stable primitives give `replace_text`
  reliable anchors — the biggest practical failure of edit-based generation
  against unfamiliar code.
- A typical revision is 3–6 turns: read the relevant view, targeted
  `replace_text`, possibly one new file, a `replace_text` on `surface.json`
  adopting the request's frozen primary-job contract, check, finish. The
  finish-time contract gate (shared `surface_primary_job_contract_diagnostics`)
  already refuses a stale manifest in-workspace before any paid submission.
- The baseline's own jobs are replaced by the first user request's frozen
  contract; that interplay is the one hardened on 2026-07-23 (id-verified
  rebinding, no stale-proof carryover).

## 7. Versioning and rollout

- New Events pin the current baseline release; a new baseline version never
  silently rewrites existing Events (same rule as all template releases).
- Baseline releases track SDK compatibility through the existing
  compatibility records; a baseline bump accompanies SDK major changes.
- Existing Events without a Surface may be backfilled lazily (materialize on
  first Surface-pane open) — optional slice, same Run, same guarantees.
- Events whose Surface was already model-authored are untouched.

## 8. Invariants (unchanged authority)

- Only the host pipeline publishes; the baseline release cannot publish
  itself (normal template rule).
- Immutable revisions, fingerprints, publication pointers, quality receipts:
  all unchanged.
- The materialization Run keeps its existing restrictions: no source-write,
  network, model, or conversation authority.
- The baseline never fabricates facts: sparse context renders as visible
  empty/setup states, never invented data (existing posture rules).

## 9. Failure modes

| Failure | Behavior |
| --- | --- |
| Baseline materialization Run fails | Event works normally without a Surface; Run retries per worker rules; terminal failure surfaces in Trail/operator evidence, never blocks the Event. |
| Baseline source drifts from SDK | Caught at release review (build + gate must pass to publish a baseline release); deployed Events keep their pinned working revision. |
| Model rewrites a primitive badly | Ordinary candidate → build/gate rejection → bounded repair on exact source; no new risk class. |
| User deletes baseline files via Builder | Allowed — it's their Surface; the gate still requires a complete, passing app. |

## 10. Implementation slices

1. **S1 — the template.** Author the baseline source + manifest as a bundled
   asset; prove it against the full browser gate in a fixture test (extend
   the existing Surface-slice suite). No behavior change shipped.
2. **S2 — the release.** Stage/publish `aloy-baseline-surface@1` through the
   protected authoring boundary (operator flow, exact checksum), plus a dev
   bootstrap path for local catalogs.
3. **S3 — the trigger.** On custom Event creation, queue the model-free
   materialization Run bound to the pinned baseline release. Regression:
   Event creation latency unchanged; failure never blocks; idempotent under
   replay.
4. **S4 — Builder routing + skill.** Assert the always-edit invariant,
   add the extend-the-primitives skill instruction, and delete/deprecate any
   remaining from-scratch prompt branches. Regression: a revision request on
   a fresh Event completes within the standard budget on an uncached
   provider.

Each slice lands independently green; S1/S2 ship no user-visible change.

## 11. Open questions

- Should the Life Event eventually get its own (different) baseline?
- Baseline variants by description classification: worth it, and where does
  the classification run (request-time heuristic vs. tiny model call)?
- Lazy backfill for existing Events: on first pane open, or operator batch?
- Should the v1 Surface visibly invite evolution ("ask Aloy to reshape this
  workspace") as product copy in the baseline's empty states?
