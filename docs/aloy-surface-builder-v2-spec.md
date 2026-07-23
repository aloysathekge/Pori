# Surface Builder v2 — the rethought generation pipeline

_Version 1.0, 2026-07-23. Supersedes the R13 workspace execution design (the
Builder loop, submission model, and legacy structured-output executor) while
keeping the trust layer of [`aloy-surface-spec.md`](./aloy-surface-spec.md)
and the warm-start invariant of
[`aloy-baseline-surface-spec.md`](./aloy-baseline-surface-spec.md). Grounded
in the 2026-07-23 live acceptance evidence and
[`research-ai-app-builder-templates.md`](./research-ai-app-builder-templates.md)._

## 0. What today proved

Seven live runs failed seven ways. None reached a user with anything broken —
the trust layer (immutable revisions, fingerprints, gates, atomic publication,
durable receipts) is proven and is **not** rethought here. Everything else is.
The generation layer failed because it assumes abundance: fast tokens, cheap
cached context, a frontier model clearing a maximal gate in three tries. The
v2 design assumes scarcity and makes the host do everything a model doesn't
have to.

## 1. Principles

- **P1 — Never from nothing.** Every Event has a published Surface before any
  Builder runs (baseline spec). Generation is always revision.
- **P2 — One representation.** The Git workspace *is* the candidate. No edit
  envelopes, no materialize/rebind round-trip, no shadow copies of source.
  The host reads workspace files directly at every boundary. (This class of
  bug — the repair-base desync — becomes unrepresentable.)
- **P3 — Deterministic before model.** Every check that can run without a
  model runs first; every fix the host can apply mechanically is applied
  silently (autofix tier). Model tokens are the last resort, never the first.
- **P4 — Cheapest gate first.** Validation → typecheck → compile → runtime
  smoke → full quality matrix → job proofs, strictly ordered; feedback
  returns from the cheapest failing tier. The expensive browser matrix runs
  only on candidates that already pass everything cheaper.
- **P5 — One budget, in money.** A session spends dollars from a cache-aware
  cost ledger. Steps, turns, tokens, and submissions stop being independent
  ceilings; they become derived telemetry.
- **P6 — Provider physics are declared.** Streaming, caching, reasoning
  control, per-call latency, and prices live in the provider profile; the
  loop adapts to them instead of assuming Claude.
- **P7 — Quality gates promotions, availability has a floor.** The strict
  gate decides what replaces the live Surface. It never decides whether the
  user has a Surface — P1 guarantees they always do, and a failed session
  leaves the previous publication untouched.

## 2. The workflow, end to end

### 2.1 Event creation (model-free)

Baseline release → model-free materialization → published v1. Unchanged from
the baseline spec. This is the availability floor and the reason every later
phase can afford to be strict.

### 2.2 The change request

The conversation agent files a `SurfaceChangeRequest`:

```
goal, experience, interaction_notes            (as today)
jobs_added:   [description...]                 (new jobs this change must prove)
jobs_removed: [job_id...]                      (optional, explicit retirement)
```

**The host computes the frozen contract**: current published jobs, minus
removals, plus additions — frozen with ids as today. This fixes the update
semantics debt structurally: a revision request no longer silently replaces
the Surface's whole job contract with the change description, and the model
never has to guess which jobs survive.

### 2.3 The Builder session (replaces Run submissions)

One durable **BuilderSession** owns: one Git workspace seeded from the
published revision, one cost budget, one frozen contract, one append-only
turn history with a byte-stable prompt prefix (system + skill + contract +
projection — nothing volatile; cache discipline is a correctness rule, not an
optimization).

The loop gives the model the same bounded vocabulary (`list/read/search/
replace_text/write/delete/check/finish`) with three structural changes:

1. **`check` is tiered and host-subsidized.** Tier A: deterministic
   validators *plus autofixers* — import normalization, manifest field
   completion, token-violation substitution, formatting — applied silently,
   reported as "the host fixed these". Tier B: typecheck. Tier C: compile.
   Tier D: runtime smoke (render + console errors in one headless page,
   ~seconds). Each tier runs only if the previous passed; the model sees one
   compact result naming the cheapest failing tier. Tiers A–C are free and
   unlimited; D is throttled.
2. **`finish` runs the full gate in place.** Quality matrix + behavioral job
   proofs execute against a build of the *workspace files directly*. On pass:
   the host persists the immutable revision from those exact files and
   publishes atomically (revisions are now created only for publishable or
   final-audit candidates — no junk revision per rejected attempt). On fail:
   the compact diagnostic bundle enters the same session as one more message,
   and the loop simply continues — there is no "submission" to consume.
3. **Provider calls are physics-aware.** Per-call timeout and single retry
   (already built), streaming when the profile supports it (true stall
   detection), reasoning controls always applied (already built), and every
   call metered into the cost ledger at cached/uncached/output rates.

**Termination** (whichever first): published; budget exhausted; converged
failure (two consecutive `finish` attempts with identical source fingerprint
or identical diagnostic fingerprint — the model is looping); or wall-clock
ceiling. On any failure the previous publication stands and the session
transcript + workspace head are retained as evidence.

### 2.4 Model roles

- **Builder** — the only required role. Qualified *together with* its
  provider profile (a model is not "qualified" in the abstract; kimi-on-
  Fireworks and kimi-elsewhere are different qualifications).
- **Fixer** *(optional, gated, later)* — a small cheap model that receives
  exactly one diagnostic and one file and returns one edit; Tier B/C
  failures route here before waking the Builder (Replit's 7B repair
  economics). Disabled until usage data justifies it.
- **No critic in the loop.** Deterministic gates carry quality (unchanged
  decision). The deferred visual critic stays deferred.

### 2.5 Publication and after

Persist → build → inspect → publish stays the trusted path it is today, with
one ordering change: persistence happens at gate-pass (2.3.2), so the
revision store contains only meaningful revisions. Reinspection, evolution
proposals, and rollback are untouched.

## 3. What is deleted

- The legacy structured-output executor and the workspace-invoker shim
  impersonating it (`_execute_claimed_surface_builder_legacy` as the outer
  frame, `SurfaceCandidateEnvelope` / `SurfaceCandidateEditEnvelope`,
  `materialize_surface_candidate_edit`, the repair-files prompt path).
  `bind_surface_manifest_primary_jobs` survives as a Tier-A autofix.
- `candidate_mode` (complete/edit) — P1 makes every session an edit session.
- The submission counter (`MAX_CANDIDATE_SUBMISSIONS`) and the five
  independent ceilings (`max_steps` / `max_tool_calls` / `max_tokens` /
  timeout / submissions) as governing limits — replaced by the cost budget +
  convergence rules; the old numbers remain as coarse safety backstops only.
- From-scratch generation as a reachable mode.

## 4. What is explicitly kept

Authoring/revision immutability, the build runner, the browser inspection
engine and quality policy, publication atomicity and rollback, HITL/authority
boundaries, run/lease durability (sessions ride the same worker), the SDK and
iframe security contract, R12 templates/materialization, and the baseline
template itself.

## 5. Migration (staged, test-equity preserving)

- **M1** — Baseline delivery (S2/S3 of the baseline spec): the P1 invariant
  must hold before v2 lands. *(S1 template already built and gate-proven.)*
- **M2** — `builder_session.py` built clean beside the old executor: session
  state, tiered check with autofix tier, in-place finish gate, cost ledger
  (provider prices in the profile). Old path stays default; sessions run
  behind a flag against the same boundaries.
- **M3** — Worker cutover to sessions; delete the legacy executor, envelopes,
  and shim; port the regression suite (repair-base, contract-gate, hang,
  duplicate tests all translate — most become simpler or moot).
- **M4** — Kernel: provider physics profile (streaming/caching/reasoning/
  prices) and streaming tool-calls where supported.

Each stage lands independently green; M2 is the bulk of the new code and
touches no live behavior until M3.

## 6. Open decisions

- **Draft-preview tier**: show the user an unpublished candidate (same
  sandbox, explicit "being built" banner) while the session runs? Pure
  product call; the security boundary already permits it, the quality
  philosophy currently forbids it.
- Fixer-role trigger thresholds and model choice.
- Whether failed-final workspace heads should be browsable in the app
  (operator-only today).
- Cost budget defaults per subscription tier.
