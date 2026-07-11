# START HERE

One map for the whole monorepo. Two audiences, two paths.

## What this repo is

**Pori** (repo root, `pori/`) — an eval-native, receipt-first, memory-native
agent kernel. **Aloy** (`products/aloy/`) — the product built on it: an
agentic operating system for life and work ([the vision](./docs/aloy-vision.md)).
Rule zero: products depend on the kernel, never the reverse (CI-enforced).

## If you are a HUMAN, read in this order

1. [`README.md`](./README.md) — the project pitch and setup.
2. [`docs/aloy-vision.md`](./docs/aloy-vision.md) — what we're building and
   why (the canon).
3. [`MONOREPO.md`](./MONOREPO.md) — repo structure and the dependency rules.
4. [`products/aloy/backend/deploy/RUNBOOK.md`](./products/aloy/backend/deploy/RUNBOOK.md)
   — how it deploys.

## If you are an AGENT, read in this order

1. [`CLAUDE.md`](./CLAUDE.md) — the kernel codebase map + working rules
   (commands, conventions, the footprint ladder).
2. [`docs/aloy-vision.md`](./docs/aloy-vision.md) — §3 primitives (use the
   nouns exactly), §4 invariants (never violate), §7 substrate map.
3. [`docs/adr/`](./docs/adr/README.md) — the decision records. **If
   something looks odd, check here before "improving" it.**
4. [`.agent/progress/current.md`](../.agent/progress/current.md) — live
   session state: what's in flight, what's next. (Work-cycle rule: read
   before meaningful work, update after.)
5. [`docs/engineering-excellence-spec.md`](./docs/engineering-excellence-spec.md)
   — the quality bar every change meets.

## To change X, read Y

| Changing… | Read first |
|---|---|
| The agent loop / kernel behavior | `CLAUDE.md` §Architecture, ADR 0002/0003/0004, `pori/agent/` |
| Kernel public API | ADR 0008 (front-door rule), `pori/__init__.py` seams block |
| Memory | ADR 0006, `docs/memory-architecture.md`, `pori/memory/` |
| Tools (adding capability) | ADR 0005 (footprint ladder) — climb the lowest rung |
| Backend routes / persistence | ADR 0007 (single finalizer), `aloy_backend/run_outcome.py` |
| Run assembly (what agents can reach) | `aloy_backend/run_surface.py` |
| Streaming / live runs / stop / resume | ADR 0009, `aloy_backend/streaming.py`, `live_runs.py`, `resumable_runs.py` |
| Storage / files / sandbox | `docs/aloy-object-storage-sandbox-spec.md`, `aloy_backend/storage.py`, `provisioning.py` |
| The app (React) | `products/aloy/app/src/` — hooks in `src/hooks/`, api layer in `src/api/` |
| Deployment | `products/aloy/backend/deploy/RUNBOOK.md` + `OAUTH-VERIFICATION.md` |
| Anything Aloy-product-shaped | `docs/aloy-vision.md` FIRST — specs cite it |

## Non-negotiables (the short list)

- Never violate a vision invariant (`docs/aloy-vision.md` §4) or an ADR
  without superseding it explicitly.
- All gates green before merge: format, lint, types, tests, build — every
  surface (see `.github/workflows/ci.yml`).
- Behavior fixes and structural refactors land in separate commits.
- The dev backend has no hot reload: restart it manually after backend or
  kernel changes, then verify (`netstat` one listener + `/openapi.json`
  route check).
