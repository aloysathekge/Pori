# Pori / Aloy Monorepo — Structure & Rules

**Status:** Scaffolding v0.1 (M0) · 2026-07-02
**See also:** [`docs/Pori.md`](./docs/Pori.md) (PRD), [`docs/Pori_Implementation_Plan.md`](./docs/Pori_Implementation_Plan.md), [`HARVEST.md`](./HARVEST.md)

> This document is the blueprint for the three-band monorepo. The skeleton is created **additively** — the working `pori/` package at the repo root is **untouched**. Code migration into `packages/` is **Phase 4**, not now.

---

## The three bands

```
repo root (this git repo: Pori/Pori — uv workspace)
├─ packages/
│  ├─ pori/        KERNEL — product-agnostic, publishable
│  │               runtime · protocol · receipts · validation · llm · tools ·
│  │               context · sandbox · memory engine · interfaces
│  │               (the `pori` package lives here at packages/pori/pori/;
│  │                pori/api moves to products/aloy/backend next)
│  └─ ext/         EXTENSION BAND (pori-*) — reusable across products, opt-in, publishable
│                  memory-scope/tenancy · skills · learning · gateway · providers · cli-kit
├─ products/
│  └─ aloy/        FLAGSHIP PRODUCT #1 (personal + org OS)
│                  backend/cli/gateway/web/desktop + Aloy-specific org policy, tenancy, branding
├─ tools/ci/       dependency-boundary enforcement + supply-chain gates
└─ (donors)        external OSS lives at ../references (NOT in this repo, never a runtime dep)
```

---

## The one rule: one-way dependencies

```
products → ext → pori        (never upward)
```

- `pori` (kernel) imports **nothing** from `ext` or `products`.
- `ext/pori-*` may import `pori` only.
- `products/*` may import `ext` and `pori`.
- Surfaces (web/desktop) talk to a product backend over **REST + SSE**, never by Python import.

Enforced by the boundary check in [`tools/ci/`](./tools/ci/). This is the single safeguard against the kernel rotting into product code.

**Dependency inversion:** the kernel defines interfaces (`Validator`, `MemoryProvider`, `MemoryStore`, `SkillProvider`, `ToolBackend`, `ContextEngine`); `ext`/`products` provide the implementations. The kernel loop imports only interfaces.

---

## Naming & publishability

- `packages/pori/` — the kernel; own `pyproject.toml`, no `ext`/`product` imports, **publishable standalone** (e.g. PyPI).
- `packages/ext/pori-*` — reusable extensions; opt-in; publishable.
- `products/<name>/` — branded/private compositions. Aloy is `products/aloy/`.
- **"The open framework" = `pori` + `ext`.** Products sit on top.

---

## Anti-speculation rule

Create a `pori-*` extension **only** when the capability is obviously generic. Otherwise build it in `products/aloy/` and **promote it into `ext/` on second use** (rule of three), not on spec.

---

## Migration staging

- **M0:** created the band directories, the dependency-boundary rule, and the harvest ledger (additive; no code moved).
- **Kernel migration — DONE:** the `pori` package moved to `packages/pori/pori/`; the root `pyproject.toml` resolves it via `packages.find where = ["packages/pori"]` (and `known_first_party = ["pori"]` for isort). Still a **single** root `pyproject.toml` — the multi-package uv-workspace split (per-package pyprojects, publishing `pori` standalone) is deferred until product/ext packages exist. Tests / black / isort / mypy all green from the new location (338 passed, 1 skipped).
- **Next (products / api):** extract `pori/api` → `products/aloy/backend/`; declare its `fastapi`/`starlette` deps and fix the `RequestResponseFunction` import; then fold in the copied-in `pori_cloud` (see copy-in plan below).
- **Later:** split into a real uv workspace with per-package `pyproject.toml`s and activate the CI boundary check.

---

## What this scaffolding pass created

- `packages/` (`README.md`), `packages/pori/README.md` (placeholder), `packages/ext/README.md`
- `products/` (`README.md`), `products/aloy/README.md`
- `tools/ci/` — `README.md`, `importlinter.ini` (the boundary contract, inert until Phase 4), `check-boundaries.sh`
- `HARVEST.md` — the provenance ledger
- this `MONOREPO.md`

---

## Absorbing the sibling projects (copy-in plan)

Four projects currently at the `Pori/` level are being folded into this monorepo by **copying their contents** into the homes below. Their standalone git histories are **not** preserved — the copy-in approach was chosen for simplicity. The source repos are left untouched until the copy happens.

| Source (external, at `../`) | → Home in this monorepo | What it is |
|---|---|---|
| `pori_cloud` (git repo) | `products/aloy/backend/` | the API — FastAPI, alembic, Docker. **"our api"** → Aloy's backend |
| `pori_cloud_client` (git repo) | `products/aloy/web/` | frontend (Vite/React); talks to backend over REST + SSE |
| `pori_website` (git repo) | `website/` | public marketing site (Vite/React/shadcn) |
| `pori_docs` (plain folder) | `docs/` | design/architecture markdown |

The homes are scaffolded with README placeholders and ready for the copy-in.

## ⚠ Open questions (topology & M0 — do not assume)

1. **Repo/workspace root.** This git repo is `Pori/Pori`; sibling projects `pori_cloud/`, `pori_cloud_client/`, `pori_website/`, `pori_docs/` live *outside* it at the `Pori/` level. Whether the monorepo root should be `Pori/` (absorbing them) or stay `Pori/Pori` — **undecided.**
2. **`pori_cloud` reconciliation.** There is both a `pori/api` inside this repo *and* a standalone `Pori/pori_cloud`. The decision "evolve `pori/api` into Aloy's backend" needs reconciling with the existing `pori_cloud` project — **undecided.**
3. **Workspace conversion details** (Python floor, lint stack, `ext/pori-*` exact naming) — deferred to the workspace-conversion step (see Implementation Plan M0 / §12).

Donor OSS location and provenance rules: see [`HARVEST.md`](./HARVEST.md).
