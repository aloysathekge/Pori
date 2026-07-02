# Pori / Aloy Monorepo — Structure & Rules

**Status:** v0.2 — flat, intent-named · 2026-07-02
**See also:** [`docs/Pori.md`](./docs/Pori.md) (PRD), [`docs/Pori_Implementation_Plan.md`](./docs/Pori_Implementation_Plan.md), [`HARVEST.md`](./HARVEST.md)

## Layout

Flat and intent-named — like Hermes (top-level packages, no generic `packages/`
wrapper), and no `name/name` nesting.

```
repo root (this git repo — one distribution for now)
├─ pori/          KERNEL — the `pori` package (import pori). Product-agnostic.
│                 (pori/api is a product/backend concern; moves to products/aloy/backend next)
├─ extensions/    reusable pori-* libraries (opt-in; created on promotion, not on spec)
├─ products/
│  └─ aloy/       Aloy — product #1: backend · cli · gateway (+ org policy, tenancy)
├─ apps/          frontend surfaces (talk to a product backend over REST + SSE)
│  ├─ web/        ← pori_cloud_client
│  └─ desktop/    ← desktop shell (later)
├─ website/       public marketing site ← pori_website
├─ docs/          PRD, plan, ALIGNMENT tracker, design docs
├─ tools/ci/      dependency-boundary contract (staged)
└─ (donors)       external OSS at ../references/ — never a runtime dep
```

## The one rule: one-way dependencies

```
products / apps → extensions → pori        (never upward)
```

- `pori` (kernel) imports **nothing** from `extensions`/`products`/`apps`.
- `extensions/*` may import `pori` only.
- `products/*` may import `extensions` and `pori`.
- Surfaces (`apps/web`, `apps/desktop`) talk to a product backend over **REST + SSE**, never by Python import.

**Dependency inversion:** the kernel defines interfaces; `extensions`/`products` implement them. Enforced (staged) by [`tools/ci/`](./tools/ci/).

## Packaging

A **single** root `pyproject.toml` builds the `pori` package (`[tool.setuptools.packages.find] where = ["."]`, `isort known_first_party = ["pori"]`). When we publish `pori` standalone or add `extensions`/`products` Python packages, we split into a uv workspace with per-package `pyproject.toml`s. Not yet.

## Absorbing the sibling projects (copy-in)

Four projects at the `Pori/` level are folded in by **copying contents** into the homes below (their standalone git histories are not preserved; sources left untouched until then).

| Source (external, `../`) | → Home | What it is |
|---|---|---|
| `pori_cloud` (git) | `products/aloy/backend/` | the API — "our api"; Aloy's backend |
| `pori_cloud_client` (git) | `apps/web/` | frontend (Vite/React) |
| `pori_website` (git) | `website/` | public marketing site |
| `pori_docs` (folder) | `docs/` | design/architecture markdown |

## ⚠ Open questions

1. **Repo topology.** Sibling projects (`pori_cloud`, `pori_cloud_client`, `pori_website`, `pori_docs`) live *outside* this git repo at the `Pori/` level; `pori/api` vs the standalone `pori_cloud` still need reconciling.
2. **Workspace split.** When to move from a single root `pyproject.toml` to a real uv workspace with per-package projects.

Donor provenance & rules: see [`HARVEST.md`](./HARVEST.md).
