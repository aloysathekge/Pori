# Pori / Aloy Monorepo — Structure & Rules

**Status:** v0.3 — platform + products · 2026-07-04
**See also:** [`docs/Pori.md`](./docs/Pori.md) (PRD), [`docs/Aloy.md`](./docs/Aloy.md) (product plan), [`docs/Pori_Implementation_Plan.md`](./docs/Pori_Implementation_Plan.md), [`HARVEST.md`](./HARVEST.md)

## The shape: one platform, many products

**Pori is a kernel; products are built on it.** So the tree separates the shared
**platform** (kernel + shared libraries) from **products**, and each product owns
its *whole* stack — backend and every surface — under one folder. A second product
is a sibling of `aloy/`, with nothing to move at the root.

```
repo root (one git repo; TS workspace + single Python distribution)
├─ pori/                 KERNEL — the `pori` package (import pori). Product-agnostic.
├─ extensions/           reusable pori-* libraries (opt-in; created on promotion)
├─ packages/             shared, product-neutral TypeScript libraries
│  └─ pori-client/       @pori/client — the REST + SSE (PoriEvent) transport every UI uses
├─ products/
│  └─ aloy/              Aloy — product #1, self-contained:
│     ├─ backend/        FastAPI (Python) — composes the kernel; tenancy/auth/persistence
│     ├─ app/            web SPA (Vite + React) — @aloy/app
│     ├─ desktop/        Electron shell wrapping app/ (later)
│     └─ website/        public marketing landing (self-contained static)
├─ docs/                 PRD, product plan, design docs
├─ tools/ci/             dependency-boundary contract (CI-enforced)
├─ package.json          root TS workspace (packages/* + products/*/{web,desktop,website})
└─ pyproject.toml        builds the `pori` package
```

**Why products own their surfaces (not a root `apps/`):** a root `apps/` and
`website/` quietly assume *one* app and *one* site. The moment a second product
lands, its surfaces have nowhere to go. Grouping by product scales to N products
with zero root churn. Only genuinely shared, product-neutral code lives at the root
(`packages/*`) — the transport client is the same for every product's UI, so it is
`@pori/client`, not `@aloy/anything`.

## The one rule: one-way dependencies

```
products (backend + surfaces) → extensions → pori        (never upward)
surfaces → (REST + SSE only) → a product backend         (never a Python import)
```

- `pori` (kernel) imports **nothing** from `extensions` / `products` / `packages`.
- `extensions/*` may import `pori` only.
- `products/*/backend` may import `extensions` and `pori`.
- Surfaces (`products/aloy/app`, `desktop`, `website`) reach the backend **only over
  REST + SSE**, via `@pori/client`. Never a Python import. This is the single
  safeguard against the Hermes-monolith trap.

The Python side of this rule is **enforced in CI**: the `boundaries` job runs
import-linter against `tools/ci/importlinter.ini` (`pori` may never import
`pori_cloud`; layering is `pori_cloud → pori`). Run it locally with
`bash tools/ci/check-boundaries.sh`.

## TypeScript workspace

The root `package.json` declares a workspace over `packages/*` and each product's
`web` / `desktop` / `website`. Shared libraries are consumed as real workspace
packages, not path-alias hacks:

```jsonc
// products/aloy/app/package.json
"dependencies": { "@pori/client": "workspace:*", … }
```

`bun install` at the root links `@pori/client` into every surface
(`node_modules/@pori/client` → `packages/pori-client`); `bun.lock` is the single
lockfile. Surfaces also keep a Vite alias / tsconfig path to the client's TS source
so there's no build step for the internal package (just-in-time source).

## Python packaging

A **single** root `pyproject.toml` builds the `pori` package
(`[tool.setuptools.packages.find] where = ["."]`). Each `products/*/backend`
depends on the kernel via an editable path source. When we publish `pori` standalone
or add `extensions` Python packages, we split into a uv workspace. Not yet.

## Extracting a product to its own repo (the exit is designed in)

A product is meant to be liftable into a standalone repo the day it outgrows the
monorepo. `products/aloy/` is self-contained; the **only** things it reaches for
outside its own folder are the two **platform** dependencies — the kernel and the
shared client — and each is a *single, replaceable dependency declaration*:

| Coupling | In the monorepo (dev) | On extraction (standalone) |
|---|---|---|
| Python → kernel | `products/aloy/backend/pyproject.toml`: `pori = { path = "../../.." }` | `pori = ">=X.Y"` from **PyPI** |
| TS → client | `products/aloy/app/package.json`: `"@pori/client": "workspace:*"` | `"@pori/client": "^X.Y"` from **npm** |

So extraction is:

```bash
git subtree split -P products/aloy -b aloy-extract   # history-preserving slice
# → push aloy-extract to the new repo, then swap the two dep sources above
```

**The rule that keeps this cheap:** nothing under `products/aloy/` may import another
product, or any root file, *except* the kernel (Python) and `@pori/client` (TS). Keep
those two the only bridges and the exit stays a two-line swap — while, in the
monorepo, many products still share one kernel and one client (no duplication, no
drift). The platform pieces (`pori`, `@pori/client`) are shaped to be published for
exactly this reason.

## Provenance

Surfaces were harvested from Hermes (MIT) and the external `pori_cloud` /
`pori_cloud_client` / `pori_website` siblings, then rebranded. Donor provenance &
rules: [`HARVEST.md`](./HARVEST.md).
