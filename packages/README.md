# packages/ — Kernel & Extension bands

Two of the three monorepo bands live here (see [`../MONOREPO.md`](../MONOREPO.md)):

- **`pori/`** — the KERNEL. Product-agnostic, publishable, imports nothing from `ext`/`products`.
  Currently a **placeholder**: the working kernel lives at the repo-root `../pori/` and migrates here in Phase 4.
- **`ext/`** — the EXTENSION BAND (`pori-*`). Reusable-across-products building blocks. May import `pori` only. Created by **promotion on second use**, not on spec.

Dependency direction (CI-enforced): `products → ext → pori`, never upward.

"The open framework" = `pori` + `ext`.
