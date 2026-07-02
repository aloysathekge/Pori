# tools/ci — Boundary & supply-chain enforcement

## The one-way dependency rule

```
products → ext → pori        (never upward)
```

`pori` (kernel) must import nothing from `ext`/`products`. This is enforced with **[import-linter](https://github.com/seddonym/import-linter)** using [`importlinter.ini`](./importlinter.ini).

### Status: STAGED (inert until Phase 4)

The contract targets the post-migration package layout (`packages/pori`, `packages/ext/pori-*`, `products/aloy`). Those packages don't exist yet (the kernel still lives at repo-root `pori/`), so the check is **staged, not active**. It activates during the Phase 4 migration.

### Run it (once packages exist)

```bash
pip install import-linter
lint-imports --config tools/ci/importlinter.ini
# or:
bash tools/ci/check-boundaries.sh
```

### Wire into CI (Phase 4)

Add a required job that runs `lint-imports --config tools/ci/importlinter.ini`. Until then, keep it out of the required set so it doesn't block on nonexistent packages.

## Supply-chain gates (to add — see Implementation Plan M0)

- OSV scan of the lockfile (donor: Hermes `osv-scanner.yml`)
- dependency-bounds check — reject unbounded `>=` deps (donor: Hermes `supply-chain-audit.yml`)
- SHA-pin all GitHub Actions
- exact/bounded dependency pins
