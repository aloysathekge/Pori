# tools/ci — Boundary & supply-chain enforcement

## The one-way dependency rule

```
products → ext → pori        (never upward)
```

`pori` (kernel) must import nothing from `ext`/`products`. This is enforced with **[import-linter](https://github.com/seddonym/import-linter)** using [`importlinter.ini`](./importlinter.ini).

### Status: ACTIVE

The contract targets the real layout — the kernel at repo-root `pori/` and the Aloy backend package `aloy_backend` under `products/aloy/backend/`. It is enforced by the `boundaries` job in `.github/workflows/ci.yml`, which fails the build if the kernel imports from any product.

### Run it locally

```bash
pip install import-linter
bash tools/ci/check-boundaries.sh
```

The script puts the repo root and `products/aloy/backend` on `PYTHONPATH` (Windows/Git Bash handled) and runs `lint-imports --config tools/ci/importlinter.ini`. As extension packages appear under `extensions/`, add them to `root_packages` and the layers in the ini.

## Supply-chain gates (to add — see Implementation Plan M0)

- OSV scan of the lockfile (donor: Hermes `osv-scanner.yml`)
- dependency-bounds check — reject unbounded `>=` deps (donor: Hermes `supply-chain-audit.yml`)
- SHA-pin all GitHub Actions
- exact/bounded dependency pins
