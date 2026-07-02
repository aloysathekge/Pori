# packages/pori — the Pori kernel

The `pori` package (the kernel) lives here at `packages/pori/pori/`. It is built
by the repo-root `pyproject.toml` (`[tool.setuptools.packages.find] where =
["packages/pori"]`) and imported as `pori`.

## Layout

- `pori/` — the kernel package (agent loop, llm, tools, memory, sandbox,
  observability, orchestrator, context, evaluation, skills, evolution, …).
- `pori/api/` — the FastAPI service. **Temporary home:** this is a product/backend
  concern and will be extracted to `products/aloy/backend/` in the next pass
  (see [`../../MONOREPO.md`](../../MONOREPO.md)); it lives here for now so the
  kernel move stayed low-risk.

## Target direction

As the kernel/product split matures (see [`../../docs/Pori.md`](../../docs/Pori.md)),
the kernel narrows to the mechanism substrate (runtime · protocol · receipts ·
validation · llm · tools · context · sandbox · memory engine · interfaces), and
product/tenancy concerns move out to `packages/ext/` and `products/`.

## Rule

`pori` imports **nothing** from `packages/ext/` or `products/`. It is the bottom
of the dependency DAG (`products → ext → pori`).
