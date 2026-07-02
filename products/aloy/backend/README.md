# products/aloy/backend — Aloy backend (the API)

**Home for the contents of `../../../../pori_cloud`** (external git repo — "our api"): a FastAPI service with alembic migrations and Docker packaging.

Copy the `pori_cloud` contents here (source repo left untouched; history not preserved). Then:

- it becomes **Aloy's backend**, composing `pori` (kernel) + `ext/pori-*`;
- grows tenancy / RBAC / **SSE** / the org policy engine (scoped kernel validators);
- **reconcile with the repo's `pori/api`** — the smaller in-kernel API surface (see [`../../../MONOREPO.md`](../../../MONOREPO.md) open questions).

Dependency rule: imports `ext` and `pori`; never imported by them.
