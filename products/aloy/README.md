# products/aloy — Aloy (flagship product #1)

**Aloy** = a personal **and** company/org **OS agent** — Hermes-class capabilities as the baseline, plus more — built on the Pori kernel + extension band.

## Composition

Aloy = `pori` (kernel) + chosen `extensions/pori-*` + Aloy-specific glue:
- **backend** — evolves from the repo's `pori/api` (tenancy, RBAC, SSE, policy). ⚠ *Reconcile with the existing `Pori/pori_cloud` sibling project — see [`../../MONOREPO.md`](../../MONOREPO.md) open questions.*
- **cli** — the daily-driver CLI surface
- **gateway** — messaging platforms (later; Slack for org, Telegram for personal — harvest Hermes architecture)
- **surfaces** (`products/aloy/app`, `products/aloy/desktop`) — harvested from Hermes, **retargeted to REST + SSE** (strip the PTY bridge)
- **org policy** — the org policy engine expressed as scoped kernel validators
- **tenancy** — org→team→personal (layered inheritance; personal populated first, tenancy-aware from day 1)

## Build order

Personal Aloy first (Hermes-grade daily-driver on the hardened kernel), org plane after — but **tenancy-aware from day one** so isolation is never retrofitted.

## Rule

Aloy imports `extensions` and `pori`; the kernel never imports Aloy. Surfaces talk to the backend over REST + SSE, not Python import.
