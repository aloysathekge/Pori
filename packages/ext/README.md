# packages/ext — EXTENSION BAND (`pori-*`)

Reusable-across-products building blocks that are **not** the kernel and **not** a single product. Opt-in, publishable, may import `pori` **only**.

## Planned extensions (created on demand, not up front)

- `pori-memory` / `pori-tenancy` — org→team→personal scope resolver, RBAC, concrete stores
- `pori-skills` — progressive-disclosure skills catalog
- `pori-learning` — learn / background-review / curator + provenance
- `pori-gateway` — thin platform-adapter ABC + adapters (Slack, Telegram, …)
- `pori-providers` — provider registry/profiles
- `pori-cli-kit` — CLI command-registry toolkit

## Anti-speculation rule

**Do not create a `pori-*` package on spec.** Build the capability inside `products/aloy/` first; when a **second** product needs it, *promote* the reusable part here (rule of three). Log the promotion in [`../../HARVEST.md`](../../HARVEST.md) if it involved harvested patterns.
