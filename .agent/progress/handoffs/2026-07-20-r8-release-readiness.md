# R8 release readiness handoff - 2026-07-20

## Outcome

Completed the locally verifiable R8 release-readiness slice on
`aloy-v1-r8-release-gate`. The working tree is intentionally uncommitted pending
user review.

## What changed

- Rewrote existing Aloy boot, product, backend, and frontend architecture docs
  around Life Conversations, durable Events, Task/Run execution, generated
  Surfaces, protected Proposals, worker recovery, budgets, context compaction,
  and read-only provider reconciliation.
- Added local no-model-credit guidance, operator failure handling, the 60-second
  Career OS demo, and an explicit manual responsive/accessibility checklist.
- Added all supported public-search environment key names to `.env.example` and
  made the local SQLite default agree with runtime configuration.
- Fixed `surface_runtime_inspection.py` so CDP waits for the navigated runtime
  document and root before transferring its MessageChannel. Each manifest UI
  interaction receives an independent bounded inspection deadline.
- Exported `ensure_budgeted_chat_model` from `pori` and moved Event bootstrap
  and Surface Builder off deep `pori.llm` / `pori.utils` imports.

## Why it matters

The browser gate now tests the Surface that actually loaded instead of racing
the initial Chrome target. Valid Add/Save forms no longer receive false bridge
failures, while render exceptions, missing visible feedback, invalid controls,
and canonical state projection remain enforced before publication. The public
kernel seam also preserves the monorepo's one-way dependency contract.

## Verification

- `uv run --no-sync pytest tests/ -q --basetemp .pytest_tmp_r8_release_kernel_final`
  -> `627 passed, 1 skipped`.
- `uv run --no-sync pytest tests/ -q --basetemp .pytest_tmp_r8_release_backend_final`
  from `products/aloy/backend` -> `401 passed`.
- Focused Surface build suite -> `17 passed`.
- Focused Event bootstrap + Surface Builder suite -> `13 passed`.
- Kernel mypy -> clean in 110 files; backend mypy -> clean in 124 files.
- App `npm run test` -> `7 passed`; ESLint -> clean; production build -> passed.
- Import linter -> all three contracts kept.
- Documentation links and `git diff --check` -> clean before the final status
  update; rerun `git diff --check` before commit.

## Remaining gates

- The user must perform the manual desktop/tablet/mobile and accessibility
  checklist; Codex browser control still fails to initialize.
- University, Madrid, Career, and the 60-second Career OS live proof need model
  credits and configured research/Builder providers.
- Hosted Surface proof needs the pinned E2B toolchain template.
