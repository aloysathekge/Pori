# packages/pori — KERNEL (placeholder)

⚠ **PLACEHOLDER — do not put code here yet.**

The Pori kernel currently lives at the **repo-root `../../pori/`** package. It migrates into this directory in **Phase 4** (see [`../../docs/Pori_Implementation_Plan.md`](../../docs/Pori_Implementation_Plan.md)).

## What lands here (Phase 4)

The eval-native, receipt-first, memory-native kernel — product-agnostic, publishable standalone:

- `runtime/` — manager/worker loop, turn lifecycle, iteration budget, the Evaluator step
- `protocol/` — streaming event contract, message/tool-call types, `NormalizedResponse`
- `receipts/` — typed, hash-chained, evidence-linked, replayable records
- `validation/` — `Validator` interface + runner + minimal non-bypassable safety floor
- `llm/` — provider-agnostic transport + adapters
- `tools/` — registry + executor engine + `ToolBackend` interface
- `context/` — `ContextEngine` interface + compression mechanism + prompt caching
- `sandbox/` — execution backends + path security + hardline command floor
- `memory/` — block model + recall→inject + write lifecycle + `MemoryStore` interface
- `interfaces/` — the ABCs (`MemoryProvider`, `SkillProvider`, `ToolBackend`, `Validator`, …)

## Rule

`pori` imports **nothing** from `ext` or `products`. It is the bottom of the dependency DAG and must build/publish on its own.
