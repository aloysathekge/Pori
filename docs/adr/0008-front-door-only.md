# 0008 — Products import only the kernel front door

Date: 2026-07 · Status: accepted · Enforced: import-linter in CI

## Context
Product code reaching into `pori.agent.x.y` internals couples it to kernel
refactors and rots the boundary (the Hermes-monolith trap).

## Decision
`aloy_backend` may import only from `pori` (the curated `__init__`). Deep
imports are CI-forbidden (tools/ci/importlinter.ini). If a product needs a
symbol, EXPORT it — don't punch through. Kernel never imports products.

## Consequences
Kernel refactors are invisible to products. The front door's "Product
integration seams" block is the de-facto kernel API contract.
