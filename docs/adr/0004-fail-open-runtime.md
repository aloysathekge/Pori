# 0004 — Fail-open runtime bookkeeping

Date: 2025-2026 · Status: accepted

## Context
The agent loop performs non-essential bookkeeping (metrics, checkpoints,
journaling, artifact observation) inline. A metrics bug must never kill a
user's run.

## Decision
Bookkeeping failures are swallowed and logged (`except Exception` +
`logger.debug/warning`), deliberately. Core execution failures are NOT
fail-open — tool errors, LLM errors, and budget exhaustion surface properly.

## Consequences
42+ intentional swallow sites exist; each should state its intent. The risk
(real bugs hiding in debug logs) is accepted for run durability; a
`fail_open` helper making the pattern explicit/greppable is the planned
ratchet.
