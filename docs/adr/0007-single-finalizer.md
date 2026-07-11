# 0007 — One finalizer persists a run's outcome

Date: 2026-07 (send_message refactor) · Status: accepted

## Context
Streaming and non-streaming paths each hand-persisted messages/usage/traces
and drifted repeatedly ("the streaming path forgot X" bug class).

## Decision
One `RunOutcome` built by `build_run_outcome`, persisted by ONE idempotent
`persist_run_outcome` (keyed by run_id) in a single transaction — messages,
run, usage, trace, event log, memory, stored artifacts. Transport handlers
never persist.

## Consequences
New outcome data (e.g. artifact rows, stopped flag) is added in exactly one
place. The durable worker must route through the same finalizer (audited
2026-07-11).
