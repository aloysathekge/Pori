# 0009 — One API worker process

Date: 2026-07 · Status: accepted (revisit at scale-out)

## Context
Live-run re-attach, clarify bridges, stop tokens, and warm resume are
in-process registries. A second uvicorn worker splits them: requests land on
the worker that doesn't hold the state, and those features fail
intermittently — the worst failure mode.

## Decision
The API runs exactly ONE uvicorn worker (Dockerfile CMD). Scale the durable
`worker` service (DB-leased), never API processes. Moving the registries to
Redis/DB is the documented path when true scale-out is needed.

## Consequences
Vertical scaling only for the API tier, for now. The constraint is
documented in the Dockerfile, deploy RUNBOOK, and here so nobody
"optimizes" it back to 2 workers (it shipped broken once).
