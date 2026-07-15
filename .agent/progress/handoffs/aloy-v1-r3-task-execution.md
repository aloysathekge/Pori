# Aloy V1 R3 - durable Task execution handoff

_Date: 2026-07-15_

## Branch

`aloy-v1-r3-task-execution`, based exactly on the R2 squash merge
`b47f0fd41a027c3a51024be0c329b1ab74d11428` in `aloy-v1`.

## Implemented

- `POST /events/{event_id}/tasks/{task_id}/work`
- related `/stop`, `/retry`, and `/resume` controls
- idempotent durable Run creation and selected-Conversation routing
- Event + Task bounded work-order assembly
- same-Run checkpoint Resume for blocked/approved work
- fresh-Run Retry for failed/cancelled work
- atomic worker Task claim and stale-Run suppression
- per-step durable Trail milestones
- Task synchronization for clarification, Proposal, stop, success, and failure
- compact lifecycle Conversation messages
- one active Task Run per Event
- one active Run per Conversation
- account-wide running cap with queued Task backpressure
- status-aware Event controls and bounded refresh

## Deliberately deferred

- R4 Event/Task SSE, reconnect cursors, and grouped Trail narratives
- R5 sourced web research providers and cited report artifact
- R6 receipt-gated Career OS Gmail completion
- R7 watchdogs and release drills

## Verification

- `uv run pytest tests/ -q --basetemp ...`: 253 passed
- focused R3/worker/proposal/resume set: 60 passed
- `uv run mypy aloy_backend --ignore-missing-imports`: 85 files clean
- backend Black/isort: clean
- app `npm run lint`: clean
- app `npm run build`: green; existing bundle-size warning only

## Remaining phase gate

Run local product QA with a real Event Task: Work on this, navigation/app closure,
queued second Task, Stop, Retry, blocked Resume, and correct Conversation routing.
Then commit/push, open a draft PR targeting `aloy-v1`, and merge only after CI.
