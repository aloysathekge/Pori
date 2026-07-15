# Aloy V1 R2 handoff - executable Task model

## State

- Base: `aloy-v1` at R1 squash merge
  `7889bc2e12560b2b9c36434a1065367cf0199af5` (#169).
- Phase branch: `aloy-v1-r2-task-model`.
- Scope: R2 in `docs/aloy-v1-plan.md`.
- Draft PR: #170, targeting `aloy-v1`.
- Status: implementation, local verification, and all seven PR checks are
  green; R2 is ready to mark for review and merge.

## Implemented

- Task status is now `open|queued|in_progress|blocked|waiting_approval|done|failed|cancelled`.
- Task stores instructions, definition of done, priority, optional due date,
  manual execution mode, optional assigned agent, origin Conversation, current
  Run, result summary, blocker, and a validated budget-policy object.
- Run stores nullable `task_id`; child Runs inherit the parent Task link.
- Task origin must belong to the Task's Event and tenant. Life Tasks may choose
  either of multiple Life Conversations; dedicated Event Tasks cannot point
  into Life or another Event.
- `task_state.py` centralizes legal transition validation, provenance checks,
  mutation Trail writes, and atomic queued-Task claims.
- Concurrent claims use compare-and-set semantics tied to a matching active
  Run. Exactly one caller advances the Task to `in_progress` and writes the
  claim Trail entry.
- User endpoints and agent Task tools use the same state boundary. Invalid
  transitions leave both Task and Trail unchanged.
- Today includes every non-terminal Task; the Event UI renders non-checklist
  statuses without offering an illegal checkbox transition.
- Migration `w9a0b1c2d3e4` safely adds and backfills the Task/Run model. Existing
  Task titles and `open|done` status are preserved; origin is populated from a
  valid Event resume Conversation when available.
- Backend `alembic.example.ini` and the boot guide now provide a working
  URL-free local migration command.

## Verification

- Full backend suite: `246 passed`.
- Focused R2/Event/context suite: `31 passed`.
- Task contract + agent-tool suite after due-date normalization: `22 passed`.
- Black: `156 files` clean.
- Backend mypy: clean across `84 source files`.
- Aloy app ESLint and production TypeScript/Vite build: passed.
- Clean SQLite database upgraded through every Alembic revision to
  `w9a0b1c2d3e4 (head)` using the checked-in configuration.
- PR CI: all seven checks passed after an import-order-only correction,
  including Python 3.10-3.12, Aloy backend, Aloy app, dependency bounds, and
  import boundaries.
- Migration backfill, cross-Event origin rejection, illegal-transition
  rollback, and concurrent one-winner claims have direct regression tests.

## Next phase boundary

R2 does not execute Tasks. After green PR CI and merge into `aloy-v1`, create
`aloy-v1-r3-task-execution`. R3 adds **Work on this**, durable Run creation,
queue/stop/retry/resume controls, one active Task Run per Event, and compact
progress messages routed to the selected Conversation.
