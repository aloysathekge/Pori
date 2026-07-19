# Event-owned Schedules handoff — 2026-07-19

## Outcome

The old global raw-cron page is now an Event-owned Schedule system. A user
chooses a dedicated Event, ordinary recurrence, IANA timezone, instruction,
Run budget, notification mode, and one of two frozen authority levels. Due
Schedules enqueue normal durable Runs into the Event's canonical Conversation,
write wake/outcome evidence to Trail, appear as active work in Today, and retain
bounded occurrence history.

## Safety and reliability

- `report_only` denies Task/file/memory-write tools, drafts, protected provider
  writes, Surface requests, skill evolution, and MCP.
- `organize` may update Event Tasks and create reversible drafts; protected
  provider writes still stage Proposals. MCP and Surface commissioning remain
  unavailable unattended.
- Dormant/archived Events stay quiet and missed occurrences do not replay.
- Schedule authority is frozen in `Run.run_profile`; `Run.cron_job_id` owns
  occurrence history. Schedule deletion is soft so receipts keep their parent.
- A migration adds Event/timezone/authority/notification/deletion fields and
  Run lineage. Local SQLite is at `h0e1f2a3b4c5`.

## Verification

- `tests/test_cron.py`: 15 passed.
- Schedule + Today focused set: 20 passed before the final added regression.
- Complete backend: 358 passed; one documented headless Surface bridge race
  failed, then passed on exact rerun.
- Backend mypy: 114 source files clean.
- Changed backend Ruff, Black, and isort: clean.
- App ESLint and production build: pass; existing large-chunk warning only.
- API `/v1/health`: `ok`; Vite `/schedules`: HTTP 200; unauthenticated `/v1/cron`:
  401 as expected.

## Remaining acceptance

The browser-control plugin still fails to initialize with `Cannot redefine
property: process`, so no authenticated visual screenshot was captured. Have the
user open Schedules, create one short Event Schedule, and verify the complete
wake/result/Trail/history experience before committing the branch.
