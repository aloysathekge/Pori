# Aloy backend

The Aloy backend is the hosted product plane around the product-neutral Pori
kernel. FastAPI exposes tenant-scoped REST and SSE; SQLAlchemy/SQLModel and
Alembic persist product truth; a leased worker executes durable work.

## Responsibilities

- verify Supabase identity and organization membership;
- enforce user, organization, Event, Conversation, and capability scope;
- own Life and dedicated Event topology;
- persist Conversations, Tasks, Runs, files, memory, evidence, records,
  Proposals, receipts, Trail, Schedules, and Surface revisions;
- assemble bounded Pori Runs from trusted product context;
- stream foreground and durable state to the app;
- compile and publish generated Surfaces outside the trusted application
  boundary;
- recover expired work and reconcile uncertain external outcomes.

Pori never imports this package. Aloy imports Pori and supplies resolved tools,
memory, policy, files, and model configuration for one Run.

## Execution rails

### Foreground Conversation

The API persists the user message, assembles the current Conversation with
accepted owning-Event state, runs Pori, streams events over SSE, and commits the
terminal outcome through the shared finalizer.

### Durable work

The worker claims pending Runs with database leases. Tasks, Schedules, Event
bootstrap, context ingestion, and Surface builds survive API closure because
their intent and progress are durable. Checkpoints and heartbeats allow another
worker to resume eligible work after a lease expires.

Before claiming new work, each loop repairs expired Runs and orphaned Tasks.
Terminal watchdog results reconcile Task, Conversation, Surface, Schedule, and
Trail projections once.

### Protected consequences

External writes follow Proposal -> user decision -> execution -> receipt. If a
provider may have accepted a write before Aloy committed the receipt, the
Proposal becomes `indeterminate`. A separate read-only reconciler uses the
stable operation identity; it never blindly repeats the consequence.

## Context longevity

Conversation hydration uses a stable host-owned token allowance rather than a
provider's maximum context. Accepted contiguous transcript prefixes become
immutable, versioned `ContextArtifact` summaries. A Run receives the latest
verified summary plus a bounded recent tail. Older Event history remains
durable and is page-faulted through a tenant/user/Event-scoped search handler.

## Budget contract

Every Run freezes ceilings for steps, tool calls, tokens, cost, and active
duration. A single kernel ledger follows the root Agent, hidden model calls,
Teams, members, and nested Teams, including checkpoint resume. Actual provider
usage is recorded even when one in-flight call crosses a ceiling; no further
model or tool action is then allowed.

## Generated Surface boundary

The Builder returns one structured source candidate without model-visible
filesystem or build tools. The host persists, validates, compiles, inspects,
and publishes it. Generated code receives the Surface SDK rather than backend
credentials or direct provider/network access. A failed candidate cannot
replace the last verified publication.

## Local development

Use [the Aloy boot and operator guide](../BOOT.md). Package entry points are:

```text
aloy-backend          API
aloy-backend-worker   durable worker
aloy-backend-gateway  optional messaging gateway
```

Run backend verification from this directory:

```bash
uv run --no-sync pytest tests/ -q
uv run --no-sync mypy aloy_backend/ --ignore-missing-imports
```
