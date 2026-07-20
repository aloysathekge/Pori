# Aloy

Aloy is the first product built on the Pori agent kernel. It is a durable life
and work operating system: Conversations are where the user and Aloy think
together; Events keep one meaningful area alive over time; Tasks and Runs do
bounded work; Surfaces turn accepted Event state into a model-authored app;
Proposals protect external consequences; Files, memory, receipts, and Trail
preserve truth and continuity.

## Product composition

```text
products/aloy/app      React host: Today, Life, Events, Workbench, Surfaces
products/aloy/desktop  Electron shell around the hosted Aloy experience
products/aloy/backend  FastAPI product plane, persistence, worker, policy
packages/pori-client   shared REST/SSE contracts
extensions/*           optional capability providers
pori/*                 product-neutral agent kernel
```

Dependencies flow downward only: `products -> extensions -> pori`. The app and
desktop client reach the backend through REST and SSE; they never import Python
product or kernel code.

## Runtime ownership

- The API owns auth, tenant scope, REST/SSE, and foreground Conversation Runs.
- The durable worker owns Task Runs, Schedules, context ingestion, Event
  bootstrap, Surface building, watchdog recovery, and Proposal reconciliation.
- The database owns canonical structured state and object storage owns durable
  blobs. Process memory and a hosted local disk are never sources of truth.
- Generated Surface code runs outside the trusted app boundary and receives
  only the capability-scoped Surface SDK. The host owns canonical data and all
  protected actions.

## Start here

- [Product vision](../../docs/aloy-vision.md)
- [Active V1 plan](../../docs/aloy-v1-plan.md)
- [Local boot, operations, and demo](./BOOT.md)
- [App architecture](./app/ARCHITECTURE.md)
- [Backend architecture](./backend/README.md)
- [Surface contract](../../docs/aloy-surface-spec.md)
- [Monorepo dependency rules](../../MONOREPO.md)
