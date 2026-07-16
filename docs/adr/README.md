# Architecture Decision Records

Architecture decisions and their current status. If you're an agent about to "improve"
something that looks odd, check here first — odd is often deliberate.

| # | Decision |
|---|---|
| [0001](./0001-no-langchain.md) | Direct SDK calls, no LangChain |
| [0002](./0002-loop-stays-whole.md) | The agent loop stays whole in `agent/core.py` |
| [0003](./0003-sibling-binding.md) | Sibling-module method binding over mixins |
| [0004](./0004-fail-open-runtime.md) | Fail-open runtime bookkeeping |
| [0005](./0005-footprint-ladder.md) | The tool footprint ladder |
| [0006](./0006-memory-as-index.md) | Memory is an index over durable things |
| [0007](./0007-single-finalizer.md) | One finalizer persists a run's outcome |
| [0008](./0008-front-door-only.md) | Products import only the kernel front door |
| [0009](./0009-single-api-worker.md) | One API worker process (in-process run registries) |
| [0010](./0010-schema-driven-surfaces.md) | **Superseded:** schema-composed Surfaces; see the canonical Surface spec |
