# pori/memory — persistent session memory (Letta-style)

## What this package owns
Everything persistent about an agent session: CoreMemory blocks,
conversation messages, tool-call records, tasks/checkpoints, archival
passages, experiences, and the pluggable persistence behind them. Split
from the old single `pori/memory.py` module for readability — the public
surface (`from pori.memory import X`) is unchanged via `__init__.py`
re-exports.

## Files
- `agent_memory.py` — the `AgentMemory` facade (the big one). Owns prompt
  assembly inputs, the write-ahead dispatch journal for tool calls,
  resumable task checkpoints, and hybrid (semantic + lexical) record
  search. Persists the whole snapshot through a `MemoryStore`.
- `models.py` — data shapes: `CoreMemory` and its named `Block`s
  (persona / human / notes — always in the prompt), `AgentMessage`,
  `ToolCallRecord`, `TaskState`, and `SerializableMemoryState` (the
  rehydration handle).
- `stores.py` — `MemoryStore` protocol (namespace → JSON snapshot),
  `InMemoryMemoryStore`, `SQLiteMemoryStore` (default file
  `.pori/memory.db`), `create_memory_store` factory, and the
  `pori.memory_stores` entry-point hook for third-party backends.
- `__init__.py` — re-exports the above plus the long-term-memory contracts
  from `pori/memory_contracts.py` (`MemoryRecord`, `MemoryCatalog`, ...),
  which are a *separate*, provider-agnostic layer used by product backends.

## Key contracts
- The agent mutates CoreMemory only through tools (`memory_insert`,
  `memory_rethink`, `core_memory_append`, `core_memory_replace`) — never
  mutate blocks directly from orchestration code.
- A `MemoryStore` stores opaque JSON snapshots by namespace; backends don't
  interpret memory semantics. Custom backends plug in via the
  `pori.memory_stores` entry point (see pyproject.toml).
- Record search is hybrid: an embedding score (sentence-transformers when
  installed, deterministic hash-embedding fallback) blended with lexical
  scoring (`_blend_scores` in `agent_memory.py`). No external vector DB.
- Serialization round-trip: `AgentMemory` ⇄ `SerializableMemoryState` —
  any new persistent field must survive it.

## Change X → look at Y
- New persistent field → `models.py` + the save/load paths in
  `agent_memory.py` (and a round-trip test).
- Prompt shows stale/missing memory → block assembly in `agent_memory.py`
  and its consumer `pori/agent/prompting.py`.
- Backend/storage issue → `stores.py`; config selection lives in
  `create_memory_store` (`memory` | `sqlite` in config.yaml).
- Long-term knowledge / scoping / conflict policy → that's
  `pori/memory_contracts.py`, not this package's stores.
