## Project State

This document summarizes the current architecture, behavior, and recent changes of the Pori agent project.

### Overview

- Goal: A tool-using agent with short-term and long-term memory, able to plan, act, and answer tasks interactively.
- Runtime: Interactive CLI via `pori/main.py` using Anthropic Chat model (configurable) and a shared memory across tasks per session.

### High-level architecture

```
User (CLI)  →  Orchestrator  →  Agent  →  Tools
                         ↘         ↗
                          EnhancedAgentMemory (working, long-term, vector)
```

### Key components

- `pori/main.py`
  - Starts interactive loop
  - Builds tool registry (via `pori/tools_builtin`) and LLM (Anthropic)
  - Creates `Orchestrator` and prints per-task summary (final answer, tool calls)

- `pori/orchestrator.py`
  - Creates a single shared `EnhancedAgentMemory` per session
  - Spawns `Agent` instances per task, injecting the shared memory
  - Tracks task lifecycle (start, run, finish)

- `pori/agent.py`
  - Core loop: plan → get actions → execute tools → reflect → repeat until complete or `max_steps`
  - Message building: adds system prompt, recent working memory, current context, and a “Retrieved Knowledge” section from memory recall
  - Planning prompt tuned to produce 1–3 steps mapped directly to tools/final answer
  - Robust structured output parsing to handle non-ideal LLM outputs
  - Logs concrete plan steps; logs top retrieved knowledge snippets
  - Records step results and final answer in memory state for the current task

- `pori/memory_v2/`
  - `enhanced_memory.py` (EnhancedAgentMemory)
    - `working`: per-task short-term buffer (reset each task)
    - `long_term`: session-long storage of experiences (task text, step results, summaries)
    - `vector`: optional semantic index (SentenceTransformers) for recall
    - Per-task isolation: working memory reset and per-task `final_answer` state cleared on `create_task`
    - Tool calls tagged with `task_id` for filtering
  - `memory_store.py`: generic `MemoryStore` interface (add/get/search/forget)
  - `stores/in_memory.py`: in-memory text store with substring search
  - `stores/vector_store.py`: local vector store using cosine similarity over normalized embeddings

- `pori/memory.py` (legacy)
  - Original list-based memory class; retained for backward-compatibility but no longer wired into `Agent`

- `pori/tools_builtin/`
  - `core_tools.py`: `answer`, `done`, `think`, `remember` (explicit fact storage helper)
  - `math_tools.py`, `number_tools.py`, `spotify_tools.py`: examples/utilities

- `pori/prompts/system/agent_core.md`
  - System prompt establishing behavior: tool use, answer as a final step, brevity on recall, avoid pasting prior final answers verbatim

### Memory behavior

- Working memory (short-term, per-task)
  - Stores simple lines `"role: content"` for recent messages
  - Reset on `create_task()` to prevent cross-task leakage
  - Used to reconstruct chat turn context for the LLM

- Long-term memory (session-durable)
  - Stores “experiences” with metadata:
    - Task text (type: `task`)
    - Step results (type: `step_result`) – excluding vector-indexing of final answers
    - Periodic summaries (type: `summary`)
  - Vector memory indexes non-final texts for semantic recall (filters out items that look like prior final answers)
  - Recall:
    - Query = last user message (fallback: task text)
    - `recall(query, k=5, min_score=0.35)` returns top items by similarity
    - Injected in prompt as “Retrieved Knowledge (for reference)” with guidance to produce fresh, concise answers
  - Logging: top retrieved snippets logged per task for observability

- Persistence
  - Current: both `working` and `long_term` are in-memory (session only)
  - After process restart: memory resets
  - Path to persistence: implement `SqliteStore` for `long_term`, and FAISS/Pinecone/Weaviate/Milvus for vector

### Recent changes (high-impact)

- Introduced `EnhancedAgentMemory` with working/long-term/vector stores
- Switched `Agent` to use enhanced memory; added semantic recall injection and logging
- Prevented replay of prior final answers by filtering/index policies and prompt guidance
- Per-task isolation: working memory reset, per-task `final_answer` cleared on new task
- Structured output parsing hardened for list/string/chunked outputs
- CLI summaries now show only current task tool calls
- Added `remember` tool (explicit fact storage) – optional to use
- Planning prompt tuned to produce tool-aligned, concise steps
- Plan steps are logged for visibility

### Known limitations / open issues

- No persistence across restarts yet (in-memory only)
- Recall can miss pronoun follow-ups (e.g., “he”) if no subject facts are stored; current design avoids indexing prior final answers to prevent replay
- Some tasks may produce pessimistic or overly cautious one-step plans if recall returns minimal context
- Facts/fun facts are model-derived unless backed by a retrieval tool/source

### Next steps (suggested roadmap)

1. Persistence
   - Implement `SqliteStore` for `long_term` (SQLAlchemy); add migrations
   - Add FAISS/Pinecone/Weaviate integration for vector memory

2. Subject and fact capture (optional)
   - Store compact subjects/entities from final answers to improve pronoun resolution in follow-ups

3. Retrieval quality & efficiency
   - Compose recall query from (last user message + plan + recent tool results)
   - Apply score thresholds and de-duplication; cap injected facts to token budget

4. Observability & testing
   - Track recall hit-rate, latency, memory growth; add unit/integration tests with large memory

5. Tooling


6. Security & hygiene
   - PII redaction pre-persistence; optional encryption at rest

### Runbook

- Requirements: see `requirements.txt` (includes `numpy`, `sentence-transformers`)
- Configure Anthropic model/key via environment variables
- Start: `python -m pori.main`
- Toggle vector memory (for limited environments): edit `pori/orchestrator.py` to set `vector=False`


