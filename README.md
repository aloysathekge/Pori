<div align="center">
  <img src="pori.png" alt="Pori" width="500"/>

  <p><strong>The agent runtime that survives. Long-running tasks that resume after a crash — and skills the agent grows from its own work.</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

  [Quick Start](#quick-start) · [Features](#features) · [Architecture](#architecture) · [Documentation](#documentation) · [Contributing](#contributing)
</div>

---

## Why Pori?

Most agent frameworks treat a run as a fire-and-forget prompt loop: if the
process dies, the work is gone, and the agent is exactly as capable on run
1,000 as on run 1. Pori is built on the opposite two bets — **runs that
survive** and **an agent that improves with use.**

### 🛡️ Durable by design — runs that survive a crash

An agent run is a checkpointed, resumable process, not a transient loop.
Every tool call is **write-ahead journaled before its side effects**; the
step counter and plan are checkpointed **every step**; and a restarted (or
re-leased) run **resumes from exactly where it stopped** — same step, same
plan, same working files — instead of starting over. A run that hits its
step or time budget still returns a **best-effort salvage summary** rather
than nothing. This is the hard-to-copy part: it's designed into the loop and
the memory model, not bolted on.

### 🌱 Self-improving — skills the agent grows from its own work

After a run, an optional background review mines the finished session for a
reusable **skill** and writes it to disk; a deterministic **curator** ages
skills (active → stale → archived) by real usage; **provenance-gating** means
autonomy has a ceiling (the curator only ever archives, never deletes). The
result compounds: the more the agent works, the better its skill library
gets — without a human in the loop.

### Everything else you'd expect

- **Memory-native** — Letta-inspired CoreMemory blocks that persist across
  conversations, plus archival semantic recall. Your agent remembers users.
- **Eval-native** — one `BaseEval` interface serves *both* offline evaluation
  and runtime guardrails (accuracy, reliability, performance) before and after
  every response.
- **Sub-agent delegation** — isolated children (single, parallel batch, or
  background), curated specialists, model-per-agent tiers — heavy work stays
  in a throwaway context; you get the summary.
- **Multi-agent teams** — router, broadcast, and delegate-DAG modes.
- **True isolation, optional** — run agent code in a cloud microVM (E2B) with
  one config line; secrets are stripped from every command.
- **Provider-agnostic + resilient** — direct SDK calls to Anthropic, OpenAI,
  Google (plus OpenRouter/Fireworks/local OSS), a cross-provider failover
  chain, and multimodal (text + image) messages. No LangChain.
- **Human-in-the-loop**, **span tracing**, and a strict **context budget** so
  every registered tool earns its place.

---

## One kernel, many products

This repository is a monorepo. **Pori is the kernel** — the product-agnostic `pori` Python package this README documents. **Products are built on top of it**; the first is **[Aloy](products/aloy/)**, a personal + org OS agent with its own FastAPI backend, web app, and website.

```
repo root
├─ pori/                  the kernel (this README) — `import pori`, product-agnostic
├─ packages/pori-client/  @pori/client — shared TypeScript REST + SSE client
└─ products/aloy/         Aloy — backend, web app, desktop shell, website
```

The dependency rule is one-way and enforced in CI with import-linter: products depend on the kernel; the kernel never imports from a product. Products consume `pori` as a normal versioned dependency, so any product can be lifted into its own repository without touching the kernel. See [MONOREPO.md](MONOREPO.md) for the layout rules and the extraction playbook.

---

## Quick Start

### Install

```bash
pip install pori
```

### Install from source

```bash
git clone https://github.com/aloysathekge/pori.git
cd pori

# Using uv (recommended)
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# Or pip
pip install -e .
```

### Configure

```bash
cp config.example.yaml config.yaml
```

Add your API keys to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
# or
GOOGLE_API_KEY=...
```

### Run

**CLI:**
```bash
pori
# or
python -m pori
```

**Python:**
```python
import asyncio
from pori import Orchestrator, AgentSettings, register_all_tools
from pori.config import create_llm, LLMConfig
from pori.tools.registry import tool_registry

async def main():
    registry = tool_registry()
    register_all_tools(registry)

    llm = create_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929"))
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)

    result = await orchestrator.execute_task(
        "What are the top 3 trending AI papers this week?",
        agent_settings=AgentSettings(max_steps=10),
    )

    agent = result.get("agent")
    answer = agent.memory.get_final_answer()
    print(answer["final_answer"])

asyncio.run(main())
```

### Docker

```bash
docker build -t pori .
docker run --env-file .env pori
```

---

## Features

### Persistent Memory

Letta-inspired three-block CoreMemory that the agent reads and writes:

| Block | Purpose |
|-------|---------|
| **persona** | Who the agent is, how it should behave |
| **human** | What the agent knows about the user |
| **notes** | Working knowledge, facts, preferences |

The agent updates these blocks autonomously via `memory_insert` and `memory_rethink` tools. Memory persists across conversations via pluggable backends (in-memory, SQLite, or bring your own).

```python
from pori import AgentMemory, create_memory_store

store = create_memory_store(backend="sqlite", sqlite_path="memory.db")
memory = AgentMemory(user_id="user_123", store=store)

# Agent remembers across sessions
memory.core_memory.get_block("human").value
# → "User is a senior engineer who prefers concise answers"
```

**Archival memory** for long-term storage with semantic search:

```python
memory.archival_memory_insert("User is building a SaaS product", tags=["context"])
results = memory.archival_memory_search("what is the user building?", k=5)
```

### Multi-Agent Teams

Three coordination modes for different use cases:

```python
from pori import Team, TeamMode, MemberConfig

team = Team(
    task="Research and write a report on quantum computing",
    coordinator_llm=llm,
    members=[
        MemberConfig(name="researcher", description="Deep web research"),
        MemberConfig(name="analyst", description="Synthesizes findings"),
        MemberConfig(name="writer", description="Writes polished output"),
    ],
    mode=TeamMode.DELEGATE,  # Multi-step plan with dependencies
)

result = await team.run()
```

| Mode | Behavior |
|------|----------|
| `ROUTER` | Coordinator picks the single best member for the task |
| `BROADCAST` | All members run in parallel, coordinator synthesizes results |
| `DELEGATE` | Coordinator creates a multi-step plan, members execute steps with dependency ordering |

### Sub-Agent Delegation

Distinct from Teams: the running agent can delegate a subtask to a focused
**sub-agent** with its own isolated context and a restricted toolset — so
context-heavy work (research, exploration, review) happens in a throwaway context
and the caller stays clean. It's the `delegate_task` tool; the model reaches for it
on its own judgment (no magic word — the *shape* of the task triggers it).

- **Single / batch / background** — one subtask, several run concurrently, or
  fire-and-forget (results re-enter on a later turn).
- **Role-based depth** — a `leaf` (default) can't delegate further; an
  `orchestrator` child can decompose its own work, bounded by a depth cap.
- **Sub-agent security** — children run non-interactively, so risky HITL-gated
  tools are auto-denied (they can't prompt a user).
- **Optional curated specialists** — drop a `.pori/agents/<name>.md` to name a
  tuned expert; a task's `agent` field selects it. Omit it for goal-driven
  delegation.
- **Model-per-agent** — a specialist can declare a provider-agnostic tier
  (`fast` / `powerful`) mapped to concrete models in `llm.tiers`, so grunt work
  runs on a cheap model while the manager reasons on a strong one.

A specialist is a markdown file with frontmatter + a system-prompt body
(`.pori/agents/code-reviewer.md`):

```markdown
---
name: code-reviewer
description: Reviews a diff or file for bugs, concurrency issues, and design smells.
tools: read_file, search_files, list_directory
model: fast
---
You are an expert code reviewer. Focus on correctness, concurrency, and error
handling. Return findings as a prioritized list with file:line references.
```

### Evaluations

Four eval types to test agent quality:

```python
from pori.eval import ReliabilityEval, AccuracyEval, PerformanceEval, AgentJudgeEval

# Did the agent call the right tools?
eval = ReliabilityEval(agent=my_agent, expected_tool_calls=["web_search", "answer"])
result = await eval.run()
result.assert_passed()

# Is the answer correct? (LLM-judged)
eval = AccuracyEval(agent=my_agent, expected_output="42", evaluator_llm=judge_llm)
result = await eval.run()
assert result.avg_score >= 7

# How fast is it?
eval = PerformanceEval(func=lambda: agent.run(), num_iterations=10)
result = await eval.run()
print(f"p95: {result.p95_run_time:.2f}s")

# Custom criteria
eval = AgentJudgeEval(
    criteria="Response must cite sources and be under 200 words",
    judge_llm=judge_llm,
)
result = await eval.run(input="...", output="...")
```

### Guardrails

Same eval interface, but runs at request time:

```python
from pori import Agent
from pori.eval import ContentPolicyGuardrail, TopicGuardrail

agent = Agent(
    task="...",
    llm=llm,
    tools_registry=registry,
    guardrails=[
        ContentPolicyGuardrail(judge_llm=llm),
        TopicGuardrail(allowed_topics=["science", "technology"], judge_llm=llm),
    ],
)

result = await agent.run()
# If guardrail fails: {"completed": False, "blocked_by": "input_guardrail", "reason": "..."}
```

### Observability

Every `agent.run()` produces a hierarchical trace:

```python
result = await agent.run()
trace = result["trace"]

# {
#   "trace_id": "abc123",
#   "duration": "3.210s",
#   "total_spans": 6,
#   "tree": [
#     {"name": "step_1", "type": "agent", "duration": "1.2s", "children": [
#       {"name": "gemini-2.5-flash.invoke", "type": "llm", "duration": "0.8s"},
#       {"name": "web_search.execute", "type": "tool", "duration": "0.4s"}
#     ]}
#   ]
# }
```

Metrics are automatic — token counts, cost estimation, latency per step:

```python
result["metrics"]
# {"duration": "3.21s", "tokens": {"input": 1200, "output": 400}, "cost_usd": "$0.0048"}
```

### Tools

Decorator-based registration with Pydantic validation:

```python
from pori import ToolRegistry
from pydantic import BaseModel, Field

registry = ToolRegistry()

class SearchParams(BaseModel):
    query: str = Field(description="Search query")

@registry.tool(name="my_search", description="Search for information")
def my_search(params: SearchParams, context: dict):
    results = do_search(params.query)
    return {"success": True, "result": results}
```

Registries resolve to an immutable `CapabilitySnapshot` when an agent is
created. Capability groups declare prerequisites and output bounds, protected
kernel tools cannot be removed accidentally, and unavailable integrations do
not enter the model-visible schema. `SkillEligibility` checks required tools,
credentials, platforms, and model capabilities before full skill instructions
are loaded.

**Built-in tools:**

| Category | Tools |
|----------|-------|
| **Core** | `answer`, `done`, `think`, `remember`, `ask_user`, `update_plan`, `conversation_search` |
| **Delegation** | `delegate_task` (isolated sub-agents — single / batch / background) |
| **Memory** | `core_memory_append`, `core_memory_replace`, `core_memory_read`, `core_memory_rethink`, `memory_insert`, `memory_rethink`, `archival_memory_insert`, `archival_memory_search` |
| **Web** | `web_search` (Tavily), `fetch_url` |
| **Files** | `read_file`, `write_file`, `edit_file`, `list_directory`, `file_info`, `create_directory`, `search_files`, `copy_file`, `move_file`, `delete_file` |
| **Skills** | `skills_list`, `skill_view`, `write_skill` |

### LLM Providers

Direct SDK integration — no middleware, no abstraction layers:

```python
from pori.config import create_llm, LLMConfig

# Anthropic
llm = create_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929"))

# OpenAI
llm = create_llm(LLMConfig(provider="openai", model="gpt-4o"))

# Google Gemini
llm = create_llm(LLMConfig(provider="google", model="gemini-2.5-flash"))
```

All providers support structured output, tool calling, and streaming.

Provider names, aliases, credential environment variables, defaults, known
models, adapter dependencies, and capability flags live in declarative
`ProviderProfile` records. `diagnose_provider()` uses the same profile checks as
`create_llm()` and reports credential presence without returning secret values.

---

## Architecture

```
pori/
├── agent/                # Core reasoning-loop package (Plan → Act → Reflect → Evaluate)
│   ├── core.py           #   the loop + lifecycle
│   ├── prompting.py      #   system-prompt / message-window / context rendering
│   ├── planning.py       #   optional plan/reflect phases + gating
│   ├── artifacts.py      #   execution-receipt / tool-artifact tracking
│   ├── authorization.py  #   tool side-effect authorization + HITL
│   └── schemas.py        #   Agent data models
├── memory.py             # Persistent memory system (CoreMemory, Archival, MemoryStore)
├── metrics.py            # Token usage, cost tracking, run metrics
├── evaluation.py         # Action result evaluation, task completion
├── config.py             # YAML + env configuration
├── capabilities.py       # Capability groups and skill eligibility
├── context.py            # Context-window policy and inclusion diagnostics
├── providers.py          # Declarative provider profiles and diagnostics
├── retrieval.py          # Provenance-preserving retrieval fusion
├── sessions.py           # Session lifecycle contracts and local SQLite store
├── subagents.py          # Sub-agent delegation (delegate_task) + specialist catalog
├── background_delegation.py  # Background (fire-and-forget) delegation registry
├── hitl.py               # Human-in-the-loop approval gates
├── clarify.py            # Structured ask_user clarify tool
├── skills.py             # Progressive on-demand skills (+ curator/learning loop)
├── sandbox/              # Sandboxed filesystem/shell execution
├── api/                  # FastAPI surface (SSE streaming) for Pori Cloud
├── orchestrator/         # Task lifecycle, concurrency, shared memory
├── team/                 # Multi-agent coordination (router, broadcast, delegate)
├── eval/                 # Evaluation framework + guardrails
│   ├── base.py           # BaseEval with pre_check/post_check
│   ├── accuracy.py       # LLM-judged answer scoring
│   ├── reliability.py    # Deterministic tool call verification
│   ├── performance.py    # Runtime + memory benchmarking
│   ├── agent_judge.py    # Custom criteria evaluation
│   └── guardrails.py     # ContentPolicy, Factuality, Topic guards
├── observability/        # Tracing and telemetry
│   ├── trace.py          # Span-based execution traces
│   ├── store.py          # Trace persistence (InMemory, extensible)
│   └── exporters.py      # Telemetry export (Console, extensible)
├── llm/                  # LLM providers (Anthropic, OpenAI, Google)
├── tools/                # Tool system with Pydantic validation
│   ├── registry.py       # Registration, immutable snapshots, execution bounds
│   └── standard/         # Built-in tools (web, math, files, memory)
└── prompts/              # System prompts
```

### Agent Loop

```
Task → Plan → LLM Call → Tool Execution → Evaluate → Reflect → Repeat
                                                          ↓
                                                    Memory Update
                                                    (CoreMemory, Archival)
```

### Memory Architecture

```
AgentMemory
├── CoreMemory (persona, human, notes) — always in context, persistent
├── Messages — conversation history
├── Archival Passages — long-term semantic storage
├── Experiences — short-term recall with embeddings
├── Tool Call History — full execution log
└── MemoryStore (pluggable: in-memory, SQLite, Postgres, custom)
```

### Durable Continuity

`DefaultContextEngine` preserves Pori's existing token-window behavior while
making inclusion, compaction, summary use, and recent-tail preservation
inspectable. Core memory and retrieved long-term evidence are frozen when a run
starts; writes remain durable but enter prompts only on the next run.

`SessionRepository` defines resume, search, export, delete, and branch lineage.
`SQLiteSessionRepository` is the local reference implementation. Session search
and typed long-term memory can be merged with `fuse_retrieval()` without losing
source IDs, session IDs, scores, or provenance.

---

## Configuration

`config.yaml`:

```yaml
llm:
  provider: anthropic          # anthropic | openai | google
  model: claude-sonnet-4-20250514
  temperature: 0.0

memory:
  backend: sqlite              # memory | sqlite
  sqlite_path: .pori/memory.db

agent:
  max_steps: 15
  max_failures: 3
```

See [`config.example.yaml`](config.example.yaml) for all options.

---

## Pori Cloud

[Pori Cloud](https://pori.aloysathekge.com) is the hosted platform built on this framework. Multi-tenant API with:

- Conversations with SSE streaming
- Persistent per-user memory (PostgreSQL-backed)
- Agent and team configuration
- Usage tracking and cost analytics
- Execution traces
- Rate limiting

---

## Documentation

- [Configuration Guide](config.example.yaml)
- [Contributing](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup dev environment
uv pip install -e ".[test]"

# Run tests
pytest

# Format
black pori/ tests/
isort pori/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
