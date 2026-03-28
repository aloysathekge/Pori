<div align="center">
  <img src="pori.png" alt="Pori" width="500"/>

  <p><strong>A lightweight, extensible AI agent framework with persistent memory, multi-agent teams, and built-in evaluations.</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

  [Quick Start](#quick-start) · [Features](#features) · [Architecture](#architecture) · [Documentation](#documentation) · [Contributing](#contributing)
</div>

---

## Why Pori?

Most agent frameworks are either too simple (no memory, no teams) or too complex (heavy abstractions, LangChain dependency graphs). Pori sits in the middle:

- **Persistent memory** — Letta-inspired CoreMemory blocks that survive across conversations. Your agent actually remembers users.
- **Multi-agent teams** — Router, broadcast, and delegate modes. Agents coordinate without sharing state.
- **Built-in evals & guardrails** — Accuracy, reliability, performance evals. Runtime safety checks before and after every response.
- **Tracing** — Hierarchical span trees for every agent run. See exactly what happened.
- **No LangChain** — Direct SDK integration with Anthropic, OpenAI, and Google. Lightweight, fast, debuggable.

---

## Quick Start

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

    llm = create_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))
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
from pori.tools.registry import Registry
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")

@Registry.tool(name="my_search", description="Search for information")
def my_search(params: SearchParams, context: dict):
    results = do_search(params.query)
    return {"success": True, "result": results}
```

**Built-in tools:**

| Category | Tools |
|----------|-------|
| **Core** | `answer`, `done`, `think`, `remember`, `conversation_search` |
| **Memory** | `core_memory_append`, `core_memory_replace`, `memory_insert`, `memory_rethink`, `archival_memory_insert`, `archival_memory_search` |
| **Web** | `web_search` (Tavily) |
| **Math** | `calculate` |
| **Files** | `read_file`, `write_file`, `list_directory`, `file_info`, `create_directory`, `search_files`, `copy_file`, `move_file`, `delete_file` |

### LLM Providers

Direct SDK integration — no middleware, no abstraction layers:

```python
from pori.config import create_llm, LLMConfig

# Anthropic
llm = create_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))

# OpenAI
llm = create_llm(LLMConfig(provider="openai", model="gpt-4o"))

# Google Gemini
llm = create_llm(LLMConfig(provider="google", model="gemini-2.5-flash"))
```

All providers support structured output, tool calling, and streaming.

---

## Architecture

```
pori/
├── agent.py              # Core reasoning loop (Plan → Act → Reflect → Evaluate)
├── memory.py             # Persistent memory system (CoreMemory, Archival, MemoryStore)
├── metrics.py            # Token usage, cost tracking, run metrics
├── evaluation.py         # Action result evaluation, task completion
├── config.py             # YAML + env configuration
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
│   ├── registry.py       # Tool registration + execution
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
