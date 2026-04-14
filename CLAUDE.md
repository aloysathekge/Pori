# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pori is a Python AI agent framework built on direct SDK calls to Anthropic, OpenAI, and Google (no LangChain). It provides a single-agent reasoning loop (`pori.Agent`), multi-agent coordination (`pori.Team`), persistent memory modeled after Letta's CoreMemory blocks, an eval/guardrail framework, and a span-based tracing system.

Entry points:
- `pori` CLI (`pori.cli:run`) → interactive REPL
- `python -m pori` → same as above via `pori/__main__.py`
- Programmatic: `from pori import Orchestrator, Agent, Team, ...`

## Common Commands

This project uses `uv` as its primary package manager, with `pyproject.toml` as the source of truth. `requirements.txt` exists but is not authoritative.

```bash
# Dev install (editable + test extras)
uv sync --extra test

# Run the CLI
uv run pori                   # or: python -m pori

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_agent.py::TestAgent::test_name   # single test
uv run pytest -m memory                                   # by marker (unit|integration|memory|tools|agent|orchestrator)
uv run pytest --cov=pori --cov-report=term-missing        # with coverage

# Formatting / lint (also run via pre-commit hook)
uv run black pori/ tests/
uv run isort pori/ tests/
uv run mypy pori/ --ignore-missing-imports

# Pre-commit (required for contributors — auto-formats on commit)
pre-commit install
```

CI (`.github/workflows/ci.yml`) runs black-check, isort-check, mypy, and pytest on Python 3.11 and 3.12. Tests run with empty API keys — tests must work via mocks.

## Configuration

Runtime config comes from two files, loaded by `pori/config.py`:

- `.env` — API keys only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`)
- `config.yaml` — LLM provider/model, agent limits, memory backend (`memory` | `sqlite`), optional `sandbox`, optional `hitl` (human-in-the-loop approval rules per tool)

`config.example.yaml` documents every option. SQLite memory defaults to `.pori/memory.db`.

## Architecture

The codebase layers responsibilities intentionally — don't bypass layers. Reading across these modules together is the fastest way to understand a change's blast radius.

### Agent loop (`pori/agent.py`)
`Agent.run()` executes a **Plan → Act → Reflect → Evaluate** loop bounded by `AgentSettings.max_steps` and `max_failures`. Each step:
1. Builds a prompt from `AgentMemory` (CoreMemory blocks + recent messages, trimmed to `context_window_tokens`).
2. Calls the LLM via `pori.llm.BaseChatModel` with tool schemas from `ToolRegistry`.
3. Executes returned tool calls through `ToolExecutor` (with HITL approval if `HITLConfig` says so).
4. `Evaluator` (`evaluation.py`) produces an `ActionResult` judging progress.
5. Emits spans into a `Trace` and `StepMetrics` with token + cost accounting (`metrics.py`).

Terminal tools are `answer` / `done` — the agent signals completion by calling them. Never remove them from the built-in registry.

### Memory (`pori/memory.py`)
`AgentMemory` bundles everything persistent about a session: `CoreMemory` (three named `Block`s — persona/human/notes — always in the prompt), conversation messages, tool-call records, archival passages (semantic-search, Chroma-backed), and experiences. `MemoryStore` is the pluggable backend (`InMemoryMemoryStore`, `SQLiteMemoryStore`, or custom via the `pori.memory_stores` entry point). The agent mutates CoreMemory through the `memory_insert` / `memory_rethink` / `core_memory_append` / `core_memory_replace` tools — don't mutate blocks directly from orchestration code.

### Orchestrator (`pori/orchestrator/core.py`)
`Orchestrator.execute_task` creates an `Agent`, threads optional `shared_memory`, sandbox, and HITL handlers, and runs it. This is the main programmatic entrypoint for single-task runs.

### Teams (`pori/team/`)
`Team` coordinates multiple `Agent`s in one of three `TeamMode`s:
- `ROUTER` — coordinator LLM picks one member.
- `BROADCAST` — all members run in parallel, coordinator synthesizes.
- `DELEGATE` — coordinator produces a DAG of steps with member assignments; team executor respects dependencies.

Members do **not** share memory by default; they communicate only through the coordinator's inputs/outputs.

### Tools (`pori/tools/`)
- `registry.py` — `ToolRegistry` + `ToolExecutor`. Tools are Pydantic-validated and registered via `@Registry.tool(...)` or by implementing the `pori.tools` entry point.
- `standard/` — built-in tools split by domain (`core_tools`, `filesystem_tools`, `internet_tools`, `math_tools`, `number_tools`). `register_all_tools(registry)` installs them all.
- `pori/sandbox/` — optional sandboxed filesystem/shell environment; sandbox-aware tool variants live in `sandbox/sandbox_tools.py` and resolve paths via `path_resolution.py`. When `config.sandbox.enabled` is true, file tools operate inside `.pori_sandbox` (or configured base).

### Eval / Guardrails (`pori/eval/`)
Same `BaseEval` interface used for both offline evaluation and runtime guardrails. Evals: `AccuracyEval` (LLM-judged), `ReliabilityEval` (deterministic tool-call checks), `PerformanceEval` (timing/memory), `AgentJudgeEval` (freeform criteria). Guardrails (`guardrails.py`): `ContentPolicyGuardrail`, `FactualityGuardrail`, `TopicGuardrail` — attached via `Agent(guardrails=[...])` and run pre/post.

### Observability (`pori/observability/`)
`Trace` is a tree of `Span`s (types: `agent`, `llm`, `tool`, etc.). Stored via `TraceStore` (`InMemoryTraceStore` default) and exported via `TelemetryExporter` (`ConsoleTelemetryExporter` default). Every `agent.run()` returns `{"trace": ..., "metrics": ...}`.

### LLM providers (`pori/llm/`)
`BaseChatModel` in `llm/base.py` defines the interface; `anthropic.py`, `openai.py`, `google.py` are direct SDK wrappers. `messages.py` holds the shared `UserMessage` / `AssistantMessage` / `SystemMessage` / tool-call types. Build provider instances via `pori.config.create_llm(LLMConfig(...))`.

### API (`pori/api/`)
FastAPI surface for the hosted Pori Cloud product — not used by the CLI. Touch only when the user is working on the API.

### Prompts (`pori/prompts/system/`)
System prompts live as Markdown files loaded by `utils/prompt_loader.py`. Package data in `pyproject.toml` includes `prompts/**/*.md` so they ship with the wheel. `agent_core.md` is the main agent system prompt.

### HITL (`pori/hitl.py`)
Per-tool human approval gate. `HITLConfig.interrupt_on` maps tool names to allowed decisions (`approve`/`edit`/`reject`). Wired into `ToolExecutor` — bypassing it means bypassing user-configured safety.

## Conventions Worth Knowing

- Async everywhere on public agent/team APIs; tests use `asyncio_mode = "auto"` so `async def test_*` works without decorators.
- Pytest markers (`unit`, `integration`, `memory`, `tools`, `agent`, `orchestrator`) are declared in `pyproject.toml` — use them on new tests.
- Black line length is 88; isort uses the `black` profile. Pre-commit will reformat on commit.
- Minimum supported Python is 3.10 (per `pyproject.toml`). CI only tests 3.11 / 3.12.
- `.pori/` and `.pori_sandbox/` are runtime state directories — not source.
