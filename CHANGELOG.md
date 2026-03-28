# Changelog

All notable changes to Pori will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.2] - 2026-03-28

### Added
- **Evaluation framework** (`pori/eval/`) — AccuracyEval, ReliabilityEval, PerformanceEval, AgentJudgeEval
- **Guardrails** — ContentPolicyGuardrail, FactualityGuardrail, TopicGuardrail with pre_check/post_check hooks on Agent
- **Observability** (`pori/observability/`) — Span-based tracing with hierarchical trace trees
- **Read-only CoreMemory** for team members — `CoreMemory.clone_read_only()` so team members see user context without modifying it
- Agent.run() now returns `trace` in result dict
- Agent accepts `guardrails` parameter

### Changed
- Team members receive read-only copy of coordinator's CoreMemory

## [1.3.0] - 2026-03-25

### Added
- **Google Gemini** LLM provider (`pori/llm/google.py`)
- **Run metrics** — token usage, cost tracking, latency per step
- Cost estimation for 30+ models (Anthropic, OpenAI, Google)

## [1.2.0] - 2026-03-20

### Added
- **Multi-agent teams** — Router, Broadcast, Delegate coordination modes
- Team configuration via YAML
- Nested team support

## [1.1.0] - 2026-03-15

### Added
- **Human-in-the-loop (HITL)** — approval gates for tool execution
- CLI and programmatic HITL handlers
- Auto-approve duplicate actions

## [1.0.0] - 2026-03-10

### Added
- Core agent loop (Plan → Act → Reflect → Evaluate)
- **Persistent memory** — Letta-inspired CoreMemory with persona, human, notes blocks
- Archival memory with semantic search
- Memory persistence via MemoryStore protocol (in-memory, SQLite)
- Tool system with Pydantic validation and decorator registration
- Built-in tools: web search, math, filesystem, memory operations
- Orchestrator for task lifecycle and concurrency
- LLM providers: Anthropic, OpenAI (direct SDK, no LangChain)
- Interactive CLI
- Docker support
