# Changelog

All notable changes to Pori will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-23

**Headline: slimmer core install.** `pip install pori` now pulls only what the core agent loop needs. Heavy ML dependencies move to optional extras.

### Changed (install surface)
- **Core dependencies reduced to six packages**: `anthropic`, `openai`, `google-genai`, `pydantic`, `python-dotenv`, `pyyaml`. Fresh install drops from ~214 transitive packages to a fraction of that — no `torch`, no `transformers`, no `chromadb`.
- **`sentence-transformers` moved to `pori[embeddings]` extra.** Pori falls back to a deterministic hash-based embedding when the extra is not installed (lower quality but zero-dependency). Install with `pip install "pori[embeddings]"` for semantic memory recall.
- **`tavily-python` moved to `pori[web]` extra.** The `web_search` tool returns a clear "install tavily-python" error when invoked without the extra. Install with `pip install "pori[web]"`.
- **`pori[all]`** extra installs both `embeddings` and `web` in one shot.

### Removed
- **`chromadb`** dependency — was declared but never imported anywhere in the codebase.
- **`numpy`** dependency — not used directly; sentence-transformers pulls it transitively when needed.
- **`requests`** dependency — not used directly; SDKs bring their own HTTP clients.

### Migration

If you relied on the implicit transitive install of these packages, pin them explicitly or install the appropriate extra:

\`\`\`bash
# Before (v1.3.x): one install pulled everything
pip install pori

# After (v1.4.0): lean core
pip install pori

# Want semantic memory recall?
pip install "pori[embeddings]"

# Want the web_search tool?
pip install "pori[web]"

# Want everything?
pip install "pori[all]"
\`\`\`

## [1.3.3] - 2026-04-22

**Headline: OpenRouter models land in Pori.** Access any OSS or hosted model (Llama, Qwen, DeepSeek, Mistral, Gemma, Claude, GPT, Gemini) through one provider, with an interactive picker for choosing at startup and a new `/model` command for swapping mid-session.

### Added
- **OpenRouter LLM provider** (`pori/llm/openrouter.py`) — access OSS models (Llama, Qwen, DeepSeek, Mistral, Gemma) and hosted frontier models through a single API
- **Curated OpenRouter catalog** (`pori/llm/openrouter_models.py`) with grouped Free / Open Source / Hosted tiers
- **Interactive model selection** at startup via `model: select` config sentinel or `PORI_SELECT_MODEL=1` env var
- **`/model` CLI command** — switch provider and model at runtime across Anthropic, OpenAI, Google, and OpenRouter without restart
- **`/new` CLI command** (aliases `/reset`, `/clear`) — start a fresh conversation; drops prior-task messages + tool calls but keeps durable memory (experiences, archival passages, core memory)
- **`ask_user` tool** for agent-driven clarification during task execution
- **Code of Conduct** and **SECURITY.md** for community governance
- `.editorconfig` for consistent editor settings across contributors

### Changed
- **Agent system prompt** tuned to respond directly to greetings, identity questions, and conversational inputs (no more `ask_user` spam on simple queries). `ask_user` now reserved for genuine blockers.
- **CoreMemory** gains `clone_read_only()` and `rewrite()` alias; team members receive read-only copy of coordinator memory
- **ChatGoogle** error handling and response processing hardened
- **LLM provider error logging** expanded for easier debugging of transport/auth failures
- **Team** refactored for cleaner task coordination and member management
- README overhauled with clearer overview and installation instructions

### Fixed
- Timestamp handling in `AgentMemory` (timezone-aware)
- Removed dead Spotify API tool registrations

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
