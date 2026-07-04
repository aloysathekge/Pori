# Pori Roadmap

Pori aims to be a small, clear, extensible agent framework for Python.

## Shipped

- **Sub-agent delegation** — `delegate_task` with isolated context: single / parallel batch / background, role-based depth (`leaf` / `orchestrator`), sub-agent security (auto-deny HITL for children), optional curated specialists (`.pori/agents/*.md`), and model-per-agent via provider-agnostic tiers
- **Human-in-the-loop** — per-tool approval gates + a structured `ask_user` clarify tool
- **Multi-agent teams** — router / broadcast / delegate
- **Skills** — progressive on-demand skills + an optional background learning loop that grows and curates its own skills
- **Cost/robustness** — prompt caching, model-aware context sizing, `edit_file` + `fetch_url` tools, tool-output bounds
- **API** — FastAPI surface with SSE streaming (Pori Cloud backend)

## Now

- Keep the core API stable and well tested
- Improve docs and examples
- Make local development and CI predictable (`uv`, `black`, `isort`, `pytest`)

## Next

- MCP support (external tool servers)
- Memory improvements (core/recall/archival separation)
- Stronger sandbox isolation backend (Docker/remote)
- Observability (metrics/dashboards)
- Delegation polish: specialist auto-discovery (roster in the tool view), a config default model for goal-driven children

## Contribution areas

- Docs and examples
- Tests and bug fixes
- New tools
- LLM provider wrappers (`pori/llm/`)

## Feature requests

Open an issue with:

- What you want to do
- A minimal example
- Why it should live in Pori
