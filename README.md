<div align="center">
  <img src="pori.png" alt="Pori Logo" width="600"/>
</div>

# Pori

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Pori** is a lightweight, extensible AI agent framework for building intelligent agents with tiered memory, tool-calling, and clean orchestration.

## ⚡ Quick Start

### Installation

**Currently, Pori must be installed from source** (PyPI publishing is planned — see [ROADMAP.md](ROADMAP.md)):

```bash
git clone https://github.com/aloysathekge/pori.git
cd pori

# Using uv (recommended)
uv venv
.venv\Scripts\activate  # On Windows: .venv\Scripts\activate
                        # On Unix/macOS: source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Configuration

Create `config.yaml` from `config.example.yaml` and add your API keys to `.env`:

```bash
cp config.example.yaml config.yaml
# Edit .env with your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### Basic Usage

**Interactive CLI:**
```bash
python -m pori
```

**Programmatic:**
```python
import asyncio
from pori import Orchestrator, AgentSettings, register_all_tools
from pori.llm import ChatAnthropic
from pori.tools.registry import tool_registry
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    registry = tool_registry()
    register_all_tools(registry)
    
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)
    
    result = await orchestrator.execute_task(
        "Calculate the sum of the first 10 Fibonacci numbers",
        agent_settings=AgentSettings(max_steps=10)
    )
    
    if result['success']:
        agent = result.get('agent')
        final_answer = agent.memory.get_final_answer()
        print(f"Answer: {final_answer['final_answer']}")

asyncio.run(main())
```

## 🧠 Core Features
- **Core Memory**: Letta-style editable blocks (persona, human, notes) — always in-context
- **Custom LLM Wrappers**: Direct SDK integration (Anthropic, OpenAI) — no LangChain dependency
- **Planning & Reflection**: Agent plans tasks and adapts based on results
- **Extensible Tools**: Simple decorator-based tool registration with Pydantic validation
- **Parallel Execution**: Orchestrate multiple tasks concurrently
- **Comprehensive Logging**: Full observability of agent decisions and tool calls

## 🏗️ Architecture
Pori follows a modular design:
- **Orchestrator**: Manages task lifecycle, concurrency, and shared memory
- **Agent**: Core reasoning loop (Plan → Act → Reflect → Evaluate)
- **Memory**: Conversation history, tool tracking, and Letta-style core memory blocks
- **Tool Registry**: Validated tool management via Pydantic models
- **LLM Wrappers**: Lightweight providers (`pori/llm/`) replacing LangChain

## 📚 Documentation
- [Roadmap](ROADMAP.md) — Planned features and contribution areas
- [Contributing](CONTRIBUTING.md) — How to contribute
- [Migration Guide](MIGRATION.md) — Moving from LangChain to custom wrappers
- [Core Memory](docs/CORE_MEMORY.md) — Letta-style memory system explained

## 📄 License
MIT License.
