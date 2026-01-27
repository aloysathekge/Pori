<p align="center">
 
  <img src="pori.png" alt="Pori Logo" width="600"/>  <!-- Larger -->
</p>

# Pori

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Pori** is a lightweight, extensible AI agent framework for building intelligent agents with tiered memory, tool-calling, and clean orchestration.

## ⚡ Quick Start

### Installation
```bash
pip install pori
```

### Basic Usage
```python
import asyncio
from pori import Orchestrator, tool_registry, register_all_tools
from pori.llm import ChatOpenAI

async def main():
    registry = tool_registry()
    register_all_tools(registry)
    
    llm = ChatOpenAI(model="gpt-4o")
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)
    
    result = await orchestrator.execute_task("Find the square root of 256")
    print(result['agent'].memory.get_final_answer())

asyncio.run(main())
```

## 🧠 Core Features
- **Tiered Memory**: Inspired by Letta (Core, Recall, Archival)
- **Extensible Tools**: Simple decorator-based tool registration
- **Parallel Execution**: Orchestrate multiple tasks concurrently
- **Observation**: Comprehensive logging and state tracking

## 🏗️ Architecture
Pori follows a modular design:
- **Orchestrator**: Manages task lifecycle and concurrency
- **Agent**: The core reasoning loop (Plan → Act → Reflect)
- **Memory**: Three-tier system for long and short-term context
- **Registry**: Validated tool management via Pydantic

See [ARCHITECTURE.md](ARCHITECTURE.md) for more details.

## 📁 Links
- [Documentation] [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md)

## 📄 License
MIT License.
