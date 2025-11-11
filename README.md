# Pori

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

>  **Pori** is a lightweight, extensible AI agent framework that makes it easy to build intelligent agents with memory, tools, and evals.

##  What Makes Pori Special

- **Intelligent Agents**: LLM-powered agents that can reason, plan, and execute complex tasks
- **🔧 Extensible Tools**: Easy-to-add custom tools for any domain
- **🧠 Smart Memory**: Conversation history and state management with automatic summarization
- **⚡ Task Orchestration**: Run single tasks or multiple tasks in parallel
- **👤 Human-in-the-Loop**: Built-in support for human oversight and intervention (coming soon)
- **✅ Evals**: Automatic retry logic and task completion assessment
- **📐 Clean Architecture**: Modular design that's easy to understand and extend

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aloysathekge/pori.git
cd pori

# Install dependencies (using uv - recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or use pip
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API key(s):
```bash
# Anthropic (Claude)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI (GPT)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# Google (Gemini)
GOOGLE_API_KEY=your_key_here

# Or any other LangChain-compatible provider
```

### Run Your First Agent

**Interactive CLI:**
```bash
python -m pori.main
```

**Programmatic Usage:**
```python
import asyncio
import os
from pori import Agent, AgentSettings, Orchestrator, register_all_tools
from pori.tools import tool_registry

# Use any LangChain-compatible LLM
from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

async def main():
    # Set up tools
    registry = tool_registry()
    register_all_tools(registry)
    
    # Choose your LLM provider
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Or use OpenAI:
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Or Google Gemini:
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create orchestrator and run task
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)
    result = await orchestrator.execute_task(
        task="Calculate the 10th Fibonacci number",
        agent_settings=AgentSettings(max_steps=10)
    )
    
    print(f"✅ Task completed: {result['success']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**FastAPI Server:**
```bash
uvicorn pori.api:app --reload
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Orchestrator   │───▶│     Agent       │
│   (main.py)     │    │  (Task Mgmt)    │    │  (Reasoning)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Memory      │◀───┤      Tools      │
                       │   (History)     │    │  (Actions)      │
                       └─────────────────┘    └─────────────────┘
```


## 🔧 Adding Custom Tools

1. **Define your tool parameters:**
```python
class WeatherParams(BaseModel):
    city: str = Field(..., description="City name")
    units: str = Field("metric", description="Temperature units")
```

2. **Create your tool function:**
```python
def get_weather(params: WeatherParams, context: dict):
    # Your tool logic here
    return f"Weather in {params.city}: 22°C, sunny"
```

3. **Register the tool:**
```python
registry.register_tool(
    name="get_weather",
    param_model=WeatherParams, 
    function=get_weather,
    description="Get current weather for a city"
)
```

## ⚙️ Configuration Options

### Environment Variables
```bash
# LLM Provider (choose one or configure multiple)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

GOOGLE_API_KEY=your_key_here

# Agent Configuration
MAX_STEPS=50                             # Max steps per task
MAX_FAILURES=3                           # Max consecutive failures
RETRY_DELAY=2                            # Delay between retries (seconds)
```

### Agent Settings
```python
settings = AgentSettings(
    max_steps=20,           # Maximum steps for this task
    max_failures=3,         # Max consecutive failures before stopping
    summary_interval=5,     # Create memory summary every N steps
    validate_output=False   # Enable output validation
)
```

## 🔄 Parallel Task Execution

```python
# Run multiple tasks simultaneously
tasks = [
    "Calculate fibonacci of 10",
    "Generate 5 random numbers", 
    "What's 15 factorial?"
]

results = await orchestrator.execute_tasks_parallel(
    tasks=tasks,
    max_concurrent=3
)

for result in results:
    if result["success"]:
        print(f"✅ Task completed: {result['task_id']}")
    else:
        print(f"❌ Task failed: {result['error']}")
```

## 📚 Documentation

- **[Architecture](ARCHITECTURE.md)** - Technical deep dive into how Pori works
- **[Contributing](CONTRIBUTING.md)** - How to contribute to Pori
- **[Roadmap](ROADMAP.md)** - Future plans and feature requests
- **[Agent Documentation](AGENT.md)** - Developer documentation and commands

## 🌟 Examples

Check out the `examples/` directory for more:

- [Basic Usage](examples/basic_usage.py) - Simple task execution
- Custom Tool Development (coming soon)
- Multi-Agent Coordination (coming soon)
- Memory Management (coming soon)


## 📁 Project Structure

```
pori/
├── agent.py          # Core agent logic
├── orchestrator.py   # Task management and parallel   
├── memory.py         # Conversation and state management
├── tools.py          # Tool registry and execution
├── evaluation.py     # Result assessment and retry logic
├── main.py          # Interactive CLI interface
├── api/             # using Pri via api 
├── tools_box/       # Built-in tools
│   ├── core_tools.py
│   ├── math_tools.py
│   └── number_tools.py
└── prompts/         # System prompts
    └── system/
        └── agent_core.md
```



## 🤝 Contributing

We welcome contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features  
- 🔧 New tools
- 📖 Documentation improvements
- 💡 Ideas and suggestions

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Quick Start for Contributors

```bash
# Fork and clone the repo
git clone https://github.com/aloysathekge/pori.git
cd pori

# Install dev dependencies
uv pip install -e ".[test]"

# Run tests
pytest

# Format code
black pori/ tests/
isort pori/ tests/
```

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and how you can help!

**Upcoming features:**
- Vector database integration for semantic memory
- Streaming support for real-time output
- Human-in-the-loop approval gates
- Multi-agent coordination


## 💬 Community

- **Issues**: [Bug reports and feature requests](https://github.com/aloysathekge/pori/issues)
- **Discussions**: [Questions and ideas](https://github.com/aloysathekge/pori/discussions)
- **Pull Requests**: [Code contributions](https://github.com/aloysathekge/pori/pulls)

## 📄 License

MIT License - feel free to use Pori in your projects!

---

**🌟 Star this repo if you find Pori interesting and potentially useful!** 