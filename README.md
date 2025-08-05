# Pori

>  **Pori** is a lightweight, extensible AI agent framework that makes it easy to build intelligent agents with memory, tools, and human-in-the-loop capabilities.

##  What Makes Pori Special

- ** Intelligent Agents**: LLM-powered agents that can reason, plan, and execute complex tasks
- ** Extensible Tools**: Easy-to-add custom tools for any domain
- ** Smart Memory**: Conversation history and state management with automatic summarization
- ** Task Orchestration**: Run single tasks or multiple tasks in parallel
- ** Human-in-the-Loop**: Built-in support for human oversight and intervention
- **  Evals**: Automatic retry logic and task completion assessment
- ** Clean Architecture**: Modular design that's easy to understand and extend

##  Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd simple_agent

# Install dependencies (using uv - recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or use pip
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API key:
```bash
ANTHROPIC_API_KEY=your_actual_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # optional
```

### Run Your First Agent

```bash
python pori/main.py
```

The agent will prompt you for a task and intelligently execute it using available tools!

## 💡 Example Usage

```python
import asyncio
from pori import Agent, AgentSettings, ToolRegistry, Orchestrator, register_all_tools
from langchain_anthropic import ChatAnthropic

async def main():
    # Set up the framework
    registry = ToolRegistry()
    register_all_tools(registry)
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    orchestrator = Orchestrator(llm=llm, tools_registry=registry)
    
    # Execute a task
    result = await orchestrator.execute_task(
        task="Generate 3 random numbers and calculate their sum",
        agent_settings=AgentSettings(max_steps=10)
    )
    
    # Get the final answer
    agent = result["agent"]
    final_answer = agent.memory.get_final_answer()
    print(f"Answer: {final_answer['final_answer']}")

asyncio.run(main())
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
ANTHROPIC_API_KEY=your_key_here          # Required
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # LLM model to use
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

## 🤝 Human-in-the-Loop (Coming Soon)

Pori is designed with human oversight in mind:

- **Approval Gates**: Ask for permission before executing sensitive tools
- **Intervention Points**: Allow humans to guide the agent's decisions
- **Real-time Monitoring**: Track agent progress and step in when needed
- **Learning from Feedback**: Improve agent behavior based on human input

## 📁 Project Structure

```
pori/
├── agent.py          # Core agent logic
├── orchestrator.py   # Task management and parallel execution  
├── memory.py         # Conversation and state management
├── tools.py          # Tool registry and execution
├── evaluation.py     # Result assessment and retry logic
├── main.py          # Interactive CLI interface
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

## 📄 License

MIT License - feel free to use Pori in your projects!

---

**🌟 Star this repo if you find Pori useful!** 