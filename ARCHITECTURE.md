# Pori Architecture

This document provides a detailed technical overview of Pori's architecture for contributors and developers who want to understand how the framework works internally.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Memory System](#memory-system)
- [Tool System](#tool-system)
- [Agent Execution Loop](#agent-execution-loop)
- [Design Patterns](#design-patterns)
- [Extension Points](#extension-points)

## Overview

Pori follows a modular, layered architecture designed for extensibility and clarity:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│                   (main.py, cli.py, api/)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                       │
│                     (orchestrator.py)                        │
│         • Task management  • Parallel execution              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                             │
│                      (agent.py)                              │
│    • Planning  • Execution  • Reflection  • LLM interface   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                      │
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Memory System │    │  Tool System   │    │  Evaluator   │
│  (memory.py)  │    │   (tools.py)   │    │(evaluation.py)│
└───────────────┘    └────────────────┘    └──────────────┘
```

### Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new tools, memory backends, and LLM providers
3. **Testability**: Components can be tested in isolation
4. **Observability**: Comprehensive logging and state tracking
5. **Type Safety**: Heavy use of Pydantic models for validation

## Core Components

### 1. Agent (`pori/agent.py`)

The Agent is the core reasoning engine that executes tasks step-by-step.

**Key Classes:**
- `Agent`: Main agent class
- `AgentState`: Tracks execution state (steps, failures, plan, reflections)
- `AgentSettings`: Configuration (max_steps, max_failures, retry_delay)
- `AgentOutput`: Structured LLM response
- `PlanOutput`: Planning phase output
- `ReflectOutput`: Reflection phase output

**Responsibilities:**
- Task initialization and execution
- LLM interaction with structured output parsing
- Planning and reflection
- Tool execution coordination
- Memory management
- State tracking

**Key Methods:**
```python
async def run() -> Dict[str, Any]:
    """Main execution loop"""
    
async def step() -> None:
    """Execute one reasoning step"""
    
async def get_next_action() -> AgentOutput:
    """Get next action from LLM"""
    
async def execute_actions(actions: List[Dict]) -> List[ActionResult]:
    """Execute list of actions"""
```

### 2. Orchestrator (`pori/orchestrator.py`)

Manages agent lifecycle and coordinates multiple tasks.

**Responsibilities:**
- Agent creation and initialization
- Task delegation
- Parallel task execution with concurrency control
- Shared memory management
- Monitoring hooks

**Key Features:**
- Semaphore-based concurrency control
- Exception handling for task failures
- Callback system for monitoring

### 3. Memory System (`pori/memory.py`)

Multi-layered memory architecture for agent state and history.

**Components:**
- **Conversation History**: System/user/assistant message tracking
- **Tool Call History**: Complete audit trail of tool executions
- **Task Management**: Track task state across execution
- **Experience Storage**: Store facts with importance levels
- **Semantic Recall**: Keyword-based retrieval (extensible to vector DBs)

**Key Classes:**
```python
class AgentMemory:
    messages: List[AgentMessage]
    tool_call_history: List[ToolCallRecord]
    tasks: Dict[str, TaskState]
    state: Dict[str, Any]
    experiences: List[Dict[str, Any]]
```

### 4. Tool System (`pori/tools.py`)

Registry-based tool management with automatic parameter validation.

**Components:**
- `ToolRegistry`: Central registry for all tools
- `ToolExecutor`: Executes tools with context injection
- `ToolInfo`: Metadata about each tool

**Tool Registration:**
```python
@registry.tool(
    name="tool_name",
    param_model=ToolParams,  # Pydantic model
    description="What the tool does"
)
def tool_function(params: ToolParams, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "result": data}
```

### 5. Evaluation System (`pori/evaluation.py`)

Determines task completion and manages retries.

**Key Classes:**
- `ActionResult`: Result of a single action
- `Evaluator`: Evaluates tool results and task completion

**Responsibilities:**
- Tool result evaluation
- Retry logic with configurable limits
- Task completion detection

## Data Flow

### 1. Task Submission Flow

```
User Input → Orchestrator.execute_task()
    ↓
Create Agent(task, llm, registry, settings, memory)
    ↓
Agent.run() → Main execution loop
```

### 2. Agent Execution Loop

```
┌─────────────────────────────────────────┐
│  1. Planning (if no plan exists)        │
│     - Generate 1-5 step plan             │
│     - Store in agent.state.current_plan │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. Get Next Action                      │
│     - Build message context              │
│     - Inject semantic recall results     │
│     - Call LLM with structured output    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. Execute Actions                      │
│     - Validate parameters with Pydantic  │
│     - Check for duplicates (cache)       │
│     - Execute tools via ToolExecutor     │
│     - Record in memory                   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  4. Reflection                           │
│     - Critique progress vs plan          │
│     - Update plan if needed              │
│     - Store reflection in state          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  5. Evaluation                           │
│     - Check for task completion          │
│     - Create periodic summaries          │
│     - Loop or exit                       │
└─────────────────────────────────────────┘
```

### 3. Tool Execution Flow

```
Action Dict → ToolExecutor.execute_tool()
    ↓
Get ToolInfo from Registry
    ↓
Validate params with Pydantic model
    ↓
Execute tool function with context
    ↓
Return {success: bool, result: Any, error: str?}
```

## Memory System

### Memory Layers

1. **Working Memory (Short-term)**
   - Recent messages (last N turns)
   - Current task context
   - Reset per task

2. **Long-term Memory**
   - Tool call history
   - Task records
   - Experiences with importance weights

3. **Semantic Memory (Optional)**
   - Vector-based recall
   - Keyword matching (default)
   - Extensible to Weaviate, Pinecone, etc.

### Memory Operations

```python
# Message tracking
memory.add_message(role="user", content="...")
memory.get_recent_messages(n=10)

# Tool call tracking
memory.add_tool_call(tool_name, parameters, result, success)

# Task management
memory.create_task(task_id, description)
memory.begin_task(task_id)

# Experience storage
memory.add_experience(text, importance=3, meta={})
memory.recall(query="...", k=5, min_score=0.3)
```

## Tool System

### Tool Lifecycle

1. **Registration**: Tools registered via decorator or explicit call
2. **Discovery**: LLM receives tool descriptions in system prompt
3. **Invocation**: LLM returns structured tool calls
4. **Validation**: Parameters validated via Pydantic
5. **Execution**: Tool function called with validated params + context
6. **Result**: Tool returns structured result dict

### Context Injection

Tools receive a context dict with:
```python
context = {
    "memory": AgentMemory,  # Access to agent memory
    "state": AgentState,    # Current agent state
}
```

### Built-in Tools

Located in `pori/tools_builtin/`:

- **Core Tools** (`core_tools.py`): answer, done, think, remember
- **Math Tools** (`math_tools.py`): Basic arithmetic, fibonacci, factorial
- **Number Tools** (`number_tools.py`): Random generation, sequences
- **Filesystem Tools** (`filesystem_tools.py`): File operations
- **Spotify Tools** (`spotify_tools.py`): Music search and playback

## Agent Execution Loop

### Step Execution

Each step follows this pattern:

```python
async def step(self):
    # 1. Create summary if interval reached
    if self.state.n_steps % self.settings.summary_interval == 0:
        summary = self.memory.create_summary(self.state.n_steps)
        self.memory.add_message("system", f"Memory summary: {summary}")
    
    # 2. Ensure we have a plan
    await self._plan_if_needed()
    
    # 3. Get next action from LLM
    model_output = await self.get_next_action()
    
    # 4. Execute actions
    tool_results = await self.execute_actions(model_output.action)
    
    # 5. Update state
    self.state.n_steps += 1
    
    # 6. Reflect and update plan
    await self._reflect_and_update_plan(tool_results)
```

### Duplicate Detection

Pori caches tool calls within a step to avoid redundant execution:

```python
# Generate signature: tool_name:json(params)
sig = f"{tool_name}:{json.dumps(params, sort_keys=True)}"

# Check if already executed
if sig in seen_signatures_this_step:
    return cached_result

# Execute and cache
result = execute_tool(...)
results_by_signature[sig] = result
```

### Error Handling

- **Tool Failures**: Tracked per tool with retry logic
- **Consecutive Failures**: Agent stops after N consecutive failures
- **Step Limit**: Agent stops after max_steps reached
- **Graceful Degradation**: Partial results returned on failure

## Design Patterns

### 1. Registry Pattern
Used for tools - allows dynamic registration and discovery.

### 2. Strategy Pattern
Memory backends can be swapped (simple keyword vs vector DB).

### 3. Observer Pattern
Orchestrator callbacks for monitoring agent progress.

### 4. Builder Pattern
AgentSettings for flexible agent configuration.

### 5. Template Method
Agent.step() defines execution skeleton, subclasses can override.

## Extension Points

### Adding New Tools

```python
from pydantic import BaseModel, Field
from pori.tools import tool_registry

Registry = tool_registry()

class MyToolParams(BaseModel):
    param1: str = Field(..., description="Parameter description")

@Registry.tool(name="my_tool", description="What it does")
def my_tool(params: MyToolParams, context: Dict[str, Any]) -> Dict[str, Any]:
    # Implementation
    return {"success": True, "result": data}
```

### Custom Memory Backend

Extend `AgentMemory` and override `recall()`:

```python
class VectorMemory(AgentMemory):
    def __init__(self, vector_db_client):
        super().__init__()
        self.vector_db = vector_db_client
    
    def recall(self, query: str, k: int, min_score: float):
        # Use vector DB for semantic search
        results = self.vector_db.search(query, k=k)
        return [(id, text, score) for id, text, score in results if score >= min_score]
```

### Custom Evaluator

Extend `Evaluator` to add custom completion logic:

```python
class CustomEvaluator(Evaluator):
    def evaluate_task_completion(self, task: str, memory: AgentMemory) -> Tuple[bool, str]:
        # Custom logic
        return is_complete, message
```

### Custom LLM Provider

Any LangChain-compatible chat model works:

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatOpenAI(model="gpt-4")
# or
llm = ChatGoogleGenerativeAI(model="gemini-pro")

orchestrator = Orchestrator(llm=llm, tools_registry=registry)
```

## Testing Architecture

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agent.py            # Agent functionality
├── test_memory.py           # Memory system
├── test_tools.py            # Tool registration/execution
├── test_filesystem_tools.py # Filesystem tool specifics
└── ...
```

### Key Fixtures

```python
@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    
@pytest.fixture
def tool_registry():
    """Fresh tool registry"""
    
@pytest.fixture
def agent_memory():
    """Clean memory instance"""
```

## Performance Considerations

1. **Message Truncation**: Keep recent messages under limit to avoid context overflow
2. **Duplicate Detection**: Prevents redundant expensive tool calls
3. **Async Execution**: Non-blocking I/O for tool execution
4. **Parallel Tasks**: Orchestrator supports concurrent task execution
5. **Memory Summarization**: Periodic summaries prevent context bloat

## Security Considerations

1. **Input Validation**: All tool parameters validated via Pydantic
2. **Sandboxing**: Tools should operate in safe contexts
3. **API Key Management**: Use environment variables, never hardcode
4. **Rate Limiting**: Implement for production deployments
5. **Audit Trail**: Complete tool call history for debugging

---

For questions about the architecture, please open a [GitHub Discussion](https://github.com/your-username/pori/discussions).
