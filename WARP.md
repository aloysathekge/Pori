# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

**Pori** is a lightweight AI agent framework built on LangChain and Anthropic's Claude models. It provides intelligent agents with memory, tool integration, task orchestration, and planning capabilities for complex task execution.

The framework follows a modular architecture with clear separation between:
- Core agent reasoning (planning, execution, reflection)
- Memory management (working memory, long-term persistence, vector storage) 
- Tool system (registry-based with Pydantic validation)
- Task orchestration (single/parallel execution)
- API layer (FastAPI-based REST interface)

## Development Commands

### Environment Setup

```powershell
# Install dependencies using uv (recommended)
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt

# Create environment file with required API key
echo "ANTHROPIC_API_KEY=your_actual_api_key_here" > .env
echo "ANTHROPIC_MODEL=claude-3-5-sonnet-20241022" >> .env
```

### Running the Application

```powershell
# Interactive CLI mode (main interface)
python pori\main.py

# Using module syntax
python -m pori.cli

# Basic usage example
python examples\basic_usage.py

# FastAPI server (development mode)
uvicorn pori.api:app --reload --host 0.0.0.0 --port 8000

# Production server
python -m uvicorn pori.api:app --workers 4
```

### Testing

```powershell
# Run all tests
pytest

# Run tests by category (using markers)
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests  
pytest -m memory         # Memory subsystem tests
pytest -m tools          # Tool system tests
pytest -m agent          # Agent functionality tests

# Coverage reporting
pytest --cov=pori --cov-report=html

# Run specific test file or method
pytest tests\test_agent.py
pytest tests\test_memory.py::TestEnhancedMemory::test_recall
```

### Code Quality

```powershell
# Format code with Black (line length: 88)
black pori\ tests\

# Sort imports with isort
isort pori\ tests\

# Type checking with mypy
mypy pori\

# All formatting in one command
black pori\ tests\ && isort pori\ tests\
```

## Architecture Deep Dive

### Core Components

**Agent (`pori/agent.py`)**
- Main reasoning engine that executes tasks step-by-step
- Implements planning-execution-reflection cycle
- Features: structured output parsing, failure recovery, memory integration
- Uses system prompts from `pori/prompts/system/agent_core.md`
- Handles max_steps, consecutive failure limits, and automatic summarization

**Orchestrator (`pori/orchestrator.py`)**
- Manages agent lifecycle and task delegation
- Supports both single task execution and parallel task execution
- Creates agents with shared memory instances
- Provides monitoring hooks (`on_step_start`, `on_step_end`) for progress tracking
- Maintains registry of active agents and running tasks

**Memory System (`pori/memory.py`, `pori/simple_memory.py`)**
- Multi-layered architecture: working memory (conversation buffer) + long-term persistence
- Working memory resets per task; long-term memory persists across sessions
- Optional vector storage for semantic recall using configurable backends
- Automatic summarization every N steps to prevent context overflow
- Tool call history with complete audit trail

**Tool System (`pori/tools.py`)**
- Registry-based tool management with decorator syntax
- Automatic parameter validation via Pydantic models
- Tools receive context including memory reference and agent state
- Built-in tools in `pori/tools_builtin/`: core (answer, done, think, remember), math, number operations
- Easy extensibility for domain-specific tools

**API Layer (`pori/api/`)**
- FastAPI-based REST API with async support
- Background task execution using asyncio.create_task
- Request tracking middleware with correlation IDs
- Structured response models and error handling
- Health check endpoint for deployment monitoring

### Agent Execution Flow

1. **Initialization**: Create agent with task, load system prompt, setup memory
2. **Planning**: Generate 1-5 step plan mapped to available tools
3. **Execution Loop**:
   - Get next action from LLM (structured output)
   - Execute tool calls with duplicate detection
   - Update memory and state
   - Handle failures with retry logic
4. **Reflection**: Critique progress and update plans if needed
5. **Memory Management**: Automatic summarization every 5 steps
6. **Completion**: Use evaluator to determine task completion

### Tool Integration Pattern

```python
@Registry.tool(name="tool_name", description="Tool description")
def tool_function(params: ParamsModel, context: Dict[str, Any]) -> Dict[str, Any]:
    # Access memory: context["memory"]
    # Access agent state: context.get("agent_state")
    # Return structured result
    return {"success": True, "result": "..."}
```

### Memory Architecture

- **Working Memory**: Short-term conversation buffer (resets per task)
- **Long-Term Memory**: Persistent experiences and summaries
- **Vector Memory**: Semantic search capabilities (optional)
- **Tool Call History**: Complete audit trail with success/failure tracking
- **State Management**: Task-scoped state for final answers and completion status

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY`: Required Claude API key
- `ANTHROPIC_MODEL`: Model selection (default: claude-3-5-sonnet-20241022)
- `MAX_STEPS`: Maximum steps per task (default: 50)
- `MAX_FAILURES`: Consecutive failure limit (default: 3)
- `RETRY_DELAY`: Delay between retries in seconds (default: 2)

### Agent Settings (Per-Task)
```python
AgentSettings(
    max_steps=20,           # Override global step limit
    max_failures=3,         # Max consecutive failures
    summary_interval=5,     # Memory summarization frequency  
    validate_output=False   # Enable output validation
)
```

## Key Files for Understanding

### Core Implementation
- `pori/agent.py` - Main agent with planning/execution logic
- `pori/orchestrator.py` - Task management and parallel execution
- `pori/memory.py` - Memory system interface
- `pori/simple_memory.py` - Basic memory implementation  
- `pori/tools.py` - Tool registry and execution framework
- `pori/evaluation.py` - Result assessment and retry logic

### API and Interface
- `pori/api/__init__.py` - FastAPI application factory
- `pori/api/models.py` - Request/response schemas
- `pori/main.py` - Interactive CLI interface
- `pori/cli.py` - Command-line entry points

### Tool System
- `pori/tools_builtin/core_tools.py` - Essential tools (answer, done, think, remember)
- `pori/tools_builtin/math_tools.py` - Mathematical operations
- `pori/tools_builtin/number_tools.py` - Number generation and statistics

### Utilities and Configuration
- `pori/utils/logging_config.py` - Structured logging setup
- `pori/utils/prompt_loader.py` - System prompt management
- `pori/utils/context.py` - Context utilities

## Development Patterns

### Adding Custom Tools
1. Create Pydantic parameter model with field descriptions
2. Implement tool function that accepts `params` and `context`
3. Register using `@Registry.tool()` decorator
4. Tool automatically appears in agent's available tools

### Memory Usage
- Use `context["memory"].add_experience()` for long-term facts
- Access conversation history via `get_recent_messages_structured()`
- Store task results using `memory.update_state()`

### Error Handling
- Tools should return `{"success": bool, "result": Any, "error": str}` format
- Agent has built-in retry logic for failed tool calls
- Consecutive failure limits prevent infinite loops

### Testing Patterns
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.memory`, etc.
- Mock LLM responses for predictable testing
- Use `async_context` fixture for async test functions
- Test tool registry and execution separately from agent logic

## API Usage

### REST Endpoints
- `POST /v1/tasks` - Submit new agent task
- `GET /v1/tasks/{task_id}` - Get task status and metadata  
- `GET /v1/tasks/{task_id}/result` - Retrieve final results
- `DELETE /v1/tasks/{task_id}` - Cancel running task
- `GET /v1/tools` - List available tools
- `GET /v1/health` - Health check for monitoring

### Local Development Server
```powershell
# Start development server with auto-reload
uvicorn pori.api:app --reload

# Test with curl
curl -X POST "http://localhost:8000/v1/tasks" \
     -H "Content-Type: application/json" \
     -d '{"task": "Calculate the fibonacci of 10"}'
```

## Project Structure Context

```
pori/
├── agent.py              # Core agent reasoning engine
├── orchestrator.py       # Task management and coordination
├── memory.py             # Memory system interface  
├── simple_memory.py      # Basic memory implementation
├── tools.py              # Tool registry and execution
├── evaluation.py         # Result assessment logic
├── main.py              # Interactive CLI
├── cli.py               # Command-line utilities
├── api/                 # FastAPI REST interface
│   ├── __init__.py      # App factory and configuration
│   ├── models.py        # Pydantic request/response schemas
│   ├── deps.py          # Dependency injection helpers
│   ├── middleware.py    # Request tracking middleware
│   └── routers/         # API route handlers
├── tools_builtin/       # Built-in tool implementations
│   ├── core_tools.py    # Essential agent tools
│   ├── math_tools.py    # Mathematical operations
│   └── number_tools.py  # Number generation/stats
└── utils/               # Shared utilities
    ├── logging_config.py  # Structured logging
    ├── prompt_loader.py   # System prompt management
    └── context.py         # Context utilities
```

The codebase is designed for extensibility and maintainability, with clear separation of concerns and comprehensive test coverage.