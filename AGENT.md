

## Overview

**Pori** is a lightweight AI agent framework  with  support for Anthropic Claude, OpenAI, Google Gemini, Oss models. It provides intelligent agents with memory, tool integration, task orchestration, and planning capabilities.

## Development Commands

### Setup and Installation
```powershell
# Install dependencies using uv (recommended)
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### Environment Configuration
```powershell
# Create .env file with the API key(s) for your chosen provider(s)

# Anthropic (Claude)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "ANTHROPIC_MODEL=claude-3-5-sonnet-20241022" >> .env

# OpenAI (GPT)
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "OPENAI_MODEL=gpt-4o-mini" >> .env

# Google (Gemini)
echo "GOOGLE_API_KEY=your_key_here" >> .env

# You can configure multiple providers and switch between them in code
```

### Running the Application

**Interactive CLI Mode:**
```powershell
 # Run as a module from the project root to avoid relative import errors
 python -m pori.main
 # Or
 python -m pori.cli

 # Using uv
 uv run -m pori.main
 uv run -m pori.cli
```

**Basic Usage Example:**
```powershell
python examples/basic_usage.py
```

**FastAPI Server:**
```powershell
uvicorn pori.api:app --host 0.0.0.0 --port 8000 --reload
# Or
python -m uvicorn pori.api:app --reload
```

### Testing
```powershell
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m memory
pytest -m tools
pytest -m agent

# Run with coverage
pytest --cov=pori --cov-report=html

# Run specific test file
pytest tests/test_agent.py
pytest tests/test_memory.py::TestEnhancedMemory::test_recall
```

### Code Quality
```powershell
# Format code
black pori/ tests/
isort pori/ tests/

# Type checking
mypy pori/
```

## Architecture

### Core Components

**Agent (pori/agent.py)**: The main reasoning engine that executes tasks step-by-step. Features planning, reflection, tool execution, and memory integration. Uses structured output parsing and handles failures gracefully.

**Orchestrator (pori/orchestrator.py)**: Manages agent lifecycle, task delegation, and parallel execution. Creates agents with shared memory and provides monitoring hooks.

**Memory System (pori/memory/)**: Multi-layered memory architecture with working memory (short-term), long-term persistence, and optional vector storage for semantic recall. Supports automatic summarization.

**Tool System (pori/tools.py)**: Registry-based tool management with automatic parameter validation via Pydantic. Tools receive context including memory and agent state.

**API Layer (pori/api/)**: FastAPI-based REST API with middleware for request tracking, async task execution, and structured responses.

### Memory Architecture

- **Working Memory**: Short-term conversation buffer that resets per task
- **Long-Term Memory**: Persistent storage for experiences and summaries  
- **Vector Memory**: Semantic search using configurable backends (Weaviate, local embeddings)
- **Tool Call History**: Complete audit trail of tool executions with success/failure tracking

### Tool Integration

Tools are registered using decorators with Pydantic parameter models:

```python
@Registry.tool(name="tool_name", description="Tool description")
def tool_function(params: ParamsModel, context: Dict[str, Any]) -> Dict[str, Any]:
    # Tool implementation
    return {"success": True, "result": "..."}
```

### Agent Flow

1. **Planning**: Creates 1-5 step plans mapped to tool calls
2. **Execution**: Executes actions with duplicate detection and caching  
3. **Reflection**: Critiques progress and updates plans if needed
4. **Memory Management**: Automatic summarization every 5 steps
5. **Completion**: Uses evaluator to determine task completion

### Configuration

Environment variables control agent behavior:
- **Provider API keys**: Configure one or more LLM providers:
  - `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`
  - `OPENAI_API_KEY` + `OPENAI_MODEL`
  - `GOOGLE_API_KEY`
  - Or any other LangChain-compatible provider
- `MAX_STEPS`: Maximum steps per task (default: 50)
- `MAX_FAILURES`: Consecutive failure limit (default: 3)

AgentSettings provide per-task configuration:
- `max_steps`: Override global step limit
- `summary_interval`: Memory summarization frequency
- `validate_output`: Enable output validation

## Key Files to Understand

- **pori/agent.py**: Core agent implementation with planning and execution logic
- **pori/orchestrator.py**: Task management and parallel execution
- **pori/memory_v2/enhanced_memory.py**: Multi-layered memory system
- **pori/tools.py**: Tool registry and execution framework
- **pori/tools_builtin/**: Built-in tools (core, math, number operations)
- **pori/api/**: FastAPI server implementation
- **pori/main.py**: Interactive CLI interface

## Tool Development

Built-in tool categories:
- **Core Tools**: answer, done, think, remember
- **Math Tools**: Basic arithmetic, advanced calculations
- **Number Tools**: Random generation, sequences, statistics

To add custom tools, create parameter models and register functions following the existing patterns in `pori/tools_builtin/`.