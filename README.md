# Simple Agent Framework

A domain-agnostic agent framework inspired by browser-use architecture, built with LangChain and Anthropic's Claude.

## Setup

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env_template.txt .env
   ```
   
   Then edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

4. **Run the example:**
   ```bash
   python main.py
   ```

### Option 2: Using pip + venv

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env_template.txt .env
   ```
   
   Then edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

4. **Run the example:**
   ```bash
   python main.py
   ```

## Configuration

The agent can be configured through environment variables:

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `ANTHROPIC_MODEL`: Model to use (default: claude-3-5-sonnet-20241022)
- `MAX_STEPS`: Maximum steps per task (default: 50)
- `MAX_FAILURES`: Maximum consecutive failures (default: 3)
- `RETRY_DELAY`: Delay between retries in seconds (default: 2)

## Architecture

- **Agent**: Main reasoning and execution engine
- **Memory**: Conversation history and task state management
- **Tools**: Extensible tool registry and execution system
- **Evaluation**: Result assessment and retry logic
- **Orchestrator**: High-level task management and coordination

## Adding Custom Tools

1. Define parameter model:
   ```python
   class MyToolParams(BaseModel):
       param1: str = Field(..., description="Description")
   ```

2. Define tool function:
   ```python
   def my_tool(params: MyToolParams, context: dict):
       # Your tool logic here
       return result
   ```

3. Register the tool:
   ```python
   registry.register_tool(
       name="my_tool",
       param_model=MyToolParams,
       function=my_tool,
       description="Tool description"
   )
   ``` 