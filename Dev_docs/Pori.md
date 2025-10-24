Pori Framework Deep Dive Summary

Pori is a lightweight, extensible AI agent framework for building intelligent agents with memory, tools, and task orchestration capabilities.

Architecture Overview

1. Agent (agent.py) - Core reasoning engine
◦  Executes tasks step-by-step using LLM (Anthropic Claude)
◦  Features planning, reflection, and tool execution
◦  Manages state with AgentState (tracks steps, failures, current plan, reflections)
◦  Uses structured output parsing (AgentOutput, PlanOutput, ReflectOutput)
◦  Implements duplicate detection for tool calls to avoid redundant execution
◦  Follows a workflow: Plan → Execute → Reflect → Update Plan
2. Orchestrator (orchestrator.py) - Task management layer
◦  Creates and manages agent lifecycle
◦  Supports parallel task execution with configurable concurrency
◦  Maintains shared memory across agents
◦  Provides monitoring hooks (on_step_start, on_step_end)
3. Memory System (memory.py) - Multi-layered memory
◦  Conversation history (messages) - tracks system/user/assistant messages
◦  Tool call tracking (tool_call_history) - complete audit trail
◦  Task management (tasks) - tracks task state (in_progress/completed/failed)
◦  Experience storage - stores facts with importance levels
◦  Semantic recall - simple keyword-based retrieval (can be extended with vector DBs)
◦  Summarization - periodic summaries to avoid context overflow
4. Tools System (tools.py + tools_builtin/) - Extensible tool registry
◦  Decorator-based registration with Pydantic validation
◦  Built-in tools: core (answer, done, think, remember), math, number generation, Spotify, filesystem
◦  Tools receive context (memory, state) and return structured results
◦  ToolExecutor handles execution and error management
5. Evaluation (evaluation.py) - Task completion assessment
◦  Tracks action results and retry counts
◦  Determines task completion by checking for final answers
◦  Implements retry logic with configurable max attempts
6. Prompt System - Template-based prompts
◦  System prompt (prompts/system/agent_core.md) defines workflow
◦  Strict 3-step workflow: Gather Info → Answer → Done
◦  JSON output format with current_state and actions

Key Features

•  Planning & Reflection: Agent creates 1-5 step plans and updates them based on progress
•  Duplicate Detection: Caches identical tool calls to avoid redundant execution
•  Memory Integration: Semantic recall injects relevant past experiences into context
•  Configurable Settings: AgentSettings controls max_steps, max_failures, summary_interval
•  Logging: Structured logging with task_id and step tracking
•  API Layer: FastAPI server for async task execution (in api/ folder)

Execution Flow

1. User provides a task
2. Agent creates a plan
3. For each step:
◦  Get next action from LLM (structured output)
◦  Execute tools with duplicate detection
◦  Record results in memory
◦  Reflect and update plan if needed
◦  Create periodic summaries
4. Agent calls answer tool with final response
5. Agent calls done tool to complete task