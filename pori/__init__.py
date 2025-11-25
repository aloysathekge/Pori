"""Pori - A simple, extensible AI agent framework."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main exports from the agent framework
from .agent import Agent, AgentSettings, AgentState, AgentOutput
from .memory import AgentMemory, SimpleMemory, TaskState, ToolCallRecord, AgentMessage
from .tools.registry import ToolRegistry, ToolExecutor, ToolInfo, tool_registry
from .evaluation import ActionResult, Evaluator
from .orchestrator import Orchestrator

# Tool registrations
from .tools.standard import register_all_tools

__all__ = [
    # Core agent classes
    "Agent",
    "AgentSettings",
    "AgentState",
    "AgentOutput",
    # Memory system
    "AgentMemory",
    "SimpleMemory",  # Backwards compatibility
    "TaskState",
    "ToolCallRecord",
    "AgentMessage",
    # Tools system
    "ToolRegistry",
    "ToolExecutor",
    "ToolInfo",
    "tool_registry",
    "register_all_tools",
    # Evaluation
    "ActionResult",
    "Evaluator",
    # Orchestration
    "Orchestrator",
]
