"""Pori - A simple, extensible AI agent framework for all ."""

__version__ = "1.0"
__author__ = "Aloy Sathekge"
__email__ = "sathekgealoy@gmail.com"

# Main exports from the agent framework
from .agent import Agent, AgentSettings, AgentState, AgentOutput
from .memory import (
    AgentMemory,
    SimpleMemory,
    TaskState,
    ToolCallRecord,
    AgentMessage,
    Block,
    CoreMemory,
)
from .tools.registry import ToolRegistry, ToolExecutor, ToolInfo, tool_registry
from .evaluation import ActionResult, Evaluator
from .orchestrator import Orchestrator
from .hitl import (
    HITLConfig,
    HITLHandler,
    CLIHITLHandler,
    AutoApproveHandler,
    InterruptConfig,
)

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
    "Block",
    "CoreMemory",
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
    # HITL
    "HITLConfig",
    "HITLHandler",
    "CLIHITLHandler",
    "AutoApproveHandler",
    "InterruptConfig",
]
