"""Pori - A simple, extensible AI agent framework for all ."""

__version__ = "1.4.0"
__author__ = "Aloy Sathekge"
__email__ = "sathekgealoy@gmail.com"

# Main exports from the agent framework
from .agent import Agent, AgentOutput, AgentSettings, AgentState

# Rebuild Config now that TeamConfig is available (deferred forward ref)
from .config import Config as _Config
from .eval import (
    AccuracyEval,
    AccuracyResult,
    AgentJudgeEval,
    AgentJudgeResult,
    BaseEval,
    ContentPolicyGuardrail,
    EvalResult,
    FactualityGuardrail,
    PerformanceEval,
    PerformanceResult,
    ReliabilityEval,
    ReliabilityResult,
    TopicGuardrail,
)
from .evaluation import ActionResult, Evaluator
from .hitl import (
    AutoApproveHandler,
    CLIHITLHandler,
    HITLConfig,
    HITLHandler,
    InterruptConfig,
)
from .memory import (
    AgentMemory,
    AgentMessage,
    Block,
    CoreMemory,
    InMemoryMemoryStore,
    MemoryStore,
    SerializableMemoryState,
    SQLiteMemoryStore,
    TaskState,
    ToolCallRecord,
    create_memory_store,
)
from .observability import (
    ConsoleTelemetryExporter,
    InMemoryTraceStore,
    Span,
    SpanStatus,
    SpanType,
    TelemetryExporter,
    Trace,
    TraceStore,
)
from .orchestrator import Orchestrator
from .team import MemberConfig, Team, TeamConfig, TeamMode
from .tools.registry import ToolExecutor, ToolInfo, ToolRegistry, tool_registry

# Tool registrations
from .tools.standard import register_all_tools

_Config.model_rebuild()

__all__ = [
    # Core agent classes
    "Agent",
    "AgentSettings",
    "AgentState",
    "AgentOutput",
    # Memory system
    "AgentMemory",
    "TaskState",
    "ToolCallRecord",
    "AgentMessage",
    "Block",
    "CoreMemory",
    "SerializableMemoryState",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "create_memory_store",
    # Tools system
    "ToolRegistry",
    "ToolExecutor",
    "ToolInfo",
    "tool_registry",
    "register_all_tools",
    # Evaluation
    "ActionResult",
    "Evaluator",
    # Eval framework
    "BaseEval",
    "EvalResult",
    "AccuracyEval",
    "AccuracyResult",
    "ReliabilityEval",
    "ReliabilityResult",
    "PerformanceEval",
    "PerformanceResult",
    "AgentJudgeEval",
    "AgentJudgeResult",
    # Guardrails
    "ContentPolicyGuardrail",
    "FactualityGuardrail",
    "TopicGuardrail",
    # Observability
    "Trace",
    "Span",
    "SpanType",
    "SpanStatus",
    "TraceStore",
    "InMemoryTraceStore",
    "TelemetryExporter",
    "ConsoleTelemetryExporter",
    # Orchestration
    "Orchestrator",
    # Team
    "Team",
    "TeamMode",
    "MemberConfig",
    "TeamConfig",
    # HITL
    "HITLConfig",
    "HITLHandler",
    "CLIHITLHandler",
    "AutoApproveHandler",
    "InterruptConfig",
]
