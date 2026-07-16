"""Pori - A simple, extensible AI agent framework for all ."""

__version__ = "1.4.0"
__author__ = "Aloy Sathekge"
__email__ = "sathekgealoy@gmail.com"

# Main exports from the agent framework
from .agent import Agent, AgentOutput, AgentSettings, AgentState
from .capabilities import (
    CapabilityGroup,
    CapabilityPrerequisites,
    EligibilityReport,
    SkillEligibility,
)

# The clarify transport seam (products render ClarificationRequests as UI and
# resume the paused run through a ClarifyBridge).
from .clarify import ClarificationRequest, ClarifyBridge

# The LLM factory — how an embedding product constructs kernel LLMs.
# Rebuild Config now that TeamConfig is available (deferred forward ref)
from .config import Config as _Config
from .config import LLMConfig, create_llm, get_configured_llm
from .context import (
    ContextDiagnostics,
    ContextEngine,
    ContextWindow,
    DefaultContextEngine,
)
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
from .evolution import (
    EvolutionActivation,
    EvolutionArtifactKind,
    EvolutionEvalCase,
    EvolutionEvalResult,
    EvolutionProposal,
    EvolutionProposalStatus,
    EvolutionRepository,
    EvolutionReviewSummary,
    FileEvolutionRepository,
    run_local_evolution_evals,
)
from .file_backends import (
    FILE_BACKEND_CONTEXT_KEY,
    CompositeFileBackend,
    FileBackend,
    FileEntry,
    FileErrorCode,
    FileListResult,
    FileMount,
    FileMutationResult,
    FileReadResult,
    MemoryFileBackend,
    ReadOnlyFileBackend,
    SandboxFileBackend,
    normalize_virtual_path,
)
from .hitl import (
    ActionRequest,
    ApprovalRequest,
    ApprovalResponse,
    AutoApproveHandler,
    CLIHITLHandler,
    Decision,
    EditedAction,
    HITLConfig,
    HITLHandler,
    InterruptConfig,
    ReviewConfig,
)

# The LLM interface type (products annotate against it; construct via
# create_llm / get_configured_llm above).
from .llm.base import BaseChatModel
from .llm.messages import (
    DocumentBlock,
    ImageBlock,
    SystemMessage,
    TextBlock,
    UserMessage,
)

# The MCP client seam (a product supplies per-run server configs).
from .mcp import McpServerConfig, McpSessionSet
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
from .memory_contracts import (
    ConflictPolicy,
    MemoryCatalog,
    MemoryHit,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRepository,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
    RetrievalEvaluation,
    evaluate_retrieval,
)

# The streaming event contract (products relay PoriEvents as SSE/websockets).
from .observability import (
    RUN_END,
    STEP_START,
    TEXT_DELTA,
    ConsoleTelemetryExporter,
    InMemoryTraceStore,
    PoriEvent,
    Span,
    SpanStatus,
    SpanType,
    TelemetryExporter,
    Trace,
    TraceStore,
)
from .orchestrator import Orchestrator
from .profiles import RunProfile, RunProfileResolutionError
from .providers import (
    PROVIDER_PROFILES,
    ProviderDiagnostic,
    ProviderProfile,
    diagnose_provider,
    get_provider_profile,
    provider_profiles,
)
from .retrieval import RetrievalEvidence, fuse_retrieval
from .runtime import (
    BudgetExceeded,
    BudgetLedger,
    CancellationToken,
    ChildRunRequest,
    ChildRunResult,
    ExecutionBudget,
    ReceiptStatus,
    RunContext,
    ToolExecutionReceipt,
    fail_open,
    stable_fingerprint,
)

# Sandbox provider hooks (a product selects/points the execution backend) +
# thread-dir resolution (a product maps artifact paths back to real files).
from .sandbox import (
    VIRTUAL_PREFIX,
    LocalSandboxProvider,
    SandboxProvider,
    ThreadData,
    create_sandbox_provider,
    get_sandbox_provider,
    get_thread_data,
    get_workspace_data,
    replace_virtual_path,
    set_sandbox_provider,
)
from .sessions import (
    SessionExport,
    SessionMessage,
    SessionRecord,
    SessionRepository,
    SessionSearchHit,
    SQLiteSessionRepository,
)
from .skills import (
    SelectedSkill,
    SkillBundle,
    SkillBundleCatalog,
    SkillCatalog,
    SkillConfigDeclaration,
    SkillFileView,
    SkillIndexEntry,
    SkillInstallPreview,
    SkillInstallResult,
    SkillInvocation,
    SkillLinkedFile,
    SkillManifest,
    SkillReadiness,
    SkillSearchHit,
    SkillSummary,
    inspect_skill_source,
    install_skill_source,
    load_skill_bundles_from_directory,
    load_skill_catalog_from_directories,
    parse_skill_markdown,
    render_selected_skills,
    uninstall_skill_from_directory,
)
from .team import MemberConfig, Team, TeamConfig, TeamMode
from .tools.registry import (
    CapabilityResolutionError,
    CapabilitySnapshot,
    CollisionPolicy,
    ToolExecutor,
    ToolInfo,
    ToolRegistry,
    tool_registry,
)

# Tool registrations + the protected kernel tool set (policy code needs it).
from .tools.standard import STANDARD_KERNEL_TOOLS, register_all_tools

# Prompt-directory override (an embedding application points the loader at its
# own prompts after loading config).
from .utils.prompt_loader import set_prompts_dir

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
    "ConflictPolicy",
    "MemoryCatalog",
    "MemoryHit",
    "MemoryKind",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryRepository",
    "MemoryRetention",
    "MemoryScope",
    "MemorySensitivity",
    "MemoryStatus",
    "RetrievalEvaluation",
    "evaluate_retrieval",
    # Runtime contracts
    "ExecutionBudget",
    "BudgetExceeded",
    "BudgetLedger",
    "CancellationToken",
    "ChildRunRequest",
    "ChildRunResult",
    "ReceiptStatus",
    "RunContext",
    "ToolExecutionReceipt",
    "fail_open",
    "stable_fingerprint",
    "FILE_BACKEND_CONTEXT_KEY",
    "CompositeFileBackend",
    "FileBackend",
    "FileEntry",
    "FileErrorCode",
    "FileListResult",
    "FileMount",
    "FileMutationResult",
    "FileReadResult",
    "MemoryFileBackend",
    "ReadOnlyFileBackend",
    "SandboxFileBackend",
    "normalize_virtual_path",
    "CapabilityGroup",
    "CapabilityPrerequisites",
    "CapabilityResolutionError",
    "CapabilitySnapshot",
    "CollisionPolicy",
    "EligibilityReport",
    "SkillEligibility",
    "ContextDiagnostics",
    "ContextEngine",
    "ContextWindow",
    "DefaultContextEngine",
    "PROVIDER_PROFILES",
    "ProviderDiagnostic",
    "ProviderProfile",
    "diagnose_provider",
    "get_provider_profile",
    "provider_profiles",
    "RunProfile",
    "RunProfileResolutionError",
    "RetrievalEvidence",
    "fuse_retrieval",
    "SQLiteSessionRepository",
    "SessionExport",
    "SessionMessage",
    "SessionRecord",
    "SessionRepository",
    "SessionSearchHit",
    "SelectedSkill",
    "SkillBundle",
    "SkillBundleCatalog",
    "SkillCatalog",
    "SkillConfigDeclaration",
    "SkillFileView",
    "SkillIndexEntry",
    "SkillInstallPreview",
    "SkillInstallResult",
    "SkillInvocation",
    "SkillLinkedFile",
    "SkillManifest",
    "SkillReadiness",
    "SkillSearchHit",
    "SkillSummary",
    "inspect_skill_source",
    "install_skill_source",
    "parse_skill_markdown",
    "load_skill_bundles_from_directory",
    "load_skill_catalog_from_directories",
    "render_selected_skills",
    "uninstall_skill_from_directory",
    # Tools system
    "ToolRegistry",
    "ToolExecutor",
    "ToolInfo",
    "tool_registry",
    # Product integration seams (the kernel front door for embedding products —
    # see docs/codebase-review-2026-07-09.md: these were forced deep imports)
    "ClarificationRequest",
    "ClarifyBridge",
    "LLMConfig",
    "BaseChatModel",
    "create_llm",
    "get_configured_llm",
    "SystemMessage",
    "UserMessage",
    "ImageBlock",
    "DocumentBlock",
    "TextBlock",
    "McpServerConfig",
    "McpSessionSet",
    "PoriEvent",
    "RUN_END",
    "STEP_START",
    "TEXT_DELTA",
    "create_sandbox_provider",
    "get_sandbox_provider",
    "set_sandbox_provider",
    "SandboxProvider",
    "LocalSandboxProvider",
    "ThreadData",
    "VIRTUAL_PREFIX",
    "get_thread_data",
    "get_workspace_data",
    "replace_virtual_path",
    "STANDARD_KERNEL_TOOLS",
    "set_prompts_dir",
    "register_all_tools",
    # Evaluation
    "ActionResult",
    "Evaluator",
    "EvolutionActivation",
    "EvolutionArtifactKind",
    "EvolutionEvalCase",
    "EvolutionEvalResult",
    "FileEvolutionRepository",
    "EvolutionProposal",
    "EvolutionProposalStatus",
    "EvolutionReviewSummary",
    "EvolutionRepository",
    "run_local_evolution_evals",
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
    "ApprovalRequest",
    "ApprovalResponse",
    "ActionRequest",
    "ReviewConfig",
    "Decision",
    "EditedAction",
]
