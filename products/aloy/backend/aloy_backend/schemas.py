from __future__ import annotations

from datetime import datetime
from typing import Literal

from pori import get_provider_profile
from pydantic import BaseModel, Field, field_validator

from .tenancy import OrganizationPolicy

# --- Organizations ---


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    role: str
    policy: OrganizationPolicy
    created_at: datetime


class OrganizationPolicyUpdate(BaseModel):
    policy: OrganizationPolicy


class MembershipCreate(BaseModel):
    user_id: str = Field(min_length=1)
    role: Literal["viewer", "member", "admin"] = "member"


class MembershipUpdate(BaseModel):
    role: Literal["viewer", "member", "admin"] | None = None
    status: Literal["active", "suspended"] | None = None


class MembershipResponse(BaseModel):
    id: str
    organization_id: str
    user_id: str
    role: str
    status: str
    created_at: datetime


# --- Runs ---


class RunRequest(BaseModel):
    task: str = Field(..., description="User task/prompt to execute")
    max_steps: int = Field(15, ge=1, le=10_000)


class ChildRunCreate(BaseModel):
    task: str = Field(..., min_length=1, max_length=100_000)
    agent_id: str = Field("child_agent", min_length=1)
    max_steps: int = Field(5, ge=1, le=10_000)
    idempotency_key: str | None = Field(default=None, max_length=200)


class RunResponse(BaseModel):
    id: str
    organization_id: str
    agent_id: str
    session_id: str
    status: str
    max_steps: int
    success: bool
    steps_taken: int
    final_answer: str | None = None
    reasoning: str | None = None
    metrics: dict | None = None
    prompt_fingerprint: str | None = None
    tool_surface_fingerprint: str | None = None
    execution_receipts: list[dict] | None = None
    selected_skills: list[str] | None = None
    artifacts: list[dict] | None = None
    plan: list[dict] | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None
    idempotency_key: str | None = None
    child_depth: int = 0
    attempt_count: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 900
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested: bool = False
    isolation_profile: str
    created_at: datetime


# --- Conversations ---


class ConversationCreate(BaseModel):
    title: str | None = None
    agent_config_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: str
    title: str | None
    agent_config_id: str | None
    parent_conversation_id: str | None = None
    branched_from_message_id: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(BaseModel):
    id: str
    title: str | None
    agent_config_id: str | None
    parent_conversation_id: str | None = None
    branched_from_message_id: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]


class ConversationBranchRequest(BaseModel):
    through_message_id: str | None = None
    title: str | None = None


class ConversationSearchHit(BaseModel):
    conversation_id: str
    message_id: str
    role: str
    content: str
    score: float
    created_at: datetime


class ContextSearchHit(BaseModel):
    source_type: Literal["session", "memory"]
    source_id: str
    session_id: str | None = None
    content: str
    score: float
    provenance: dict


class ConversationExportResponse(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse]
    context_artifacts: list[dict]
    exported_at: datetime


# --- Messages ---


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict | None = None
    created_at: datetime


class ImageAttachment(BaseModel):
    """One user-attached image, inline base64 (v1: no media store)."""

    data: str = Field(..., min_length=1, max_length=7_000_000)  # ~5MB decoded
    media_type: str = Field(..., pattern="^image/(png|jpeg|gif|webp)$")


class FileAttachment(BaseModel):
    """One user-attached text file (code, markdown, csv, …), inline."""

    name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., max_length=200_000)  # ~200KB of text


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)
    max_steps: int = Field(15, ge=1, le=10_000)
    stream: bool = False
    team_id: str | None = None  # If set, route through a multi-agent team
    # Multimodal turn: up to 3 inline images ride with the message.
    images: list[ImageAttachment] = Field(default_factory=list, max_length=3)
    # Text-file attachments: content is embedded into the task for the model.
    files: list[FileAttachment] = Field(default_factory=list, max_length=3)


# --- Agent Configs ---


# --- User Profiles ---


class UserProfileResponse(BaseModel):
    id: str
    display_name: str | None
    avatar_url: str | None
    default_agent_config_id: str | None
    preferences: dict | None
    created_at: datetime
    updated_at: datetime


class UserProfileUpdate(BaseModel):
    display_name: str | None = None
    avatar_url: str | None = None
    default_agent_config_id: str | None = None
    preferences: dict | None = None


class UsageStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    total_runs: int
    member_since: datetime


# --- Agent Configs ---


# --- Team Configs ---


class MemberConfigSchema(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    llm_config: dict | None = None  # {"provider": "...", "model": "...", ...}
    agent_settings: dict | None = None  # {"max_steps": 10, ...}
    tools: list[str] | None = None


class TeamConfigCreate(BaseModel):
    name: str = Field(..., min_length=1)
    mode: str = Field("router", pattern="^(router|broadcast|delegate)$")
    members: list[MemberConfigSchema] = Field(..., min_length=1)
    max_delegation_steps: int = Field(10, ge=1, le=100)
    max_concurrent_members: int = Field(5, ge=1, le=20)


class TeamConfigUpdate(BaseModel):
    name: str | None = None
    mode: str | None = Field(None, pattern="^(router|broadcast|delegate)$")
    members: list[MemberConfigSchema] | None = None
    max_delegation_steps: int | None = Field(None, ge=1, le=100)
    max_concurrent_members: int | None = Field(None, ge=1, le=20)


class TeamConfigResponse(BaseModel):
    id: str
    name: str
    mode: str
    members: list[dict]
    max_delegation_steps: int
    max_concurrent_members: int
    created_at: datetime


class TeamRunRequest(BaseModel):
    task: str = Field(..., min_length=1)


class TeamRunResponse(BaseModel):
    task: str
    completed: bool
    steps_taken: int
    final_answer: str | None = None
    mode: str
    metrics: dict | None = None


# --- Skills ---


class SkillCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
    version: str = Field("1", min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=500)
    instructions: str = Field(min_length=1, max_length=50_000)
    tags: list[str] = []
    category: str = "organization"
    author: str = ""
    license: str = ""
    commands: list[str] = []
    argument_hint: str = ""
    provenance: str = "organization"
    trust_level: str = "organization"
    required_commands: list[str] = []
    setup_help: str = ""
    source_url: str = ""
    install_command: str = ""
    readiness_warnings: list[str] = []
    required_tools: list[str] = []
    required_credentials: list[str] = []
    required_platforms: list[str] = []
    required_model_capabilities: list[str] = []
    sensitivity: str = "internal"


class SkillUpdate(BaseModel):
    name: str | None = None
    summary: str | None = None
    instructions: str | None = Field(default=None, min_length=1, max_length=50_000)
    tags: list[str] | None = None
    category: str | None = None
    author: str | None = None
    license: str | None = None
    commands: list[str] | None = None
    argument_hint: str | None = None
    provenance: str | None = None
    trust_level: str | None = None
    required_commands: list[str] | None = None
    setup_help: str | None = None
    source_url: str | None = None
    install_command: str | None = None
    readiness_warnings: list[str] | None = None
    required_tools: list[str] | None = None
    required_credentials: list[str] | None = None
    required_platforms: list[str] | None = None
    required_model_capabilities: list[str] | None = None
    sensitivity: str | None = None
    status: Literal["draft", "approved", "disabled"] | None = None


class SkillGrantCreate(BaseModel):
    principal_type: Literal["role", "user"]
    principal_id: str = Field(min_length=1)


class SkillResponse(BaseModel):
    id: str
    organization_id: str
    slug: str
    version: str
    name: str
    summary: str
    instructions: str
    tags: list[str]
    category: str
    author: str
    license: str
    commands: list[str]
    argument_hint: str
    provenance: str
    trust_level: str
    required_commands: list[str]
    setup_help: str
    source_url: str
    install_command: str
    readiness_warnings: list[str]
    required_tools: list[str]
    required_credentials: list[str]
    required_platforms: list[str]
    required_model_capabilities: list[str]
    sensitivity: str
    status: str
    created_at: datetime
    updated_at: datetime


class SkillGrantResponse(BaseModel):
    id: str
    skill_id: str
    principal_type: str
    principal_id: str
    created_at: datetime


# --- Evolution ---


class EvolutionEvalCase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    input: str = Field(min_length=1)
    expected: str | None = None
    criteria: str = Field(min_length=1, max_length=1000)


class EvolutionEvalResult(BaseModel):
    case_name: str = Field(min_length=1, max_length=120)
    passed: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=2000)
    evaluator: str = Field(default="manual", min_length=1, max_length=120)


class EvolutionProposalCreate(BaseModel):
    artifact_kind: Literal["skill", "prompt", "policy", "eval", "code", "config"]
    target: str = Field(min_length=1, max_length=240)
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=4000)
    proposed_version: str = Field(min_length=1, max_length=80)
    current_version: str | None = Field(default=None, max_length=80)
    proposed_content: str = Field(min_length=1)
    eval_cases: list[EvolutionEvalCase] = Field(min_length=1)
    supersedes_proposal_id: str | None = None


class EvolutionEvalRecord(BaseModel):
    results: list[EvolutionEvalResult] = Field(min_length=1)


class EvolutionProposalResponse(BaseModel):
    id: str
    organization_id: str
    artifact_kind: str
    target: str
    title: str
    summary: str
    rationale: str
    current_version: str | None
    proposed_version: str
    proposed_content: str
    eval_cases: list[dict]
    eval_results: list[dict]
    status: str
    approved_by: str | None
    activated_at: datetime | None
    supersedes_proposal_id: str | None
    created_at: datetime
    updated_at: datetime


class EvolutionActivationResponse(BaseModel):
    id: str
    organization_id: str
    target: str
    proposal_id: str
    version: str
    activated_by: str
    activated_at: datetime
    rolled_back_at: datetime | None = None


# --- Memory ---


class CoreMemoryBlockResponse(BaseModel):
    label: str
    value: str
    limit: int
    read_only: bool


class CoreMemoryResponse(BaseModel):
    blocks: list[CoreMemoryBlockResponse]


class CoreMemoryBlockUpdate(BaseModel):
    value: str


class KnowledgeEntryResponse(BaseModel):
    id: str
    organization_id: str
    user_id: str
    agent_id: str | None = None
    session_id: str | None = None
    content: str
    tags: list[str] | None = None
    importance: int = 1
    kind: Literal["semantic", "episodic", "procedural"] = "semantic"
    confidence: float = 1.0
    sensitivity: Literal["public", "internal", "confidential", "restricted"] = (
        "internal"
    )
    source: str = "agent"
    provenance: dict = Field(default_factory=dict)
    retention: dict = Field(default_factory=dict)
    conflict_key: str | None = None
    status: str = "active"
    superseded_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    event_at: str | None = None


class KnowledgeEntryCreate(BaseModel):
    content: str = Field(..., min_length=1)
    agent_id: str | None = None
    session_id: str | None = None
    tags: list[str] | None = None
    importance: int = Field(1, ge=1, le=5)
    kind: Literal["semantic", "episodic", "procedural"] = "semantic"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    sensitivity: Literal["public", "internal", "confidential", "restricted"] = (
        "internal"
    )
    source: str = "user"
    source_id: str | None = None
    conversation_id: str | None = None
    run_id: str | None = None
    retention: dict = Field(default_factory=dict)
    conflict_key: str | None = None
    conflict_policy: Literal["keep_both", "reject", "supersede"] = "keep_both"
    event_at: datetime | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=100)
    agent_id: str | None = None
    session_id: str | None = None
    kinds: list[Literal["semantic", "episodic", "procedural"]] | None = None
    tags: list[str] | None = None
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class MemoryExportResponse(BaseModel):
    records: list[KnowledgeEntryResponse]
    exported_at: datetime


# Keep old names as aliases for backward compatibility
ArchivalPassageResponse = KnowledgeEntryResponse
ArchivalSearchRequest = KnowledgeSearchRequest


# --- Usage & Billing ---


class UsageRecordResponse(BaseModel):
    id: str
    run_id: str | None
    conversation_id: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    created_at: datetime


class UsageSummaryResponse(BaseModel):
    total_tokens: int
    total_cost: float
    total_requests: int
    by_model: dict  # {model: {tokens, cost, requests}}


class DailyUsageResponse(BaseModel):
    date: str
    tokens: int
    cost: float
    requests: int


# --- Traces ---


class TraceResponse(BaseModel):
    id: str
    run_id: str | None
    conversation_id: str | None
    trace_data: dict
    duration_seconds: float
    total_spans: int
    status: str
    created_at: datetime


class TraceListResponse(BaseModel):
    id: str
    run_id: str | None
    conversation_id: str | None
    duration_seconds: float
    total_spans: int
    status: str
    created_at: datetime


# --- Agent Configs ---


class AgentConfigCreate(BaseModel):
    name: str = Field(..., min_length=1)
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_steps: int = Field(15, ge=1, le=10_000)
    system_prompt: str | None = None
    tools: list[str] | None = None
    is_default: bool = False

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return get_provider_profile(value).name


class AgentConfigUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_steps: int | None = Field(None, ge=1, le=10_000)
    system_prompt: str | None = None
    tools: list[str] | None = None
    is_default: bool | None = None

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        return get_provider_profile(value).name if value is not None else None


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    provider: str
    model: str
    temperature: float
    max_steps: int
    system_prompt: str | None
    tools: list[str] | None
    is_default: bool
    created_at: datetime


# --- Cron jobs (recurring runs, marathon Phase 3) ---


class CronJobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    task: str = Field(min_length=1, max_length=100_000)
    # 5-field cron expression or "@every:SECONDS" (validated in the route)
    schedule: str = Field(min_length=1, max_length=120)
    max_steps: int = Field(15, ge=1, le=10_000)
    conversation_id: str | None = None


class CronJobUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    task: str | None = Field(None, min_length=1, max_length=100_000)
    schedule: str | None = Field(None, min_length=1, max_length=120)
    max_steps: int | None = Field(None, ge=1, le=10_000)
    conversation_id: str | None = None
    enabled: bool | None = None


class CronJobResponse(BaseModel):
    id: str
    name: str
    task: str
    schedule: str
    enabled: bool
    max_steps: int
    conversation_id: str | None
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_run_id: str | None
    created_at: datetime
    updated_at: datetime


# --- Gateway (external chat pairing) ---


class GatewayPairResponse(BaseModel):
    code: str
    platform: str
    expires_at: datetime


class GatewayLinkResponse(BaseModel):
    id: str
    platform: str
    chat_id: str
    chat_title: str | None
    conversation_id: str | None
    created_at: datetime


# --- Run event log (read-only replay) ---


class RunEventLogResponse(BaseModel):
    run_id: str
    conversation_id: str | None
    events: list[dict]
    event_count: int
    created_at: datetime


# --- Account connections (native OAuth) ---


class ProviderInfo(BaseModel):
    provider: str
    label: str
    description: str
    # The member's own (user-scoped) connection.
    connected: bool
    status: str | None = None
    account_email: str | None = None
    # The organization's shared (org-scoped) connection.
    org_connected: bool = False
    org_status: str | None = None
    org_account_email: str | None = None
    can_manage_org: bool = False


class ConnectionStartResponse(BaseModel):
    authorize_url: str


class ConnectionResponse(BaseModel):
    provider: str
    connected: bool


class McpServerCreate(BaseModel):
    name: str
    url: str
    transport: str = "http"  # "http" | "sse"
    auth_kind: str = "none"  # "none" | "static"
    static_secret: str | None = None  # raw Bearer (encrypted server-side)
    tools_include: list[str] | None = None
    tools_exclude: list[str] = []
    scope: str = "user"  # "user" | "org"


class McpServerUpdate(BaseModel):
    enabled: bool


class McpServerInfo(BaseModel):
    id: str
    name: str
    url: str
    transport: str
    auth_kind: str
    scope: str
    enabled: bool
    account_managed: bool  # can the caller manage this row?
