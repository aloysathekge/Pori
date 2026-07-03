from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, UniqueConstraint
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Organization(SQLModel, table=True):
    __tablename__ = "organizations"

    id: str = Field(default_factory=lambda: f"org_{uuid.uuid4().hex}", primary_key=True)
    name: str
    slug: str = Field(unique=True, index=True)
    created_by: str = Field(index=True)
    policy: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class OrganizationMembership(SQLModel, table=True):
    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_org_membership"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    role: str = Field(default="member", index=True)
    status: str = Field(default="active", index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    title: str | None = None
    agent_config_id: str | None = None
    parent_conversation_id: str | None = Field(default=None, index=True)
    branched_from_message_id: str | None = None
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    conversation_id: str = Field(index=True)
    role: str
    content: str
    metadata_: dict | None = Field(default=None, sa_column=Column("metadata", JSON))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ContextArtifact(SQLModel, table=True):
    __tablename__ = "context_artifacts"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    conversation_id: str = Field(index=True)
    run_id: str | None = Field(default=None, index=True)
    artifact_type: str = Field(default="summary", index=True)
    content: str
    source_message_ids: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    diagnostics: dict = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class AgentConfig(SQLModel, table=True):
    __tablename__ = "agent_configs"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    name: str
    provider: str = "google"  # "anthropic" | "openai" | "google"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_steps: int = 15
    system_prompt: str | None = None
    tools: list[str] | None = Field(default=None, sa_column=Column(JSON))
    is_default: bool = False
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class UserProfile(SQLModel, table=True):
    __tablename__ = "user_profiles"

    id: str = Field(primary_key=True)  # user_id from Supabase
    display_name: str | None = None
    avatar_url: str | None = None
    default_agent_config_id: str | None = None
    preferences: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class TeamConfig(SQLModel, table=True):
    __tablename__ = "team_configs"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    name: str
    mode: str = "router"  # "router" | "broadcast" | "delegate"
    members: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    max_delegation_steps: int = 10
    max_concurrent_members: int = 5
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SkillDefinition(SQLModel, table=True):
    __tablename__ = "skill_definitions"
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", "version", name="uq_skill_version"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    created_by: str = Field(index=True)
    slug: str = Field(index=True)
    version: str = Field(default="1", index=True)
    name: str
    summary: str
    instructions: str
    tags: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    category: str = "organization"
    author: str = ""
    license: str = ""
    commands: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    argument_hint: str = ""
    provenance: str = "organization"
    trust_level: str = "organization"
    required_commands: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    setup_help: str = ""
    source_url: str = ""
    install_command: str = ""
    readiness_warnings: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    required_tools: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    required_credentials: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    required_platforms: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    required_model_capabilities: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    source: str = "organization"
    sensitivity: str = "internal"
    status: str = Field(default="draft", index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SkillGrant(SQLModel, table=True):
    __tablename__ = "skill_grants"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "skill_id",
            "principal_type",
            "principal_id",
            name="uq_skill_grant",
        ),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    skill_id: str = Field(index=True)
    principal_type: str = Field(index=True)  # role | user
    principal_id: str = Field(index=True)
    created_by: str = Field(index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class EvolutionProposal(SQLModel, table=True):
    __tablename__ = "evolution_proposals"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "target",
            "proposed_version",
            name="uq_evolution_proposal_version",
        ),
    )

    id: str = Field(default_factory=lambda: f"evo_{uuid.uuid4().hex}", primary_key=True)
    organization_id: str = Field(index=True)
    created_by: str = Field(index=True)
    artifact_kind: str = Field(index=True)
    target: str = Field(index=True)
    title: str
    summary: str
    rationale: str
    current_version: str | None = None
    proposed_version: str = Field(index=True)
    proposed_content: str
    eval_cases: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    eval_results: list[dict] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    status: str = Field(default="proposed", index=True)
    approved_by: str | None = Field(default=None, index=True)
    activated_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    supersedes_proposal_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class EvolutionActivation(SQLModel, table=True):
    __tablename__ = "evolution_activations"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    target: str = Field(index=True)
    proposal_id: str = Field(index=True)
    version: str
    activated_by: str = Field(index=True)
    activated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    rolled_back_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class UsageRecord(SQLModel, table=True):
    __tablename__ = "usage_records"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    run_id: str | None = Field(default=None, index=True)
    conversation_id: str | None = None
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class TraceRecord(SQLModel, table=True):
    __tablename__ = "trace_records"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    user_id: str = Field(index=True)
    organization_id: str = Field(index=True)
    run_id: str | None = Field(default=None, index=True)
    conversation_id: str | None = None
    trace_data: dict = Field(sa_column=Column(JSON, nullable=False))
    duration_seconds: float = 0.0
    total_spans: int = 0
    status: str = "ok"
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class CoreMemoryBlock(SQLModel, table=True):
    """Persistent core memory block per user (persona, human, notes)."""

    __tablename__ = "core_memory_blocks"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    label: str  # "persona", "human", "notes", or custom
    value: str = ""
    char_limit: int = 2000
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class KnowledgeEntry(SQLModel, table=True):
    """Searchable long-term knowledge store (replaces experiences + archival)."""

    __tablename__ = "knowledge_entries"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    agent_id: str | None = Field(default=None, index=True)
    session_id: str | None = Field(default=None, index=True)
    content: str
    tags: list[str] | None = Field(default=None, sa_column=Column(JSON))
    importance: int = 1
    kind: str = "semantic"
    confidence: float = 1.0
    sensitivity: str = "internal"
    source: str = "agent"  # "agent", "user", "admin"
    provenance: dict | None = Field(default=None, sa_column=Column(JSON))
    retention: dict | None = Field(default=None, sa_column=Column(JSON))
    conflict_key: str | None = Field(default=None, index=True)
    status: str = "active"
    superseded_by: str | None = None
    metadata_: dict | None = Field(default=None, sa_column=Column("metadata", JSON))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    event_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    deleted_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    user_id: str = Field(index=True)
    organization_id: str = Field(index=True)
    agent_id: str = Field(index=True)
    session_id: str = Field(index=True)
    conversation_id: str | None = Field(default=None, index=True)
    team_config_id: str | None = Field(default=None, index=True)
    parent_run_id: str | None = Field(default=None, index=True)
    root_run_id: str | None = Field(default=None, index=True)
    idempotency_key: str | None = Field(default=None, index=True)
    child_depth: int = 0
    status: str = "pending"  # pending, running, completed, failed
    task: str
    max_steps: int = 15
    success: bool = False
    steps_taken: int = 0
    final_answer: str | None = None
    reasoning: str | None = None
    metrics: dict | None = Field(default=None, sa_column=Column(JSON))
    prompt_fingerprint: str | None = None
    tool_surface_fingerprint: str | None = None
    execution_receipts: list[dict] | None = Field(default=None, sa_column=Column(JSON))
    selected_skills: list[str] | None = Field(default=None, sa_column=Column(JSON))
    artifacts: list[dict] | None = Field(default=None, sa_column=Column(JSON))
    plan: list[dict] | None = Field(default=None, sa_column=Column(JSON))
    attempt_count: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 900
    lease_owner: str | None = Field(default=None, index=True)
    lease_expires_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    completed_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    cancel_requested: bool = False
    isolation_profile: str = "worker-process"
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
