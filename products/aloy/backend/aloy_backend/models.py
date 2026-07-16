"""All SQLModel table definitions for the backend: organizations and
memberships, conversations/messages, runs and event logs, agent/team configs,
core-memory blocks and knowledge entries, skills and grants, evolution
proposals/activations, traces, usage, and user profiles. Invariants: schema
changes here must ship with an Alembic migration, and timestamps are
timezone-aware UTC (``_utcnow``).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Index, UniqueConstraint, text
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


class Event(SQLModel, table=True):
    """Durable aggregate root for a user's life or project work."""

    __tablename__ = "events"
    __table_args__ = (
        Index(
            "uq_events_life_per_user",
            "organization_id",
            "user_id",
            unique=True,
            sqlite_where=text("is_life = 1"),
            postgresql_where=text("is_life = true"),
        ),
    )

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex}", primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    type: str = Field(default="project", index=True)
    title: str
    lifecycle: str = Field(default="active", index=True)
    phase: str = ""
    summary: str = ""
    is_life: bool = Field(default=False, index=True)
    # A dedicated Event's canonical conversation. For Life, this is the most
    # recent resume target among its intentionally many conversations.
    primary_conversation_id: str | None = Field(default=None, index=True)
    metadata_: dict = Field(
        default_factory=dict, sa_column=Column("metadata", JSON, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SurfaceProject(SQLModel, table=True):
    """One model-authored application project owned by one Event."""

    __tablename__ = "surface_projects"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "event_id",
            name="uq_surface_project_event",
        ),
    )

    id: str = Field(
        default_factory=lambda: f"surface_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    draft_revision_id: str | None = Field(default=None, index=True)
    published_revision_id: str | None = Field(default=None, index=True)
    sdk_version: str = "1"
    lifecycle: str = Field(default="draft", index=True)
    user_lock_state: str = Field(default="editable", index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SurfaceRevision(SQLModel, table=True):
    """Immutable source and manifest snapshot for a Surface project."""

    __tablename__ = "surface_revisions"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "revision_number",
            name="uq_surface_revision_number",
        ),
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_revision_idempotency",
        ),
    )

    id: str = Field(
        default_factory=lambda: f"srev_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    project_id: str = Field(foreign_key="surface_projects.id", index=True)
    revision_number: int = Field(index=True)
    parent_revision_id: str | None = Field(default=None, index=True)
    creator_run_id: str | None = Field(default=None, index=True)
    idempotency_key: str
    request_fingerprint: str
    manifest: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    files: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    checksum: str = Field(index=True)
    file_count: int = 0
    total_bytes: int = 0
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SurfaceBuild(SQLModel, table=True):
    """Durable diagnostics and artifact pointer for one isolated build."""

    __tablename__ = "surface_builds"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_surface_build_idempotency",
        ),
    )

    id: str = Field(
        default_factory=lambda: f"sbuild_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    project_id: str = Field(foreign_key="surface_projects.id", index=True)
    revision_id: str = Field(foreign_key="surface_revisions.id", index=True)
    creator_run_id: str | None = Field(default=None, index=True)
    idempotency_key: str
    request_fingerprint: str
    status: str = Field(default="queued", index=True)
    source_checksum: str = Field(index=True)
    toolchain_version: str
    validation_result: dict = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    diagnostics: list[dict] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    build_log: str = ""
    bundle_key: str | None = None
    bundle_sha256: str | None = Field(default=None, index=True)
    bundle_size_bytes: int = 0
    preview_artifacts: list[dict] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    resource_metrics: dict = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    completed_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class Task(SQLModel, table=True):
    """Durable executable work owned by an Event."""

    __tablename__ = "tasks"

    id: str = Field(
        default_factory=lambda: f"task_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    origin_conversation_id: str | None = Field(default=None, index=True)
    title: str
    status: str = Field(default="open", index=True)
    instructions: str = ""
    definition_of_done: str = ""
    priority: str = Field(default="normal", index=True)
    due_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    execution_mode: str = Field(default="manual", index=True)
    assigned_agent_id: str | None = Field(default=None, index=True)
    current_run_id: str | None = Field(default=None, index=True)
    result_summary: str = ""
    blocker: str = ""
    budget_policy: dict = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    order: int = 0
    created_by: str
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ActionProposal(SQLModel, table=True):
    """A staged external action awaiting policy routing or a decision."""

    __tablename__ = "proposals"

    id: str = Field(
        default_factory=lambda: f"prop_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    origin_session_id: str | None = Field(default=None, index=True)
    origin_run_id: str | None = Field(default=None, index=True)
    tool: str = Field(index=True)
    args: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    tool_schema_fingerprint: str
    reason: str
    impact: str
    risk: str = Field(index=True)
    routing: str = Field(index=True)
    status: str = Field(default="proposed", index=True)
    expires_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    safe_default: dict | None = Field(default=None, sa_column=Column(JSON))
    decided_by: str | None = Field(default=None, index=True)
    decided_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    receipt: dict | None = Field(default=None, sa_column=Column(JSON))
    execution_attempt_id: str | None = Field(default=None, index=True)
    provider_operation_id: str | None = Field(default=None, index=True)
    error: str | None = None
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class EventTrailEntry(SQLModel, table=True):
    """Append-only activity and evidence for an Event."""

    __tablename__ = "event_trail_entries"

    id: str = Field(
        default_factory=lambda: f"trail_{uuid.uuid4().hex}", primary_key=True
    )
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    actor_id: str = Field(index=True)
    kind: str = Field(index=True)
    summary: str
    run_id: str | None = Field(default=None, index=True)
    proposal_id: str | None = Field(default=None, index=True)
    task_id: str | None = Field(default=None, index=True)
    evidence_refs: list[dict] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
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


class StoredFile(SQLModel, table=True):
    """A durable blob in the object store — pointer row only, never bytes.

    kind='artifact': a file an agent run wrote (extracted in the run
    finalizer). kind='upload': a user attachment (Phase 2)."""

    __tablename__ = "stored_files"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    origin_session_id: str | None = Field(default=None, index=True)
    conversation_id: str | None = Field(default=None, index=True)
    run_id: str | None = Field(default=None, index=True)
    kind: str = "artifact"  # artifact | upload
    # In the user's file library: durable beyond its conversation — a
    # KnowledgeEntry pointer makes every future run aware it exists.
    in_library: bool = Field(default=False, index=True)
    name: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    sha256: str = ""
    storage_key: str
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ContextArtifact(SQLModel, table=True):
    __tablename__ = "context_artifacts"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
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
    event_id: str = Field(foreign_key="events.id", index=True)
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
    event_id: str | None = Field(default=None, foreign_key="events.id", index=True)
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
    # Aloy layered-knowledge scope (the moat): "org" | "team" | "personal". Existing
    # rows default to personal; org/team populate later. See scope_resolver.py.
    scope_level: str = Field(default="personal", index=True)
    team_id: str | None = Field(default=None, index=True)
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
    event_id: str = Field(foreign_key="events.id", index=True)
    task_id: str | None = Field(default=None, index=True)
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
    # Live loop checkpoint written every step while the run executes:
    # {kernel_task_id, n_steps, consecutive_failures, current_activity, plan,
    #  updated_at}. On a re-claim after a crash/expired lease, the worker
    # injects this into memory and RESUMES the kernel task from its last step
    # instead of restarting from zero (docs/long-running.md Phase 2).
    progress: dict | None = Field(default=None, sa_column=Column(JSON))
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


# Reserved user_id for ORG-SHARED connections (scope="org"): the credential
# belongs to the organization, not a member. A sentinel (rather than NULL) keeps
# the (org, user, provider) unique constraint dedup'ing org rows per provider
# and avoids a NOT-NULL column rebuild.
ORG_CONNECTION_USER = "__org__"


class OAuthConnection(SQLModel, table=True):
    """A connected external account (e.g. Gmail).

    scope="user": a member's own connection, keyed (org, user, provider) — private
    even from org admins. scope="org": org-shared (user_id=ORG_CONNECTION_USER),
    provisioned by an admin and usable by permitted members.

    Access/refresh tokens are stored ENCRYPTED (see connections/crypto.py) — the
    DB never holds plaintext; tools use a freshly resolved token."""

    __tablename__ = "oauth_connections"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "user_id", "provider", name="uq_oauth_conn"
        ),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)  # a member id, or ORG_CONNECTION_USER
    scope: str = Field(default="user", index=True)  # "user" | "org"
    created_by: str | None = None  # who provisioned an org-shared connection
    provider: str = Field(index=True)  # "google" (more later)
    access_token_enc: str
    refresh_token_enc: str | None = None
    scopes: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    account_email: str | None = None
    expires_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    status: str = "active"  # active | error | revoked
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class McpServer(SQLModel, table=True):
    """An MCP server a member (scope='user') or the org (scope='org') has added.

    Resolved per run into a kernel ``McpServerConfig`` (with auth headers) and
    passed to the agent — the same scope/union machinery as OAuthConnection.
    Static Bearer secrets are ENCRYPTED at rest; OAuth-authed servers (later)
    reference an OAuthConnection."""

    __tablename__ = "mcp_servers"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "user_id", "name", name="uq_mcp_server_name"
        ),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)  # a member id, or ORG_CONNECTION_USER
    scope: str = Field(default="user", index=True)  # "user" | "org"
    created_by: str | None = None
    name: str  # namespaces tools as mcp__<name>__<tool>
    transport: str = "http"  # "http" | "sse"
    url: str = ""
    auth_kind: str = "none"  # "none" | "static" | "oauth"
    static_secret_enc: str | None = None  # encrypted Bearer for auth_kind=static
    oauth_connection_id: str | None = None  # for auth_kind=oauth (later)
    tools_include: list[str] | None = Field(default=None, sa_column=Column(JSON))
    tools_exclude: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    enabled: bool = True
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class OAuthFlowState(SQLModel, table=True):
    """Short-lived CSRF/PKCE state for an in-flight connect (10-min TTL)."""

    __tablename__ = "oauth_flow_states"

    state: str = Field(primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    scope: str = Field(default="user")  # "user" | "org" (target of this connect)
    provider: str
    pkce_verifier: str
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class RunEventLog(SQLModel, table=True):
    """A coalesced, replayable log of a run's kernel PoriEvents.

    One row per run: the ordered event stream (streamed token deltas coalesced
    into text/thinking blocks; tool calls, steps, retries kept verbatim) so the
    app can offer a read-only replay of what the agent did. Tenant-scoped;
    written in the same transaction that persists the Run."""

    __tablename__ = "run_event_logs"

    run_id: str = Field(primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    conversation_id: str | None = Field(default=None, index=True)
    events: list[dict] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    event_count: int = 0
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class GatewayLink(SQLModel, table=True):
    """A paired external chat (e.g. one Telegram chat) bound to an Aloy user.

    Created by the pairing flow: the user mints a code in the product
    (POST /v1/gateway/pair), sends it to the bot, and the gateway service
    exchanges it for this link. All inbound messages from the chat run as
    that user; all outbound delivery for the user can target the chat.
    """

    __tablename__ = "gateway_links"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    platform: str = Field(index=True)  # "telegram" (more adapters later)
    chat_id: str = Field(index=True)  # platform-native chat identifier
    chat_title: str | None = None
    # Dedicated conversation this chat maps to — inbound messages become runs
    # in it, so history/memory behave exactly like the web app.
    conversation_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class GatewayPairingCode(SQLModel, table=True):
    """A short-lived one-time code that links an external chat to a user."""

    __tablename__ = "gateway_pairing_codes"

    code: str = Field(primary_key=True)  # 8 chars, uppercase
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    platform: str = "telegram"
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class CronJob(SQLModel, table=True):
    """A recurring task: each due tick enqueues a normal Run on the worker
    queue (docs/long-running.md Phase 3). At-most-once firing is guaranteed by
    advancing next_run_at inside the same transaction that enqueues the Run,
    before commit — the Hermes cron pattern."""

    __tablename__ = "cron_jobs"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    organization_id: str = Field(index=True)
    user_id: str = Field(index=True)
    name: str
    task: str
    # Either a 5-field cron expression ("0 7 * * 1-5") or "@every:SECONDS".
    schedule: str
    enabled: bool = True
    max_steps: int = 15
    # When set, completed runs deliver their answer into this conversation as
    # an assistant message — the existing surface the app already renders.
    conversation_id: str | None = Field(default=None, index=True)
    next_run_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    last_run_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    last_run_id: str | None = None
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
