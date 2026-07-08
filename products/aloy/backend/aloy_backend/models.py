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
