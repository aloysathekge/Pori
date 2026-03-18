"""Data models for the multi-agent Team system."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pori.config import LLMConfig


class TeamMode(str, Enum):
    """How the team coordinator routes work to members."""

    ROUTER = "router"
    BROADCAST = "broadcast"
    DELEGATE = "delegate"


class MemberConfig(BaseModel):
    """Blueprint for a team member.

    Each member is instantiated as a fresh Agent (or nested Team) at execution time.
    """

    name: str = Field(description="Unique name used by the coordinator for routing")
    description: str = Field(
        description="What this member specialises in (shown to coordinator LLM)"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None, description="Override LLM; inherits team coordinator LLM if None"
    )
    agent_settings: Optional[Dict[str, Any]] = Field(
        default=None, description="AgentSettings overrides (max_steps, etc.)"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="Tool name filter (None = all tools). 'answer' and 'done' are always included.",
    )
    team_config: Optional["TeamConfig"] = Field(
        default=None, description="If set, this member is a nested Team instead of an Agent"
    )
    hitl_config: Optional[Dict[str, Any]] = Field(default=None)
    sandbox_base_dir: Optional[str] = Field(default=None)


# --- Coordinator structured outputs ---


class RoutingDecision(BaseModel):
    """Coordinator output for Router mode: pick one member."""

    chosen_member: str = Field(description="Name of the member to handle this task")
    reasoning: str = Field(description="Why this member was chosen")
    rewritten_task: Optional[str] = Field(
        default=None,
        description="Optionally rewritten task tailored to the chosen member",
    )


class DelegationStep(BaseModel):
    """A single step in a delegation plan."""

    step_number: int
    member_name: str = Field(description="Which member executes this step")
    subtask: str = Field(description="The subtask description for this step")
    depends_on: List[int] = Field(
        default_factory=list,
        description="Step numbers that must complete before this step",
    )


class DelegationPlan(BaseModel):
    """Coordinator output for Delegate mode: multi-step execution plan."""

    rationale: str = Field(description="Overall reasoning for the plan")
    steps: List[DelegationStep] = Field(description="Ordered list of delegation steps")


class BroadcastSummary(BaseModel):
    """Coordinator output for Broadcast mode: combine all member results."""

    combined_answer: str = Field(description="Synthesised answer from all members")
    reasoning: str = Field(description="How the results were combined")


# --- Execution results ---


class MemberRunResult(BaseModel):
    """Result from executing one member."""

    member_name: str
    task: str
    completed: bool = False
    steps_taken: int = 0
    final_answer: Optional[str] = None
    reasoning: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- Top-level config for YAML / nesting ---


class TeamConfig(BaseModel):
    """Configuration for a Team, usable in YAML config and for nesting."""

    name: str = Field(default="team", description="Team name for logging")
    mode: TeamMode = Field(default=TeamMode.ROUTER)
    coordinator_llm: Optional[LLMConfig] = Field(
        default=None,
        description="LLM config for the coordinator. If None, must be supplied at runtime.",
    )
    members: List[MemberConfig] = Field(default_factory=list)
    max_delegation_steps: int = Field(default=10, ge=1)
    max_concurrent_members: int = Field(default=5, ge=1)
    agent_defaults: Optional[Dict[str, Any]] = Field(
        default=None, description="Default AgentSettings applied to all members"
    )


# Resolve forward references
MemberConfig.model_rebuild()
TeamConfig.model_rebuild()
