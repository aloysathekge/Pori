"""
Human-in-the-Loop (HITL) module for Pori agents.

Provides approval gates that pause tool execution so a human can
approve, edit, or reject actions before they run.

Design inspired by LangChain's HITL middleware, adapted for Pori's
own architecture (no LangGraph / checkpointing dependency).
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("pori.hitl")


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class InterruptConfig(BaseModel):
    """Per-tool interrupt configuration."""

    allowed_decisions: List[Literal["approve", "edit", "reject"]] = Field(
        default=["approve", "edit", "reject"],
        description="Which decisions the human can make for this tool",
    )
    description: Optional[str] = Field(
        default=None,
        description="Custom description shown to the human for this tool",
    )


class HITLConfig(BaseModel):
    """Top-level Human-in-the-Loop configuration."""

    enabled: bool = Field(default=False, description="Enable HITL approval gates")
    interrupt_on: Dict[str, Union[bool, InterruptConfig]] = Field(
        default_factory=dict,
        description=(
            "Per-tool interrupt rules. "
            "True = all decisions allowed; "
            "False = no approval needed; "
            "InterruptConfig = fine-grained control"
        ),
    )
    auto_approve_duplicates: bool = Field(
        default=True,
        description="Auto-approve tool calls previously approved with the same params",
    )
    timeout_seconds: int = Field(
        default=300, ge=1, description="Seconds to wait for a human decision"
    )
    description_prefix: str = Field(
        default="Tool execution pending approval",
        description="Prefix for interrupt descriptions shown to the human",
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ActionRequest(BaseModel):
    """A single tool call awaiting human review."""

    name: str
    arguments: Dict[str, Any]
    description: str


class ReviewConfig(BaseModel):
    """Describes what decisions are allowed for one action."""

    action_name: str
    allowed_decisions: List[Literal["approve", "edit", "reject"]]


class ApprovalRequest(BaseModel):
    """Sent to the human handler for review."""

    action_requests: List[ActionRequest]
    review_configs: List[ReviewConfig]
    task_id: str
    step_number: int


class EditedAction(BaseModel):
    """An edited tool call (name + args)."""

    name: str
    args: Dict[str, Any]


class Decision(BaseModel):
    """A single human decision."""

    type: Literal["approve", "edit", "reject"]
    edited_action: Optional[EditedAction] = None
    message: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Human's response with one decision per action."""

    decisions: List[Decision]


# ---------------------------------------------------------------------------
# Handler protocol (ABC)
# ---------------------------------------------------------------------------


class HITLHandler(ABC):
    """Interface that CLI, API, or test handlers implement."""

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Present an approval request and return the human's decisions."""
        ...

    async def notify(self, message: str, task_id: str) -> None:
        """Optional: notify the human of something (default: log only)."""
        logger.info(f"[HITL notify] {message}", extra={"task_id": task_id})


# ---------------------------------------------------------------------------
# AutoApproveHandler (for tests / non-interactive use)
# ---------------------------------------------------------------------------


class AutoApproveHandler(HITLHandler):
    """Always approves every action — useful for tests and batch runs."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            decisions=[Decision(type="approve") for _ in request.action_requests]
        )


# ---------------------------------------------------------------------------
# CLIHITLHandler (interactive terminal)
# ---------------------------------------------------------------------------


class CLIHITLHandler(HITLHandler):
    """Interactive CLI handler that prompts the user via stdin."""

    def __init__(self, timeout_seconds: int = 300):
        self._timeout = timeout_seconds

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        decisions: List[Decision] = []

        for action, review in zip(request.action_requests, request.review_configs):
            decision = await self._prompt_for_decision(action, review)
            decisions.append(decision)

        return ApprovalResponse(decisions=decisions)

    async def _prompt_for_decision(
        self, action: ActionRequest, review: ReviewConfig
    ) -> Decision:
        """Prompt the human for a single decision in the terminal."""
        # Build the prompt
        print("\n" + "=" * 60)
        print(f"🔒 {action.description}")
        print(f"\n  Tool: {action.name}")
        try:
            args_str = json.dumps(action.arguments, indent=4)
        except (TypeError, ValueError):
            args_str = str(action.arguments)
        print(f"  Args: {args_str}")
        print()

        # Build allowed choices
        choices = []
        choice_map = {}
        if "approve" in review.allowed_decisions:
            choices.append("[A]pprove")
            choice_map["a"] = "approve"
        if "edit" in review.allowed_decisions:
            choices.append("[E]dit")
            choice_map["e"] = "edit"
        if "reject" in review.allowed_decisions:
            choices.append("[R]eject")
            choice_map["r"] = "reject"

        prompt_str = f"  {', '.join(choices)}: "

        # Get input (run in executor to avoid blocking the event loop)
        loop = asyncio.get_event_loop()
        while True:
            try:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, input, prompt_str),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                print("  Timeout — auto-rejecting.")
                return Decision(type="reject", message="Approval timed out")

            key = raw.strip().lower()[:1]
            if key in choice_map:
                break
            print(f"  Invalid choice '{raw.strip()}'. Please enter one of: {', '.join(choice_map.keys())}")

        decision_type = choice_map[key]

        if decision_type == "approve":
            return Decision(type="approve")

        if decision_type == "reject":
            reason = await loop.run_in_executor(
                None, input, "  Reason (optional, press Enter to skip): "
            )
            return Decision(type="reject", message=reason.strip() or None)

        # Edit
        print(f"\n  Current args: {args_str}")
        print("  Enter new args as JSON (or press Enter to keep current):")
        new_args_raw = await loop.run_in_executor(None, input, "  > ")

        new_args = action.arguments
        if new_args_raw.strip():
            try:
                new_args = json.loads(new_args_raw.strip())
            except json.JSONDecodeError:
                print("  ⚠ Invalid JSON — keeping original args.")

        new_name_raw = await loop.run_in_executor(
            None, input, f"  Tool name [{action.name}] (press Enter to keep): "
        )
        new_name = new_name_raw.strip() or action.name

        return Decision(
            type="edit",
            edited_action=EditedAction(name=new_name, args=new_args),
        )


# ---------------------------------------------------------------------------
# Helper: resolve interrupt config for a tool
# ---------------------------------------------------------------------------


def resolve_interrupt_config(
    tool_name: str, hitl_config: HITLConfig
) -> Optional[InterruptConfig]:
    """
    Look up whether a tool requires approval.

    Returns an InterruptConfig if the tool needs approval, or None if not.
    """
    cfg = hitl_config.interrupt_on.get(tool_name)

    if cfg is None or cfg is False:
        return None

    if cfg is True:
        return InterruptConfig()  # all decisions allowed

    if isinstance(cfg, InterruptConfig):
        return cfg

    # dict from YAML → parse as InterruptConfig
    if isinstance(cfg, dict):
        return InterruptConfig(**cfg)

    return None
