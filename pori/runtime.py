"""Portable runtime identity, evidence, and fingerprint contracts."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExecutionBudget(BaseModel):
    """Limits shared by a run and, later, its child runs."""

    model_config = ConfigDict(frozen=True)

    max_steps: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_cost_usd: Optional[float] = Field(default=None, ge=0.0)
    max_duration_seconds: Optional[float] = Field(default=None, gt=0.0)


class RunContext(BaseModel):
    """Immutable identity and authority context for one agent run."""

    model_config = ConfigDict(frozen=True)

    organization_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    permissions: Tuple[str, ...] = ()
    credential_scope: Optional[str] = None
    isolation_profile: str = "local"
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    metadata: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def local(
        cls,
        *,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: str = "local-user",
    ) -> "RunContext":
        generated_run_id = run_id or f"run_{uuid.uuid4().hex[:16]}"
        return cls(
            organization_id=f"local:{user_id}",
            user_id=user_id,
            agent_id=agent_id or generated_run_id,
            session_id=session_id or generated_run_id,
            run_id=generated_run_id,
        )


class ReceiptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    REUSED = "reused"


class ToolExecutionReceipt(BaseModel):
    """Auditable evidence for an attempted or reused tool action."""

    receipt_id: str = Field(default_factory=lambda: f"rcpt_{uuid.uuid4().hex[:16]}")
    run_id: str
    tool_name: str
    status: ReceiptStatus
    backend: str = "pori"
    parameters_fingerprint: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ExecutionBudget",
    "ReceiptStatus",
    "RunContext",
    "ToolExecutionReceipt",
    "stable_fingerprint",
    "utc_now",
]
