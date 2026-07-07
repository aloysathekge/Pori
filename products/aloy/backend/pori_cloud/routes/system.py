"""System status the app reflects to the user (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..config import settings
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/system", tags=["system"])

# Which backends give real isolation (vs. running on the host).
_ISOLATED = {"e2b"}


class ExecutionStatus(BaseModel):
    enabled: bool
    backend: str
    isolated: bool
    label: str
    detail: str


@router.get("/execution", response_model=ExecutionStatus)
async def execution_status(
    _: OrganizationContext = Depends(require_permission(Permission.RUN_READ)),
) -> ExecutionStatus:
    """How the agent's code/shell runs — surfaced read-only in Settings so a
    user can see whether execution is sandboxed."""
    enabled = settings.sandbox_enabled
    backend = settings.sandbox_backend if enabled else "none"
    isolated = enabled and backend in _ISOLATED
    if not enabled:
        label = "Direct execution"
        detail = "The agent runs commands without a sandbox."
    elif isolated:
        label = "Isolated cloud sandbox"
        detail = (
            f"Agent code runs in a per-session {backend.upper()} microVM — "
            "your machine and other sessions are never touched."
        )
    else:
        label = "Host sandbox"
        detail = "The agent runs in a cooperative sandbox on Aloy's server."
    return ExecutionStatus(
        enabled=enabled,
        backend=backend,
        isolated=isolated,
        label=label,
        detail=detail,
    )
