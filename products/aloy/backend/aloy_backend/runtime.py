"""Cloud derivation of Pori runtime identity."""

from __future__ import annotations

from typing import Iterable, Optional

from pori import ExecutionBudget, RunContext


def personal_organization_id(user_id: str) -> str:
    return f"user:{user_id}"


def authenticated_run_context(
    *,
    user_id: str,
    run_id: str,
    session_id: str,
    event_id: str | None = None,
    workspace_id: str | None = None,
    agent_id: str,
    permissions: Iterable[str] = (),
    organization_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    isolation_profile: str = "worker-process",
) -> RunContext:
    """Build run identity from trusted authentication and server-owned data."""
    tenant_id = organization_id or personal_organization_id(user_id)
    return RunContext(
        organization_id=tenant_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        run_id=run_id,
        event_id=event_id,
        workspace_id=workspace_id or event_id,
        permissions=tuple(sorted(set(permissions))),
        credential_scope=f"organization:{tenant_id}",
        isolation_profile=isolation_profile,
        budget=ExecutionBudget(max_steps=max_steps),
    )


__all__ = ["authenticated_run_context", "personal_organization_id"]
