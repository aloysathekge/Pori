"""Database-backed organization authorization and policy resolution."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Callable

from fastapi import Depends, Header, HTTPException, status
from pori.providers import get_provider_profile
from pori.tools.standard import STANDARD_KERNEL_TOOLS
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .auth import get_current_user
from .database import get_session
from .models import Organization, OrganizationMembership


class Permission(str, Enum):
    ORG_READ = "org:read"
    ORG_MANAGE = "org:manage"
    MEMBER_MANAGE = "member:manage"
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    RUN_READ = "run:read"
    RUN_CREATE = "run:create"
    RUN_CANCEL = "run:cancel"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    TRACE_READ = "trace:read"
    USAGE_READ = "usage:read"
    POLICY_MANAGE = "policy:manage"


ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "viewer": frozenset(
        {
            Permission.ORG_READ,
            Permission.AGENT_READ,
            Permission.RUN_READ,
            Permission.MEMORY_READ,
            Permission.TRACE_READ,
        }
    ),
    "member": frozenset(
        {
            Permission.ORG_READ,
            Permission.AGENT_READ,
            Permission.AGENT_WRITE,
            Permission.RUN_READ,
            Permission.RUN_CREATE,
            Permission.MEMORY_READ,
            Permission.MEMORY_WRITE,
            Permission.TRACE_READ,
        }
    ),
    "admin": frozenset(Permission) - {Permission.ORG_MANAGE},
    "owner": frozenset(Permission),
}


class OrganizationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps_per_run: int = Field(default=50, ge=1, le=10_000)
    max_concurrent_runs: int = Field(default=5, ge=1, le=1_000)
    max_attempts: int = Field(default=3, ge=1, le=20)
    max_child_runs_per_run: int = Field(default=10, ge=0, le=1_000)
    max_child_depth: int = Field(default=2, ge=0, le=20)
    run_timeout_seconds: int = Field(default=900, ge=30, le=86_400)
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    allowed_capability_groups: tuple[str, ...] = ()
    allowed_provider_profiles: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    require_tool_receipts: bool = True
    allow_shared_process_execution: bool = False
    allow_shared_session_search: bool = False
    memory_retention_days: int | None = Field(default=None, ge=1)

    @field_validator("allowed_provider_profiles")
    @classmethod
    def normalize_provider_profiles(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(get_provider_profile(value).name for value in values)
        )

    @model_validator(mode="after")
    def validate_tool_policy(self) -> "OrganizationPolicy":
        overlap = set(self.allowed_tools).intersection(self.denied_tools)
        if overlap:
            raise ValueError(
                "Tools cannot be both allowed and denied: " + ", ".join(sorted(overlap))
            )
        protected = set(self.denied_tools).intersection(STANDARD_KERNEL_TOOLS)
        if protected:
            raise ValueError(
                "Protected kernel tools cannot be denied: "
                + ", ".join(sorted(protected))
            )
        return self


class OrganizationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str
    user_id: str
    role: str
    permissions: tuple[str, ...]
    policy: OrganizationPolicy

    def permits(self, permission: Permission) -> bool:
        return permission.value in self.permissions


def _personal_slug(user_id: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", user_id.lower()).strip("-") or "user"
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:10]
    return f"personal-{safe[:32]}-{digest}"


async def ensure_personal_organization(
    session: AsyncSession, user_id: str
) -> OrganizationMembership:
    organization_id = f"user:{user_id}"
    organization = await session.get(Organization, organization_id)
    if organization is None:
        organization = Organization(
            id=organization_id,
            name="Personal Workspace",
            slug=_personal_slug(user_id),
            created_by=user_id,
            policy=OrganizationPolicy().model_dump(mode="json"),
        )
        session.add(organization)

    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.user_id == user_id,
        )
    )
    membership = result.scalars().first()
    if membership is None:
        membership = OrganizationMembership(
            organization_id=organization_id,
            user_id=user_id,
            role="owner",
        )
        session.add(membership)
    await session.commit()
    return membership


async def get_organization_context(
    selected_organization_id: str | None = Header(
        default=None, alias="X-Pori-Organization"
    ),
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> OrganizationContext:
    selected_id = selected_organization_id
    if selected_id is None:
        personal = await ensure_personal_organization(session, user_id)
        selected_id = personal.organization_id

    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == selected_id,
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.status == "active",
        )
    )
    membership = result.scalars().first()
    if membership is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    organization = await session.get(Organization, selected_id)
    if organization is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    permissions = ROLE_PERMISSIONS.get(membership.role)
    if permissions is None:
        raise HTTPException(status_code=403, detail="Invalid organization role")
    return OrganizationContext(
        organization_id=organization.id,
        user_id=user_id,
        role=membership.role,
        permissions=tuple(sorted(permission.value for permission in permissions)),
        policy=OrganizationPolicy.model_validate(organization.policy or {}),
    )


def require_permission(
    permission: Permission,
) -> Callable[..., OrganizationContext]:
    async def dependency(
        context: OrganizationContext = Depends(get_organization_context),
    ) -> OrganizationContext:
        if not context.permits(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission.value}",
            )
        return context

    return dependency


__all__ = [
    "OrganizationContext",
    "OrganizationPolicy",
    "Permission",
    "ROLE_PERMISSIONS",
    "ensure_personal_organization",
    "get_organization_context",
    "require_permission",
]
