from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pori import (
    CapabilityResolutionError,
    diagnose_provider,
    get_provider_profile,
    provider_profiles,
    register_all_tools,
    tool_registry,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import AgentConfig
from ..schemas import AgentConfigCreate, AgentConfigResponse, AgentConfigUpdate
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/agent-configs", tags=["agent-configs"])


def _validate_provider_grant(
    provider: str, model: str, context: OrganizationContext
) -> None:
    profile = get_provider_profile(provider)
    policy = context.policy
    if not profile.implemented:
        raise HTTPException(status_code=422, detail="Provider adapter is unavailable")
    if (
        policy.allowed_provider_profiles
        and profile.name not in policy.allowed_provider_profiles
    ):
        raise HTTPException(
            status_code=403, detail="Provider denied by organization policy"
        )
    if policy.allowed_models and model not in policy.allowed_models:
        raise HTTPException(
            status_code=403, detail="Model denied by organization policy"
        )
    if not profile.accepts_model(model):
        raise HTTPException(
            status_code=422, detail="Model is not supported by provider"
        )


def _resolve_capabilities(
    context: OrganizationContext, requested_tools: list[str] | None = None
):
    registry = tool_registry()
    register_all_tools(registry)
    if requested_tools and context.policy.allowed_tools:
        denied = set(requested_tools).difference(context.policy.allowed_tools)
        if denied:
            raise HTTPException(
                status_code=403,
                detail="Tools denied by organization policy: "
                + ", ".join(sorted(denied)),
            )
    try:
        return registry.snapshot(
            include_tools=(requested_tools or context.policy.allowed_tools or None),
            exclude_tools=context.policy.denied_tools,
            include_groups=context.policy.allowed_capability_groups or None,
        )
    except CapabilityResolutionError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Organization capability policy cannot be resolved: {exc}",
        ) from exc


@router.get("/info/models", tags=["info"])
async def list_models(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
):
    """List provider models after organization policy is applied."""
    return {
        profile.name: list(profile.models)
        for profile in provider_profiles()
        if (
            not context.policy.allowed_provider_profiles
            or profile.name in context.policy.allowed_provider_profiles
        )
    }


@router.get("/info/tools", tags=["info"])
async def list_tools(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
):
    """List the exact model-visible tool snapshot for this organization."""
    snapshot = _resolve_capabilities(context)
    return {
        "fingerprint": snapshot.fingerprint,
        "tools": [
            {"name": name, "description": info.description}
            for name, info in snapshot.tool_items
        ],
        "groups": [group.name for group in snapshot.groups],
        "excluded": dict(snapshot.excluded),
    }


@router.get("/info/setup", tags=["info"])
async def setup_diagnostics(
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
):
    """Report provider and capability readiness without exposing credentials."""
    selected = (get_provider_profile(provider),) if provider else provider_profiles()
    provider_results = []
    for profile in selected:
        diagnostic = diagnose_provider(profile.name, model=model)
        reasons = list(diagnostic.reasons)
        if (
            context.policy.allowed_provider_profiles
            and profile.name not in context.policy.allowed_provider_profiles
        ):
            reasons.append("organization_policy:provider_denied")
        if (
            context.policy.allowed_models
            and diagnostic.model not in context.policy.allowed_models
        ):
            reasons.append("organization_policy:model_denied")
        item = diagnostic.model_dump(mode="json")
        item["reasons"] = reasons
        item["available"] = not reasons
        provider_results.append(item)

    registry = tool_registry()
    register_all_tools(registry)
    snapshot = _resolve_capabilities(context)
    groups = []
    for name, group in sorted(registry.groups.items()):
        missing = list(group.prerequisites.missing())
        if (
            context.policy.allowed_capability_groups
            and name not in context.policy.allowed_capability_groups
            and not group.protected
        ):
            missing.append("organization_policy:group_denied")
        groups.append(
            {
                "name": name,
                "available": not missing,
                "reasons": missing,
                "protected": group.protected,
            }
        )
    return {
        "ready": any(item["available"] for item in provider_results),
        "providers": provider_results,
        "capabilities": {
            "fingerprint": snapshot.fingerprint,
            "groups": groups,
            "tool_count": len(snapshot.tool_items),
        },
    }


@router.post("", response_model=AgentConfigResponse, status_code=201)
async def create_agent_config(
    req: AgentConfigCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    _validate_provider_grant(req.provider, req.model, context)
    if req.tools:
        _resolve_capabilities(context, req.tools)
    # If setting as default, unset any existing default
    if req.is_default:
        result = await session.execute(
            select(AgentConfig).where(
                AgentConfig.organization_id == context.organization_id,
                AgentConfig.user_id == context.user_id,
                AgentConfig.is_default == True,
            )
        )
        for existing in result.scalars().all():
            existing.is_default = False
            session.add(existing)

    config = AgentConfig(
        organization_id=context.organization_id,
        user_id=context.user_id,
        **req.model_dump(),
    )
    session.add(config)
    await session.commit()
    await session.refresh(config)
    logger.info("AgentConfig %s created for user %s", config.id, context.user_id)
    return config


@router.get("", response_model=list[AgentConfigResponse])
async def list_agent_configs(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(AgentConfig)
        .where(AgentConfig.organization_id == context.organization_id)
        .order_by(AgentConfig.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{config_id}", response_model=AgentConfigResponse)
async def get_agent_config(
    config_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    config = await session.get(AgentConfig, config_id)
    if not config or config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Agent config not found")
    return config


@router.patch("/{config_id}", response_model=AgentConfigResponse)
async def update_agent_config(
    config_id: str,
    req: AgentConfigUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    config = await session.get(AgentConfig, config_id)
    if not config or config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Agent config not found")

    updates = req.model_dump(exclude_unset=True)
    _validate_provider_grant(
        updates.get("provider", config.provider),
        updates.get("model", config.model),
        context,
    )
    selected_tools = updates.get("tools", config.tools)
    if selected_tools:
        _resolve_capabilities(context, selected_tools)

    # If setting as default, unset any existing default
    if updates.get("is_default"):
        result = await session.execute(
            select(AgentConfig).where(
                AgentConfig.organization_id == context.organization_id,
                AgentConfig.is_default == True,
                AgentConfig.id != config_id,
            )
        )
        for existing in result.scalars().all():
            existing.is_default = False
            session.add(existing)

    for key, value in updates.items():
        setattr(config, key, value)

    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config


@router.delete("/{config_id}", status_code=204)
async def delete_agent_config(
    config_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    config = await session.get(AgentConfig, config_id)
    if not config or config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Agent config not found")
    await session.delete(config)
    await session.commit()
    logger.info("AgentConfig %s deleted", config_id)
