from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pori import RunContext, register_all_tools, tool_registry
from pori.agent import AgentSettings
from pori.config import LLMConfig, create_llm, get_configured_llm
from pori.team.core import Team
from pori.team.models import MemberConfig, TeamMode
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import Run, TeamConfig
from ..schemas import (
    TeamConfigCreate,
    TeamConfigResponse,
    TeamConfigUpdate,
    TeamRunRequest,
)
from ..tenancy import OrganizationContext, Permission, require_permission

logger = logging.getLogger("aloy_backend")

router = APIRouter(prefix="/teams", tags=["teams"])


# ---- CRUD ----


@router.post("", response_model=TeamConfigResponse, status_code=201)
async def create_team(
    req: TeamConfigCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    team = TeamConfig(
        organization_id=context.organization_id,
        user_id=context.user_id,
        name=req.name,
        mode=req.mode,
        members=[m.model_dump() for m in req.members],
        max_delegation_steps=req.max_delegation_steps,
        max_concurrent_members=req.max_concurrent_members,
    )
    session.add(team)
    await session.commit()
    await session.refresh(team)
    logger.info("Team config %s created", team.id)
    return team


@router.get("", response_model=list[TeamConfigResponse])
async def list_teams(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(TeamConfig)
        .where(TeamConfig.organization_id == context.organization_id)
        .order_by(TeamConfig.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{team_id}", response_model=TeamConfigResponse)
async def get_team(
    team_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    team = await session.get(TeamConfig, team_id)
    if not team or team.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Team config not found")
    return team


@router.patch("/{team_id}", response_model=TeamConfigResponse)
async def update_team(
    team_id: str,
    req: TeamConfigUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    team = await session.get(TeamConfig, team_id)
    if not team or team.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Team config not found")

    update_data = req.model_dump(exclude_unset=True)
    if "members" in update_data:
        update_data["members"] = [m.model_dump() for m in req.members]
    for key, value in update_data.items():
        setattr(team, key, value)

    session.add(team)
    await session.commit()
    await session.refresh(team)
    return team


@router.delete("/{team_id}", status_code=204)
async def delete_team(
    team_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    team = await session.get(TeamConfig, team_id)
    if not team or team.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Team config not found")
    await session.delete(team)
    await session.commit()
    logger.info("Team config %s deleted", team_id)


# ---- Execution ----


def _build_team_from_config(
    team_config: TeamConfig,
    task: str,
    memory: "AgentMemory | None" = None,
    run_context: RunContext | None = None,
) -> Team:
    """Construct a pori Team from a stored TeamConfig."""
    llm, _ = get_configured_llm()

    registry = tool_registry()
    register_all_tools(registry)

    members = []
    for m in team_config.members:
        llm_config = None
        if m.get("llm_config"):
            llm_config = LLMConfig(**m["llm_config"])

        members.append(
            MemberConfig(
                name=m["name"],
                description=m["description"],
                llm_config=llm_config,
                agent_settings=m.get("agent_settings"),
                tools=m.get("tools"),
            )
        )

    return Team(
        task=task,
        coordinator_llm=llm,
        members=members,
        mode=TeamMode(team_config.mode),
        tools_registry=registry,
        memory=memory,
        max_delegation_steps=team_config.max_delegation_steps,
        max_concurrent_members=team_config.max_concurrent_members,
        name=team_config.name,
        run_context=run_context,
    )


@router.post("/{team_id}/run", status_code=202)
async def run_team(
    team_id: str,
    req: TeamRunRequest,
    context: OrganizationContext = Depends(require_permission(Permission.RUN_CREATE)),
    session: AsyncSession = Depends(get_session),
):
    team_config = await session.get(TeamConfig, team_id)
    if not team_config or team_config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Team config not found")

    active_count = (
        await session.execute(
            select(func.count())
            .select_from(Run)
            .where(
                Run.organization_id == context.organization_id,
                Run.status.in_(["pending", "running"]),
            )
        )
    ).scalar_one()
    if active_count >= context.policy.max_concurrent_runs:
        raise HTTPException(status_code=429, detail="Organization run limit reached")

    run = Run(
        organization_id=context.organization_id,
        user_id=context.user_id,
        agent_id=f"team:{team_id}",
        session_id="pending",
        team_config_id=team_id,
        task=req.task,
        max_steps=min(
            team_config.max_delegation_steps, context.policy.max_steps_per_run
        ),
        max_attempts=context.policy.max_attempts,
        timeout_seconds=context.policy.run_timeout_seconds,
        status="pending",
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    run.session_id = run.id
    session.add(run)
    await session.commit()
    return {"run_id": run.id, "status": run.status}
