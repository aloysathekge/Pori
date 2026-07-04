from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..models import SkillDefinition, SkillGrant
from ..schemas import (
    SkillCreate,
    SkillGrantCreate,
    SkillGrantResponse,
    SkillResponse,
    SkillUpdate,
)
from ..tenancy import OrganizationContext, Permission, require_permission

router = APIRouter(prefix="/skills", tags=["skills"])


@router.post("", response_model=SkillResponse, status_code=201)
async def create_skill(
    body: SkillCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    skill = SkillDefinition(
        organization_id=context.organization_id,
        created_by=context.user_id,
        **body.model_dump(),
    )
    session.add(skill)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Skill version exists") from exc
    await session.refresh(skill)
    return skill


@router.get("", response_model=list[SkillResponse])
async def list_skills(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(SkillDefinition)
        .where(SkillDefinition.organization_id == context.organization_id)
        .order_by(SkillDefinition.slug, SkillDefinition.version)
    )
    return result.scalars().all()


@router.get("/{skill_id}", response_model=SkillResponse)
async def get_skill(
    skill_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@router.patch("/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: str,
    body: SkillUpdate,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
):
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(skill, key, value)
    skill.updated_at = datetime.now(timezone.utc)
    session.add(skill)
    await session.commit()
    await session.refresh(skill)
    return skill


@router.post("/{skill_id}/grants", response_model=SkillGrantResponse, status_code=201)
async def create_skill_grant(
    skill_id: str,
    body: SkillGrantCreate,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
):
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    grant = SkillGrant(
        organization_id=context.organization_id,
        skill_id=skill_id,
        principal_type=body.principal_type,
        principal_id=body.principal_id,
        created_by=context.user_id,
    )
    session.add(grant)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Skill grant exists") from exc
    await session.refresh(grant)
    return grant


@router.get("/{skill_id}/grants", response_model=list[SkillGrantResponse])
async def list_skill_grants(
    skill_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    skill = await session.get(SkillDefinition, skill_id)
    if not skill or skill.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Skill not found")
    result = await session.execute(
        select(SkillGrant).where(
            SkillGrant.organization_id == context.organization_id,
            SkillGrant.skill_id == skill_id,
        )
    )
    return result.scalars().all()
