"""Organization, membership, RBAC, and policy endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..auth import get_current_user
from ..database import get_session
from ..models import Organization, OrganizationMembership
from ..schemas import (
    MembershipCreate,
    MembershipResponse,
    MembershipUpdate,
    OrganizationCreate,
    OrganizationPolicyUpdate,
    OrganizationResponse,
)
from ..tenancy import (
    OrganizationContext,
    Permission,
    get_organization_context,
    require_permission,
)

router = APIRouter(prefix="/organizations", tags=["organizations"])


def _organization_response(
    organization: Organization, membership: OrganizationMembership
) -> OrganizationResponse:
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        slug=organization.slug,
        role=membership.role,
        policy=organization.policy,
        created_at=organization.created_at,
    )


@router.post("", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    body: OrganizationCreate,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> OrganizationResponse:
    organization = Organization(
        id=f"org_{uuid.uuid4().hex}",
        name=body.name,
        slug=body.slug,
        created_by=user_id,
        policy={},
    )
    membership = OrganizationMembership(
        organization_id=organization.id,
        user_id=user_id,
        role="owner",
    )
    session.add(organization)
    session.add(membership)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Organization slug exists") from exc
    await session.refresh(organization)
    await session.refresh(membership)
    return _organization_response(organization, membership)


@router.get("", response_model=list[OrganizationResponse])
async def list_organizations(
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[OrganizationResponse]:
    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.status == "active",
        )
    )
    memberships = result.scalars().all()
    responses = []
    for membership in memberships:
        organization = await session.get(Organization, membership.organization_id)
        if organization is not None:
            responses.append(_organization_response(organization, membership))
    return responses


@router.get("/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: str,
    context: OrganizationContext = Depends(get_organization_context),
    session: AsyncSession = Depends(get_session),
) -> OrganizationResponse:
    if context.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    organization = await session.get(Organization, organization_id)
    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.user_id == context.user_id,
        )
    )
    membership = result.scalars().first()
    if organization is None or membership is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return _organization_response(organization, membership)


@router.patch("/{organization_id}/policy", response_model=OrganizationResponse)
async def update_policy(
    organization_id: str,
    body: OrganizationPolicyUpdate,
    context: OrganizationContext = Depends(
        require_permission(Permission.POLICY_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
) -> OrganizationResponse:
    if context.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    organization = await session.get(Organization, organization_id)
    if organization is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    organization.policy = body.policy.model_dump(mode="json")
    organization.updated_at = datetime.now(timezone.utc)
    session.add(organization)
    await session.commit()
    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == organization_id,
            OrganizationMembership.user_id == context.user_id,
        )
    )
    return _organization_response(organization, result.scalars().one())


@router.get("/{organization_id}/members", response_model=list[MembershipResponse])
async def list_members(
    organization_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.ORG_READ)),
    session: AsyncSession = Depends(get_session),
):
    if context.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == organization_id
        )
    )
    return result.scalars().all()


@router.post(
    "/{organization_id}/members",
    response_model=MembershipResponse,
    status_code=201,
)
async def add_member(
    organization_id: str,
    body: MembershipCreate,
    context: OrganizationContext = Depends(
        require_permission(Permission.MEMBER_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
):
    if context.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    membership = OrganizationMembership(
        organization_id=organization_id,
        user_id=body.user_id,
        role=body.role,
    )
    session.add(membership)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Membership exists") from exc
    await session.refresh(membership)
    return membership


@router.patch(
    "/{organization_id}/members/{membership_id}",
    response_model=MembershipResponse,
)
async def update_member(
    organization_id: str,
    membership_id: str,
    body: MembershipUpdate,
    context: OrganizationContext = Depends(
        require_permission(Permission.MEMBER_MANAGE)
    ),
    session: AsyncSession = Depends(get_session),
):
    if context.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Organization not found")
    membership = await session.get(OrganizationMembership, membership_id)
    if membership is None or membership.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Membership not found")
    if membership.role == "owner":
        raise HTTPException(status_code=409, detail="Owner membership is immutable")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(membership, key, value)
    membership.updated_at = datetime.now(timezone.utc)
    session.add(membership)
    await session.commit()
    await session.refresh(membership)
    return membership
