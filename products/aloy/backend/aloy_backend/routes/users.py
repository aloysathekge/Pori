from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from ..auth import get_current_user
from ..database import get_session
from ..models import Conversation, Message, Run, UserProfile
from ..schemas import UsageStatsResponse, UserProfileResponse, UserProfileUpdate

router = APIRouter(prefix="/me", tags=["users"])


async def _get_or_create_profile(user_id: str, session: AsyncSession) -> UserProfile:
    """Return existing profile or auto-create one."""
    profile = await session.get(UserProfile, user_id)
    if profile is None:
        profile = UserProfile(id=user_id)
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
    return profile


@router.get("", response_model=UserProfileResponse)
async def get_my_profile(
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get the current user's profile (auto-creates if missing)."""
    profile = await _get_or_create_profile(user_id, session)
    return profile


@router.patch("", response_model=UserProfileResponse)
async def update_my_profile(
    body: UserProfileUpdate,
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Update the current user's profile."""
    profile = await _get_or_create_profile(user_id, session)

    update_data = body.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(profile, key, value)
    profile.updated_at = datetime.now(timezone.utc)

    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


@router.get("/usage", response_model=UsageStatsResponse)
async def get_my_usage(
    user_id: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get usage statistics for the current user."""
    profile = await _get_or_create_profile(user_id, session)

    conv_count = (
        await session.execute(
            select(func.count()).where(col(Conversation.user_id) == user_id)
        )
    ).scalar_one()

    msg_count = (
        await session.execute(
            select(func.count())
            .select_from(Message)
            .join(Conversation, col(Message.conversation_id) == col(Conversation.id))
            .where(col(Conversation.user_id) == user_id)
        )
    ).scalar_one()

    run_count = (
        await session.execute(select(func.count()).where(col(Run.user_id) == user_id))
    ).scalar_one()

    return UsageStatsResponse(
        total_conversations=conv_count,
        total_messages=msg_count,
        total_runs=run_count,
        member_since=profile.created_at,
    )
