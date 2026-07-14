"""Event aggregate identity helpers."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from .models import Event


async def ensure_life_event(
    session: AsyncSession, *, organization_id: str, user_id: str
) -> Event:
    """Return the user's singleton Life Event, creating it race-safely.

    The partial unique index is the authority under concurrent first requests;
    the nested transaction keeps a losing insert from rolling back unrelated
    work already staged by the caller.
    """
    statement = select(Event).where(
        Event.organization_id == organization_id,
        Event.user_id == user_id,
        Event.is_life == True,  # noqa: E712 - SQLAlchemy expression
    )
    existing = (await session.execute(statement)).scalars().first()
    if existing is not None:
        return existing

    candidate = Event(
        organization_id=organization_id,
        user_id=user_id,
        type="life",
        title="Life",
        is_life=True,
    )
    try:
        async with session.begin_nested():
            session.add(candidate)
            await session.flush()
        return candidate
    except IntegrityError:
        existing = (await session.execute(statement)).scalars().first()
        if existing is None:
            raise
        return existing


__all__ = ["ensure_life_event"]
