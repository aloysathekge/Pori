"""Event aggregate identity helpers."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import Conversation, Event


async def ensure_life_event(
    session: AsyncSession, *, organization_id: str, user_id: str
) -> Event:
    """Return the user's singleton Life Event, creating it race-safely."""
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


async def ensure_event_conversation(
    session: AsyncSession, *, event: Event
) -> Conversation:
    """Return the one user-facing conversation that lives with an Event.

    Older databases can contain several conversations for an Event. We adopt
    the most recently active one once, persist that choice on the Event, and
    keep every other row as provenance. New Events receive their conversation
    immediately.
    """
    if event.primary_conversation_id:
        primary = await session.get(Conversation, event.primary_conversation_id)
        if (
            primary is not None
            and primary.event_id == event.id
            and primary.organization_id == event.organization_id
            and primary.user_id == event.user_id
        ):
            return primary

    existing = (
        (
            await session.execute(
                select(Conversation)
                .where(
                    Conversation.event_id == event.id,
                    Conversation.organization_id == event.organization_id,
                    Conversation.user_id == event.user_id,
                )
                .order_by(
                    col(Conversation.updated_at).desc(),
                    col(Conversation.created_at).desc(),
                )
            )
        )
        .scalars()
        .first()
    )
    conversation = existing or Conversation(
        organization_id=event.organization_id,
        user_id=event.user_id,
        event_id=event.id,
        title=event.title,
    )
    session.add(conversation)
    await session.flush()
    event.primary_conversation_id = conversation.id
    session.add(event)
    await session.flush()
    return conversation


__all__ = ["ensure_event_conversation", "ensure_life_event"]
