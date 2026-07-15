"""Conversation lifecycle endpoints: create, list, get, update, delete,
branch, and export.

Contract: pure row CRUD over ``Conversation`` and its dependents — no agent
execution. Delete cascades to related records and (best-effort) the
conversation's blobs and sandbox scratch dir, sparing library files.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from ...database import get_session
from ...events import ensure_life_event
from ...models import ContextArtifact, Conversation, Event, Message
from ...pagination import CursorError, decode_cursor, encode_cursor
from ...schemas import (
    ConversationBranchRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationExportResponse,
    ConversationResponse,
    ConversationUpdate,
    MessagePage,
    MessageResponse,
)
from ...tenancy import OrganizationContext, Permission, require_permission
from ._helpers import _load_conv

logger = logging.getLogger("aloy_backend")

router = APIRouter()

DEFAULT_MESSAGE_PAGE = 100


def _message_payload(message: Message) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        metadata=message.metadata_,
        created_at=message.created_at,
    )


async def _message_page(
    session: AsyncSession,
    conversation_id: str,
    *,
    cursor: str | None,
    limit: int,
) -> MessagePage:
    stmt = select(Message).where(Message.conversation_id == conversation_id)
    if cursor:
        try:
            decoded = decode_cursor(cursor)
        except CursorError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        stmt = stmt.where(
            or_(
                col(Message.created_at) < decoded.created_at,
                and_(
                    col(Message.created_at) == decoded.created_at,
                    col(Message.id) < decoded.row_id,
                ),
            )
        )
    rows = list(
        (
            await session.execute(
                stmt.order_by(
                    col(Message.created_at).desc(), col(Message.id).desc()
                ).limit(limit + 1)
            )
        )
        .scalars()
        .all()
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = (
        encode_cursor(rows[-1].created_at, rows[-1].id) if has_more and rows else None
    )
    rows.reverse()
    return MessagePage(
        messages=[_message_payload(message) for message in rows],
        next_cursor=next_cursor,
    )


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    req: ConversationCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> ConversationResponse:
    if req.event_id:
        event = await session.get(Event, req.event_id)
        if (
            event is None
            or event.organization_id != context.organization_id
            or event.user_id != context.user_id
            or event.lifecycle == "archived"
        ):
            raise HTTPException(status_code=404, detail="Event not found")
    else:
        event = await ensure_life_event(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
    conv = Conversation(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=event.id,
        title=req.title,
        agent_config_id=req.agent_config_id,
    )
    session.add(conv)
    await session.flush()
    # Life is a multi-conversation personal space: the newest explicit chat is
    # its default resume target. Dedicated Events keep their canonical row.
    if event.is_life or event.primary_conversation_id is None:
        event.primary_conversation_id = conv.id
        session.add(event)
    await session.commit()
    await session.refresh(conv)
    logger.info("Conversation %s created", conv.id)
    return ConversationResponse(
        id=conv.id,
        event_id=conv.event_id,
        title=conv.title,
        agent_config_id=conv.agent_config_id,
        parent_conversation_id=conv.parent_conversation_id,
        branched_from_message_id=conv.branched_from_message_id,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
    limit: int = 20,
    offset: int = 0,
    event_id: str | None = None,
) -> list[ConversationResponse]:
    if event_id is None:
        target_event = await ensure_life_event(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        await session.commit()
    else:
        event = await session.get(Event, event_id)
        if (
            event is None
            or event.organization_id != context.organization_id
            or event.user_id != context.user_id
        ):
            raise HTTPException(status_code=404, detail="Event not found")
        target_event = event

    stmt = (
        select(
            Conversation,
            func.count(col(Message.id)).label("message_count"),
        )
        .outerjoin(Message, col(Message.conversation_id) == col(Conversation.id))
        .where(
            Conversation.organization_id == context.organization_id,
            Conversation.user_id == context.user_id,
            Conversation.event_id == target_event.id,
        )
        .group_by(Conversation.id)
        .order_by(col(Conversation.updated_at).desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()

    return [
        ConversationResponse(
            id=conv.id,
            event_id=conv.event_id,
            title=conv.title,
            agent_config_id=conv.agent_config_id,
            parent_conversation_id=conv.parent_conversation_id,
            branched_from_message_id=conv.branched_from_message_id,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=count,
        )
        for conv, count in rows
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    conv = await _load_conv(session, context, conversation_id)

    page = await _message_page(
        session, conversation_id, cursor=None, limit=DEFAULT_MESSAGE_PAGE
    )

    return ConversationDetail(
        id=conv.id,
        event_id=conv.event_id,
        title=conv.title,
        agent_config_id=conv.agent_config_id,
        parent_conversation_id=conv.parent_conversation_id,
        branched_from_message_id=conv.branched_from_message_id,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=page.messages,
        messages_next_cursor=page.next_cursor,
    )


@router.get("/{conversation_id}/messages", response_model=MessagePage)
async def get_conversation_messages(
    conversation_id: str,
    cursor: str | None = None,
    limit: int = Query(DEFAULT_MESSAGE_PAGE, ge=1, le=200),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> MessagePage:
    await _load_conv(session, context, conversation_id)
    return await _message_page(session, conversation_id, cursor=cursor, limit=limit)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    req: ConversationUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> ConversationResponse:
    conv = await _load_conv(session, context, conversation_id)

    conv.title = req.title
    conv.updated_at = datetime.now(timezone.utc)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)

    result = await session.execute(
        select(func.count(col(Message.id))).where(
            Message.conversation_id == conversation_id
        )
    )
    count = result.scalar() or 0

    return ConversationResponse(
        id=conv.id,
        event_id=conv.event_id,
        title=conv.title,
        agent_config_id=conv.agent_config_id,
        parent_conversation_id=conv.parent_conversation_id,
        branched_from_message_id=conv.branched_from_message_id,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=count,
    )


@router.get("/{conversation_id}/export", response_model=ConversationExportResponse)
async def export_conversation(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> ConversationExportResponse:
    detail = await get_conversation(conversation_id, context, session)
    messages_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(col(Message.created_at), col(Message.id))
    )
    all_messages = [
        _message_payload(message) for message in messages_result.scalars().all()
    ]
    artifacts = await session.execute(
        select(ContextArtifact)
        .where(
            ContextArtifact.organization_id == context.organization_id,
            ContextArtifact.conversation_id == conversation_id,
        )
        .order_by(col(ContextArtifact.created_at))
    )
    return ConversationExportResponse(
        conversation=ConversationResponse(
            id=detail.id,
            event_id=detail.event_id,
            title=detail.title,
            agent_config_id=detail.agent_config_id,
            parent_conversation_id=detail.parent_conversation_id,
            branched_from_message_id=detail.branched_from_message_id,
            created_at=detail.created_at,
            updated_at=detail.updated_at,
            message_count=len(all_messages),
        ),
        messages=all_messages,
        context_artifacts=[
            {
                "id": artifact.id,
                "type": artifact.artifact_type,
                "content": artifact.content,
                "source_message_ids": artifact.source_message_ids,
                "diagnostics": artifact.diagnostics,
                "created_at": artifact.created_at.isoformat(),
            }
            for artifact in artifacts.scalars().all()
        ],
        exported_at=datetime.now(timezone.utc),
    )


@router.post(
    "/{conversation_id}/branch",
    response_model=ConversationDetail,
    status_code=201,
)
async def branch_conversation(
    conversation_id: str,
    body: ConversationBranchRequest,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> ConversationDetail:
    parent = await _load_conv(session, context, conversation_id)
    messages_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(col(Message.created_at), col(Message.id))
    )
    messages = list(messages_result.scalars().all())
    if body.through_message_id:
        ids = [message.id for message in messages]
        if body.through_message_id not in ids:
            raise HTTPException(status_code=404, detail="Branch message not found")
        messages = messages[: ids.index(body.through_message_id) + 1]
    child = Conversation(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=parent.event_id,
        title=body.title or parent.title,
        agent_config_id=parent.agent_config_id,
        parent_conversation_id=parent.id,
        branched_from_message_id=body.through_message_id,
    )
    session.add(child)
    for message in messages:
        session.add(
            Message(
                conversation_id=child.id,
                role=message.role,
                content=message.content,
                metadata_={
                    **(message.metadata_ or {}),
                    "copied_from_message_id": message.id,
                },
                created_at=message.created_at,
            )
        )
    await session.commit()
    return await get_conversation(child.id, context, session)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
) -> None:
    conv = await _load_conv(session, context, conversation_id)

    event = await session.get(Event, conv.event_id)
    if (
        event is not None
        and not event.is_life
        and event.primary_conversation_id == conv.id
    ):
        raise HTTPException(
            status_code=409,
            detail="A dedicated Event's continuous conversation cannot be deleted",
        )

    if event is not None and event.is_life and event.primary_conversation_id == conv.id:
        replacement = (
            (
                await session.execute(
                    select(Conversation)
                    .where(
                        Conversation.event_id == event.id,
                        Conversation.organization_id == context.organization_id,
                        Conversation.user_id == context.user_id,
                        Conversation.id != conv.id,
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
        event.primary_conversation_id = replacement.id if replacement else None
        session.add(event)

    # A Session owns only its messages. Event-owned history, files, and
    # workspace data retain this session id as provenance.
    related = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id)
    )
    for message in related.scalars().all():
        await session.delete(message)

    await session.delete(conv)
    await session.commit()

    logger.info("Conversation %s deleted", conversation_id)
