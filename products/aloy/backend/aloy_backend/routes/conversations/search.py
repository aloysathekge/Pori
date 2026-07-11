"""Conversation search endpoints.

Contract: org-boundary-scoped full-conversation search (``/search``) and
continuity-context fusion (``/context/search``: session hits + retrievable
memory records, fused by ``pori.fuse_retrieval``). These literal paths MUST
be registered before the ``/{conversation_id}`` routes (see package
``__init__``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from pori import RetrievalEvidence, fuse_retrieval

from ...database import get_session
from ...memory_records import request_scope, row_to_record
from ...models import KnowledgeEntry
from ...schemas import ContextSearchHit, ConversationSearchHit
from ...session_repository import CloudSessionRepository
from ...tenancy import OrganizationContext, Permission, require_permission

router = APIRouter()


@router.get("/search", response_model=list[ConversationSearchHit])
async def search_conversations(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=20, ge=1, le=100),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[ConversationSearchHit]:
    """Search only after applying the organization boundary in SQL."""
    repository = CloudSessionRepository(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        allow_shared_search=context.policy.allow_shared_session_search,
    )
    return [
        ConversationSearchHit(
            conversation_id=hit.session_id,
            message_id=hit.message_id,
            role=hit.role,
            content=hit.content,
            score=hit.score,
            created_at=hit.created_at,
        )
        for hit in await repository.search(q, limit)
    ]


@router.get("/context/search", response_model=list[ContextSearchHit])
async def search_continuity_context(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
    context: OrganizationContext = Depends(require_permission(Permission.MEMORY_READ)),
    session: AsyncSession = Depends(get_session),
) -> list[ContextSearchHit]:
    session_hits = await search_conversations(q, limit, context, session)
    session_evidence = [
        RetrievalEvidence(
            source_type="session",
            source_id=hit.message_id,
            session_id=hit.conversation_id,
            content=hit.content,
            score=hit.score,
            provenance={"role": hit.role, "created_at": hit.created_at.isoformat()},
        )
        for hit in session_hits
    ]
    memory_rows = await session.execute(
        select(KnowledgeEntry).where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
            func.lower(KnowledgeEntry.content).contains(q.lower()),
        )
    )
    terms = set(q.lower().split())
    memory_evidence = []
    scope = request_scope(context.user_id, organization_id=context.organization_id)
    for row in memory_rows.scalars().all():
        record = row_to_record(row)
        if not scope.can_access(record.scope) or not record.is_retrievable():
            continue
        words = set(record.content.lower().split())
        score = len(terms.intersection(words)) / max(1, len(terms))
        memory_evidence.append(
            RetrievalEvidence(
                source_type="memory",
                source_id=record.id,
                session_id=record.scope.session_id,
                content=record.content,
                score=score,
                provenance=record.provenance.model_dump(mode="json"),
            )
        )
    return [
        ContextSearchHit(**item.model_dump())
        for item in fuse_retrieval(session_evidence, memory_evidence, limit=limit)
    ]
