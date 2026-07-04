from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pori import AgentMemory, AgentSettings, RetrievalEvidence, fuse_retrieval
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..database import get_session
from ..memory_records import record_to_row, request_scope, row_to_record
from ..models import (
    AgentConfig,
    ContextArtifact,
    Conversation,
    CoreMemoryBlock,
    KnowledgeEntry,
    Message,
    Run,
    TeamConfig,
    TraceRecord,
    UsageRecord,
)
from ..orchestrator import build_orchestrator
from ..rate_limit import check_rate_limit
from ..runtime import authenticated_run_context
from ..schemas import (
    ContextSearchHit,
    ConversationBranchRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationExportResponse,
    ConversationResponse,
    ConversationSearchHit,
    ConversationUpdate,
    MessageResponse,
    SendMessageRequest,
)
from ..session_repository import CloudSessionRepository
from ..skills import load_skill_catalog
from ..streaming import resolve_clarification, stream_agent_execution
from ..tenancy import OrganizationContext, Permission, require_permission
from .teams import _build_team_from_config

logger = logging.getLogger("pori_cloud")

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    req: ConversationCreate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    conv = Conversation(
        organization_id=context.organization_id,
        user_id=context.user_id,
        title=req.title,
        agent_config_id=req.agent_config_id,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    logger.info("Conversation %s created", conv.id)
    return ConversationResponse(
        id=conv.id,
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
):
    stmt = (
        select(
            Conversation,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .where(Conversation.organization_id == context.organization_id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()

    return [
        ConversationResponse(
            id=conv.id,
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


@router.get("/search", response_model=list[ConversationSearchHit])
async def search_conversations(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=20, ge=1, le=100),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
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
):
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


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        agent_config_id=conv.agent_config_id,
        parent_conversation_id=conv.parent_conversation_id,
        branched_from_message_id=conv.branched_from_message_id,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                metadata=m.metadata_,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    req: ConversationUpdate,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_WRITE)),
    session: AsyncSession = Depends(get_session),
):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.title = req.title
    conv.updated_at = datetime.now(timezone.utc)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)

    result = await session.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    )
    count = result.scalar() or 0

    return ConversationResponse(
        id=conv.id,
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
):
    detail = await get_conversation(conversation_id, context, session)
    artifacts = await session.execute(
        select(ContextArtifact)
        .where(
            ContextArtifact.organization_id == context.organization_id,
            ContextArtifact.conversation_id == conversation_id,
        )
        .order_by(ContextArtifact.created_at)
    )
    return ConversationExportResponse(
        conversation=ConversationResponse(
            id=detail.id,
            title=detail.title,
            agent_config_id=detail.agent_config_id,
            parent_conversation_id=detail.parent_conversation_id,
            branched_from_message_id=detail.branched_from_message_id,
            created_at=detail.created_at,
            updated_at=detail.updated_at,
            message_count=len(detail.messages),
        ),
        messages=detail.messages,
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
):
    parent = await session.get(Conversation, conversation_id)
    if not parent or parent.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages_result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
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
):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete all related records to avoid orphans
    for model in (Message, Run, UsageRecord, TraceRecord, ContextArtifact):
        related = await session.execute(
            select(model).where(model.conversation_id == conversation_id)
        )
        for record in related.scalars().all():
            await session.delete(record)

    await session.delete(conv)
    await session.commit()
    logger.info("Conversation %s deleted", conversation_id)


# ---- helpers shared by streaming and non-streaming paths ----


async def _prepare_message(
    session: AsyncSession,
    context: OrganizationContext,
    conversation_id: str,
    content: str,
) -> tuple[Conversation, AgentMemory]:
    """Validate conversation, save user message, seed memory from DB tables."""
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=content,
    )
    session.add(user_msg)
    await session.commit()

    # Auto-set title from first message
    if not conv.title:
        conv.title = content[:80]
        session.add(conv)
        await session.commit()

    # Create in-memory AgentMemory (no persistent store — we manage persistence)
    agent_id = conv.agent_config_id or "default_agent"
    memory = AgentMemory(
        organization_id=context.organization_id,
        user_id=context.user_id,
        agent_id=agent_id,
        session_id=conversation_id,
    )

    # Seed core memory blocks from the core_memory_blocks table
    for label in ("persona", "human", "notes"):
        result = await session.execute(
            select(CoreMemoryBlock).where(
                CoreMemoryBlock.organization_id == context.organization_id,
                CoreMemoryBlock.user_id == context.user_id,
                CoreMemoryBlock.label == label,
            )
        )
        row = result.scalars().first()
        if row and row.value:
            memory.core_memory.get_block(label).set_value(row.value)

    # Seed knowledge entries as experiences (for recall during agent run)
    result = await session.execute(
        select(KnowledgeEntry)
        .where(
            KnowledgeEntry.organization_id == context.organization_id,
            KnowledgeEntry.user_id == context.user_id,
        )
        .order_by(KnowledgeEntry.created_at.desc())
        .limit(100)
    )
    for entry in result.scalars().all():
        record = row_to_record(entry)
        if memory.scope.can_access(record.scope) and record.is_retrievable():
            record.metadata["legacy_collection"] = "experience"
            memory.memory_records.append(record)

    # Load conversation message history from the messages table
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    history = result.scalars().all()
    for msg in history[:-1]:  # Exclude the user message we just saved
        memory.add_message(msg.role, msg.content)

    return conv, memory


async def _flush_memory_to_db(
    session: AsyncSession, context: OrganizationContext, memory: AgentMemory
) -> None:
    """Persist any core memory changes the agent made back to the DB."""
    now = datetime.now(timezone.utc)
    for label in ("persona", "human", "notes"):
        block = memory.core_memory.get_block(label)
        result = await session.execute(
            select(CoreMemoryBlock).where(
                CoreMemoryBlock.organization_id == context.organization_id,
                CoreMemoryBlock.user_id == context.user_id,
                CoreMemoryBlock.label == label,
            )
        )
        existing = result.scalars().first()
        if existing:
            if existing.value != block.value:
                existing.value = block.value
                existing.updated_at = now
                session.add(existing)
        elif block.value:
            session.add(
                CoreMemoryBlock(
                    organization_id=context.organization_id,
                    user_id=context.user_id,
                    label=label,
                    value=block.value,
                )
            )

    # Persist typed long-term memory through the shared Pori contract.
    for record in memory.memory_records:
        if not request_scope(
            context.user_id,
            organization_id=context.organization_id,
            agent_id=memory.agent_id,
            session_id=memory.session_id,
        ).can_access(record.scope):
            continue
        existing = await session.get(KnowledgeEntry, record.id)
        row = record_to_row(record, existing)
        session.add(row)


async def _save_result(
    session: AsyncSession,
    conv: Conversation,
    context: OrganizationContext,
    task: str,
    max_steps: int,
    agent_result: dict,
    memory: AgentMemory | None = None,
) -> Message:
    """Save assistant message, run record, and flush memory changes."""
    agent = agent_result.get("agent")
    final = agent.memory.get_final_answer() if agent else {}
    answer = (
        agent_result.get("final_answer")
        or (final or {}).get("final_answer")
        or "I could not generate a response."
    )
    reasoning = agent_result.get("reasoning") or (final or {}).get("reasoning")
    metrics = agent_result.get("metrics") or agent_result.get("result", {}).get(
        "metrics"
    )
    result_data = agent_result.get("result") or {}
    context_data = result_data.get("run_context") or {}
    trace_data = agent_result.get("trace") or result_data.get("trace")
    selected_skills = (
        agent_result.get("selected_skills") or result_data.get("selected_skills") or []
    )
    artifacts = agent_result.get("artifacts") or result_data.get("artifacts") or []
    plan = agent_result.get("plan") or result_data.get("plan") or []

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        metadata_={
            "reasoning": reasoning,
            "steps_taken": int(agent_result.get("steps_taken") or 0),
            "metrics": metrics,
            "selected_skills": selected_skills,
            "artifacts": artifacts,
            "plan": plan,
        },
    )
    session.add(assistant_msg)

    run = Run(
        id=context_data.get("run_id") or uuid.uuid4().hex,
        user_id=context.user_id,
        organization_id=context_data.get("organization_id") or context.organization_id,
        agent_id=context_data.get("agent_id")
        or conv.agent_config_id
        or "default_agent",
        session_id=context_data.get("session_id") or conv.id,
        conversation_id=conv.id,
        task=task,
        max_steps=max_steps,
        success=bool(agent_result.get("success")),
        steps_taken=int(agent_result.get("steps_taken") or 0),
        final_answer=answer,
        reasoning=reasoning,
        metrics=metrics,
        prompt_fingerprint=(trace_data or {}).get("prompt_fingerprint"),
        tool_surface_fingerprint=(trace_data or {}).get("tool_surface_fingerprint"),
        execution_receipts=(trace_data or {}).get("execution_receipts") or [],
        selected_skills=selected_skills,
        artifacts=artifacts,
        plan=plan,
    )
    session.add(run)

    # Save usage record from metrics
    if metrics and isinstance(metrics, dict):
        tokens = metrics.get("tokens") or {}
        usage = UsageRecord(
            organization_id=context.organization_id,
            user_id=context.user_id,
            run_id=run.id,
            conversation_id=conv.id,
            provider=(metrics.get("model") or "").split("/")[0],
            model=(metrics.get("model") or "").split("/")[-1],
            input_tokens=int(tokens.get("input", 0)),
            output_tokens=int(tokens.get("output", 0)),
            total_tokens=int(tokens.get("total", 0)),
            estimated_cost=float(
                (metrics.get("cost_usd") or "$0").replace("$", "") or 0
            ),
        )
        session.add(usage)

    # Save trace record
    if trace_data and isinstance(trace_data, dict):
        trace_record = TraceRecord(
            organization_id=run.organization_id,
            user_id=context.user_id,
            run_id=run.id,
            conversation_id=conv.id,
            trace_data=trace_data,
            duration_seconds=float(
                (trace_data.get("duration") or "0s").replace("s", "") or 0
            ),
            total_spans=int(trace_data.get("total_spans", 0)),
            status=trace_data.get("status", "ok"),
        )
        session.add(trace_record)

    # Flush core memory and archival changes to DB tables
    agent_memory = (agent.memory if agent else None) or memory
    if agent_memory:
        await _flush_memory_to_db(session, context, agent_memory)

    conv.updated_at = datetime.now(timezone.utc)
    session.add(conv)

    await session.commit()
    await session.refresh(assistant_msg)

    return assistant_msg


# ---- endpoints ----


@router.post("/{conversation_id}/messages", status_code=202)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    context: OrganizationContext = Depends(check_rate_limit),
    session: AsyncSession = Depends(get_session),
):
    conv, memory = await _prepare_message(
        session, context, conversation_id, req.content
    )
    user_id = context.user_id

    if not context.policy.allow_shared_process_execution:
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
            raise HTTPException(
                status_code=429, detail="Organization run limit reached"
            )
        team_config = None
        if req.team_id:
            team_config = await session.get(TeamConfig, req.team_id)
            if (
                team_config is None
                or team_config.organization_id != context.organization_id
            ):
                raise HTTPException(status_code=404, detail="Team config not found")
        if conv.agent_config_id:
            agent_config = await session.get(AgentConfig, conv.agent_config_id)
            if (
                agent_config is None
                or agent_config.organization_id != context.organization_id
            ):
                raise HTTPException(status_code=404, detail="Agent config not found")
        requested_steps = (
            team_config.max_delegation_steps
            if team_config is not None
            else req.max_steps
        )
        run = Run(
            organization_id=context.organization_id,
            user_id=context.user_id,
            agent_id=(
                f"team:{team_config.id}"
                if team_config is not None
                else conv.agent_config_id or "default_agent"
            ),
            session_id=conv.id,
            conversation_id=conv.id,
            team_config_id=team_config.id if team_config is not None else None,
            task=req.content,
            max_steps=min(requested_steps, context.policy.max_steps_per_run),
            max_attempts=context.policy.max_attempts,
            timeout_seconds=context.policy.run_timeout_seconds,
            status="pending",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return {
            "run_id": run.id,
            "conversation_id": conv.id,
            "status": run.status,
            "execution": "durable-worker",
        }

    # ---- team path ----
    if req.team_id:
        team_config = await session.get(TeamConfig, req.team_id)
        if not team_config or team_config.organization_id != context.organization_id:
            raise HTTPException(status_code=404, detail="Team config not found")

        team_context = authenticated_run_context(
            user_id=context.user_id,
            organization_id=context.organization_id,
            run_id=uuid.uuid4().hex,
            session_id=conv.id,
            agent_id=f"team:{team_config.id}",
            permissions=context.permissions,
            max_steps=min(
                team_config.max_delegation_steps,
                context.policy.max_steps_per_run,
            ),
            isolation_profile="shared-process",
        )
        team = _build_team_from_config(
            team_config,
            req.content,
            memory=memory,
            run_context=team_context,
        )

        try:
            result = await team.run()
        except Exception as e:
            logger.exception("Team failed for conversation %s", conversation_id)
            error_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=f"Sorry, the team encountered an error: {e}",
                metadata_={"error": True, "team_id": req.team_id},
            )
            session.add(error_msg)
            conv.updated_at = datetime.now(timezone.utc)
            session.add(conv)
            await session.commit()
            await session.refresh(error_msg)
            return MessageResponse(
                id=error_msg.id,
                role=error_msg.role,
                content=error_msg.content,
                metadata=error_msg.metadata_,
                created_at=error_msg.created_at,
            )

        final_state = result.get("final_state", {})
        answer = final_state.get(
            "final_answer", "The team could not generate a response."
        )

        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=answer,
            metadata_={
                "team_id": req.team_id,
                "mode": team_config.mode,
                "steps_taken": result.get("steps_taken", 0),
                "metrics": result.get("metrics"),
            },
        )
        session.add(assistant_msg)
        conv.updated_at = datetime.now(timezone.utc)
        session.add(conv)
        await session.commit()
        await session.refresh(assistant_msg)

        # Flush memory changes from team run
        await _flush_memory_to_db(session, context, memory)
        await session.commit()

        return MessageResponse(
            id=assistant_msg.id,
            role=assistant_msg.role,
            content=assistant_msg.content,
            metadata=assistant_msg.metadata_,
            created_at=assistant_msg.created_at,
        )

    # ---- single agent path ----

    # Look up agent config if the conversation has one
    agent_config = None
    if conv.agent_config_id:
        agent_config = await session.get(AgentConfig, conv.agent_config_id)
        if (
            agent_config is None
            or agent_config.organization_id != context.organization_id
        ):
            raise HTTPException(status_code=404, detail="Agent config not found")

    orchestrator = build_orchestrator(
        shared_memory=memory,
        agent_config=agent_config,
        allowed_tools=context.policy.allowed_tools or None,
        denied_tools=context.policy.denied_tools,
        allowed_capability_groups=(context.policy.allowed_capability_groups or None),
        allowed_provider_profiles=(context.policy.allowed_provider_profiles or None),
        allowed_models=context.policy.allowed_models or None,
        skill_catalog=await load_skill_catalog(
            session,
            organization_id=context.organization_id,
            user_id=context.user_id,
            role=context.role,
        ),
    )
    agent_settings = AgentSettings(
        max_steps=min(
            agent_config.max_steps if agent_config else req.max_steps,
            context.policy.max_steps_per_run,
        )
    )

    # ---- streaming path ----
    if req.stream:
        stream_context = authenticated_run_context(
            user_id=user_id,
            organization_id=context.organization_id,
            run_id=uuid.uuid4().hex,
            session_id=conv.id,
            agent_id=conv.agent_config_id or "default_agent",
            max_steps=agent_settings.max_steps,
            permissions=context.permissions,
            isolation_profile="shared-process",
        )

        async def _stream():
            result_holder: dict = {}

            async for event in stream_agent_execution(
                orchestrator=orchestrator,
                task=req.content,
                settings=agent_settings,
                run_context=stream_context,
            ):
                # Capture the final message event so we can persist it
                if event.startswith("event: message\n"):
                    try:
                        data_line = event.split("data: ", 1)[1].split("\n")[0]
                        result_holder["message_data"] = json.loads(data_line)
                    except (json.JSONDecodeError, IndexError):
                        logger.warning("Failed to parse SSE message event")

                yield event

            # After stream completes, persist to DB
            msg_data = result_holder.get("message_data")
            if msg_data:
                assistant_msg = Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=msg_data.get("content", ""),
                    metadata_={
                        "reasoning": msg_data.get("reasoning"),
                        "steps_taken": msg_data.get("steps_taken", 0),
                        "metrics": msg_data.get("metrics"),
                        "selected_skills": msg_data.get("selected_skills") or [],
                        "artifacts": msg_data.get("artifacts") or [],
                        "plan": msg_data.get("plan") or [],
                    },
                )
                session.add(assistant_msg)

                run = Run(
                    id=stream_context.run_id,
                    user_id=user_id,
                    organization_id=stream_context.organization_id,
                    agent_id=stream_context.agent_id,
                    session_id=stream_context.session_id,
                    conversation_id=conv.id,
                    task=req.content,
                    max_steps=req.max_steps,
                    success=msg_data.get("success", False),
                    steps_taken=msg_data.get("steps_taken", 0),
                    final_answer=msg_data.get("content", ""),
                    reasoning=msg_data.get("reasoning"),
                    metrics=msg_data.get("metrics"),
                    prompt_fingerprint=(msg_data.get("trace") or {}).get(
                        "prompt_fingerprint"
                    ),
                    tool_surface_fingerprint=(msg_data.get("trace") or {}).get(
                        "tool_surface_fingerprint"
                    ),
                    execution_receipts=(msg_data.get("trace") or {}).get(
                        "execution_receipts"
                    )
                    or [],
                    selected_skills=msg_data.get("selected_skills") or [],
                    artifacts=msg_data.get("artifacts") or [],
                    plan=msg_data.get("plan") or [],
                )
                session.add(run)

                conv.updated_at = datetime.now(timezone.utc)
                session.add(conv)
                await session.commit()

            # Flush core memory changes after stream completes
            await _flush_memory_to_db(session, context, memory)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- non-streaming path ----
    try:
        run_context = authenticated_run_context(
            user_id=user_id,
            organization_id=context.organization_id,
            run_id=uuid.uuid4().hex,
            session_id=conv.id,
            agent_id=conv.agent_config_id or "default_agent",
            max_steps=agent_settings.max_steps,
            permissions=context.permissions,
            isolation_profile="shared-process",
        )
        agent_result = await orchestrator.execute_task(
            task=req.content,
            agent_settings=agent_settings,
            run_context=run_context,
        )
    except Exception as e:
        logger.exception("Agent failed for conversation %s", conversation_id)
        error_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=f"Sorry, I encountered an error: {e}",
            metadata_={"error": True},
        )
        session.add(error_msg)
        conv.updated_at = datetime.now(timezone.utc)
        session.add(conv)
        await session.commit()
        await session.refresh(error_msg)
        return MessageResponse(
            id=error_msg.id,
            role=error_msg.role,
            content=error_msg.content,
            metadata=error_msg.metadata_,
            created_at=error_msg.created_at,
        )

    assistant_msg = await _save_result(
        session, conv, context, req.content, req.max_steps, agent_result, memory
    )

    logger.info(
        "Message in conversation %s, steps=%d",
        conversation_id,
        assistant_msg.metadata_.get("steps_taken", 0),
    )

    return MessageResponse(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        metadata=assistant_msg.metadata_,
        created_at=assistant_msg.created_at,
    )


class ClarifyBody(BaseModel):
    value: str


@router.post("/clarify/{clarification_id}")
async def submit_clarification(
    clarification_id: str,
    body: ClarifyBody,
    context: OrganizationContext = Depends(check_rate_limit),
):
    """Resolve a paused ``ask_user`` by delivering the user's answer (a tapped
    option or free text) to the waiting stream, routed via the module-level
    clarify-bridge registry."""
    if resolve_clarification(clarification_id, body.value):
        return {"ok": True}
    raise HTTPException(
        status_code=404, detail="Unknown or already-answered clarification"
    )
