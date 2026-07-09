from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pori import (
    AgentMemory,
    AgentSettings,
    ImageBlock,
    RetrievalEvidence,
    fuse_retrieval,
)
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from ..connections.mcp_store import resolve_run_mcp_servers
from ..connections.store import resolve_run_connections
from ..database import async_session, get_session
from ..event_log import EventLogCollector
from ..memory_records import request_scope, row_to_record
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
from ..run_outcome import build_run_outcome, flush_memory_to_db, persist_run_outcome
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
from ..tools import GOOGLE_TOOL_NAMES
from .teams import _build_team_from_config

logger = logging.getLogger("aloy_backend")

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
    images: list | None = None,
    files: list | None = None,
) -> tuple[Conversation, AgentMemory]:
    """Validate conversation, save user message, seed memory from DB tables."""
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save user message. Attached images/files ride in metadata so history
    # renders them AND follow-up turns can rebuild the file context.
    meta: dict = {}
    if images:
        meta["images"] = [{"data": i.data, "media_type": i.media_type} for i in images]
    if files:
        meta["files"] = [
            {"name": f.name, "size": len(f.content), "content": f.content}
            for f in files
        ]
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=content,
        metadata_=meta or None,
    )
    session.add(user_msg)
    await session.commit()
    # NOTE: the conversation title is set AFTER the first exchange by
    # _maybe_generate_title (a topic title, not the raw first message).

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
        body = msg.content
        for f in (msg.metadata_ or {}).get("files", []) or []:
            if f.get("content"):
                body += _render_file_block(f["name"], f["content"])
        memory.add_message(msg.role, body)

    return conv, memory


def _render_file_block(name: str, content: str) -> str:
    """How an attached text file is shown to the model, appended to the task."""
    return f'\n\n<attached-file name="{name}">\n{content}\n</attached-file>'


async def _maybe_generate_title(session, conv, llm, first_user_content: str) -> None:
    """Give an untitled conversation a short topic title from its first message
    (like ChatGPT/Claude). Best-effort: an LLM title, else a clean heuristic."""
    if conv.title:
        return
    title = ""
    try:
        from pori import SystemMessage, UserMessage

        raw = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Generate a concise 3-6 word title for a conversation that "
                        "opens with the user's message. Reply with ONLY the title — "
                        "no quotes, no trailing punctuation."
                    )
                ),
                UserMessage(content=first_user_content[:1000]),
            ]
        )
        title = (raw or "").strip().strip("\"'").splitlines()[0][:60].strip()
    except Exception:
        logger.debug("Title generation failed; using heuristic", exc_info=True)
    if not title:
        words = first_user_content.strip().split()
        title = " ".join(words[:6])[:60].strip() or "New conversation"
        title = title[:1].upper() + title[1:]
    conv.title = title
    session.add(conv)
    await session.commit()


# ---- artifacts ----

_LANG_BY_EXT = {
    ".py": "python",
    ".md": "markdown",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".txt": "text",
    ".toml": "toml",
    ".env": "bash",
}
_ARTIFACT_MAX_BYTES = 200_000


def _conversation_artifacts(messages) -> dict:
    """path -> {path, tool_name, bytes_written, message_id} across a conversation
    (the allowlist of files its runs actually wrote)."""
    out: dict = {}
    for m in messages:
        for a in (m.metadata_ or {}).get("artifacts", []) or []:
            path = a.get("path")
            if path:
                out[path] = {
                    "path": path,
                    "tool_name": a.get("tool_name"),
                    "bytes_written": a.get("bytes_written"),
                    "message_id": m.id,
                }
    return out


async def _load_conv(session, context, conversation_id):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/{conversation_id}/artifacts")
async def list_artifacts(
    conversation_id: str,
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Files the conversation's agent runs wrote (from message receipts)."""
    await _load_conv(session, context, conversation_id)
    msgs = (
        (
            await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
        )
        .scalars()
        .all()
    )
    return list(_conversation_artifacts(msgs).values())


@router.get("/{conversation_id}/artifacts/content")
async def get_artifact_content(
    conversation_id: str,
    path: str = Query(..., max_length=1024),
    context: OrganizationContext = Depends(require_permission(Permission.AGENT_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Read one artifact's text, allowlisted to paths this conversation wrote
    (so no arbitrary-file read) and confined to the server working dir."""
    from pathlib import Path

    await _load_conv(session, context, conversation_id)
    msgs = (
        (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    if path not in _conversation_artifacts(msgs):
        raise HTTPException(
            status_code=404, detail="Not an artifact of this conversation"
        )

    base = Path.cwd().resolve()
    target = (base / path).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(
            status_code=400, detail="Path outside the working directory"
        )
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact file no longer available")
    try:
        raw = target.read_bytes()
    except OSError:
        raise HTTPException(status_code=400, detail="Could not read artifact")
    truncated = len(raw) > _ARTIFACT_MAX_BYTES
    content = raw[:_ARTIFACT_MAX_BYTES].decode("utf-8", errors="replace")
    return {
        "path": path,
        "content": content,
        "language": _LANG_BY_EXT.get(target.suffix.lower(), "text"),
        "truncated": truncated,
    }


# ---- endpoints ----


@router.post("/{conversation_id}/messages", status_code=202)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    context: OrganizationContext = Depends(check_rate_limit),
    session: AsyncSession = Depends(get_session),
):
    conv, memory = await _prepare_message(
        session,
        context,
        conversation_id,
        req.content,
        images=req.images,
        files=req.files,
    )
    user_id = context.user_id
    # Kernel-side image blocks for this turn (multimodal task message).
    task_images = [
        ImageBlock(source="base64", media_type=i.media_type, data=i.data)
        for i in req.images
    ]
    # Attached text files are embedded into the task the model receives; the
    # stored user message keeps only the typed text (chips render from metadata).
    task_content = req.content + "".join(
        _render_file_block(f.name, f.content) for f in req.files
    )

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
        await flush_memory_to_db(session, context, memory)
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

    # Resolve the user's connected accounts (Gmail, …) into fresh tokens for the
    # run, and exclude a provider's tools when it isn't connected (per-user gate).
    run_connections = await resolve_run_connections(
        session, context.organization_id, context.user_id
    )
    connection_denied = () if "google" in run_connections else tuple(GOOGLE_TOOL_NAMES)
    # Resolve this member's + the org's MCP servers (union) for the run.
    run_mcp_servers = await resolve_run_mcp_servers(
        session, context.organization_id, context.user_id
    )

    orchestrator = build_orchestrator(
        shared_memory=memory,
        agent_config=agent_config,
        allowed_tools=context.policy.allowed_tools or None,
        denied_tools=tuple(context.policy.denied_tools) + connection_denied,
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

        event_collector = EventLogCollector()

        async def _stream():
            result_holder: dict = {}
            # The generator's only job: emit frames + capture the result. All
            # persistence happens once, in the finally, through the shared
            # finalizer — so a client disconnect still records the turn, and the
            # streaming and non-streaming paths can never drift.
            try:
                async for event in stream_agent_execution(
                    orchestrator=orchestrator,
                    task=task_content,
                    settings=agent_settings,
                    run_context=stream_context,
                    collector=event_collector,
                    tool_context_extra={"connections": run_connections},
                    mcp_servers=run_mcp_servers,
                    task_images=task_images,
                    result_holder=result_holder,
                ):
                    yield event
            finally:
                result = result_holder.get("result")
                if result is not None:
                    try:
                        outcome = build_run_outcome(
                            result,
                            memory,
                            stream_context,
                            req.content,
                            fallback_org=context.organization_id,
                            events=event_collector.finalize(),
                        )
                        # Persist on a session the GENERATOR owns, not the
                        # request-scoped one: the request's dependency teardown
                        # order (esp. under BaseHTTPMiddleware, and on client
                        # disconnect-as-cancellation) is not something run
                        # durability should be pinned to.
                        async with async_session() as persist_session:
                            fresh_conv = await persist_session.get(
                                Conversation, conv.id
                            )
                            await persist_run_outcome(
                                persist_session, fresh_conv or conv, context, outcome
                            )
                            await _maybe_generate_title(
                                persist_session,
                                fresh_conv or conv,
                                orchestrator.llm,
                                req.content,
                            )
                        logger.info(
                            "Persisted streamed run %s (conv %s)",
                            stream_context.run_id,
                            conv.id,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to persist streamed run %s", stream_context.run_id
                        )
                else:
                    logger.warning(
                        "Stream for conv %s ended with NO result — run did not "
                        "complete (interrupted, errored, or an unanswered "
                        "clarification). Nothing persisted.",
                        conv.id,
                    )

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
            task=task_content,
            agent_settings=agent_settings,
            run_context=run_context,
            tool_context_extra={"connections": run_connections},
            mcp_servers=run_mcp_servers,
            task_images=task_images,
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

    outcome = build_run_outcome(
        agent_result,
        memory,
        run_context,
        req.content,
        fallback_org=context.organization_id,
    )
    assistant_msg = await persist_run_outcome(session, conv, context, outcome)
    await _maybe_generate_title(session, conv, orchestrator.llm, req.content)

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
    option or free text) to the waiting stream — but only if the awaiting run
    belongs to the caller (ownership enforced in resolve_clarification)."""
    if resolve_clarification(
        clarification_id,
        body.value,
        organization_id=context.organization_id,
        user_id=context.user_id,
    ):
        return {"ok": True}
    raise HTTPException(
        status_code=404, detail="Unknown or already-answered clarification"
    )
