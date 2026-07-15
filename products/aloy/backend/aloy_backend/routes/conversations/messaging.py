"""POST /{conversation_id}/messages — the send-message pipeline.

Contract: one endpoint, four execution modes, dispatched in this order:

1. durable-worker enqueue — org policy forbids shared-process execution;
2. inline team run — ``req.team_id`` routes through a multi-agent team;
3. SSE streaming — ``req.stream`` (persistence via ``StreamPersister``);
4. blocking single-agent — everything else.

``_prepare_message`` validates + persists the user turn and seeds an
``AgentMemory`` from the DB for all modes; ``_assemble_task`` builds the
multimodal task the model receives (attachment ladder).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, NamedTuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from pori import AgentMemory, AgentSettings, DocumentBlock, ImageBlock

from ... import resumable_runs
from ...approvals import proposal_write_gate
from ...conversation_runtime import load_event_memory
from ...database import async_session, get_session
from ...doc_extract import ExtractionError, extract_docx_text, extract_xlsx_text
from ...event_log import EventLogCollector
from ...models import (
    AgentConfig,
    Conversation,
    Message,
    Run,
    StoredFile,
    TeamConfig,
)
from ...orchestrator import build_orchestrator, sandbox_base_dir
from ...provisioning import (
    provision_event_uploads,
    resolve_upload_refs,
    uploads_task_block,
)
from ...rate_limit import rate_limited_permission
from ...run_outcome import build_run_outcome, flush_memory_to_db, persist_run_outcome
from ...run_surface import resolve_run_surface
from ...runtime import authenticated_run_context
from ...schemas import XLSX_MIME, MessageResponse, SendMessageRequest
from ...skills import load_skill_catalog
from ...storage import safe_name
from ...streaming import stream_agent_execution
from ...team_execution import build_team_from_config
from ...tenancy import OrganizationContext, Permission
from ._helpers import _load_conv, _maybe_generate_title, _render_file_block

logger = logging.getLogger("aloy_backend")

router = APIRouter()


# ---- helpers shared by streaming and non-streaming paths ----


async def _prepare_message(
    session: AsyncSession,
    context: OrganizationContext,
    conversation_id: str,
    content: str,
    images: list | None = None,
    files: list | None = None,
    documents: list | None = None,
    uploaded: list[StoredFile] | None = None,
) -> tuple[Conversation, AgentMemory]:
    """Validate conversation, save user message, seed memory from DB tables."""
    conv = await _load_conv(session, context, conversation_id)

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
    if documents:
        # Chips only (name/size) — PDF bytes aren't re-attached on later turns,
        # and extracted docx/xlsx text already rides in the model task.
        meta.setdefault("files", []).extend(
            {"name": d.name, "size": len(d.data) * 3 // 4} for d in documents
        )
    if uploaded:
        # Durable uploads: a file that ALSO rode inline/native this turn gets
        # its file_id merged onto the existing chip (no duplicate); pure
        # durable uploads get their own chip.
        chips = meta.setdefault("files", [])
        by_name = {safe_name(c["name"]): c for c in chips if c.get("name")}
        for u in uploaded:
            chip = by_name.get(u.name)  # StoredFile names are safe_name'd
            if chip is not None:
                chip["file_id"] = u.id
            else:
                chips.append({"name": u.name, "size": u.size_bytes, "file_id": u.id})
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
    memory = await load_event_memory(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        conversation=conv,
        exclude_message_id=user_msg.id,
    )

    return conv, memory


async def _assemble_task(
    session: AsyncSession,
    conversation: Conversation,
    req: SendMessageRequest,
) -> tuple[str, list]:
    """Build the multimodal task the model receives: (task_content, blocks).

    The attachment ladder — inline images ride as kernel ImageBlocks, PDFs
    natively as DocumentBlocks; DOCX/XLSX are text-extracted server-side
    (Hermes-harvested stdlib OOXML parsing) and join the text-file flow;
    every durable upload rides as a ~1-line reference block, provisioned
    into the sandbox — never as bytes in the task.
    """
    # Kernel-side image blocks for this turn (multimodal task message).
    task_attachments: list = [
        ImageBlock(source="base64", media_type=i.media_type, data=i.data)
        for i in req.images
    ]
    # Binary documents: PDFs ride natively (kernel DocumentBlock — providers
    # accept PDF blocks directly); DOCX/XLSX are text-extracted server-side
    # (Hermes-harvested stdlib OOXML parsing) and join the text-file flow.
    extracted_files: list[tuple[str, str]] = []  # (name, extracted text)
    for doc in req.documents:
        if doc.media_type == "application/pdf":
            task_attachments.append(
                DocumentBlock(
                    media_type="application/pdf", data=doc.data, name=doc.name
                )
            )
            continue
        try:
            raw = base64.b64decode(doc.data)
            text = (
                extract_docx_text(raw)
                if doc.media_type != XLSX_MIME
                else extract_xlsx_text(raw)
            )
        except (ExtractionError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not read {doc.name}: {exc}"
            )
        extracted_files.append((doc.name, text[:200_000]))

    # Attached text files are embedded into the task the model receives; the
    # stored user message keeps only the typed text (chips render from metadata).
    task_content = (
        req.content
        + "".join(_render_file_block(f.name, f.content) for f in req.files)
        + "".join(_render_file_block(n, t) for n, t in extracted_files)
    )

    # Every durable upload of this conversation rides as a REFERENCE block
    # (~1 line per file), provisioned into the sandbox — never as bytes in
    # the task. Eager provisioning at upload time makes this a hash verify.
    conv_uploads = (
        (
            await session.execute(
                select(StoredFile).where(
                    StoredFile.event_id == conversation.event_id,
                    StoredFile.kind == "upload",
                )
            )
        )
        .scalars()
        .all()
    )
    if conv_uploads:
        try:
            task_content += uploads_task_block(
                provision_event_uploads(conversation.event_id, conv_uploads)
            )
        except Exception:
            logger.exception(
                "Upload provisioning failed for Event %s", conversation.event_id
            )

    return task_content, task_attachments


# ---- execution modes ----


async def _enqueue_durable_run(
    session: AsyncSession,
    context: OrganizationContext,
    conv: Conversation,
    req: SendMessageRequest,
) -> dict:
    """Durable-worker mode: create a pending Run row and return 202 —
    a worker process claims and executes it (org policy forbids
    shared-process execution)."""
    active_count = (
        await session.execute(
            select(func.count())
            .select_from(Run)
            .where(
                Run.organization_id == context.organization_id,
                col(Run.status).in_(["pending", "running"]),
            )
        )
    ).scalar_one()
    if active_count >= context.policy.max_concurrent_runs:
        raise HTTPException(status_code=429, detail="Organization run limit reached")
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
        team_config.max_delegation_steps if team_config is not None else req.max_steps
    )
    run = Run(
        organization_id=context.organization_id,
        user_id=context.user_id,
        event_id=conv.event_id,
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


async def _run_team_inline(
    session: AsyncSession,
    context: OrganizationContext,
    conv: Conversation,
    memory: AgentMemory,
    req: SendMessageRequest,
) -> MessageResponse:
    """Team mode: build the configured Team, run it inline, persist the
    answer (or error) as an assistant message, and flush memory changes."""
    team_config = await session.get(TeamConfig, req.team_id)
    if not team_config or team_config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Team config not found")

    team_context = authenticated_run_context(
        user_id=context.user_id,
        organization_id=context.organization_id,
        run_id=uuid.uuid4().hex,
        session_id=conv.id,
        event_id=conv.event_id,
        workspace_id=conv.event_id,
        agent_id=f"team:{team_config.id}",
        permissions=context.permissions,
        max_steps=min(
            team_config.max_delegation_steps,
            context.policy.max_steps_per_run,
        ),
        isolation_profile="shared-process",
    )
    team = build_team_from_config(
        team_config,
        req.content,
        memory=memory,
        run_context=team_context,
    )

    try:
        result = await team.run()
    except Exception as e:
        logger.exception("Team failed for conversation %s", conv.id)
        error_msg = Message(
            conversation_id=conv.id,
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
    answer = final_state.get("final_answer", "The team could not generate a response.")

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


async def _load_agent_config(
    session: AsyncSession,
    context: OrganizationContext,
    conv: Conversation,
) -> AgentConfig | None:
    """The conversation's agent config, ownership-checked (404 on foreign)."""
    if not conv.agent_config_id:
        return None
    agent_config = await session.get(AgentConfig, conv.agent_config_id)
    if agent_config is None or agent_config.organization_id != context.organization_id:
        raise HTTPException(status_code=404, detail="Agent config not found")
    return agent_config


def _claim_warm_resume(
    conversation_id: str,
    req: SendMessageRequest,
    memory: AgentMemory,
    task_content: str,
) -> tuple[AgentMemory, str, str | None]:
    """Continue a stopped run: claim its warm state (live AgentMemory + kernel
    task checkpoint) for a TRUE resume — the run picks up at the step it was
    stopped on, tool work intact. A cold cache needs no special handling:
    the continuation content runs as a normal turn over persisted history
    (which includes the stopped run's partial text). A normal turn instead
    invalidates any warm state — resuming after the conversation moved on
    would fork history.
    """
    resume_task_id: str | None = None
    if req.resume_run_id:
        warm = resumable_runs.claim(conversation_id, req.resume_run_id)
        if warm is not None:
            memory = warm.memory
            task_content = warm.task
            resume_task_id = warm.task_id
            logger.info(
                "Resuming stopped run %s (kernel task %s) for conversation %s",
                req.resume_run_id,
                warm.task_id,
                conversation_id,
            )
    else:
        resumable_runs.discard(conversation_id)
    return memory, task_content, resume_task_id


class _AgentRunSetup(NamedTuple):
    """Everything the single-agent modes (streaming, blocking) run with."""

    memory: AgentMemory
    task_content: str
    resume_task_id: str | None
    orchestrator: Any
    agent_settings: AgentSettings
    surface: Any


async def _setup_single_agent(
    session: AsyncSession,
    context: OrganizationContext,
    conv: Conversation,
    req: SendMessageRequest,
    memory: AgentMemory,
    task_content: str,
) -> _AgentRunSetup:
    """Resolve everything the single-agent modes share: agent config, the
    run's capability surface, warm-resume state, orchestrator, and settings."""
    agent_config = await _load_agent_config(session, context, conv)

    # The run's capability surface (connections + MCP + library + gated
    # denials) — resolved by the ONE shared service, same as the worker path.
    surface = await resolve_run_surface(
        session,
        organization_id=context.organization_id,
        user_id=context.user_id,
        policy=context.policy,
    )

    memory, task_content, resume_task_id = _claim_warm_resume(
        conv.id, req, memory, task_content
    )

    orchestrator = build_orchestrator(
        shared_memory=memory,
        agent_config=agent_config,
        allowed_tools=context.policy.allowed_tools or None,
        denied_tools=surface.denied_tools,
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
    return _AgentRunSetup(
        memory, task_content, resume_task_id, orchestrator, agent_settings, surface
    )


class StreamPersister:
    """Post-stream persistence for one streamed turn (formerly closures in
    ``send_message``).

    Single-finalizer rule: ALL persistence flows through
    ``persist_streamed_outcome`` exactly once — whether the stream completed
    normally or the client disconnected mid-run (the run is then awaited in
    the background and persisted, ChatGPT behavior). Resumable warm-state
    registration flows through here indirectly via ``streaming.py``.
    """

    def __init__(
        self,
        *,
        conv: Conversation,
        context: OrganizationContext,
        memory: AgentMemory,
        orchestrator,
        event_collector: EventLogCollector,
        stream_context,
        req: SendMessageRequest,
    ) -> None:
        self.conv = conv
        self.context = context
        self.memory = memory
        self.orchestrator = orchestrator
        self.event_collector = event_collector
        self.stream_context = stream_context
        self.req = req

    async def persist_streamed_outcome(self, result: dict) -> None:
        try:
            outcome = build_run_outcome(
                result,
                self.memory,
                self.stream_context,
                self.req.content,
                fallback_org=self.context.organization_id,
                events=self.event_collector.finalize(),
            )
            # Persist on a session WE own, not the request-scoped one: the
            # request's dependency teardown order (esp. under
            # BaseHTTPMiddleware, and on disconnect-as-cancellation) is not
            # something run durability should be pinned to.
            async with async_session() as persist_session:
                fresh_conv = await persist_session.get(Conversation, self.conv.id)
                await persist_run_outcome(
                    persist_session, fresh_conv or self.conv, self.context, outcome
                )
                await _maybe_generate_title(
                    persist_session,
                    fresh_conv or self.conv,
                    self.orchestrator.llm,
                    self.req.content,
                )
            logger.info(
                "Persisted streamed run %s (conv %s)",
                self.stream_context.run_id,
                self.conv.id,
            )
        except Exception:
            logger.exception(
                "Failed to persist streamed run %s", self.stream_context.run_id
            )

    async def finish_disconnected_run(self, run_future) -> None:
        try:
            result = await run_future
        except Exception:
            logger.exception(
                "Backgrounded run %s failed after disconnect",
                self.stream_context.run_id,
            )
            return
        if result:
            await self.persist_streamed_outcome(result)


def _stream_response(
    *,
    conv: Conversation,
    context: OrganizationContext,
    req: SendMessageRequest,
    setup: _AgentRunSetup,
    task_attachments: list,
) -> StreamingResponse:
    """Streaming mode: run the agent behind an SSE frame generator; all
    persistence funnels through the ``StreamPersister`` finalizer."""
    memory, task_content, resume_task_id, orchestrator, agent_settings, surface = setup
    stream_context = authenticated_run_context(
        user_id=context.user_id,
        organization_id=context.organization_id,
        run_id=uuid.uuid4().hex,
        session_id=conv.id,
        event_id=conv.event_id,
        workspace_id=conv.event_id,
        agent_id=conv.agent_config_id or "default_agent",
        max_steps=agent_settings.max_steps,
        permissions=context.permissions,
        isolation_profile="shared-process",
    )

    event_collector = EventLogCollector()
    persister = StreamPersister(
        conv=conv,
        context=context,
        memory=memory,
        orchestrator=orchestrator,
        event_collector=event_collector,
        stream_context=stream_context,
        req=req,
    )

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
                tool_context_extra=surface.tool_context_extra,
                mcp_servers=surface.mcp_servers,
                task_attachments=task_attachments,
                result_holder=result_holder,
                conversation_id=conv.id,
                resume_task_id=resume_task_id,
                sandbox_base_dir=sandbox_base_dir(),
            ):
                yield event
        finally:
            result = result_holder.get("result")
            if result is not None:
                await persister.persist_streamed_outcome(result)
            else:
                run_future = result_holder.get("run_future")
                if run_future is not None and not run_future.cancelled():
                    # The client went away (navigated to another page,
                    # closed the tab) but the agent is still working.
                    # Don't lose the turn: await the run in the background
                    # and persist its outcome — ChatGPT behavior. The user
                    # sees the answer when they come back.
                    logger.info(
                        "Client disconnected mid-run %s — continuing in the background",
                        stream_context.run_id,
                    )
                    asyncio.get_running_loop().create_task(
                        persister.finish_disconnected_run(run_future)
                    )
                else:
                    logger.warning(
                        "Stream for conv %s ended with NO result and no "
                        "in-flight run. Nothing persisted.",
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


async def _run_blocking(
    *,
    session: AsyncSession,
    context: OrganizationContext,
    conv: Conversation,
    req: SendMessageRequest,
    setup: _AgentRunSetup,
    task_attachments: list,
) -> MessageResponse:
    """Blocking mode: run the agent to completion, persist the outcome (or
    error message), and return the assistant message."""
    memory, task_content, _, orchestrator, agent_settings, surface = setup
    try:
        run_context = authenticated_run_context(
            user_id=context.user_id,
            organization_id=context.organization_id,
            run_id=uuid.uuid4().hex,
            session_id=conv.id,
            event_id=conv.event_id,
            workspace_id=conv.event_id,
            agent_id=conv.agent_config_id or "default_agent",
            max_steps=agent_settings.max_steps,
            permissions=context.permissions,
            isolation_profile="shared-process",
        )
        proposal_handler, proposal_config = proposal_write_gate(
            run_context=run_context,
            tools_registry=orchestrator.tools_registry,
            session_factory=async_session,
        )
        agent_result = await orchestrator.execute_task(
            task=task_content,
            agent_settings=agent_settings,
            run_context=run_context,
            tool_context_extra=surface.tool_context_extra,
            mcp_servers=surface.mcp_servers,
            task_attachments=task_attachments,
            sandbox_base_dir=sandbox_base_dir(),
            hitl_handler=proposal_handler,
            hitl_config=proposal_config,
        )
    except Exception as e:
        logger.exception("Agent failed for conversation %s", conv.id)
        error_msg = Message(
            conversation_id=conv.id,
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
        conv.id,
        (assistant_msg.metadata_ or {}).get("steps_taken", 0),
    )

    return MessageResponse(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        metadata=assistant_msg.metadata_,
        created_at=assistant_msg.created_at,
    )


# ---- endpoint ----


# response_model=None: the honest return is a union with Response
# subclasses, which FastAPI cannot build a response model for.
@router.post("/{conversation_id}/messages", status_code=202, response_model=None)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    context: OrganizationContext = Depends(
        rate_limited_permission(Permission.RUN_CREATE)
    ),
    session: AsyncSession = Depends(get_session),
) -> dict | MessageResponse | StreamingResponse:
    """Dispatcher: prepare the turn, assemble the task, then hand off to one
    of the four execution modes (see module docstring)."""
    conv_for_refs = await _load_conv(session, context, conversation_id)
    # Durable upload refs for this turn (already in the store; validate they
    # are THIS org+conversation's uploads — a foreign id is dropped, not an
    # oracle).
    ref_rows = [await session.get(StoredFile, fid) for fid in req.file_refs]
    turn_uploads = resolve_upload_refs(
        ref_rows,
        organization_id=context.organization_id,
        event_id=conv_for_refs.event_id,
    )

    conv, memory = await _prepare_message(
        session,
        context,
        conversation_id,
        req.content,
        images=req.images,
        files=req.files,
        documents=req.documents,
        uploaded=turn_uploads,
    )
    task_content, task_attachments = await _assemble_task(session, conv, req)

    if not context.policy.allow_shared_process_execution:
        return await _enqueue_durable_run(session, context, conv, req)

    if req.team_id:
        return await _run_team_inline(session, context, conv, memory, req)

    # ---- single agent path ----
    setup = await _setup_single_agent(session, context, conv, req, memory, task_content)

    if req.stream:
        return _stream_response(
            conv=conv,
            context=context,
            req=req,
            setup=setup,
            task_attachments=task_attachments,
        )

    return await _run_blocking(
        session=session,
        context=context,
        conv=conv,
        req=req,
        setup=setup,
        task_attachments=task_attachments,
    )
