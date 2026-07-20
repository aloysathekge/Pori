"""Background task execution for fire-and-forget runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from sqlmodel import select

from pori import Agent, AgentMemory, AgentSettings, tool_registry

from .approvals import proposal_write_gate
from .conversation_runtime import (
    flush_context_artifact,
    flush_event_memory,
    load_event_memory,
)
from .database import async_session
from .event_bootstrap import (
    EVENT_BOOTSTRAP_RUN_KIND,
    execute_claimed_event_bootstrap,
)
from .models import (
    AgentConfig,
    Conversation,
    EventTrailEntry,
    Message,
    Organization,
    OrganizationMembership,
    Run,
    Task,
    TeamConfig,
)
from .orchestrator import build_orchestrator, sandbox_base_dir
from .research_outcomes import gate_and_index_research_run
from .run_outcome import (
    RunOutcome,
    json_safe,
    make_trace_record,
    make_usage_record,
    store_run_artifacts,
)
from .run_profiles import resolve_persisted_run_profile
from .run_surface import resolve_run_surface
from .runtime import authenticated_run_context
from .schedule_runtime import (
    record_schedule_terminal_trail,
    scheduled_denied_tools,
)
from .skills import load_skill_catalog
from .surface_builder import execute_claimed_surface_builder
from .surface_lifecycle import (
    mark_surface_run_started,
    reconcile_surface_run,
    surface_interaction_for_run,
)
from .surface_requests import (
    SURFACE_BUILDER_RUN_KIND,
    SurfaceRequestHandler,
)
from .surface_run_gate import evaluate_surface_interaction_context
from .task_execution import (
    DurableClarificationRecorder,
    add_task_lifecycle_message,
    synchronize_task_after_run,
    task_has_pending_proposal,
)
from .task_state import claim_task
from .team_execution import build_team_from_config
from .tenancy import ROLE_PERMISSIONS, OrganizationPolicy
from .tools import (
    EVENT_RECORD_HANDLER_CONTEXT_KEY,
    SURFACE_STATE_CONTEXT_KEY,
    EventEvidenceRecorder,
    EventRecordHandler,
    EventWebPageReader,
    SurfaceStateReader,
    TaskMutationHandler,
)
from .tools.surface_requests import SURFACE_REQUEST_CONTEXT_KEY

logger = logging.getLogger("aloy_backend")


def kernel_task_id_for_run(run_id: str) -> str:
    """Stable kernel task id derived from the Run row.

    Passing this as ``resume_task_id`` on every attempt means: first attempt
    creates the kernel task under this id; a re-claimed attempt resumes it
    from the checkpoint instead of minting a new task and starting over.
    """
    return f"run-{run_id[:12]}"


def _make_progress_checkpointer(
    run_id: str, worker_id: str, kernel_task_id: str
) -> Callable[[Agent], Awaitable[None]]:
    """Per-step callback: persist the loop checkpoint AND renew the lease.

    This is the heartbeat (docs/long-running.md Phase 2): while steps advance,
    the checkpoint lands in ``runs.progress`` (its own short session + commit,
    so it survives a later rollback of the main run transaction) and the lease
    is extended. A hung or dead worker stops renewing, the lease expires, and
    another worker re-claims + resumes from the last persisted step. Failures
    here must never break the run itself.
    """
    from .config import settings as app_settings

    async def _checkpoint(agent: Agent) -> None:
        try:
            async with async_session() as beat_session:
                run = await beat_session.get(Run, run_id)
                if run is None or run.lease_owner != worker_id:
                    return
                previous_steps = int((run.progress or {}).get("n_steps") or 0)
                run.progress = {
                    "kernel_task_id": kernel_task_id,
                    "n_steps": agent.state.n_steps,
                    "consecutive_failures": agent.state.consecutive_failures,
                    "current_activity": agent.state.current_activity,
                    "plan": agent._plan_snapshot(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                run.steps_taken = agent.state.n_steps
                # Heartbeat: progress is proof of life — extend the lease so a
                # long-but-advancing run is never re-claimed out from under us.
                run.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=max(
                        app_settings.worker_lease_seconds,
                        run.timeout_seconds + 30,
                    )
                )
                beat_session.add(run)
                if run.task_id and agent.state.n_steps > previous_steps:
                    task = await beat_session.get(Task, run.task_id)
                    if (
                        task is not None
                        and task.status == "in_progress"
                        and task.current_run_id == run.id
                    ):
                        beat_session.add(
                            EventTrailEntry(
                                organization_id=run.organization_id,
                                user_id=run.user_id,
                                event_id=run.event_id,
                                actor_id="worker:task-execution",
                                kind="task_progress",
                                summary=(
                                    agent.state.current_activity
                                    or f"Advanced {task.title}"
                                )[:1000],
                                run_id=run.id,
                                task_id=task.id,
                                payload={
                                    "step": agent.state.n_steps,
                                    "activity": agent.state.current_activity,
                                    "plan": agent._plan_snapshot(),
                                },
                            )
                        )
                await beat_session.commit()
        except Exception:  # pragma: no cover - heartbeat must never kill a run
            logger.warning(
                "Progress checkpoint failed for run %s", run_id, exc_info=True
            )

    return _checkpoint


def _inject_resume_checkpoint(
    memory: AgentMemory, run: Run, kernel_task_id: str
) -> bool:
    """Seed a reconstructed AgentMemory with the persisted loop checkpoint.

    The worker's AgentMemory is rebuilt from the database each attempt, so the
    kernel's own per-step checkpoint is not in it. If this run has prior
    progress, recreate the task record and restore the checkpoint so
    ``resume_task_id`` continues from the recorded step. Returns True when a
    resume was injected.
    """
    progress = run.progress or {}
    n_steps = int(progress.get("n_steps") or 0)
    if progress.get("kernel_task_id") != kernel_task_id or n_steps <= 0:
        return False
    if kernel_task_id not in memory.tasks:
        memory.create_task(kernel_task_id, run.task)
    memory.checkpoint_task_progress(
        kernel_task_id,
        n_steps=n_steps,
        consecutive_failures=0,  # fresh attempt: don't inherit a failure streak
        current_activity=str(progress.get("current_activity") or ""),
        plan=list(progress.get("plan") or []),
    )
    logger.info(
        "Run %s resuming kernel task %s from step %s",
        run.id,
        kernel_task_id,
        n_steps,
    )
    return True


async def execute_claimed_run(run_id: str, worker_id: str) -> None:
    """Execute a leased run and persist only while this worker owns the lease."""
    async with async_session() as session:
        run = await session.get(Run, run_id)
        if not run or run.lease_owner != worker_id or run.status != "running":
            logger.error("Background run %s not found", run_id)
            return
        if run.run_kind == EVENT_BOOTSTRAP_RUN_KIND:
            # Purpose-specific structured generation owns its own short
            # transaction lifecycle and never receives the general tool surface.
            await session.rollback()
            await execute_claimed_event_bootstrap(run_id, worker_id)
            return
        if run.run_kind == SURFACE_BUILDER_RUN_KIND:
            # Surface source is one structured model output. The dedicated
            # executor gives the model zero lifecycle tools and hands its
            # complete candidate to Aloy's trusted host pipeline.
            await session.rollback()
            await execute_claimed_surface_builder(run_id, worker_id)
            return

        metrics: dict | None = None

        task: Task | None = None
        task_started = False
        if run.task_id:
            task = await session.get(Task, run.task_id)
            if (
                task is not None
                and task.status == "queued"
                and task.current_run_id == run.id
            ):
                task = await claim_task(
                    session,
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    task_id=run.task_id,
                    run_id=run.id,
                    actor_id="worker:task-execution",
                )
                task_started = task is not None
                if task is None:
                    # Another Run won the compare-and-set claim. This stale
                    # queue row must terminate without touching tools.
                    run = await session.get(Run, run_id, populate_existing=True)
                    if run is not None and run.lease_owner == worker_id:
                        run.status = "cancelled"
                        run.success = False
                        run.cancel_requested = True
                        run.completed_at = datetime.now(timezone.utc)
                        run.lease_owner = None
                        run.lease_expires_at = None
                        session.add(run)
                        await session.commit()
                    return
            elif not (
                task is not None
                and task.status == "in_progress"
                and task.current_run_id == run.id
            ):
                # A stopped/replaced Task must never let its stale queue row run.
                run.status = "cancelled"
                run.success = False
                run.cancel_requested = True
                run.completed_at = datetime.now(timezone.utc)
                run.lease_owner = None
                run.lease_expires_at = None
                session.add(run)
                await session.commit()
                return

        clarification_recorder = DurableClarificationRecorder() if run.task_id else None
        evidence_recorder: EventEvidenceRecorder | None = None
        event_record_handler: EventRecordHandler | None = None
        surface_state_reader: SurfaceStateReader | None = None
        required_surface_interaction_id: str | None = None
        surface_gate_error: str | None = None
        conversation: Conversation | None = None

        try:
            surface_interaction = await surface_interaction_for_run(session, run.id)
            if surface_interaction is not None:
                required_surface_interaction_id = surface_interaction.id
            if await mark_surface_run_started(session, run=run):
                # Surface reasoning has its own semantic start transition so
                # Event SSE can refresh generated UI while the Run works.
                await session.commit()
            membership_result = await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == run.organization_id,
                    OrganizationMembership.user_id == run.user_id,
                    OrganizationMembership.status == "active",
                )
            )
            membership = membership_result.scalars().first()
            organization = await session.get(Organization, run.organization_id)
            if membership is None or organization is None:
                raise PermissionError("Run owner no longer has organization access")
            policy = OrganizationPolicy.model_validate(organization.policy or {})
            permissions = ROLE_PERMISSIONS.get(membership.role, frozenset())
            skill_catalog = await load_skill_catalog(
                session,
                organization_id=run.organization_id,
                user_id=run.user_id,
                role=membership.role,
            )
            run_context = authenticated_run_context(
                user_id=run.user_id,
                organization_id=run.organization_id,
                run_id=run.id,
                session_id=run.session_id,
                event_id=run.event_id,
                workspace_id=run.event_id,
                agent_id=run.agent_id,
                max_steps=run.max_steps,
                permissions=(permission.value for permission in permissions),
                isolation_profile="worker-process",
            )
            agent_config = None
            agent = None
            if run.conversation_id:
                conversation = await session.get(Conversation, run.conversation_id)
                if (
                    conversation is None
                    or conversation.organization_id != run.organization_id
                ):
                    raise ValueError("Conversation is unavailable")
                if conversation.agent_config_id:
                    agent_config = await session.get(
                        AgentConfig, conversation.agent_config_id
                    )
                    if (
                        agent_config is None
                        or agent_config.organization_id != run.organization_id
                    ):
                        raise ValueError("Agent config is unavailable")
            if task_started and conversation is not None and task is not None:
                add_task_lifecycle_message(
                    session,
                    conversation_id=conversation.id,
                    task=task,
                    run_id=run.id,
                    status="in_progress",
                    content=f"Started **{task.title}**.",
                )
                conversation.updated_at = datetime.now(timezone.utc)
                session.add(conversation)
                await session.commit()
            memory = await load_event_memory(
                session,
                organization_id=run.organization_id,
                user_id=run.user_id,
                conversation=conversation,
                event_id=run.event_id,
                session_id=run.session_id,
                agent_id=run.agent_id,
            )
            await session.commit()
            if run.team_config_id:
                team_config = await session.get(TeamConfig, run.team_config_id)
                if (
                    team_config is None
                    or team_config.organization_id != run.organization_id
                ):
                    raise ValueError("Team config is unavailable")
                team = build_team_from_config(
                    team_config,
                    run.task,
                    memory=memory,
                    run_context=run_context,
                )
                team_result = await asyncio.wait_for(
                    team.run(), timeout=run.timeout_seconds
                )
                final_state = team_result.get("final_state") or {}
                final: dict | None = {
                    "final_answer": final_state.get("final_answer"),
                    "reasoning": final_state.get("reasoning"),
                }
                metrics = team_result.get("metrics")
                result = {
                    "success": team_result.get("completed", False),
                    "steps_taken": team_result.get("steps_taken", 0),
                    "trace": None,
                }
            else:
                # The worker path resolves the SAME capability surface as the
                # chat path — connections, MCP servers, file library, gated
                # denials. (It used to build runs without any of them: the
                # drift the 2026-07-11 audit flagged.)
                surface = await resolve_run_surface(
                    session,
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    event_id=run.event_id,
                    policy=policy,
                )
                execution_memory = memory
                schedule_denials = scheduled_denied_tools(run)
                orchestrator = build_orchestrator(
                    shared_memory=memory,
                    agent_config=agent_config,
                    allowed_tools=policy.allowed_tools or None,
                    denied_tools=tuple(
                        set(surface.denied_tools).union(schedule_denials)
                    ),
                    allowed_capability_groups=(
                        policy.allowed_capability_groups or None
                    ),
                    allowed_provider_profiles=(
                        policy.allowed_provider_profiles or None
                    ),
                    allowed_models=policy.allowed_models or None,
                    skill_catalog=skill_catalog,
                    run_profile=resolve_persisted_run_profile(run.run_profile),
                    enable_surface_requests=bool(run.event_id and not run.cron_job_id),
                )
                proposal_handler, proposal_config = proposal_write_gate(
                    run_context=run_context,
                    tools_registry=getattr(
                        orchestrator, "tools_registry", tool_registry()
                    ),
                    session_factory=async_session,
                )
                evidence_recorder = EventEvidenceRecorder(
                    run_context=run_context,
                    task_id=run.task_id,
                    session_factory=async_session,
                )
                event_record_handler = EventRecordHandler(
                    run_context=run_context,
                    task_id=run.task_id,
                    session_factory=async_session,
                )
                surface_state_reader = SurfaceStateReader(
                    run_context=run_context,
                    session_factory=async_session,
                )
                tool_context = {
                    **surface.tool_context_extra,
                    "task_mutator": TaskMutationHandler(
                        run_context=run_context,
                        session_factory=async_session,
                    ),
                    SURFACE_STATE_CONTEXT_KEY: surface_state_reader,
                    "web_evidence_recorder": evidence_recorder,
                    "web_page_reader": EventWebPageReader(),
                    EVENT_RECORD_HANDLER_CONTEXT_KEY: event_record_handler,
                    **(
                        {
                            SURFACE_REQUEST_CONTEXT_KEY: SurfaceRequestHandler(
                                run_context=run_context,
                                session_factory=async_session,
                            )
                        }
                        if run.event_id
                        else {}
                    ),
                }
                # MCP tools do not yet expose host-verifiable authority metadata.
                # They are therefore unavailable to unattended Schedule runs.
                mcp_servers = () if run.cron_job_id else surface.mcp_servers
                # Resume-not-restart: run under a stable kernel task id. A
                # first attempt creates it; a re-claim after a crash/expired
                # lease injects the persisted checkpoint and continues from
                # the recorded step (docs/long-running.md Phase 2).
                kernel_task_id = kernel_task_id_for_run(run.id)
                if execution_memory is None:
                    execution_memory = AgentMemory(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        agent_id=run.agent_id,
                        session_id=run.session_id,
                    )
                _inject_resume_checkpoint(execution_memory, run, kernel_task_id)
                checkpoint = _make_progress_checkpointer(
                    run.id, worker_id, kernel_task_id
                )
                if clarification_recorder is not None:
                    tool_context["clarify_handler"] = clarification_recorder
                result = await asyncio.wait_for(
                    orchestrator.execute_task(
                        task=run.task,
                        agent_settings=AgentSettings(max_steps=run.max_steps),
                        run_context=run_context,
                        memory=execution_memory,
                        resume_task_id=kernel_task_id,
                        on_step_end=checkpoint,
                        sandbox_base_dir=sandbox_base_dir(),
                        tool_context_extra=tool_context,
                        mcp_servers=mcp_servers,
                        hitl_handler=proposal_handler,
                        hitl_config=proposal_config,
                    ),
                    timeout=run.timeout_seconds,
                )
                agent = result.get("agent")
                final = agent.memory.get_final_answer() if agent else None
                metrics = result.get("result", {}).get("metrics")
                # Salvage passthrough: a run that exhausted its budget without
                # an answer still surfaces its best-effort handoff (Phase 1).
                partial = (result.get("result") or {}).get("partial_result")
                if partial and not (final or {}).get("final_answer"):
                    final = {
                        "final_answer": partial.get("summary"),
                        "reasoning": (
                            f"Partial result — run stopped early "
                            f"({partial.get('reason', 'unknown')})."
                        ),
                    }

            # Stop is written by the API through a different session while the
            # agent is running. Refresh the cancellation bit before deciding
            # the terminal Run/Task state.
            await session.refresh(run, attribute_names=["cancel_requested"])
            if run.cancel_requested:
                run.status = "cancelled"
                run.success = False
            else:
                run.status = "completed"
                run.success = bool(result.get("success"))
            run.steps_taken = int(result.get("steps_taken") or 0)
            run.final_answer = (final or {}).get("final_answer")
            run.reasoning = (final or {}).get("reasoning")
            # json_safe everything destined for JSON columns — rich objects
            # (TokenUsage etc.) otherwise throw at flush and lose the message
            # (the drift class the run_outcome refactor exists to prevent).
            run.metrics = json_safe(metrics)
            run.selected_skills = json_safe(result.get("selected_skills")) or []
            run.artifacts = json_safe(result.get("artifacts")) or []
            run.plan = json_safe(result.get("plan")) or []
            usage = make_usage_record(
                organization_id=run.organization_id,
                user_id=run.user_id,
                run_id=run.id,
                conversation_id=run.conversation_id,
                metrics=run.metrics,
            )
            if usage is not None:
                session.add(usage)
            trace_data = json_safe(result.get("trace")) or {}
            run.prompt_fingerprint = trace_data.get("prompt_fingerprint")
            run.tool_surface_fingerprint = trace_data.get("tool_surface_fingerprint")
            run.execution_receipts = trace_data.get("execution_receipts") or []
            if required_surface_interaction_id and not run.cancel_requested:
                observed_interaction_ids = set(
                    surface_state_reader.interaction_ids
                    if surface_state_reader is not None
                    else ()
                )
                # The reader also persists proof on the interaction. A fresh
                # session makes a resumed worker see a successful read from a
                # prior process even when all in-memory instrumentation is gone.
                async with async_session() as read_session:
                    durable_interaction = await surface_interaction_for_run(
                        read_session, run.id
                    )
                    if (
                        durable_interaction is not None
                        and durable_interaction.context_read_run_id == run.id
                    ):
                        observed_interaction_ids.add(durable_interaction.id)
                surface_gate = evaluate_surface_interaction_context(
                    required_interaction_id=required_surface_interaction_id,
                    observed_interaction_ids=observed_interaction_ids,
                )
                gate_receipt = surface_gate.receipt()
                run.execution_receipts = [
                    *(run.execution_receipts or []),
                    gate_receipt,
                ]
                run.metrics = run.metrics if isinstance(run.metrics, dict) else {}
                run.metrics["surface_interaction_context_gate"] = gate_receipt
                if not surface_gate.accepted:
                    surface_gate_error = surface_gate.errors[0]
                    run.success = False
                    run.artifacts = []
                    run.final_answer = (
                        "Aloy could not safely load the selection from this "
                        "Surface. Please retry the action."
                    )
                    run.reasoning = surface_gate_error
                    final = {
                        "final_answer": run.final_answer,
                        "reasoning": run.reasoning,
                    }
            if run.artifacts:
                store_run_artifacts(
                    session,
                    conversation,
                    run,
                    RunOutcome(
                        task=run.task,
                        final_answer=run.final_answer or "",
                        reasoning=run.reasoning,
                        success=run.success,
                        steps_taken=run.steps_taken,
                        metrics=run.metrics,
                        trace=trace_data,
                        artifacts=run.artifacts,
                        run_id=run.id,
                        organization_id=run.organization_id,
                        event_id=run.event_id,
                        session_id=run.session_id,
                        agent_id=run.agent_id,
                    ),
                )
            if (
                (run.run_profile or {}).get("profile_id") == "aloy.sourced-research"
                and evidence_recorder is not None
                and event_record_handler is not None
                and not run.cancel_requested
            ):
                research_gate = await gate_and_index_research_run(
                    session,
                    run=run,
                    artifacts=run.artifacts,
                    evidence_ids=evidence_recorder.evidence_ids,
                    record_ids=event_record_handler.record_ids,
                )
                receipts = list(run.execution_receipts or [])
                receipts.append(research_gate.receipt())
                run.execution_receipts = receipts
                run.metrics = dict(run.metrics or {})
                run.metrics["research_gate"] = research_gate.receipt()
                if not research_gate.accepted:
                    run.success = False
                    failure = "Research quality gate failed: " + "; ".join(
                        research_gate.errors
                    )
                    run.final_answer = (
                        f"{run.final_answer}\n\n{failure}"
                        if run.final_answer
                        else failure
                    )
                    final = {
                        "final_answer": run.final_answer,
                        "reasoning": run.reasoning,
                    }
            trace_record = make_trace_record(
                organization_id=run.organization_id,
                user_id=run.user_id,
                event_id=run.event_id,
                run_id=run.id,
                conversation_id=run.conversation_id,
                trace=trace_data,
            )
            if trace_record is not None:
                session.add(trace_record)

            pending_proposal = False
            if run.task_id:
                # Proposal staging commits through its own short session while
                # this execution session is open. Read through a fresh session
                # so SQLite and Postgres both see the committed Proposal.
                async with async_session() as proposal_session:
                    pending_proposal = await task_has_pending_proposal(
                        proposal_session, run_id=run.id
                    )
                task = await synchronize_task_after_run(
                    session,
                    run=run,
                    clarification=(
                        clarification_recorder.request
                        if clarification_recorder is not None
                        else None
                    ),
                    pending_proposal=pending_proposal,
                )

            surface_outcome_message: Message | None = None
            if conversation is not None and run.status == "completed":
                answer = (final or {}).get("final_answer") or ""
                if task is not None and task.status == "blocked":
                    request = (
                        clarification_recorder.request
                        if clarification_recorder is not None
                        else None
                    )
                    question = str((request or {}).get("question") or task.blocker)
                    options = list((request or {}).get("options") or [])
                    choices = "\n\nOptions: " + ", ".join(options) if options else ""
                    add_task_lifecycle_message(
                        session,
                        conversation_id=conversation.id,
                        task=task,
                        run_id=run.id,
                        status="blocked",
                        content=f"**{task.title}** needs your input: {question}{choices}",
                    )
                elif task is not None and task.status == "waiting_approval":
                    add_task_lifecycle_message(
                        session,
                        conversation_id=conversation.id,
                        task=task,
                        run_id=run.id,
                        status="waiting_approval",
                        content=(
                            f"**{task.title}** is waiting for your approval. "
                            "Review the pending decision in this Event."
                        ),
                    )
                elif task is not None and task.status == "failed":
                    add_task_lifecycle_message(
                        session,
                        conversation_id=conversation.id,
                        task=task,
                        run_id=run.id,
                        status="failed",
                        content=(
                            answer
                            or f"**{task.title}** did not satisfy its definition "
                            "of done. You can retry it."
                        ),
                    )
                else:
                    surface_outcome_message = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=answer,
                        metadata_={
                            "run_id": run.id,
                            "reasoning": (final or {}).get("reasoning"),
                            "steps_taken": run.steps_taken,
                            "metrics": run.metrics,
                            "selected_skills": run.selected_skills,
                            "artifacts": run.artifacts,
                            "plan": run.plan,
                            **(
                                {
                                    "kind": "task_result",
                                    "task_id": task.id,
                                    "task_status": task.status,
                                }
                                if task is not None
                                else {}
                            ),
                        },
                    )
                    session.add(surface_outcome_message)
                conversation.updated_at = datetime.now(timezone.utc)
                session.add(conversation)
                if memory is not None:
                    await flush_event_memory(
                        session,
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        memory=memory,
                    )
                    await flush_context_artifact(
                        session,
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        event_id=run.event_id,
                        conversation_id=conversation.id,
                        run_id=run.id,
                        memory=memory,
                        diagnostics=(
                            agent.context_diagnostics.model_dump(mode="json")
                            if agent is not None and agent.context_diagnostics
                            else None
                        ),
                    )

            await reconcile_surface_run(
                session,
                run=run,
                outcome_message=surface_outcome_message,
                error=surface_gate_error,
            )
            await record_schedule_terminal_trail(session, run=run)

        except Exception:
            logger.exception("Background run %s failed", run_id)
            await session.rollback()
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                logger.warning("Worker %s lost lease for run %s", worker_id, run_id)
                return
            run.status = "pending" if run.attempt_count < run.max_attempts else "failed"
            run.success = False
            run.metrics = json_safe(metrics)
            if run.status == "failed" and run.task_id:
                task = await synchronize_task_after_run(session, run=run)
                if task is not None and run.conversation_id:
                    conversation = await session.get(Conversation, run.conversation_id)
                    if conversation is not None:
                        add_task_lifecycle_message(
                            session,
                            conversation_id=conversation.id,
                            task=task,
                            run_id=run.id,
                            status="failed",
                            content=(
                                f"**{task.title}** could not finish after its safe "
                                "retry attempts. You can retry it."
                            ),
                        )
                        conversation.updated_at = datetime.now(timezone.utc)
                        session.add(conversation)
            if run.status == "failed":
                await reconcile_surface_run(
                    session,
                    run=run,
                    error="The worker exhausted its safe retry attempts.",
                )
                await record_schedule_terminal_trail(session, run=run)

        try:
            current = await session.get(Run, run_id)
            if current is None or current.lease_owner != worker_id:
                logger.warning("Worker %s lost lease for run %s", worker_id, run_id)
                await session.rollback()
                return
            run.completed_at = (
                datetime.now(timezone.utc)
                if run.status in {"completed", "failed", "cancelled"}
                else None
            )
            run.lease_owner = None
            run.lease_expires_at = None
            session.add(run)
            await session.commit()
            logger.info("Background run %s finished: %s", run_id, run.status)
        except Exception:
            # The commit itself failed (e.g. an unserializable payload). Never
            # let this escape — an uncaught error here crashed the worker, the
            # lease expired, the run was re-claimed, and the next worker crashed
            # identically: a poison-pill loop. Mark the run failed with a
            # minimal write instead of retrying the poisoned payload.
            logger.exception("Final commit failed for run %s", run_id)
            await session.rollback()
            try:
                poisoned = await session.get(Run, run_id)
                if poisoned is not None and poisoned.lease_owner == worker_id:
                    poisoned.status = "failed"
                    poisoned.success = False
                    poisoned.completed_at = datetime.now(timezone.utc)
                    poisoned.lease_owner = None
                    poisoned.lease_expires_at = None
                    session.add(poisoned)
                    await reconcile_surface_run(
                        session,
                        run=poisoned,
                        error="The Run outcome could not be persisted safely.",
                    )
                    await session.commit()
            except Exception:
                logger.exception("Could not mark poisoned run %s failed", run_id)
                await session.rollback()
