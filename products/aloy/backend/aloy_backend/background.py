"""Background task execution for fire-and-forget runs."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from pori import AgentMemory, AgentSettings
from sqlmodel import select

from .conversation_runtime import (
    flush_context_artifact,
    flush_conversation_memory,
    load_conversation_memory,
)
from .database import async_session
from .models import (
    AgentConfig,
    Conversation,
    Message,
    Organization,
    OrganizationMembership,
    Run,
    TeamConfig,
    TraceRecord,
    UsageRecord,
)
from .orchestrator import build_orchestrator
from .runtime import authenticated_run_context
from .skills import load_skill_catalog
from .tenancy import ROLE_PERMISSIONS, OrganizationPolicy

logger = logging.getLogger("aloy_backend")


def _serialize_metrics(metrics: dict | None) -> dict | None:
    """Convert metrics to a JSON-safe dict (handles pydantic/dataclass objects)."""
    if metrics is None:
        return None

    # Round-trip through JSON with a custom serializer
    def _default(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.loads(json.dumps(metrics, default=_default))


def kernel_task_id_for_run(run_id: str) -> str:
    """Stable kernel task id derived from the Run row.

    Passing this as ``resume_task_id`` on every attempt means: first attempt
    creates the kernel task under this id; a re-claimed attempt resumes it
    from the checkpoint instead of minting a new task and starting over.
    """
    return f"run-{run_id[:12]}"


def _make_progress_checkpointer(run_id: str, worker_id: str, kernel_task_id: str):
    """Per-step callback: persist the loop checkpoint AND renew the lease.

    This is the heartbeat (docs/long-running.md Phase 2): while steps advance,
    the checkpoint lands in ``runs.progress`` (its own short session + commit,
    so it survives a later rollback of the main run transaction) and the lease
    is extended. A hung or dead worker stops renewing, the lease expires, and
    another worker re-claims + resumes from the last persisted step. Failures
    here must never break the run itself.
    """
    from .config import settings as app_settings

    async def _checkpoint(agent) -> None:
        try:
            async with async_session() as beat_session:
                run = await beat_session.get(Run, run_id)
                if run is None or run.lease_owner != worker_id:
                    return
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
                await beat_session.commit()
        except Exception:  # pragma: no cover - heartbeat must never kill a run
            logger.warning(
                "Progress checkpoint failed for run %s", run_id, exc_info=True
            )

    return _checkpoint


def _inject_resume_checkpoint(memory, run: Run, kernel_task_id: str) -> bool:
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

        try:
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
                agent_id=run.agent_id,
                max_steps=run.max_steps,
                permissions=(permission.value for permission in permissions),
                isolation_profile="worker-process",
            )
            conversation = None
            memory = None
            agent_config = None
            agent = None
            if run.conversation_id:
                conversation = await session.get(Conversation, run.conversation_id)
                if (
                    conversation is None
                    or conversation.organization_id != run.organization_id
                ):
                    raise ValueError("Conversation is unavailable")
                memory = await load_conversation_memory(
                    session,
                    organization_id=run.organization_id,
                    user_id=run.user_id,
                    conversation=conversation,
                )
                if conversation.agent_config_id:
                    agent_config = await session.get(
                        AgentConfig, conversation.agent_config_id
                    )
                    if (
                        agent_config is None
                        or agent_config.organization_id != run.organization_id
                    ):
                        raise ValueError("Agent config is unavailable")
            if run.team_config_id:
                from .routes.teams import _build_team_from_config

                team_config = await session.get(TeamConfig, run.team_config_id)
                if (
                    team_config is None
                    or team_config.organization_id != run.organization_id
                ):
                    raise ValueError("Team config is unavailable")
                team = _build_team_from_config(
                    team_config,
                    run.task,
                    memory=memory,
                    run_context=run_context,
                )
                team_result = await asyncio.wait_for(
                    team.run(), timeout=run.timeout_seconds
                )
                final_state = team_result.get("final_state") or {}
                final = {
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
                orchestrator = build_orchestrator(
                    shared_memory=memory,
                    agent_config=agent_config,
                    allowed_tools=policy.allowed_tools or None,
                    denied_tools=policy.denied_tools,
                    allowed_capability_groups=(
                        policy.allowed_capability_groups or None
                    ),
                    allowed_provider_profiles=(
                        policy.allowed_provider_profiles or None
                    ),
                    allowed_models=policy.allowed_models or None,
                    skill_catalog=skill_catalog,
                )
                # Resume-not-restart: run under a stable kernel task id. A
                # first attempt creates it; a re-claim after a crash/expired
                # lease injects the persisted checkpoint and continues from
                # the recorded step (docs/long-running.md Phase 2).
                kernel_task_id = kernel_task_id_for_run(run.id)
                if memory is None:
                    memory = AgentMemory(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        agent_id=run.agent_id,
                        session_id=run.session_id,
                    )
                _inject_resume_checkpoint(memory, run, kernel_task_id)
                checkpoint = _make_progress_checkpointer(
                    run.id, worker_id, kernel_task_id
                )
                result = await asyncio.wait_for(
                    orchestrator.execute_task(
                        task=run.task,
                        agent_settings=AgentSettings(max_steps=run.max_steps),
                        run_context=run_context,
                        memory=memory,
                        resume_task_id=kernel_task_id,
                        on_step_end=checkpoint,
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

            if run.cancel_requested:
                run.status = "cancelled"
                run.success = False
            else:
                run.status = "completed"
                run.success = bool(result.get("success"))
            run.steps_taken = int(result.get("steps_taken") or 0)
            run.final_answer = (final or {}).get("final_answer")
            run.reasoning = (final or {}).get("reasoning")
            run.metrics = _serialize_metrics(metrics)
            run.selected_skills = result.get("selected_skills") or []
            run.artifacts = result.get("artifacts") or []
            run.plan = result.get("plan") or []
            if run.metrics and isinstance(run.metrics, dict):
                tokens = run.metrics.get("tokens") or {}
                session.add(
                    UsageRecord(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        run_id=run.id,
                        conversation_id=run.conversation_id,
                        provider=(run.metrics.get("model") or "").split("/")[0],
                        model=(run.metrics.get("model") or "").split("/")[-1],
                        input_tokens=int(tokens.get("input", 0)),
                        output_tokens=int(tokens.get("output", 0)),
                        total_tokens=int(tokens.get("total", 0)),
                        estimated_cost=float(
                            (run.metrics.get("cost_usd") or "$0").replace("$", "") or 0
                        ),
                    )
                )
            trace_data = result.get("trace") or {}
            run.prompt_fingerprint = trace_data.get("prompt_fingerprint")
            run.tool_surface_fingerprint = trace_data.get("tool_surface_fingerprint")
            run.execution_receipts = trace_data.get("execution_receipts") or []
            if trace_data:
                session.add(
                    TraceRecord(
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        run_id=run.id,
                        conversation_id=run.conversation_id,
                        trace_data=trace_data,
                        duration_seconds=float(
                            (trace_data.get("duration") or "0s").replace("s", "") or 0
                        ),
                        total_spans=int(trace_data.get("total_spans", 0)),
                        status=trace_data.get("status", "ok"),
                    )
                )

            if conversation is not None and run.status == "completed":
                answer = (final or {}).get("final_answer") or ""
                session.add(
                    Message(
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
                        },
                    )
                )
                conversation.updated_at = datetime.now(timezone.utc)
                session.add(conversation)
                if memory is not None:
                    await flush_conversation_memory(
                        session,
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        memory=memory,
                    )
                    await flush_context_artifact(
                        session,
                        organization_id=run.organization_id,
                        user_id=run.user_id,
                        conversation_id=conversation.id,
                        run_id=run.id,
                        memory=memory,
                        diagnostics=(
                            agent.context_diagnostics.model_dump(mode="json")
                            if agent is not None and agent.context_diagnostics
                            else None
                        ),
                    )

        except Exception:
            logger.exception("Background run %s failed", run_id)
            await session.rollback()
            run = await session.get(Run, run_id)
            if run is None or run.lease_owner != worker_id:
                logger.warning("Worker %s lost lease for run %s", worker_id, run_id)
                return
            run.status = "pending" if run.attempt_count < run.max_attempts else "failed"
            run.success = False

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
