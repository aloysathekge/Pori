"""Durable execution controls and lifecycle synchronization for Event Tasks.

R2 made Task an executable state object. This module is the R3 command boundary:
explicit user controls create or resume durable ``Run`` rows, compact lifecycle
messages return to the selected Conversation, and worker outcomes advance Task
state without letting HTTP handlers or the agent loop invent transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from .models import ActionProposal, Conversation, Event, Message, Run, Task
from .run_budgets import resolve_run_budget
from .run_profiles import SOURCED_RESEARCH_RUN_PROFILE
from .task_state import mutate_task
from .tenancy import OrganizationContext

TaskControl = Literal["work", "retry", "resume"]


class TaskExecutionError(ValueError):
    """A Task execution command cannot be applied to the current durable state."""


@dataclass(frozen=True)
class TaskExecutionResult:
    task: Task
    run: Run
    idempotent: bool = False


class DurableClarificationRecorder:
    """Non-blocking ``ask_user`` handler for worker Runs.

    A durable worker must never wait on stdin or an in-memory bridge. The first
    question is recorded and returned to the Task lifecycle after the bounded
    Run yields; Resume later continues the same Run from its checkpoint.
    """

    RESPONSE = (
        "Clarification has been recorded for the user. Do not guess or continue "
        "the blocked part of the task; summarize useful progress and stop."
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._request: dict[str, Any] | None = None

    def __call__(self, question: str, options: list[str]) -> str:
        with self._lock:
            if self._request is None:
                self._request = {
                    "question": question.strip() or "Aloy needs more information.",
                    "options": list(options),
                }
        return self.RESPONSE

    @property
    def request(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._request) if self._request is not None else None


def assemble_task_instructions(event: Event, task: Task) -> str:
    """Build the bounded work order; Conversation history is hydrated separately."""
    instructions = task.instructions.strip() or task.title.strip()
    definition = task.definition_of_done.strip() or (
        "Complete the requested work and clearly explain the result."
    )
    return "\n".join(
        [
            "<aloy-task>",
            f"Event: {event.title}",
            f"Event summary: {event.summary or '(none)'}",
            f"Task: {task.title}",
            f"Instructions: {instructions}",
            f"Definition of done: {definition}",
            f"Priority: {task.priority}",
            f"Execution profile: {task.execution_profile}",
            "Work only within this Event and its selected Conversation context.",
            "If required information is missing, use ask_user instead of guessing.",
            (
                "Stage external consequences as Proposals. Never claim that an "
                "external action happened until a committed receipt exists."
            ),
            "</aloy-task>",
        ]
    )


def add_task_lifecycle_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    task: Task,
    run_id: str,
    status: str,
    content: str,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        metadata_={
            "kind": "task_lifecycle",
            "task_id": task.id,
            "run_id": run_id,
            "task_status": status,
        },
    )
    session.add(message)
    return message


async def _selected_conversation(
    session: AsyncSession,
    *,
    event: Event,
    task: Task,
) -> Conversation:
    conversation_id = task.origin_conversation_id or event.primary_conversation_id
    if conversation_id is None:
        raise TaskExecutionError(
            "Choose a Conversation for this Task before starting work"
        )
    conversation = await session.get(Conversation, conversation_id)
    if (
        conversation is None
        or conversation.event_id != event.id
        or conversation.organization_id != event.organization_id
        or conversation.user_id != event.user_id
    ):
        raise TaskExecutionError("The Task's selected Conversation is unavailable")
    if not event.is_life and event.primary_conversation_id != conversation.id:
        raise TaskExecutionError(
            "Dedicated Event Tasks must report to the canonical Conversation"
        )
    return conversation


async def _active_current_run(
    session: AsyncSession,
    *,
    task: Task,
) -> Run | None:
    if not task.current_run_id:
        return None
    run = await session.get(Run, task.current_run_id)
    if (
        run is not None
        and run.task_id == task.id
        and run.event_id == task.event_id
        and run.organization_id == task.organization_id
        and run.user_id == task.user_id
        and run.status in {"pending", "running"}
    ):
        return run
    return None


async def _transition_to_queued(
    session: AsyncSession,
    *,
    event: Event,
    task: Task,
    run: Run,
    actor_id: str,
    action: str,
) -> None:
    entry = await mutate_task(
        session,
        event=event,
        task=task,
        changes={"status": "queued", "blocker": ""},
        actor_id=actor_id,
        source_run_id=run.id,
    )
    task.current_run_id = run.id
    entry.summary = f"Queued task {task.title}"
    entry.payload["action"] = action
    entry.payload["after"]["current_run_id"] = run.id
    session.add_all([task, entry])


async def queue_task_run(
    session: AsyncSession,
    *,
    event: Event,
    task: Task,
    context: OrganizationContext,
    control: TaskControl,
    response: str | None = None,
) -> TaskExecutionResult:
    """Create or resume the one durable Run represented by a Task command."""
    active = await _active_current_run(session, task=task)
    if active is not None and task.status in {"queued", "in_progress"}:
        return TaskExecutionResult(task=task, run=active, idempotent=True)

    expected: dict[TaskControl, frozenset[str]] = {
        "work": frozenset({"open"}),
        "retry": frozenset({"failed", "cancelled"}),
        "resume": frozenset({"blocked", "waiting_approval"}),
    }
    if task.status not in expected[control]:
        raise TaskExecutionError(
            f"Cannot {control} a Task while it is {task.status.replace('_', ' ')}"
        )
    if event.lifecycle == "archived":
        raise TaskExecutionError("Event is archived")

    conversation = await _selected_conversation(session, event=event, task=task)
    budget = resolve_run_budget(context.policy, task.budget_policy or {})

    if control == "resume":
        run = (
            await session.get(Run, task.current_run_id) if task.current_run_id else None
        )
        if (
            run is None
            or run.task_id != task.id
            or run.event_id != event.id
            or run.conversation_id != conversation.id
        ):
            raise TaskExecutionError("The blocked Task no longer has a resumable Run")
        if task.status == "waiting_approval":
            pending = (
                await session.execute(
                    select(ActionProposal.id).where(
                        col(ActionProposal.origin_run_id) == run.id,
                        col(ActionProposal.status).in_(
                            [
                                "proposed",
                                "routed",
                                "pending",
                                "approved",
                                "executing",
                                "indeterminate",
                            ]
                        ),
                    )
                )
            ).first()
            if pending is not None:
                raise TaskExecutionError(
                    "This Task is still waiting for its Proposal decision or receipt"
                )
        prompt = assemble_task_instructions(event, task)
        if response and response.strip():
            prompt += f"\n<user-response>\n{response.strip()}\n</user-response>"
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=response.strip(),
                    metadata_={"kind": "task_resume", "task_id": task.id},
                )
            )
        run.task = prompt
        run.status = "pending"
        run.success = False
        run.cancel_requested = False
        run.completed_at = None
        run.lease_owner = None
        run.lease_expires_at = None
        run.attempt_count = 0
        run.max_steps = budget.max_steps
        run.max_tool_calls = budget.max_tool_calls
        run.max_tokens = budget.max_tokens
        run.max_cost_usd = budget.max_cost_usd
        run.timeout_seconds = budget.timeout_seconds
        session.add(run)
    else:
        if control == "retry" and task.status == "cancelled":
            # Preserve the state-machine contract rather than inventing a direct
            # cancelled -> queued edge.
            await mutate_task(
                session,
                event=event,
                task=task,
                changes={"status": "open"},
                actor_id=context.user_id,
            )
        run = Run(
            organization_id=context.organization_id,
            user_id=context.user_id,
            event_id=event.id,
            task_id=task.id,
            agent_id=(
                task.assigned_agent_id
                or conversation.agent_config_id
                or "default_agent"
            ),
            session_id=conversation.id,
            conversation_id=conversation.id,
            task=assemble_task_instructions(event, task),
            max_steps=budget.max_steps,
            max_tool_calls=budget.max_tool_calls,
            max_tokens=budget.max_tokens,
            max_cost_usd=budget.max_cost_usd,
            max_attempts=context.policy.max_attempts,
            timeout_seconds=budget.timeout_seconds,
            status="pending",
            run_profile=(
                SOURCED_RESEARCH_RUN_PROFILE.descriptor()
                if task.execution_profile == "sourced_research"
                else None
            ),
        )
        run.root_run_id = run.id
        session.add(run)

    await _transition_to_queued(
        session,
        event=event,
        task=task,
        run=run,
        actor_id=context.user_id,
        action={"work": "queued", "retry": "retried", "resume": "resumed"}[control],
    )
    add_task_lifecycle_message(
        session,
        conversation_id=conversation.id,
        task=task,
        run_id=run.id,
        status="queued",
        content=(
            f"Queued **{task.title}**. Aloy will start when this Event's active "
            "work slot is available."
        ),
    )
    conversation.updated_at = datetime.now(timezone.utc)
    session.add(conversation)
    await session.commit()
    await session.refresh(task)
    await session.refresh(run)
    return TaskExecutionResult(task=task, run=run)


async def stop_task_run(
    session: AsyncSession,
    *,
    event: Event,
    task: Task,
    actor_id: str,
) -> TaskExecutionResult | None:
    if task.status == "cancelled":
        return None
    if task.status not in {"queued", "in_progress", "blocked", "waiting_approval"}:
        raise TaskExecutionError(
            f"Cannot stop a Task while it is {task.status.replace('_', ' ')}"
        )
    run = await session.get(Run, task.current_run_id) if task.current_run_id else None
    if run is None or run.task_id != task.id:
        raise TaskExecutionError("The Task's current Run is unavailable")
    run.cancel_requested = True
    if run.status == "pending":
        run.status = "cancelled"
        run.completed_at = datetime.now(timezone.utc)
    session.add(run)
    conversation = await _selected_conversation(session, event=event, task=task)
    await mutate_task(
        session,
        event=event,
        task=task,
        changes={"status": "cancelled", "blocker": ""},
        actor_id=actor_id,
        source_run_id=run.id,
    )
    add_task_lifecycle_message(
        session,
        conversation_id=conversation.id,
        task=task,
        run_id=run.id,
        status="cancelled",
        content=f"Stopped **{task.title}**. You can retry it when you are ready.",
    )
    conversation.updated_at = datetime.now(timezone.utc)
    session.add(conversation)
    await session.commit()
    await session.refresh(task)
    await session.refresh(run)
    return TaskExecutionResult(task=task, run=run)


async def task_has_pending_proposal(session: AsyncSession, *, run_id: str) -> bool:
    return (
        await session.execute(
            select(ActionProposal.id).where(
                ActionProposal.origin_run_id == run_id,
                col(ActionProposal.status).in_(
                    [
                        "proposed",
                        "routed",
                        "pending",
                        "approved",
                        "executing",
                        "indeterminate",
                    ]
                ),
            )
        )
    ).first() is not None


async def synchronize_task_after_run(
    session: AsyncSession,
    *,
    run: Run,
    clarification: dict[str, Any] | None = None,
    pending_proposal: bool = False,
) -> Task | None:
    """Advance the owning Task after a worker attempt, preserving newer state."""
    if not run.task_id:
        return None
    task = await session.get(Task, run.task_id, populate_existing=True)
    event = await session.get(Event, run.event_id, populate_existing=True)
    if (
        task is None
        or event is None
        or task.current_run_id != run.id
        or task.organization_id != run.organization_id
        or task.user_id != run.user_id
    ):
        return task
    if task.status != "in_progress":
        return task

    status: str | None = None
    changes: dict[str, Any] = {}
    if run.status == "cancelled" or run.cancel_requested:
        status = "cancelled"
    elif clarification is not None:
        status = "blocked"
        changes["blocker"] = str(clarification.get("question") or "")
    elif pending_proposal:
        status = "waiting_approval"
    elif run.status == "completed":
        status = "done" if run.success else "failed"
    elif run.status == "failed":
        status = "failed"
    if status is None:
        return task

    changes["status"] = status
    if status in {"done", "failed"}:
        changes["result_summary"] = (run.final_answer or run.reasoning or "")[:50_000]
    await mutate_task(
        session,
        event=event,
        task=task,
        changes=changes,
        actor_id="worker:task-execution",
        source_run_id=run.id,
    )
    return task


__all__ = [
    "DurableClarificationRecorder",
    "TaskExecutionError",
    "TaskExecutionResult",
    "add_task_lifecycle_message",
    "assemble_task_instructions",
    "queue_task_run",
    "stop_task_run",
    "synchronize_task_after_run",
    "task_has_pending_proposal",
]
