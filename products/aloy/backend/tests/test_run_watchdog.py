from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from aloy_backend.models import (
    Conversation,
    Event,
    EventTrailEntry,
    Message,
    Organization,
    Run,
    Task,
)
from aloy_backend.run_watchdog import (
    reconcile_orphaned_tasks,
    reconcile_stale_runs,
)
from aloy_backend.worker import claim_next_run

pytestmark = pytest.mark.asyncio


async def _seed_event(session, *, event_id: str, conversation_id: str) -> None:
    session.add(
        Organization(
            id="org-watchdog",
            name="Watchdog",
            slug="watchdog",
            created_by="alice",
            policy={},
        )
    )
    session.add(
        Event(
            id=event_id,
            organization_id="org-watchdog",
            user_id="alice",
            title="Reliable work",
            primary_conversation_id=conversation_id,
        )
    )
    session.add(
        Conversation(
            id=conversation_id,
            organization_id="org-watchdog",
            user_id="alice",
            event_id=event_id,
            title="Reliable work",
        )
    )


async def test_final_expired_attempt_fails_run_and_task_exactly_once(
    db_session_maker,
):
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        await _seed_event(
            session,
            event_id="evt-watchdog-final",
            conversation_id="conv-watchdog-final",
        )
        run = Run(
            id="run-watchdog-final",
            organization_id="org-watchdog",
            user_id="alice",
            event_id="evt-watchdog-final",
            task_id="task-watchdog-final",
            agent_id="default_agent",
            session_id="conv-watchdog-final",
            conversation_id="conv-watchdog-final",
            task="Finish the report",
            status="running",
            attempt_count=3,
            max_attempts=3,
            lease_owner="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
        )
        task = Task(
            id="task-watchdog-final",
            organization_id="org-watchdog",
            user_id="alice",
            event_id="evt-watchdog-final",
            origin_conversation_id="conv-watchdog-final",
            title="Finish the report",
            status="in_progress",
            current_run_id=run.id,
            created_by="alice",
        )
        session.add_all([run, task])
        await session.commit()

    assert await reconcile_stale_runs(session_factory=db_session_maker, now=now) == 1
    assert await reconcile_stale_runs(session_factory=db_session_maker, now=now) == 0

    async with db_session_maker() as session:
        run = await session.get(Run, "run-watchdog-final")
        task = await session.get(Task, "task-watchdog-final")
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.run_id == "run-watchdog-final",
                        EventTrailEntry.kind == "run_watchdog_terminal",
                    )
                )
            )
            .scalars()
            .all()
        )
        messages = (
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == "conv-watchdog-final"
                    )
                )
            )
            .scalars()
            .all()
        )

    assert run is not None
    assert run.status == "failed"
    assert run.lease_owner is None
    assert run.completed_at is not None
    assert run.progress["watchdog"]["code"] == (
        "worker_lease_expired_attempts_exhausted"
    )
    assert task is not None
    assert task.status == "failed"
    assert task.current_run_id is None
    assert len(trail) == 1
    assert len(messages) == 1
    assert messages[0].metadata_["kind"] == "task_lifecycle"


async def test_cancelled_expired_run_is_not_reclaimed(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        await _seed_event(
            session,
            event_id="evt-watchdog-cancel",
            conversation_id="conv-watchdog-cancel",
        )
        run = Run(
            id="run-watchdog-cancel",
            organization_id="org-watchdog",
            user_id="alice",
            event_id="evt-watchdog-cancel",
            task_id="task-watchdog-cancel",
            agent_id="default_agent",
            session_id="conv-watchdog-cancel",
            conversation_id="conv-watchdog-cancel",
            task="Stop this work",
            status="running",
            attempt_count=1,
            max_attempts=3,
            cancel_requested=True,
            lease_owner="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
        )
        task = Task(
            id="task-watchdog-cancel",
            organization_id="org-watchdog",
            user_id="alice",
            event_id="evt-watchdog-cancel",
            origin_conversation_id="conv-watchdog-cancel",
            title="Stop this work",
            status="in_progress",
            current_run_id=run.id,
            created_by="alice",
        )
        session.add_all([run, task])
        await session.commit()

    assert await reconcile_stale_runs(session_factory=db_session_maker, now=now) == 1
    async with db_session_maker() as session:
        run = await session.get(Run, "run-watchdog-cancel")
        task = await session.get(Task, "task-watchdog-cancel")

    assert run is not None and run.status == "cancelled"
    assert task is not None and task.status == "cancelled"
    assert await claim_next_run("replacement-worker") is None


async def test_recoverable_expired_run_is_reclaimed_with_trail(
    db_session_maker,
    monkeypatch,
):
    monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        await _seed_event(
            session,
            event_id="evt-watchdog-recover",
            conversation_id="conv-watchdog-recover",
        )
        session.add(
            Run(
                id="run-watchdog-recover",
                organization_id="org-watchdog",
                user_id="alice",
                event_id="evt-watchdog-recover",
                agent_id="default_agent",
                session_id="conv-watchdog-recover",
                conversation_id="conv-watchdog-recover",
                task="Resume safely",
                status="running",
                attempt_count=1,
                max_attempts=3,
                lease_owner="dead-worker",
                lease_expires_at=now - timedelta(seconds=1),
                progress={"n_steps": 4},
            )
        )
        await session.commit()

    assert await reconcile_stale_runs(session_factory=db_session_maker, now=now) == 0
    assert await claim_next_run("replacement-worker") == "run-watchdog-recover"

    async with db_session_maker() as session:
        run = await session.get(Run, "run-watchdog-recover")
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.run_id == "run-watchdog-recover",
                        EventTrailEntry.kind == "run_watchdog_recovered",
                    )
                )
            )
            .scalars()
            .all()
        )

    assert run is not None
    assert run.status == "running"
    assert run.attempt_count == 2
    assert run.lease_owner == "replacement-worker"
    assert len(trail) == 1
    assert trail[0].payload["checkpoint_steps"] == 4


async def test_orphaned_task_becomes_retryable_failure_exactly_once(
    db_session_maker,
):
    async with db_session_maker() as session:
        await _seed_event(
            session,
            event_id="evt-watchdog-orphan",
            conversation_id="conv-watchdog-orphan",
        )
        session.add(
            Task(
                id="task-watchdog-orphan",
                organization_id="org-watchdog",
                user_id="alice",
                event_id="evt-watchdog-orphan",
                origin_conversation_id="conv-watchdog-orphan",
                title="Lost work",
                status="queued",
                current_run_id="missing-run",
                created_by="alice",
            )
        )
        await session.commit()

    assert await reconcile_orphaned_tasks(session_factory=db_session_maker) == 1
    assert await reconcile_orphaned_tasks(session_factory=db_session_maker) == 0

    async with db_session_maker() as session:
        task = await session.get(Task, "task-watchdog-orphan")
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.task_id == "task-watchdog-orphan",
                        EventTrailEntry.kind == "task_watchdog_reconciled",
                    )
                )
            )
            .scalars()
            .all()
        )

    assert task is not None
    assert task.status == "failed"
    assert task.current_run_id is None
    assert len(trail) == 1
