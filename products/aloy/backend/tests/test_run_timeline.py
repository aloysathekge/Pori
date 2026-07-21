"""Durable public Run timeline projection, persistence, and API."""

import asyncio

import pytest
from pydantic import ValidationError
from sqlmodel import select

from aloy_backend.models import Event, Run, RunTimelineCursor, RunTimelineEvent
from aloy_backend.run_timeline import (
    MAX_TIMELINE_EVENTS_PER_RUN,
    LocalTimelineNotifier,
    RunTimelineRecorder,
    project_pori_event,
    reconcile_terminal_run_timeline,
    validate_public_payload,
)
from pori.observability import (
    ACTIVITY_CHANGED,
    PLAN_CHANGED,
    TOOL_CALL_END,
    TOOL_CALL_START,
    PoriEvent,
)

pytestmark = pytest.mark.asyncio

ORG = "user:test-user"
USER = "test-user"


def _event(event_type: str, **payload):
    return PoriEvent(type=event_type, payload=payload, step=2)


def test_projection_keeps_public_labels_and_drops_raw_arguments():
    projected = project_pori_event(
        _event(
            TOOL_CALL_START,
            name="send_secret",
            call_id="call-1",
            label="Sending the approved draft",
            args={"token": "must-not-leak", "body": "private"},
        )
    )
    assert projected == (
        "action_started",
        {"call_id": "call-1", "label": "Sending the approved draft"},
    )


def test_projection_carries_host_owned_status_and_duration():
    projected = project_pori_event(
        _event(
            TOOL_CALL_END,
            name="web_search",
            call_id="call-2",
            label="Searched the web: AI roles",
            success=True,
            duration_seconds=1.25,
            result={"raw": "must-not-leak"},
        )
    )
    assert projected == (
        "action_finished",
        {
            "call_id": "call-2",
            "label": "Searched the web: AI roles",
            "success": True,
            "duration_seconds": 1.25,
        },
    )


async def test_recorder_appends_monotonic_projected_events(db_session_maker):
    async with db_session_maker() as session:
        session.add(
            Event(
                id="evt-timeline",
                organization_id=ORG,
                user_id=USER,
                title="Timeline Event",
            )
        )
        await session.commit()

    recorder = RunTimelineRecorder(
        organization_id=ORG,
        user_id=USER,
        event_id="evt-timeline",
        conversation_id="conv-timeline",
        run_id="run-timeline",
        session_factory=db_session_maker,
    )
    await recorder.record(_event(ACTIVITY_CHANGED, activity="Comparing roles"))
    await recorder.record(
        _event(
            PLAN_CHANGED,
            plan=[{"id": "1", "content": "Compare roles", "status": "in_progress"}],
            summary={"in_progress": 1},
        )
    )

    async with db_session_maker() as session:
        rows = list(
            (
                await session.execute(
                    select(RunTimelineEvent)
                    .where(RunTimelineEvent.run_id == "run-timeline")
                    .order_by(RunTimelineEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [row.sequence for row in rows] == [1, 2]
    assert [row.kind for row in rows] == ["activity_changed", "plan_changed"]


async def test_replayed_kernel_event_is_idempotent(db_session_maker):
    async with db_session_maker() as session:
        session.add(
            Event(
                id="evt-timeline-replay",
                organization_id=ORG,
                user_id=USER,
                title="Replay Event",
            )
        )
        await session.commit()

    recorder = RunTimelineRecorder(
        organization_id=ORG,
        user_id=USER,
        event_id="evt-timeline-replay",
        conversation_id=None,
        run_id="run-timeline-replay",
        session_factory=db_session_maker,
    )
    event = _event(ACTIVITY_CHANGED, activity="Checking sources")
    first = await recorder.record(event)
    second = await recorder.record(event)

    assert first is not None
    assert second is not None
    assert first.id == second.id
    async with db_session_maker() as session:
        rows = list(
            (
                await session.execute(
                    select(RunTimelineEvent).where(
                        RunTimelineEvent.run_id == "run-timeline-replay"
                    )
                )
            )
            .scalars()
            .all()
        )
        cursor = await session.get(RunTimelineCursor, "run-timeline-replay")
    assert len(rows) == 1
    assert cursor is not None and cursor.last_sequence == 1


async def test_overlapping_recorders_allocate_unique_contiguous_sequences(
    db_session_maker,
):
    async with db_session_maker() as session:
        session.add(
            Event(
                id="evt-timeline-overlap",
                organization_id=ORG,
                user_id=USER,
                title="Overlap Event",
            )
        )
        await session.commit()

    recorders = [
        RunTimelineRecorder(
            organization_id=ORG,
            user_id=USER,
            event_id="evt-timeline-overlap",
            conversation_id=None,
            run_id="run-timeline-overlap",
            session_factory=db_session_maker,
        )
        for _ in range(2)
    ]
    await asyncio.gather(
        *[
            recorders[index % 2].append(
                "activity_changed",
                {"activity": f"Step {index}"},
                source_event_key=f"test:overlap:{index}",
            )
            for index in range(12)
        ]
    )

    async with db_session_maker() as session:
        rows = list(
            (
                await session.execute(
                    select(RunTimelineEvent)
                    .where(RunTimelineEvent.run_id == "run-timeline-overlap")
                    .order_by(RunTimelineEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
    assert [row.sequence for row in rows] == list(range(1, 13))
    assert len({row.source_event_key for row in rows}) == 12


async def test_timeline_limit_fails_without_advancing_cursor(db_session_maker):
    async with db_session_maker() as session:
        event = Event(
            id="evt-timeline-limit",
            organization_id=ORG,
            user_id=USER,
            title="Bounded Event",
        )
        session.add(event)
        session.add(
            RunTimelineCursor(
                run_id="run-timeline-limit",
                last_sequence=MAX_TIMELINE_EVENTS_PER_RUN,
            )
        )
        await session.commit()
    recorder = RunTimelineRecorder(
        organization_id=ORG,
        user_id=USER,
        event_id="evt-timeline-limit",
        conversation_id=None,
        run_id="run-timeline-limit",
        session_factory=db_session_maker,
    )
    with pytest.raises(ValueError, match="durable event limit"):
        await recorder.append(
            "activity_changed",
            {"activity": "One too many"},
            source_event_key="test:limit",
        )
    async with db_session_maker() as session:
        cursor = await session.get(RunTimelineCursor, "run-timeline-limit")
    assert cursor is not None
    assert cursor.last_sequence == MAX_TIMELINE_EVENTS_PER_RUN


async def test_local_notifier_wakes_readers_without_replacing_cursor_replay():
    notifier = LocalTimelineNotifier()
    waiter = asyncio.create_task(notifier.wait("run-notify", timeout=1))
    await asyncio.sleep(0)
    notifier.publish("run-notify")
    await asyncio.wait_for(waiter, timeout=0.2)


def test_public_payload_schema_rejects_drift_and_oversized_values():
    with pytest.raises(ValidationError):
        validate_public_payload(
            "action_started",
            {"call_id": "call", "label": "Safe", "args": {"secret": True}},
        )
    with pytest.raises(ValidationError):
        validate_public_payload("activity_changed", {"activity": "x" * 1001})
    with pytest.raises(ValueError, match="Unsupported"):
        validate_public_payload("private_reasoning", {"text": "no"})


@pytest.mark.parametrize(
    ("status", "expected_kind"),
    [
        ("completed", "run_finished"),
        ("failed", "run_failed"),
        ("cancelled", "run_cancelled"),
    ],
)
async def test_terminal_reconciliation_repairs_missing_story_once(
    db_session_maker, status, expected_kind
):
    async with db_session_maker() as session:
        event = Event(
            id=f"evt-terminal-{status}",
            organization_id=ORG,
            user_id=USER,
            title="Terminal Event",
        )
        session.add_all(
            [
                event,
                Run(
                    id=f"run-terminal-{status}",
                    organization_id=ORG,
                    user_id=USER,
                    event_id=event.id,
                    agent_id="default_agent",
                    session_id=f"run-terminal-{status}",
                    task="Finish safely",
                    status=status,
                    success=status == "completed",
                    steps_taken=3,
                ),
            ]
        )
        await session.commit()

    first = await reconcile_terminal_run_timeline(
        f"run-terminal-{status}", session_factory=db_session_maker
    )
    second = await reconcile_terminal_run_timeline(
        f"run-terminal-{status}", session_factory=db_session_maker
    )
    assert first is not None and first.kind == expected_kind
    assert second is not None and second.id == first.id


async def test_timeline_endpoint_returns_cursor_ordered_public_events(
    client, db_session_maker
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-timeline-api",
            organization_id=ORG,
            user_id=USER,
            title="Timeline API Event",
        )
        run = Run(
            id="run-timeline-api",
            organization_id=ORG,
            user_id=USER,
            event_id=event.id,
            agent_id="default_agent",
            session_id="run-timeline-api",
            task="Research roles",
            status="completed",
        )
        session.add_all(
            [
                event,
                run,
                RunTimelineEvent(
                    id="rtl-api-1",
                    organization_id=ORG,
                    user_id=USER,
                    event_id=event.id,
                    run_id=run.id,
                    sequence=1,
                    kind="run_started",
                    source_event_key="test:api:1",
                    public_payload={"status": "running"},
                ),
                RunTimelineEvent(
                    id="rtl-api-2",
                    organization_id=ORG,
                    user_id=USER,
                    event_id=event.id,
                    run_id=run.id,
                    sequence=2,
                    kind="run_finished",
                    source_event_key="test:api:2",
                    public_payload={"completed": True, "steps": 2},
                ),
            ]
        )
        await session.commit()

    response = await client.get("/v1/runs/run-timeline-api/timeline?after=1")
    assert response.status_code == 200
    body = response.json()
    assert body["next_cursor"] == 2
    assert [entry["kind"] for entry in body["entries"]] == ["run_finished"]
    assert body["entries"][0]["schema_version"] == 1
    assert "technical_payload" not in body["entries"][0]


async def test_timeline_paginates_without_losing_cursor_position(
    client, db_session_maker
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-timeline-pages",
            organization_id=ORG,
            user_id=USER,
            title="Paged Event",
        )
        run = Run(
            id="run-timeline-pages",
            organization_id=ORG,
            user_id=USER,
            event_id=event.id,
            agent_id="default_agent",
            session_id="run-timeline-pages",
            task="Long bounded story",
            status="completed",
        )
        session.add(event)
        session.add(run)
        session.add_all(
            [
                RunTimelineEvent(
                    id=f"rtl-page-{sequence}",
                    organization_id=ORG,
                    user_id=USER,
                    event_id=event.id,
                    run_id=run.id,
                    sequence=sequence,
                    kind="activity_changed",
                    source_event_key=f"test:page:{sequence}",
                    public_payload={"activity": f"Milestone {sequence}"},
                )
                for sequence in range(1, 502)
            ]
        )
        await session.commit()

    first = await client.get("/v1/runs/run-timeline-pages/timeline?limit=500")
    assert first.status_code == 200
    assert len(first.json()["entries"]) == 500
    assert first.json()["next_cursor"] == 500
    second = await client.get(
        "/v1/runs/run-timeline-pages/timeline?after=500&limit=500"
    )
    assert second.status_code == 200
    assert [entry["sequence"] for entry in second.json()["entries"]] == [501]


async def test_timeline_does_not_reveal_a_run_from_another_organization(
    client, db_session_maker
):
    async with db_session_maker() as session:
        event = Event(
            id="evt-foreign-timeline",
            organization_id="user:someone-else",
            user_id="someone-else",
            title="Private Event",
        )
        session.add_all(
            [
                event,
                Run(
                    id="run-foreign-timeline",
                    organization_id="user:someone-else",
                    user_id="someone-else",
                    event_id=event.id,
                    agent_id="default_agent",
                    session_id="run-foreign-timeline",
                    task="Private work",
                    status="completed",
                ),
            ]
        )
        await session.commit()

    response = await client.get("/v1/runs/run-foreign-timeline/timeline")
    assert response.status_code == 404
