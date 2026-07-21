"""Durable public Run timeline projection, persistence, and API."""

import pytest
from sqlmodel import select

from aloy_backend.models import Event, Run, RunTimelineEvent
from aloy_backend.run_timeline import RunTimelineRecorder, project_pori_event
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
    assert "technical_payload" not in body["entries"][0]
