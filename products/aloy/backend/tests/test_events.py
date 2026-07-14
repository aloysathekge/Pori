from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from aloy_backend.models import ActionProposal, Event, EventTrailEntry, Task


async def test_sessions_and_direct_runs_share_the_users_life_event(
    client, db_session_maker
):
    first = await client.post("/v1/conversations", json={"title": "First"})
    second = await client.post("/v1/conversations", json={"title": "Second"})
    assert first.status_code == second.status_code == 201
    event_id = first.json()["event_id"]
    assert event_id.startswith("evt_")
    assert second.json()["event_id"] == event_id

    branch = await client.post(
        f"/v1/conversations/{first.json()['id']}/branch", json={}
    )
    assert branch.status_code == 201
    assert branch.json()["event_id"] == event_id

    run = await client.post("/v1/runs", json={"task": "background work"})
    assert run.status_code == 202
    assert run.json()["event_id"] == event_id

    async with db_session_maker() as session:
        events = (
            (
                await session.execute(
                    select(Event).where(
                        Event.organization_id == "user:test-user",
                        Event.user_id == "test-user",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert [(event.id, event.type, event.is_life) for event in events] == [
            (event_id, "life", True)
        ]


async def test_database_enforces_one_life_event_per_user(db_session_maker):
    async with db_session_maker() as session:
        session.add_all(
            [
                Event(
                    organization_id="org-1",
                    user_id="alice",
                    type="life",
                    title="Life",
                    is_life=True,
                ),
                Event(
                    organization_id="org-1",
                    user_id="alice",
                    type="life",
                    title="Another Life",
                    is_life=True,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_event_owned_working_and_activity_rows_round_trip(db_session_maker):
    async with db_session_maker() as session:
        event = Event(organization_id="org-1", user_id="alice", title="Ship Aloy")
        session.add(event)
        await session.flush()
        task = Task(
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            title="Build aggregate",
            created_by="alice",
        )
        proposal = ActionProposal(
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            tool="gmail_send",
            args={"to": "team@example.com"},
            tool_schema_fingerprint="schema-v1",
            reason="Send the update",
            impact="Creates an external email",
            risk="medium",
            routing="ask",
        )
        trail = EventTrailEntry(
            organization_id="org-1",
            user_id="alice",
            event_id=event.id,
            actor_id="alice",
            kind="task_changed",
            summary="Created Build aggregate",
            task_id=task.id,
        )
        session.add_all([task, proposal, trail])
        await session.commit()

        assert (await session.get(Task, task.id)).event_id == event.id
        assert (await session.get(ActionProposal, proposal.id)).args["to"] == (
            "team@example.com"
        )
        assert (await session.get(EventTrailEntry, trail.id)).task_id == task.id
