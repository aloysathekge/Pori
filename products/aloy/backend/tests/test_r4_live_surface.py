from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import col, select

from aloy_backend.models import (
    ActionProposal,
    Conversation,
    EventTrailEntry,
    Message,
    Run,
    StoredFile,
    Task,
)
from aloy_backend.pagination import encode_cursor
from aloy_backend.routes.events import stream_event_changes
from aloy_backend.tenancy import OrganizationContext, OrganizationPolicy

pytestmark = pytest.mark.asyncio

ORG = "user:test-user"
USER = "test-user"


def _context() -> OrganizationContext:
    return OrganizationContext(
        organization_id=ORG,
        user_id=USER,
        role="owner",
        permissions=("run:read",),
        policy=OrganizationPolicy(),
    )


async def _project(client) -> dict:
    response = await client.post("/v1/events", json={"title": "R4 Event"})
    assert response.status_code == 201
    return response.json()


async def test_trail_and_conversation_histories_use_stable_cursor_pages(
    client, db_session_maker
):
    event = await _project(client)
    conversation_id = event["conversation_id"]
    base = datetime.now(timezone.utc) - timedelta(hours=2)
    async with db_session_maker() as session:
        session.add_all(
            [
                EventTrailEntry(
                    id=f"trail-page-{index:03d}",
                    organization_id=ORG,
                    user_id=USER,
                    event_id=event["id"],
                    actor_id=USER,
                    kind="task_progress",
                    summary=f"Trail {index}",
                    created_at=base + timedelta(seconds=index),
                )
                for index in range(65)
            ]
            + [
                Message(
                    id=f"message-page-{index:03d}",
                    conversation_id=conversation_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"Message {index}",
                    created_at=base + timedelta(seconds=index),
                )
                for index in range(105)
            ]
        )
        await session.commit()

    surface = (await client.get(f"/v1/events/{event['id']}")).json()
    activity = next(
        section
        for section in surface["surface"]["sections"]
        if section["kind"] == "activity"
    )
    assert len(activity["entries"]) == 50
    assert activity["next_cursor"]
    older = (
        await client.get(
            f"/v1/events/{event['id']}/trail",
            params={"cursor": activity["next_cursor"], "limit": 50},
        )
    ).json()
    assert not (
        {row["id"] for row in activity["entries"]}
        & {row["id"] for row in older["entries"]}
    )
    assert len(activity["entries"]) + len(older["entries"]) == 66
    assert older["next_cursor"] is None

    detail = (await client.get(f"/v1/conversations/{conversation_id}")).json()
    assert len(detail["messages"]) == 100
    assert detail["messages"][0]["content"] == "Message 5"
    message_page = (
        await client.get(
            f"/v1/conversations/{conversation_id}/messages",
            params={"cursor": detail["messages_next_cursor"]},
        )
    ).json()
    assert [message["content"] for message in message_page["messages"]] == [
        f"Message {index}" for index in range(5)
    ]
    exported = (await client.get(f"/v1/conversations/{conversation_id}/export")).json()
    assert len(exported["messages"]) == 105
    assert exported["conversation"]["message_count"] == 105


async def test_surface_groups_one_task_run_with_provenance_and_evidence(
    client, db_session_maker
):
    event = await _project(client)
    task_response = await client.post(
        f"/v1/events/{event['id']}/tasks", json={"title": "Research companies"}
    )
    task_id = task_response.json()["id"]
    async with db_session_maker() as session:
        task = await session.get(Task, task_id)
        assert task is not None
        run = Run(
            id="run-r4-group",
            organization_id=ORG,
            user_id=USER,
            event_id=event["id"],
            task_id=task.id,
            conversation_id=task.origin_conversation_id,
            agent_id="aloy",
            session_id="session-r4",
            task=task.title,
            status="completed",
            completed_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.add(
            EventTrailEntry(
                organization_id=ORG,
                user_id=USER,
                event_id=event["id"],
                actor_id="aloy",
                kind="task_progress",
                summary="Compared target companies",
                task_id=task.id,
                run_id=run.id,
            )
        )
        session.add(
            StoredFile(
                organization_id=ORG,
                user_id=USER,
                event_id=event["id"],
                conversation_id=task.origin_conversation_id,
                run_id=run.id,
                name="companies.md",
                storage_key="tests/companies.md",
            )
        )
        session.add(
            ActionProposal(
                organization_id=ORG,
                user_id=USER,
                event_id=event["id"],
                origin_run_id=run.id,
                tool="gmail_send",
                args={},
                tool_schema_fingerprint="schema",
                reason="Send research",
                impact="Email",
                risk="high",
                routing="ask",
                status="committed",
                receipt={"status": "succeeded"},
            )
        )
        await session.commit()

    surface = (await client.get(f"/v1/events/{event['id']}")).json()["surface"]
    group = next(
        item for item in surface["execution_groups"] if item["run_id"] == run.id
    )
    assert group["conversation_id"] == task.origin_conversation_id
    assert [entry["summary"] for entry in group["entries"]] == [
        "Compared target companies"
    ]
    assert [artifact["name"] for artifact in group["artifacts"]] == ["companies.md"]
    assert group["receipts"] == [{"status": "succeeded"}]


class _ConnectedRequest:
    headers: dict[str, str] = {}

    async def is_disconnected(self) -> bool:
        return False


async def test_event_stream_replays_missed_change_with_origin_conversation(
    client, db_session_maker
):
    origin = (await client.post("/v1/conversations", json={"title": "Origin"})).json()
    sibling = (await client.post("/v1/conversations", json={"title": "Sibling"})).json()
    assert sibling["event_id"] == origin["event_id"]
    event_id = origin["event_id"]
    task_response = await client.post(
        f"/v1/events/{event_id}/tasks",
        json={
            "title": "Durable work",
            "origin_conversation_id": origin["id"],
        },
    )
    task_id = task_response.json()["id"]
    async with db_session_maker() as session:
        task = await session.get(Task, task_id)
        assert task is not None
        origin_conversation_id = task.origin_conversation_id
        first = (
            (
                await session.execute(
                    select(EventTrailEntry)
                    .where(EventTrailEntry.event_id == event_id)
                    .order_by(col(EventTrailEntry.created_at).desc())
                )
            )
            .scalars()
            .first()
        )
        assert first is not None
        missed = EventTrailEntry(
            organization_id=ORG,
            user_id=USER,
            event_id=event_id,
            actor_id="aloy",
            kind="task_changed",
            summary="Task completed after disconnect",
            task_id=task.id,
            payload={"action": "completed", "after": {"status": "done"}},
            created_at=first.created_at + timedelta(seconds=1),
        )
        session.add(missed)
        await session.commit()
        response = await stream_event_changes(
            event_id,
            _ConnectedRequest(),  # type: ignore[arg-type]
            encode_cursor(first.created_at, first.id),
            _context(),
            session,
        )
        iterator = response.body_iterator
        ready = await anext(iterator)
        change = await anext(iterator)
        await iterator.aclose()

    assert "event: ready" in ready
    assert "event: event_change" in change
    assert "Task completed after disconnect" in change
    assert origin_conversation_id is not None
    assert origin_conversation_id in change
    assert sibling["id"] not in change


async def test_today_calls_out_blocked_and_stale_tasks(client, db_session_maker):
    event = await _project(client)
    blocked = (
        await client.post(
            f"/v1/events/{event['id']}/tasks", json={"title": "Needs founder input"}
        )
    ).json()
    stale = (
        await client.post(
            f"/v1/events/{event['id']}/tasks", json={"title": "Old research"}
        )
    ).json()
    async with db_session_maker() as session:
        blocked_row = await session.get(Task, blocked["id"])
        stale_row = await session.get(Task, stale["id"])
        assert blocked_row is not None and stale_row is not None
        blocked_row.status = "blocked"
        blocked_row.blocker = "Choose a market"
        stale_row.updated_at = datetime.now(timezone.utc) - timedelta(days=2)
        session.add_all([blocked_row, stale_row])
        await session.commit()

    today = (await client.get("/v1/today")).json()
    group = next(item for item in today["events"] if item["event"]["id"] == event["id"])
    assert [task["id"] for task in group["blocked"]] == [blocked["id"]]
    assert [task["id"] for task in group["stale"]] == [stale["id"]]
