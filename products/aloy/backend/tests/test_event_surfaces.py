from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from aloy_backend.models import (
    ActionProposal,
    Event,
    EventTrailEntry,
    StoredFile,
    Task,
)
from aloy_backend.runtime import authenticated_run_context
from aloy_backend.tools.tasks import TaskMutationHandler, register_task_tools
from pori.tools.registry import ToolExecutor, ToolRegistry

pytestmark = pytest.mark.asyncio


async def _create_project(client, title: str = "Building Aloy") -> dict:
    response = await client.post(
        "/v1/events",
        json={
            "title": title,
            "summary": "Ship the event loop",
            "phase": "building",
            "notes": "Working state, not external proof.",
        },
    )
    assert response.status_code == 201
    return response.json()


async def test_project_event_two_sessions_and_task_mutations_share_one_surface(
    client, db_session_maker
):
    event = await _create_project(client)
    first = await client.post(
        "/v1/conversations",
        json={"title": "Plan", "event_id": event["id"]},
    )
    second = await client.post(
        "/v1/conversations",
        json={"title": "Build", "event_id": event["id"]},
    )
    assert first.status_code == second.status_code == 201
    assert first.json()["event_id"] == second.json()["event_id"] == event["id"]

    created = await client.post(
        f"/v1/events/{event['id']}/tasks",
        json={"title": "Build the Surface"},
    )
    assert created.status_code == 201
    task = created.json()
    completed = await client.patch(
        f"/v1/events/{event['id']}/tasks/{task['id']}",
        json={"status": "done"},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "done"

    surface = await client.get(f"/v1/events/{event['id']}")
    assert surface.status_code == 200
    payload = surface.json()
    assert payload["event"]["type"] == "project"
    assert payload["surface"]["type"] == "project"
    sections = {section["kind"]: section for section in payload["surface"]["sections"]}
    assert sections["status"] == {
        "kind": "status",
        "summary": "Ship the event loop",
        "phase": "building",
    }
    assert sections["notes"]["notes"] == "Working state, not external proof."
    assert sections["tasks"]["tasks"][0]["status"] == "done"
    assert [entry["kind"] for entry in sections["activity"]["entries"]] == [
        "task_changed",
        "task_changed",
        "event_created",
    ]

    async with db_session_maker() as session:
        tasks = (
            (await session.execute(select(Task).where(Task.event_id == event["id"])))
            .scalars()
            .all()
        )
        trails = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == event["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(tasks) == 1
        assert len(trails) == 3


async def test_surface_reads_only_durable_files_and_pending_proposals(
    client, db_session_maker
):
    event = await _create_project(client, "Evidence")
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        pending = ActionProposal(
            id="prop-surface-pending",
            organization_id="user:test-user",
            user_id="test-user",
            event_id=event["id"],
            tool="gmail_send",
            args={"to": "team@example.com", "subject": "Update", "body": "Done"},
            tool_schema_fingerprint="schema",
            reason="Send the update",
            impact="Sends an external email",
            risk="high",
            routing="ask",
            status="pending",
        )
        committed = ActionProposal(
            id="prop-surface-committed",
            organization_id="user:test-user",
            user_id="test-user",
            event_id=event["id"],
            tool="gmail_send",
            args={"to": "done@example.com"},
            tool_schema_fingerprint="schema",
            reason="Already sent",
            impact="Email",
            risk="high",
            routing="ask",
            status="committed",
            receipt={"receipt_id": "rcpt-1", "status": "succeeded"},
            provider_operation_id="provider-1",
            updated_at=now,
        )
        file = StoredFile(
            organization_id="user:test-user",
            user_id="test-user",
            event_id=event["id"],
            origin_session_id="session-1",
            run_id="run-1",
            kind="artifact",
            name="status.md",
            content_type="text/markdown",
            size_bytes=42,
            sha256="abc",
            storage_key="events/evidence/status.md",
        )
        trail = EventTrailEntry(
            organization_id="user:test-user",
            user_id="test-user",
            event_id=event["id"],
            actor_id="worker",
            kind="proposal_committed",
            summary="Committed gmail_send",
            proposal_id=committed.id,
            evidence_refs=[
                {"receipt_id": "rcpt-1"},
                {"provider_operation_id": "provider-1"},
            ],
        )
        session.add_all([pending, committed, file, trail])
        await session.commit()

    response = await client.get(f"/v1/events/{event['id']}")
    payload = response.json()["surface"]
    sections = {section["kind"]: section for section in payload["sections"]}
    assert [proposal["id"] for proposal in payload["proposals"]] == [pending.id]
    assert sections["files"]["files"] == [
        {
            "id": file.id,
            "name": "status.md",
            "kind": "artifact",
            "content_type": "text/markdown",
            "size_bytes": 42,
            "origin_session_id": "session-1",
            "origin_run_id": "run-1",
            "created_at": sections["files"]["files"][0]["created_at"],
        }
    ]
    committed_entry = sections["activity"]["entries"][0]
    assert committed_entry["evidence_refs"] == [
        {"receipt_id": "rcpt-1"},
        {"provider_operation_id": "provider-1"},
    ]


async def test_today_groups_life_first_and_resolves_proposal_from_both_lenses(
    client, db_session_maker
):
    life_session = await client.post("/v1/conversations", json={"title": "Life"})
    life_event_id = life_session.json()["event_id"]
    project = await _create_project(client, "Today Project")
    open_task = await client.post(
        f"/v1/events/{project['id']}/tasks", json={"title": "Open work"}
    )
    done_task = await client.post(
        f"/v1/events/{project['id']}/tasks", json={"title": "Done work"}
    )
    await client.patch(
        f"/v1/events/{project['id']}/tasks/{done_task.json()['id']}",
        json={"status": "done"},
    )
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        pending = ActionProposal(
            id="prop-today-pending",
            organization_id="user:test-user",
            user_id="test-user",
            event_id=project["id"],
            tool="gmail_send",
            args={"to": "today@example.com"},
            tool_schema_fingerprint="schema",
            reason="Needs a decision",
            impact="Email",
            risk="high",
            routing="ask",
            status="pending",
        )
        committed = ActionProposal(
            id="prop-today-committed",
            organization_id="user:test-user",
            user_id="test-user",
            event_id=project["id"],
            tool="calendar_create_event",
            args={"summary": "Review"},
            tool_schema_fingerprint="schema",
            reason="Committed",
            impact="Calendar",
            risk="high",
            routing="ask",
            status="committed",
            receipt={"receipt_id": "rcpt-today", "status": "succeeded"},
            updated_at=now - timedelta(hours=1),
        )
        session.add_all([pending, committed])
        await session.commit()

    today = await client.get("/v1/today")
    assert today.status_code == 200
    groups = today.json()["events"]
    assert groups[0]["event"]["id"] == life_event_id
    project_group = next(
        group for group in groups if group["event"]["id"] == project["id"]
    )
    assert [proposal["id"] for proposal in project_group["needs_decision"]] == [
        pending.id
    ]
    assert [proposal["id"] for proposal in project_group["changed_proposals"]] == [
        committed.id
    ]
    assert [task["id"] for task in project_group["upcoming"]] == [
        open_task.json()["id"]
    ]

    surface = await client.get(f"/v1/events/{project['id']}")
    assert surface.json()["surface"]["proposals"][0]["id"] == pending.id
    decided = await client.post(
        f"/v1/events/{project['id']}/proposals/{pending.id}/decision",
        json={"decision": "reject"},
    )
    assert decided.status_code == 200
    assert (await client.get("/v1/today")).json()["events"][1]["needs_decision"] == []
    assert (await client.get(f"/v1/events/{project['id']}")).json()["surface"][
        "proposals"
    ] == []


async def test_event_and_task_routes_are_tenant_scoped(client):
    event = await _create_project(client, "Private")
    denied_surface = await client.get(
        f"/v1/events/{event['id']}", headers={"X-Test-User": "other-user"}
    )
    denied_task = await client.post(
        f"/v1/events/{event['id']}/tasks",
        json={"title": "Intrude"},
        headers={"X-Test-User": "other-user"},
    )
    denied_session = await client.post(
        "/v1/conversations",
        json={"event_id": event["id"]},
        headers={"X-Test-User": "other-user"},
    )
    assert denied_surface.status_code == 404
    assert denied_task.status_code == 404
    assert denied_session.status_code == 404


async def test_agent_task_tools_mutate_working_state_and_trail_atomically(
    client, db_session_maker
):
    event = await _create_project(client, "Agent Tasks")
    context = authenticated_run_context(
        user_id="test-user",
        organization_id="user:test-user",
        run_id="run-agent-tasks",
        session_id="session-agent-tasks",
        event_id=event["id"],
        workspace_id=event["id"],
        agent_id="planner-agent",
    )
    handler = TaskMutationHandler(
        run_context=context,
        session_factory=db_session_maker,
    )
    registry = ToolRegistry()
    register_task_tools(registry)
    executor = ToolExecutor(registry)

    created = await executor.execute_tool_async(
        "task_create",
        {"title": "Agent-created work"},
        {"task_mutator": handler},
    )
    assert created["success"] is True
    task_id = created["result"]["id"]
    assert created["result"]["created_by"] == "planner-agent"

    completed = await executor.execute_tool_async(
        "task_update",
        {"task_id": task_id, "status": "done"},
        {"task_mutator": handler},
    )
    assert completed["result"]["status"] == "done"

    async with db_session_maker() as session:
        task = await session.get(Task, task_id)
        entries = (
            (
                await session.execute(
                    select(EventTrailEntry).where(EventTrailEntry.task_id == task_id)
                )
            )
            .scalars()
            .all()
        )
    assert task is not None and task.status == "done"
    assert [entry.payload["action"] for entry in entries] == ["created", "updated"]
    assert all(entry.run_id == "run-agent-tasks" for entry in entries)

    other_event = await _create_project(client, "Other Agent Scope")
    other_task = await client.post(
        f"/v1/events/{other_event['id']}/tasks", json={"title": "Keep isolated"}
    )
    denied = await executor.execute_tool_async(
        "task_update",
        {"task_id": other_task.json()["id"], "status": "done"},
        {"task_mutator": handler},
    )
    assert denied == {"success": False, "error": "Task is unavailable"}
    other_surface = (await client.get(f"/v1/events/{other_event['id']}")).json()
    other_sections = {
        section["kind"]: section for section in other_surface["surface"]["sections"]
    }
    assert other_sections["tasks"]["tasks"][0]["status"] == "open"
