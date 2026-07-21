from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from aloy_backend.models import (
    ActionProposal,
    EventTrailEntry,
    StoredFile,
    Task,
)
from aloy_backend.routes import today as today_routes
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


async def test_project_event_keeps_one_canonical_session_and_task_surface(
    client, db_session_maker
):
    event = await _create_project(client)
    canonical_id = event["conversation_id"]
    assert canonical_id

    # Compatibility paths may still create provenance conversations, but they
    # never replace the Event's lifetime working Session.
    legacy = await client.post(
        "/v1/conversations",
        json={"title": "Legacy branch", "event_id": event["id"]},
    )
    assert legacy.status_code == 201
    assert legacy.json()["id"] != canonical_id
    protected = await client.delete(f"/v1/conversations/{canonical_id}")
    assert protected.status_code == 409

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
    assert payload["event"]["conversation_id"] == canonical_id
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
            "in_library": False,
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
    today_payload = today.json()
    groups = today_payload["events"]
    notifications = {item["id"]: item for item in today_payload["notifications"]}
    assert notifications[f"proposal:{pending.id}:pending"]["title"] == (
        "Approval requested"
    )
    assert notifications[f"proposal:{committed.id}:committed"]["title"] == (
        "Action completed"
    )
    assert notifications[f"proposal:{pending.id}:pending"]["event_title"] == (
        "Today Project"
    )
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


async def test_today_email_brief_is_bounded_and_provider_owned(client, monkeypatch):
    async def connected(*args, **kwargs):
        return {
            "google": {
                "access_token": "token-for-test",
                "account_email": "karabo@example.com",
                "scope": "user",
            }
        }

    def search(params, context):
        assert params.max_results == 5
        assert "is:important" in params.query
        assert context["connections"]["google"]["access_token"] == "token-for-test"
        return {
            "messages": [
                {
                    "id": "gmail-message-1",
                    "from": "Vertex Systems <jobs@vertex.example>",
                    "subject": "Next steps",
                    "date": "Sat, 18 Jul 2026 08:47:00 +0200",
                    "snippet": "Choose a time for the next interview.",
                }
            ]
        }

    monkeypatch.setattr(today_routes, "resolve_run_connections", connected)
    monkeypatch.setattr(today_routes, "gmail_search_tool", search)

    response = await client.get("/v1/today/emails")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["account_email"] == "karabo@example.com"
    assert payload["messages"] == [
        {
            "id": "gmail-message-1",
            "sender": "Vertex Systems <jobs@vertex.example>",
            "subject": "Next steps",
            "snippet": "Choose a time for the next interview.",
            "received_at": "2026-07-18T06:47:00Z",
            "event_id": None,
            "event_title": None,
            "provider_url": ("https://mail.google.com/mail/u/0/#all/gmail-message-1"),
        }
    ]


async def test_today_email_brief_fails_independently_of_today(client, monkeypatch):
    async def disconnected(*args, **kwargs):
        return {}

    monkeypatch.setattr(today_routes, "resolve_run_connections", disconnected)
    email_response = await client.get("/v1/today/emails")
    today_response = await client.get("/v1/today")

    assert email_response.status_code == 200
    assert email_response.json() == {
        "status": "not_connected",
        "account_email": None,
        "messages": [],
    }
    assert today_response.status_code == 200


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
        {
            "title": "Agent-created work",
            "instructions": "Produce the implementation notes.",
            "definition_of_done": "Notes are attached to the Event.",
            "priority": "high",
            "budget_policy": {"max_steps": 8},
        },
        {"task_mutator": handler},
    )
    assert created["success"] is True
    task_id = created["result"]["id"]
    assert created["result"]["created_by"] == "planner-agent"
    assert created["result"]["origin_conversation_id"] == event["conversation_id"]
    assert created["result"]["instructions"] == "Produce the implementation notes."
    assert created["result"]["definition_of_done"] == (
        "Notes are attached to the Event."
    )
    assert created["result"]["priority"] == "high"
    assert created["result"]["budget_policy"] == {"max_steps": 8}

    completed = await executor.execute_tool_async(
        "task_update",
        {"task_id": task_id, "status": "done"},
        {"task_mutator": handler},
    )
    assert completed["result"]["status"] == "done"

    illegal = await executor.execute_tool_async(
        "task_update",
        {"task_id": task_id, "status": "queued"},
        {"task_mutator": handler},
    )
    assert illegal == {
        "success": False,
        "error": "Illegal Task transition: done -> queued",
    }

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
