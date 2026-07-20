from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import select

from aloy_backend.models import EventTrailEntry, Message, Run, Task
from aloy_backend.task_execution import synchronize_task_after_run
from aloy_backend.worker import claim_next_run


async def _project_task(client, *, title: str = "Research startup jobs"):
    event = (
        await client.post(
            "/v1/events",
            json={
                "title": "Career OS",
                "summary": "Find a strong US startup role",
            },
        )
    ).json()
    task = (
        await client.post(
            f"/v1/events/{event['id']}/tasks",
            json={
                "title": title,
                "instructions": "Research current US startup openings.",
                "definition_of_done": "Produce a concise, sourced shortlist.",
                "budget_policy": {"max_steps": 12, "timeout_seconds": 300},
            },
        )
    ).json()
    return event, task


async def test_work_queues_one_idempotent_durable_run(client, db_session_maker):
    event, task = await _project_task(client)

    started = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    assert started.status_code == 202
    payload = started.json()
    assert payload["task"]["status"] == "queued"
    assert payload["run"]["status"] == "pending"
    assert payload["run"]["conversation_id"] == event["conversation_id"]
    assert payload["idempotent"] is False

    duplicate = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    assert duplicate.status_code == 202
    assert duplicate.json()["idempotent"] is True
    assert duplicate.json()["run"]["id"] == payload["run"]["id"]

    async with db_session_maker() as session:
        stored_task = await session.get(Task, task["id"])
        run = await session.get(Run, payload["run"]["id"])
        messages = (
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == event["conversation_id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        entries = (
            (
                await session.execute(
                    select(EventTrailEntry).where(EventTrailEntry.task_id == task["id"])
                )
            )
            .scalars()
            .all()
        )

    assert stored_task is not None
    assert stored_task.current_run_id == run.id
    assert run is not None
    assert run.task_id == task["id"]
    assert run.session_id == event["conversation_id"]
    assert "Event: Career OS" in run.task
    assert "Research current US startup openings." in run.task
    assert "Produce a concise, sourced shortlist." in run.task
    assert [message.metadata_.get("task_status") for message in messages] == ["queued"]
    assert [entry.payload["action"] for entry in entries] == ["created", "queued"]


async def test_task_run_freezes_complete_execution_budget(client, db_session_maker):
    event = (
        await client.post(
            "/v1/events",
            json={"title": "Budgeted event", "summary": "Bounded work"},
        )
    ).json()
    task = (
        await client.post(
            f"/v1/events/{event['id']}/tasks",
            json={
                "title": "Bounded task",
                "budget_policy": {
                    "max_steps": 7,
                    "max_tool_calls": 9,
                    "max_tokens": 12_345,
                    "max_cost_usd": 0.75,
                    "timeout_seconds": 180,
                },
            },
        )
    ).json()

    queued = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    assert queued.status_code == 202
    run_id = queued.json()["run"]["id"]

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)

    assert run is not None
    assert run.max_steps == 7
    assert run.max_tool_calls == 9
    assert run.max_tokens == 12_345
    assert run.max_cost_usd == 0.75
    assert run.timeout_seconds == 180


async def test_life_task_reports_to_its_selected_origin_conversation(
    client, db_session_maker
):
    first = (await client.post("/v1/conversations", json={"title": "First"})).json()
    second = (
        await client.post("/v1/conversations", json={"title": "Startup search"})
    ).json()
    task = (
        await client.post(
            f"/v1/events/{first['event_id']}/tasks",
            json={
                "title": "Search roles",
                "origin_conversation_id": second["id"],
            },
        )
    ).json()

    response = await client.post(
        f"/v1/events/{first['event_id']}/tasks/{task['id']}/work"
    )
    assert response.status_code == 202
    run_id = response.json()["run"]["id"]
    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
    assert run is not None
    assert run.conversation_id == second["id"]
    assert run.conversation_id != first["id"]


async def test_stop_then_retry_creates_one_fresh_run(client, db_session_maker):
    event, task = await _project_task(client, title="Recoverable work")
    first = (
        await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    ).json()
    stopped = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["task"]["status"] == "cancelled"
    assert stopped.json()["run"]["status"] == "cancelled"

    retried = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/retry")
    assert retried.status_code == 202
    assert retried.json()["task"]["status"] == "queued"
    assert retried.json()["run"]["id"] != first["run"]["id"]

    duplicate = await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/retry")
    assert duplicate.status_code == 202
    assert duplicate.json()["idempotent"] is True
    assert duplicate.json()["run"]["id"] == retried.json()["run"]["id"]

    async with db_session_maker() as session:
        runs = (
            (await session.execute(select(Run).where(Run.task_id == task["id"])))
            .scalars()
            .all()
        )
    assert len(runs) == 2


async def test_blocked_resume_reuses_checkpointed_run_and_records_answer(
    client, db_session_maker
):
    event, task = await _project_task(client, title="Clarify target")
    queued = (
        await client.post(f"/v1/events/{event['id']}/tasks/{task['id']}/work")
    ).json()
    run_id = queued["run"]["id"]
    async with db_session_maker() as session:
        stored = await session.get(Task, task["id"])
        run = await session.get(Run, run_id)
        assert stored is not None and run is not None
        stored.status = "blocked"
        stored.blocker = "Which job level should I prioritize?"
        run.status = "completed"
        run.progress = {
            "kernel_task_id": f"run-{run.id[:12]}",
            "n_steps": 2,
            "current_activity": "Compared role levels",
            "plan": [],
        }
        run.completed_at = datetime.now(timezone.utc)
        session.add_all([stored, run])
        await session.commit()

    resumed = await client.post(
        f"/v1/events/{event['id']}/tasks/{task['id']}/resume",
        json={"response": "Prioritize senior backend roles."},
    )
    assert resumed.status_code == 202
    assert resumed.json()["run"]["id"] == run_id
    assert resumed.json()["task"]["status"] == "queued"

    async with db_session_maker() as session:
        run = await session.get(Run, run_id)
        answers = (
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == event["conversation_id"],
                        Message.role == "user",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert run is not None
    assert run.status == "pending"
    assert run.progress["n_steps"] == 2
    assert "Prioritize senior backend roles." in run.task
    assert any(
        (message.metadata_ or {}).get("kind") == "task_resume" for message in answers
    )


async def test_worker_keeps_second_task_run_queued_for_same_event(
    client, db_session_maker, monkeypatch
):
    first_conversation = (
        await client.post("/v1/conversations", json={"title": "First work"})
    ).json()
    second_conversation = (
        await client.post("/v1/conversations", json={"title": "Second work"})
    ).json()
    event_id = first_conversation["event_id"]
    first_task = (
        await client.post(
            f"/v1/events/{event_id}/tasks",
            json={
                "title": "First task",
                "origin_conversation_id": first_conversation["id"],
            },
        )
    ).json()
    second_task = (
        await client.post(
            f"/v1/events/{event_id}/tasks",
            json={
                "title": "Second task",
                "origin_conversation_id": second_conversation["id"],
            },
        )
    ).json()
    first_run = (
        await client.post(f"/v1/events/{event_id}/tasks/{first_task['id']}/work")
    ).json()["run"]["id"]
    second_run = (
        await client.post(f"/v1/events/{event_id}/tasks/{second_task['id']}/work")
    ).json()["run"]["id"]
    monkeypatch.setattr("aloy_backend.worker.async_session", db_session_maker)

    assert await claim_next_run("worker-r3-a") == first_run
    assert await claim_next_run("worker-r3-b") is None

    async with db_session_maker() as session:
        first = await session.get(Run, first_run)
        second = await session.get(Run, second_run)
        assert first is not None and second is not None
        assert first.status == "running"
        assert second.status == "pending"
        first.status = "completed"
        first.completed_at = datetime.now(timezone.utc)
        first.lease_owner = None
        first.lease_expires_at = None
        session.add(first)
        await session.commit()

    assert await claim_next_run("worker-r3-b") == second_run


async def test_worker_outcome_advances_task_to_done_or_blocked(
    client, db_session_maker
):
    event, done_task = await _project_task(client, title="Finish cleanly")
    done_run_id = (
        await client.post(f"/v1/events/{event['id']}/tasks/{done_task['id']}/work")
    ).json()["run"]["id"]
    blocked_task = (
        await client.post(
            f"/v1/events/{event['id']}/tasks",
            json={"title": "Needs a choice"},
        )
    ).json()
    blocked_run_id = (
        await client.post(f"/v1/events/{event['id']}/tasks/{blocked_task['id']}/work")
    ).json()["run"]["id"]

    async with db_session_maker() as session:
        done = await session.get(Task, done_task["id"])
        done_run = await session.get(Run, done_run_id)
        blocked = await session.get(Task, blocked_task["id"])
        blocked_run = await session.get(Run, blocked_run_id)
        assert done is not None and done_run is not None
        assert blocked is not None and blocked_run is not None
        done.status = "in_progress"
        done_run.status = "completed"
        done_run.success = True
        done_run.final_answer = "A sourced shortlist is ready."
        blocked.status = "in_progress"
        blocked_run.status = "completed"
        blocked_run.success = True
        session.add_all([done, done_run, blocked, blocked_run])
        await session.flush()

        completed = await synchronize_task_after_run(session, run=done_run)
        waiting = await synchronize_task_after_run(
            session,
            run=blocked_run,
            clarification={
                "question": "Which location should I prioritize?",
                "options": ["Remote", "New York"],
            },
        )
        await session.commit()

    assert completed is not None
    assert completed.status == "done"
    assert completed.current_run_id is None
    assert completed.result_summary == "A sourced shortlist is ready."
    assert waiting is not None
    assert waiting.status == "blocked"
    assert waiting.current_run_id == blocked_run_id
    assert waiting.blocker == "Which location should I prioritize?"


async def test_conversation_rejects_a_second_foreground_run_before_saving_message(
    client, db_session_maker
):
    conversation = (
        await client.post("/v1/conversations", json={"title": "One run at a time"})
    ).json()
    async with db_session_maker() as session:
        session.add(
            Run(
                organization_id="user:test-user",
                user_id="test-user",
                event_id=conversation["event_id"],
                agent_id="default_agent",
                session_id=conversation["id"],
                conversation_id=conversation["id"],
                task="Existing foreground work",
                status="pending",
            )
        )
        await session.commit()

    denied = await client.post(
        f"/v1/conversations/{conversation['id']}/messages",
        json={"content": "Start another run", "stream": False},
    )
    assert denied.status_code == 409
    assert "queued or active work" in denied.json()["detail"]

    async with db_session_maker() as session:
        messages = (
            (
                await session.execute(
                    select(Message).where(Message.conversation_id == conversation["id"])
                )
            )
            .scalars()
            .all()
        )
    assert messages == []
