from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aloy_backend.models import Run


async def _event(client) -> dict:
    response = await client.post(
        "/v1/events",
        json={"title": "Career OS", "summary": "Find an AI engineering role"},
    )
    assert response.status_code == 201
    return response.json()


async def test_surface_status_exposes_prebuild_progress(client, db_session_maker):
    event = await _event(client)
    empty = await client.get(f"/v1/events/{event['id']}/surface/status")
    assert empty.status_code == 200
    assert empty.json() is None

    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        session.add(
            Run(
                id="surface-status-running",
                organization_id="user:test-user",
                user_id="test-user",
                event_id=event["id"],
                agent_id="surface-builder",
                session_id=event["id"],
                run_kind="surface_builder",
                task="Build an application manager",
                status="running",
                attempt_count=2,
                started_at=now - timedelta(seconds=42),
                lease_owner="worker-test",
                lease_expires_at=now + timedelta(minutes=10),
                progress={
                    "stage": "generating_candidate",
                    "submission": 1,
                    "updated_at": now.isoformat(),
                },
            )
        )
        await session.commit()

    response = await client.get(f"/v1/events/{event['id']}/surface/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "surface-status-running"
    assert payload["status"] == "running"
    assert payload["active"] is True
    assert payload["stage"] == "generating_candidate"
    assert payload["message"] == "Designing and writing your Surface"
    assert payload["attempt_count"] == 2
    assert payload["submission"] == 1
    assert payload["elapsed_seconds"] >= 42


async def test_surface_status_marks_expired_worker_lease_overdue(
    client, db_session_maker
):
    event = await _event(client)
    now = datetime.now(timezone.utc)
    async with db_session_maker() as session:
        session.add(
            Run(
                id="surface-status-overdue",
                organization_id="user:test-user",
                user_id="test-user",
                event_id=event["id"],
                agent_id="surface-builder",
                session_id=event["id"],
                run_kind="surface_builder",
                task="Build an application manager",
                status="running",
                attempt_count=1,
                started_at=now - timedelta(minutes=20),
                lease_owner="worker-test",
                lease_expires_at=now - timedelta(seconds=1),
                progress={"stage": "building_bundle", "submission": 1},
            )
        )
        await session.commit()

    response = await client.get(f"/v1/events/{event['id']}/surface/status")
    payload = response.json()
    assert payload["status"] == "overdue"
    assert payload["active"] is False
    assert payload["message"] == "The Surface Builder stopped reporting progress"
