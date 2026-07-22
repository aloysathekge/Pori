from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from aloy_backend.models import (
    ActionProposal,
    Conversation,
    Event,
    EventTrailEntry,
    Task,
)


async def test_event_creation_queues_cover_without_blocking_and_upload_can_replace_it(
    client, monkeypatch, tmp_path
):
    from aloy_backend import config as config_mod
    from aloy_backend import storage as storage_mod

    monkeypatch.setattr(config_mod.settings, "storage_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(storage_mod, "_STORE", None)

    created = await client.post(
        "/v1/events",
        json={"title": "University", "setup_mode": "simple"},
    )
    assert created.status_code == 201
    assert created.json()["cover"] == {
        "status": "queued",
        "source": "automatic",
        "alt_text": "",
        "url": None,
    }

    event_id = created.json()["id"]
    uploaded = await client.post(
        f"/v1/events/{event_id}/cover",
        files={"file": ("campus.png", b"\x89PNG\r\n\x1a\ncover-body", "image/png")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["cover"]["status"] == "ready"
    assert uploaded.json()["cover"]["source"] == "user_upload"
    assert uploaded.json()["cover"]["url"] == f"/events/{event_id}/cover"

    response = await client.get(f"/v1/events/{event_id}/cover")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == b"\x89PNG\r\n\x1a\ncover-body"

    monkeypatch.setattr(storage_mod, "_STORE", None)


async def test_event_archive_restore_and_listing_are_explicit(client):
    created = await client.post(
        "/v1/events",
        json={"title": "Career OS", "cover_mode": "none"},
    )
    event_id = created.json()["id"]

    archived = await client.patch(
        f"/v1/events/{event_id}", json={"lifecycle": "archived"}
    )
    assert archived.status_code == 200
    assert archived.json()["lifecycle"] == "archived"
    assert event_id not in {
        row["id"] for row in (await client.get("/v1/events")).json()
    }
    assert event_id in {
        row["id"]
        for row in (
            await client.get("/v1/events", params={"lifecycle": "archived"})
        ).json()
    }

    restored = await client.patch(
        f"/v1/events/{event_id}", json={"lifecycle": "active"}
    )
    assert restored.status_code == 200
    assert restored.json()["lifecycle"] == "active"
    assert event_id in {row["id"] for row in (await client.get("/v1/events")).json()}


async def test_life_cannot_be_archived_or_permanently_deleted(client):
    life = next(
        row for row in (await client.get("/v1/events")).json() if row["is_life"]
    )

    archived = await client.patch(
        f"/v1/events/{life['id']}", json={"lifecycle": "archived"}
    )
    assert archived.status_code == 409
    assert archived.json()["detail"] == "Life cannot be archived"

    deleted = await client.request(
        "DELETE",
        f"/v1/events/{life['id']}",
        json={"confirmation": life["title"]},
    )
    assert deleted.status_code == 409
    assert deleted.json()["detail"] == "Life cannot be permanently deleted"


async def test_permanent_delete_requires_archive_and_exact_name(
    client, db_session_maker
):
    created = await client.post(
        "/v1/events",
        json={"title": "Career OS", "cover_mode": "none"},
    )
    event_id = created.json()["id"]
    conversation_id = created.json()["conversation_id"]

    not_archived = await client.request(
        "DELETE",
        f"/v1/events/{event_id}",
        json={"confirmation": "Career OS"},
    )
    assert not_archived.status_code == 409

    assert (
        await client.patch(f"/v1/events/{event_id}", json={"lifecycle": "archived"})
    ).status_code == 200
    mismatch = await client.request(
        "DELETE",
        f"/v1/events/{event_id}",
        json={"confirmation": "career os"},
    )
    assert mismatch.status_code == 422

    deleted = await client.request(
        "DELETE",
        f"/v1/events/{event_id}",
        json={"confirmation": "Career OS"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["event_id"] == event_id
    assert (await client.get(f"/v1/events/{event_id}")).status_code == 404

    async with db_session_maker() as session:
        assert await session.get(Event, event_id) is None
        assert await session.get(Conversation, conversation_id) is None


async def test_reading_life_does_not_create_an_empty_conversation(client):
    events = await client.get("/v1/events")
    life = next(row for row in events.json() if row["is_life"])
    assert life["conversation_id"] is None

    surface = await client.get(f"/v1/events/{life['id']}")
    assert surface.status_code == 200
    assert surface.json()["event"]["conversation_id"] is None
    assert (await client.get("/v1/conversations")).json() == []


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


async def test_chat_list_defaults_to_life_and_event_scope_is_explicit(client):
    life = await client.post("/v1/conversations", json={"title": "Personal chat"})
    project = await client.post("/v1/events", json={"title": "Career OS"})
    project_id = project.json()["id"]
    canonical_id = project.json()["conversation_id"]
    provenance = await client.post(
        "/v1/conversations",
        json={"title": "Imported provenance", "event_id": project_id},
    )

    personal_list = await client.get("/v1/conversations")
    assert [row["id"] for row in personal_list.json()] == [life.json()["id"]]

    project_list = await client.get(
        "/v1/conversations", params={"event_id": project_id}
    )
    assert {row["id"] for row in project_list.json()} == {
        canonical_id,
        provenance.json()["id"],
    }


async def test_life_conversation_delete_retargets_default_without_deleting_life(
    client,
):
    first = await client.post("/v1/conversations", json={"title": "First"})
    second = await client.post("/v1/conversations", json={"title": "Second"})
    life_event_id = first.json()["event_id"]

    events = await client.get("/v1/events")
    life = next(row for row in events.json() if row["id"] == life_event_id)
    assert life["conversation_id"] == second.json()["id"]

    assert (
        await client.delete(f"/v1/conversations/{second.json()['id']}")
    ).status_code == 204
    life = next(
        row
        for row in (await client.get("/v1/events")).json()
        if row["id"] == life_event_id
    )
    assert life["conversation_id"] == first.json()["id"]

    assert (
        await client.delete(f"/v1/conversations/{first.json()['id']}")
    ).status_code == 204
    life = next(
        row
        for row in (await client.get("/v1/events")).json()
        if row["id"] == life_event_id
    )
    assert life["conversation_id"] is None
    assert (await client.get("/v1/conversations")).json() == []


async def test_project_event_can_preserve_life_conversation_origin(client):
    source = await client.post("/v1/conversations", json={"title": "Career direction"})
    created = await client.post(
        "/v1/events",
        json={
            "title": "Career OS",
            "summary": "Find the next role",
            "origin_conversation_id": source.json()["id"],
        },
    )
    assert created.status_code == 201
    assert created.json()["origin_conversation_id"] == source.json()["id"]
    assert created.json()["conversation_id"] != source.json()["id"]

    surface = await client.get(f"/v1/events/{created.json()['id']}")
    event = surface.json()["event"]
    trail = next(
        section
        for section in surface.json()["surface"]["sections"]
        if section["kind"] == "activity"
    )["entries"]
    assert event["origin_conversation_id"] == source.json()["id"]
    assert trail[-1]["payload"]["origin_conversation_id"] == source.json()["id"]
    assert (
        await client.get(f"/v1/conversations/{source.json()['id']}")
    ).status_code == 200


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
