from __future__ import annotations

from sqlmodel import col, select

from aloy_backend.event_context import refresh_event_context_snapshot
from aloy_backend.models import (
    Event,
    EventBrief,
    EventContextSnapshot,
    EventTrailEntry,
    KnowledgeEntry,
)


async def _create_event(client, title: str) -> dict:
    response = await client.post("/v1/events", json={"title": title})
    assert response.status_code == 201
    return response.json()


async def _event_owner(db_session_maker, event_id: str) -> Event:
    async with db_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None
        session.expunge(event)
        return event


async def _add_memory(
    db_session_maker,
    *,
    event: Event,
    memory_id: str,
    content: str,
    event_id: str | None,
    conflict_key: str | None = None,
    retention: dict | None = None,
) -> None:
    async with db_session_maker() as session:
        session.add(
            KnowledgeEntry(
                id=memory_id,
                organization_id=event.organization_id,
                user_id=event.user_id,
                event_id=event_id,
                content=content,
                conflict_key=conflict_key,
                retention=retention,
                source="user",
            )
        )
        await session.commit()


async def test_event_memory_lists_only_same_event_and_inherited_global(
    client, db_session_maker
):
    university = await _create_event(client, "University")
    madrid = await _create_event(client, "Madrid")
    owner = await _event_owner(db_session_maker, university["id"])
    await _add_memory(
        db_session_maker,
        event=owner,
        memory_id="global-home",
        content="Home country is South Africa",
        event_id=None,
    )
    await _add_memory(
        db_session_maker,
        event=owner,
        memory_id="university-course",
        content="Computer Science lecture is Monday at 10",
        event_id=university["id"],
    )
    await _add_memory(
        db_session_maker,
        event=owner,
        memory_id="madrid-hotel",
        content="Madrid hotel is near the stadium",
        event_id=madrid["id"],
    )

    response = await client.get(f"/v1/events/{university['id']}/memory")

    assert response.status_code == 200
    payload = response.json()
    assert [row["id"] for row in payload["event_records"]] == ["university-course"]
    assert [row["id"] for row in payload["inherited_global_records"]] == ["global-home"]
    assert payload["event_records"][0]["scope"] == "event"
    assert payload["event_records"][0]["can_correct"] is True
    assert payload["inherited_global_records"][0]["scope"] == "global"
    assert payload["inherited_global_records"][0]["can_correct"] is False
    assert "Madrid" not in str(payload)

    cross_event = await client.post(
        f"/v1/events/{university['id']}/memory/madrid-hotel/corrections",
        json={"content": "Try to leak Madrid"},
    )
    assert cross_event.status_code == 404


async def test_event_memory_correction_supersedes_and_forget_soft_deletes(
    client, db_session_maker
):
    university = await _create_event(client, "University")
    owner = await _event_owner(db_session_maker, university["id"])
    await _add_memory(
        db_session_maker,
        event=owner,
        memory_id="exam-date",
        content="The statistics exam is Tuesday",
        event_id=university["id"],
        conflict_key="statistics-exam-date",
    )
    async with db_session_maker() as session:
        snapshot, _pack, _created = await refresh_event_context_snapshot(
            session,
            organization_id=owner.organization_id,
            user_id=owner.user_id,
            event_id=university["id"],
        )
        session.add(
            EventBrief(
                organization_id=owner.organization_id,
                user_id=owner.user_id,
                event_id=university["id"],
                version=1,
                source_context_snapshot_id=snapshot.id,
                fingerprint="initial-exam-brief",
                payload={"summary": "Exam is Tuesday"},
                evidence_refs=[{"kind": "knowledge_entry", "id": "exam-date"}],
            )
        )
        await session.commit()

    correction = await client.post(
        f"/v1/events/{university['id']}/memory/exam-date/corrections",
        json={
            "content": "The statistics exam is Thursday",
            "reason": "The lecturer changed the date",
        },
    )

    assert correction.status_code == 200
    replacement = correction.json()["record"]
    assert replacement["content"] == "The statistics exam is Thursday"
    assert replacement["source"] == "user_correction"
    assert replacement["event_id"] == university["id"]

    async with db_session_maker() as session:
        original = await session.get(KnowledgeEntry, "exam-date")
        assert original is not None
        assert original.status == "superseded"
        assert original.superseded_by == replacement["id"]
        brief = (
            (
                await session.execute(
                    select(EventBrief).where(EventBrief.event_id == university["id"])
                )
            )
            .scalars()
            .one()
        )
        assert brief.status == "stale"
        snapshots = (
            (
                await session.execute(
                    select(EventContextSnapshot).where(
                        EventContextSnapshot.event_id == university["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert snapshots

    visible = await client.get(f"/v1/events/{university['id']}/memory")
    assert [row["id"] for row in visible.json()["event_records"]] == [replacement["id"]]

    forgotten = await client.delete(
        f"/v1/events/{university['id']}/memory/{replacement['id']}"
    )
    assert forgotten.status_code == 204
    assert (await client.get(f"/v1/events/{university['id']}/memory")).json()[
        "event_records"
    ] == []

    async with db_session_maker() as session:
        deleted = await session.get(KnowledgeEntry, replacement["id"])
        assert deleted is not None
        assert deleted.status == "deleted"
        trail = (
            (
                await session.execute(
                    select(EventTrailEntry)
                    .where(EventTrailEntry.event_id == university["id"])
                    .order_by(col(EventTrailEntry.created_at))
                )
            )
            .scalars()
            .all()
        )
        kinds = [entry.kind for entry in trail]
        assert "event_memory_corrected" in kinds
        assert "event_memory_forgotten" in kinds


async def test_event_memory_promotion_is_idempotent_and_global(
    client, db_session_maker
):
    university = await _create_event(client, "University")
    madrid = await _create_event(client, "Madrid")
    owner = await _event_owner(db_session_maker, university["id"])
    await _add_memory(
        db_session_maker,
        event=owner,
        memory_id="dietary-preference",
        content="I prefer vegetarian meals",
        event_id=university["id"],
    )

    first = await client.post(
        f"/v1/events/{university['id']}/memory/dietary-preference/promote"
    )
    second = await client.post(
        f"/v1/events/{university['id']}/memory/dietary-preference/promote"
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["created"] is True
    assert second.json()["created"] is False
    global_record = first.json()["record"]
    assert global_record["id"] == second.json()["record"]["id"]
    assert global_record["event_id"] is None
    assert global_record["scope"] == "global"
    university_memory = (
        await client.get(f"/v1/events/{university['id']}/memory")
    ).json()
    assert university_memory["event_records"][0]["can_promote"] is False
    assert (
        university_memory["event_records"][0]["promoted_global_id"]
        == global_record["id"]
    )
    madrid_memory = (await client.get(f"/v1/events/{madrid['id']}/memory")).json()
    assert [row["content"] for row in madrid_memory["inherited_global_records"]] == [
        "I prefer vegetarian meals"
    ]

    forgotten_global = await client.delete(
        f"/v1/me/memory/knowledge/{global_record['id']}"
    )
    assert forgotten_global.status_code == 204
    university_after_forget = (
        await client.get(f"/v1/events/{university['id']}/memory")
    ).json()
    assert university_after_forget["event_records"][0]["can_promote"] is True
    restored = await client.post(
        f"/v1/events/{university['id']}/memory/dietary-preference/promote"
    )
    assert restored.status_code == 200
    assert restored.json()["created"] is True
    assert restored.json()["record"]["id"] == global_record["id"]

    async with db_session_maker() as session:
        global_rows = (
            (
                await session.execute(
                    select(KnowledgeEntry).where(
                        KnowledgeEntry.id == global_record["id"]
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(global_rows) == 1
        promotion_trail = (
            (
                await session.execute(
                    select(EventTrailEntry).where(
                        EventTrailEntry.event_id == university["id"],
                        EventTrailEntry.kind == "event_memory_promoted",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(promotion_trail) == 2


async def test_event_memory_rejects_another_users_event(client):
    private_event = await _create_event(client, "Private")

    response = await client.get(
        f"/v1/events/{private_event['id']}/memory",
        headers={"X-Test-User": "another-user"},
    )

    assert response.status_code == 404
