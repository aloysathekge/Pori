import pytest

pytestmark = pytest.mark.asyncio


async def test_health_is_public(client):
    response = await client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1"}


async def test_conversation_crud_scopes_to_current_user(client):
    created = await client.post("/v1/conversations", json={"title": "First"})

    assert created.status_code == 201
    conversation = created.json()
    assert conversation["title"] == "First"
    assert conversation["message_count"] == 0

    listed = await client.get("/v1/conversations")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [conversation["id"]]

    updated = await client.patch(
        f"/v1/conversations/{conversation['id']}",
        json={"title": "Renamed"},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Renamed"

    deleted = await client.delete(f"/v1/conversations/{conversation['id']}")
    assert deleted.status_code == 204

    retained = await client.get(f"/v1/conversations/{conversation['id']}")
    assert retained.status_code == 404


async def test_conversation_without_active_run_has_quiet_live_probe(client):
    created = await client.post("/v1/conversations", json={"title": "Idle"})
    conversation_id = created.json()["id"]

    response = await client.get(f"/v1/conversations/{conversation_id}/live")

    assert response.status_code == 204
    assert response.content == b""


async def test_core_memory_blocks_can_be_read_and_updated(client):
    initial = await client.get("/v1/me/memory")
    assert initial.status_code == 200
    blocks = initial.json()["blocks"]
    assert {block["label"] for block in blocks} == {"persona", "human", "notes"}

    updated = await client.patch(
        "/v1/me/memory/human",
        json={"value": "User prefers concise API responses."},
    )
    assert updated.status_code == 200
    assert updated.json()["value"] == "User prefers concise API responses."

    block = await client.get("/v1/me/memory/human")
    assert block.status_code == 200
    assert block.json()["value"] == "User prefers concise API responses."


async def test_knowledge_entries_and_usage_summary(client):
    created = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "User is building Pori Cloud.",
            "tags": ["project"],
            "importance": 3,
        },
    )
    assert created.status_code == 201
    assert created.json()["source"] == "user"

    listed = await client.get("/v1/me/memory/knowledge")
    assert listed.status_code == 200
    assert listed.json()[0]["content"] == "User is building Pori Cloud."

    usage = await client.get("/v1/me/usage")
    assert usage.status_code == 200
    assert usage.json() == {
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_requests": 0,
        "by_model": {},
    }


async def test_memory_contract_conflict_search_export_and_delete(client):
    first = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "Preferred database is SQLite.",
            "kind": "semantic",
            "confidence": 0.7,
            "sensitivity": "confidential",
            "source": "user",
            "source_id": "message-1",
            "conflict_key": "preferred-database",
        },
    )
    assert first.status_code == 201
    first_record = first.json()
    assert first_record["organization_id"] == "user:test-user"
    assert first_record["provenance"]["source_id"] == "message-1"

    second = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "Preferred database is PostgreSQL.",
            "kind": "semantic",
            "conflict_key": "preferred-database",
            "conflict_policy": "supersede",
        },
    )
    assert second.status_code == 201
    second_record = second.json()

    searched = await client.post(
        "/v1/me/memory/archival/search",
        json={"query": "PostgreSQL", "k": 5},
    )
    assert searched.status_code == 200
    assert [record["id"] for record in searched.json()] == [second_record["id"]]

    exported = await client.get("/v1/me/memory/export/all")
    assert exported.status_code == 200
    records = {record["id"]: record for record in exported.json()["records"]}
    assert records[first_record["id"]]["status"] == "superseded"
    assert records[first_record["id"]]["superseded_by"] == second_record["id"]

    deleted = await client.delete(f"/v1/me/memory/knowledge/{second_record['id']}")
    assert deleted.status_code == 204

    searched_after_delete = await client.post(
        "/v1/me/memory/archival/search",
        json={"query": "PostgreSQL", "k": 5},
    )
    assert searched_after_delete.status_code == 200
    assert searched_after_delete.json() == []


async def test_expired_memory_is_not_returned(client):
    created = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "Temporary launch code",
            "retention": {"delete_after": "2000-01-01T00:00:00Z"},
        },
    )
    assert created.status_code == 201

    listed = await client.get("/v1/me/memory/knowledge")
    assert listed.status_code == 200
    assert listed.json() == []

    searched = await client.post(
        "/v1/me/memory/archival/search",
        json={"query": "launch code", "k": 5},
    )
    assert searched.status_code == 200
    assert searched.json() == []


async def test_owner_can_export_and_delete_agent_session_memory(client):
    created = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "Session-specific investigation note",
            "agent_id": "agent-1",
            "session_id": "session-1",
        },
    )
    assert created.status_code == 201
    record_id = created.json()["id"]

    listed = await client.get("/v1/me/memory/knowledge")
    assert record_id in {record["id"] for record in listed.json()}

    exported = await client.get("/v1/me/memory/export/all")
    assert record_id in {record["id"] for record in exported.json()["records"]}

    deleted = await client.delete(f"/v1/me/memory/knowledge/{record_id}")
    assert deleted.status_code == 204


async def test_memory_api_does_not_leak_between_users(client):
    alice_headers = {"X-Test-User": "alice"}
    bob_headers = {"X-Test-User": "bob"}
    created = await client.post(
        "/v1/me/memory/knowledge",
        headers=alice_headers,
        json={"content": "Alice private launch plan"},
    )
    assert created.status_code == 201
    alice_record_id = created.json()["id"]

    bob_list = await client.get(
        "/v1/me/memory/knowledge",
        headers=bob_headers,
    )
    assert bob_list.status_code == 200
    assert bob_list.json() == []

    bob_search = await client.post(
        "/v1/me/memory/archival/search",
        headers=bob_headers,
        json={"query": "launch plan", "k": 10},
    )
    assert bob_search.status_code == 200
    assert bob_search.json() == []

    bob_delete = await client.delete(
        f"/v1/me/memory/knowledge/{alice_record_id}",
        headers=bob_headers,
    )
    assert bob_delete.status_code == 404


async def test_retention_prune_removes_expired_records(client):
    created = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "Expired operational note",
            "retention": {"delete_after": "2000-01-01T00:00:00Z"},
        },
    )
    record_id = created.json()["id"]

    pruned = await client.post("/v1/me/memory/retention/prune")
    assert pruned.status_code == 200
    assert pruned.json() == {"deleted": 1, "record_ids": [record_id]}
