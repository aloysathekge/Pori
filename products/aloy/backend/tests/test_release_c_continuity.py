import pytest

pytestmark = pytest.mark.asyncio


async def test_session_search_branch_export_and_delete(client):
    created = await client.post("/v1/conversations", json={"title": "Continuity"})
    conversation_id = created.json()["id"]
    queued = await client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": "PostgreSQL migration checklist", "max_steps": 1},
    )
    assert queued.status_code == 202

    detail = await client.get(f"/v1/conversations/{conversation_id}")
    message_id = detail.json()["messages"][0]["id"]
    searched = await client.get(
        "/v1/conversations/search", params={"q": "PostgreSQL migration"}
    )
    assert searched.status_code == 200
    assert searched.json()[0]["conversation_id"] == conversation_id

    branched = await client.post(
        f"/v1/conversations/{conversation_id}/branch",
        json={"through_message_id": message_id, "title": "Alternative"},
    )
    assert branched.status_code == 201
    branch = branched.json()
    assert branch["parent_conversation_id"] == conversation_id
    assert branch["branched_from_message_id"] == message_id
    assert branch["messages"][0]["metadata"]["copied_from_message_id"] == message_id

    exported = await client.get(f"/v1/conversations/{branch['id']}/export")
    assert exported.status_code == 200
    assert len(exported.json()["messages"]) == 1

    deleted = await client.delete(f"/v1/conversations/{branch['id']}")
    assert deleted.status_code == 204
    assert (await client.get(f"/v1/conversations/{branch['id']}")).status_code == 404


async def test_session_search_is_organization_scoped_before_ranking(client):
    alice = {"X-Test-User": "alice"}
    created = await client.post(
        "/v1/conversations", headers=alice, json={"title": "Private"}
    )
    await client.post(
        f"/v1/conversations/{created.json()['id']}/messages",
        headers=alice,
        json={"content": "confidential launch phrase", "max_steps": 1},
    )

    bob_search = await client.get(
        "/v1/conversations/search",
        headers={"X-Test-User": "bob"},
        params={"q": "confidential launch phrase"},
    )
    assert bob_search.status_code == 200
    assert bob_search.json() == []


async def test_shared_org_session_search_requires_explicit_policy(client):
    created_org = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Shared", "slug": "shared-continuity"},
    )
    organization_id = created_org.json()["id"]
    owner_headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    await client.post(
        f"/v1/organizations/{organization_id}/members",
        headers=owner_headers,
        json={"user_id": "bob", "role": "member"},
    )
    conversation = await client.post(
        "/v1/conversations", headers=owner_headers, json={"title": "Owner session"}
    )
    await client.post(
        f"/v1/conversations/{conversation.json()['id']}/messages",
        headers=owner_headers,
        json={"content": "shared incident timeline", "max_steps": 1},
    )
    bob_headers = {
        "X-Test-User": "bob",
        "X-Pori-Organization": organization_id,
    }
    private = await client.get(
        "/v1/conversations/search",
        headers=bob_headers,
        params={"q": "shared incident timeline"},
    )
    assert private.json() == []

    await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=owner_headers,
        json={"policy": {"allow_shared_session_search": True}},
    )
    shared = await client.get(
        "/v1/conversations/search",
        headers=bob_headers,
        params={"q": "shared incident timeline"},
    )
    assert len(shared.json()) == 1


async def test_context_search_fuses_session_and_memory_with_provenance(client):
    conversation = await client.post("/v1/conversations", json={"title": "Fusion"})
    await client.post(
        f"/v1/conversations/{conversation.json()['id']}/messages",
        json={"content": "deployment rollback signal", "max_steps": 1},
    )
    memory = await client.post(
        "/v1/me/memory/knowledge",
        json={
            "content": "deployment rollback signal",
            "source": "user",
            "source_id": "message-source",
        },
    )
    assert memory.status_code == 201

    response = await client.get(
        "/v1/conversations/context/search",
        params={"q": "deployment rollback signal"},
    )
    assert response.status_code == 200
    hits = response.json()
    assert {hit["source_type"] for hit in hits} == {"session", "memory"}
    memory_hit = next(hit for hit in hits if hit["source_type"] == "memory")
    assert memory_hit["source_id"] == memory.json()["id"]
    assert memory_hit["provenance"]["source_id"] == "message-source"
