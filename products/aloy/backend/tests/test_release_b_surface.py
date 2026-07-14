import pytest

pytestmark = pytest.mark.asyncio


async def _create_org(client, slug="release-b"):
    response = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Release B", "slug": slug},
    )
    assert response.status_code == 201
    return response.json()["id"]


async def test_provider_and_capability_policy_controls_agent_surface(client):
    organization_id = await _create_org(client)
    headers = {
        "X-Test-User": "alice",
        "X-Pori-Organization": organization_id,
    }
    policy = await client.patch(
        f"/v1/organizations/{organization_id}/policy",
        headers=headers,
        json={
            "policy": {
                "allowed_provider_profiles": ["google"],
                "allowed_models": ["gemini-2.5-flash"],
                "allowed_capability_groups": ["kernel"],
            }
        },
    )
    assert policy.status_code == 200

    denied = await client.post(
        "/v1/agent-configs",
        headers=headers,
        json={"name": "Denied", "provider": "anthropic", "model": "claude-x"},
    )
    assert denied.status_code == 403

    allowed = await client.post(
        "/v1/agent-configs",
        headers=headers,
        json={
            "name": "Allowed",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        },
    )
    assert allowed.status_code == 201
    assert allowed.json()["provider"] == "google"

    tools = await client.get("/v1/agent-configs/info/tools", headers=headers)
    assert tools.status_code == 200
    assert tools.json()["groups"] == ["kernel"]
    assert {item["name"] for item in tools.json()["tools"]} == {
        "answer",
        "ask_user",
        "done",
        "think",
        "skills_list",
        "skill_view",
        "update_plan",
    }


async def test_setup_diagnostics_reuse_runtime_availability_without_secrets(
    client, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-never-be-returned")
    response = await client.get(
        "/v1/agent-configs/info/setup",
        params={
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"][0]["available"] is True
    assert payload["providers"][0]["credential_configured"] is True
    assert "must-never-be-returned" not in response.text
    assert payload["capabilities"]["fingerprint"]


async def test_unavailable_explicit_capability_is_rejected(client, monkeypatch):
    # web_search is available if ANY backend key is set (Tavily/Serper/SerpApi),
    # so clear them all — otherwise a dev .env key makes the capability
    # available and the request is accepted (201) instead of rejected.
    for key in ("TAVILY_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    response = await client.post(
        "/v1/agent-configs",
        json={"name": "Web", "tools": ["web_search"]},
    )

    assert response.status_code == 409
    assert "unavailable" in response.json()["detail"]
