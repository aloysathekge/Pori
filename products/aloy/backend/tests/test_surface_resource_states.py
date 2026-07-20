from __future__ import annotations

import pytest

from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_resource_states import (
    REQUIRED_SURFACE_STATE_FIXTURES,
    surface_resource_states,
    surface_state_fixture_context,
)


def _context() -> dict:
    manifest = SurfaceManifest(
        capabilities=["event", "tasks", "data:career", "records:course"]
    )
    data = {
        "event": {"id": "event-1"},
        "tasks": [],
        "surface": {"career": [{"key": "job-1"}]},
        "records": {"course": []},
        "interactions": [],
    }
    return {
        "capabilities": manifest.capabilities,
        "resource_state_version": "1",
        "resource_states": surface_resource_states(manifest, data),
        "data": data,
    }


def test_resource_states_distinguish_ready_from_empty():
    states = _context()["resource_states"]

    assert states["event"]["status"] == "ready"
    assert states["tasks"]["status"] == "empty"
    assert states["data:career"]["status"] == "ready"
    assert states["records:course"]["status"] == "empty"


@pytest.mark.parametrize("status", REQUIRED_SURFACE_STATE_FIXTURES)
def test_state_fixtures_use_the_public_resource_contract(status: str):
    original = _context()
    fixture = surface_state_fixture_context(original, status)  # type: ignore[arg-type]

    assert original["resource_states"]["event"]["status"] == "ready"
    assert {item["status"] for item in fixture["resource_states"].values()} == {status}
    if status in {"loading", "empty", "error", "permission_denied"}:
        assert "event" not in fixture["data"]
        assert fixture["data"]["tasks"] == []
        assert fixture["data"]["surface"]["career"] == []
        assert fixture["data"]["records"]["course"] == []


def test_unknown_state_fixture_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Surface state fixture"):
        surface_state_fixture_context(_context(), "mystery")  # type: ignore[arg-type]
