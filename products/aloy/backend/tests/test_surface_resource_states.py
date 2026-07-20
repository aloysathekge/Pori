from __future__ import annotations

import pytest

from aloy_backend.surface_manifest import SurfaceManifest
from aloy_backend.surface_resource_states import (
    REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES,
    REQUIRED_SURFACE_SCENARIO_FIXTURES,
    REQUIRED_SURFACE_STATE_FIXTURES,
    surface_fixture_applicable,
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


def test_action_only_capabilities_are_not_projected_as_data_resources():
    manifest = SurfaceManifest(capabilities=["ask_aloy", "event"])
    states = surface_resource_states(manifest, {"event": {"id": "event-1"}})

    assert set(states) == {"event"}


@pytest.mark.parametrize("status", REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES)
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


def test_long_content_fixture_populates_public_shapes_without_mutating_context():
    original = _context()
    fixture = surface_state_fixture_context(original, "long_content")

    assert original["data"]["tasks"] == []
    assert len(fixture["data"]["tasks"]) == 24
    assert len(fixture["data"]["surface"]["career"]) == 24
    assert len(fixture["data"]["records"]["course"]) == 24
    assert len(fixture["data"]["event"]["summary"]) > 500
    assert {item["status"] for item in fixture["resource_states"].values()} == {"ready"}


def test_approval_fixture_uses_pending_proposal_and_interaction_truth():
    original = _context()
    original["capabilities"].append("proposals")
    original["data"]["proposals"] = []
    manifest = SurfaceManifest(capabilities=original["capabilities"])
    fixture = surface_state_fixture_context(
        original,
        "approval_required",
        manifest=manifest,
    )

    assert original["data"]["proposals"] == []
    assert fixture["data"]["proposals"][0]["status"] == "pending"
    assert fixture["data"]["interactions"][0]["status"] == "waiting_approval"
    assert surface_fixture_applicable(
        manifest,
        "approval_required",
        original["capabilities"],
    )


def test_scenario_fixture_set_is_stable_and_separate_from_resource_statuses():
    assert REQUIRED_SURFACE_SCENARIO_FIXTURES == (
        "long_content",
        "approval_required",
    )
    assert REQUIRED_SURFACE_STATE_FIXTURES == (
        *REQUIRED_SURFACE_RESOURCE_STATE_FIXTURES,
        *REQUIRED_SURFACE_SCENARIO_FIXTURES,
    )


def test_unknown_state_fixture_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Surface state fixture"):
        surface_state_fixture_context(_context(), "mystery")  # type: ignore[arg-type]
