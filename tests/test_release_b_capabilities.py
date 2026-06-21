from pydantic import BaseModel

from pori.capabilities import CapabilityGroup, SkillEligibility
from pori.providers import diagnose_provider, get_provider_profile
from pori.tools.registry import CapabilityResolutionError, CollisionPolicy, ToolRegistry


class EmptyParams(BaseModel):
    pass


def _noop(params, context):
    return None


def test_snapshot_is_immutable_after_registry_changes():
    registry = ToolRegistry()
    registry.register_tool("first", EmptyParams, _noop, "First")
    snapshot = registry.snapshot()

    registry.register_tool("second", EmptyParams, _noop, "Second")

    assert snapshot.tool_names == frozenset({"first"})
    assert "second" not in snapshot.tools
    assert snapshot.fingerprint != registry.snapshot().fingerprint


def test_protected_kernel_cannot_be_removed():
    registry = ToolRegistry()
    registry.register_tool("answer", EmptyParams, _noop, "Answer")
    registry.register_tool("optional", EmptyParams, _noop, "Optional")
    registry.define_group(
        CapabilityGroup(
            name="kernel",
            description="Protected",
            tool_names=frozenset({"answer"}),
            protected=True,
        )
    )

    snapshot = registry.snapshot(include_tools={"optional"})
    assert snapshot.tool_names == frozenset({"answer", "optional"})

    try:
        registry.snapshot(exclude_tools={"answer"})
    except CapabilityResolutionError as exc:
        assert "cannot be excluded" in str(exc)
    else:  # pragma: no cover - regression assertion
        raise AssertionError("protected kernel exclusion should fail")


def test_collision_policy_and_skill_eligibility(monkeypatch):
    registry = ToolRegistry(collision_policy=CollisionPolicy.ERROR)
    registry.register_tool("search", EmptyParams, _noop, "Search")
    try:
        registry.register_tool("search", EmptyParams, _noop, "Duplicate")
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover - regression assertion
        raise AssertionError("duplicate registration should fail")

    monkeypatch.delenv("SEARCH_TOKEN", raising=False)
    eligibility = SkillEligibility(
        required_tools=frozenset({"search"}),
        required_credentials=("SEARCH_TOKEN",),
        required_model_capabilities=frozenset({"tools"}),
        version="2",
        source="org-catalog",
    )
    report = eligibility.evaluate(
        available_tools=frozenset({"search"}),
        model_capabilities=frozenset({"tools"}),
    )
    assert report.eligible is False
    assert report.reasons == ("missing_credential:SEARCH_TOKEN",)


def test_group_output_limit_is_enforced_by_executor():
    from pori.tools.registry import ToolExecutor

    registry = ToolRegistry()
    registry.register_tool("large", EmptyParams, lambda p, c: "x" * 100, "Large")
    registry.define_group(
        CapabilityGroup(
            name="bounded",
            description="Bounded output",
            tool_names=frozenset({"large"}),
            max_output_chars=10,
        )
    )

    result = ToolExecutor(registry).execute_tool("large", {}, {})
    assert result["success"] is True
    assert result["result"] == {
        "truncated": True,
        "max_output_chars": 10,
        "preview": '"xxxxxxxxx',
    }


def test_provider_profiles_drive_aliases_and_diagnostics(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    profile = get_provider_profile("claude")
    diagnostic = diagnose_provider("claude", model=profile.default_model)

    assert profile.name == "anthropic"
    assert diagnostic.provider == "anthropic"
    assert diagnostic.credential_configured is False
    assert diagnostic.available is False
    assert diagnostic.reasons == ("missing_credential:ANTHROPIC_API_KEY",)
