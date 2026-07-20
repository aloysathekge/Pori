"""Tests for declarative tool side effects and the authorization policy."""

import pytest
from pydantic import BaseModel

from pori.tools.policy import (
    AuthorizationDecision,
    ToolAuthorizationPolicy,
    task_requests_artifact,
)
from pori.tools.registry import (
    ReconciliationStatus,
    SideEffect,
    ToolReconciliation,
    ToolRegistry,
)
from pori.tools.standard import register_all_tools


@pytest.fixture
def snapshot():
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry.snapshot()


def test_filesystem_tools_declare_write_side_effect(snapshot):
    """Filesystem-mutating tools advertise their side effect declaratively."""
    assert SideEffect.FILESYSTEM_WRITE in snapshot.get_tool("write_file").side_effects
    assert (
        SideEffect.FILESYSTEM_WRITE
        in snapshot.get_tool("create_directory").side_effects
    )


def test_read_only_tools_declare_no_side_effects(snapshot):
    assert snapshot.get_tool("read_file").side_effects == ()
    assert snapshot.get_tool("answer").side_effects == ()


def test_side_effects_survive_snapshot_round_trip(snapshot):
    """Reconstructing a registry from a snapshot preserves side effects."""
    rebuilt = snapshot.to_registry()
    assert SideEffect.FILESYSTEM_WRITE in rebuilt.get_tool("write_file").side_effects


def test_provider_reconciler_survives_snapshot_round_trip():
    class EmptyParams(BaseModel):
        pass

    def execute(params, context):
        return {"sent": True}

    def reconcile(params, context):
        return ToolReconciliation(status=ReconciliationStatus.SUCCEEDED)

    registry = ToolRegistry()
    registry.register_tool(
        "send",
        EmptyParams,
        execute,
        "Send",
        reconcile_fn=reconcile,
    )

    rebuilt = registry.snapshot(protect_kernel=False).to_registry()
    assert rebuilt.get_tool("send").reconcile_fn is reconcile


@pytest.mark.parametrize(
    "task,expected",
    [
        ("create an HTML lesson file", True),
        ("write the report to a file", True),
        ("teach me division step by step", False),
        ("what is 2 + 2", False),
    ],
)
def test_task_requests_artifact(task, expected):
    assert task_requests_artifact(task) is expected


def test_policy_allows_writes_by_default():
    """The default trusts the model; receipts (not keywords) enforce honesty."""
    policy = ToolAuthorizationPolicy()
    decision = policy.authorize(
        tool_name="write_file",
        side_effects=(SideEffect.FILESYSTEM_WRITE,),
        task="create me a python script",
    )
    assert decision == AuthorizationDecision(allowed=True)


def test_strict_policy_blocks_write_without_artifact_intent():
    policy = ToolAuthorizationPolicy(require_artifact_intent=True)
    decision = policy.authorize(
        tool_name="write_file",
        side_effects=(SideEffect.FILESYSTEM_WRITE,),
        task="teach me division step by step",
    )
    assert decision.allowed is False
    assert "did not explicitly ask for a file artifact" in decision.reason


def test_strict_policy_allows_write_with_artifact_intent():
    policy = ToolAuthorizationPolicy(require_artifact_intent=True)
    decision = policy.authorize(
        tool_name="write_file",
        side_effects=(SideEffect.FILESYSTEM_WRITE,),
        task="create an HTML lesson file for division",
    )
    assert decision == AuthorizationDecision(allowed=True)


def test_strict_policy_allows_read_only_tools_regardless_of_task():
    policy = ToolAuthorizationPolicy(require_artifact_intent=True)
    decision = policy.authorize(
        tool_name="read_file",
        side_effects=(),
        task="teach me division step by step",
    )
    assert decision.allowed is True
