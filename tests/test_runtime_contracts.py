import pytest
from pydantic import BaseModel

from pori import Agent, AgentMemory, AgentSettings
from pori.runtime import ReceiptStatus, RunContext, ToolExecutionReceipt
from pori.team import Team
from pori.tools.registry import ToolRegistry


class EchoParams(BaseModel):
    text: str


def test_run_context_is_immutable_and_explicitly_scoped():
    context = RunContext(
        organization_id="org-1",
        user_id="user-1",
        agent_id="agent-1",
        session_id="session-1",
        run_id="run-1",
        permissions=("tools:read",),
    )

    assert context.organization_id == "org-1"
    try:
        context.user_id = "other"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("RunContext must be immutable")


def test_tool_surface_fingerprint_is_deterministic_and_sensitive_to_schema():
    first = ToolRegistry()
    second = ToolRegistry()

    def echo(params, context):
        return params.text

    for registry in (first, second):
        registry.register_tool("echo", EchoParams, echo, "Echo text")

    assert first.surface_fingerprint() == second.surface_fingerprint()

    second.register_tool("other", EchoParams, echo, "Another tool")
    assert first.surface_fingerprint() != second.surface_fingerprint()


def test_execution_receipt_serializes_auditable_evidence():
    receipt = ToolExecutionReceipt(
        run_id="run-1",
        tool_name="web_search",
        status=ReceiptStatus.FAILED,
        parameters_fingerprint="abc",
        error="backend unavailable",
    )

    data = receipt.model_dump(mode="json")
    assert data["status"] == "failed"
    assert data["error"] == "backend unavailable"


async def test_agent_records_receipt_and_passes_run_context_to_tools(test_agent):
    result = await test_agent.execute_actions(
        [{"test_tool": {"param1": "evidence", "param2": 7}}]
    )

    assert result[0].success is True
    assert len(test_agent.execution_receipts) == 1
    receipt = test_agent.execution_receipts[0]
    assert receipt.tool_name == "test_tool"
    assert receipt.status == ReceiptStatus.SUCCEEDED
    assert receipt.run_id == test_agent.run_context.run_id


def test_team_child_context_preserves_tenant_and_reduces_identity(mock_llm):
    parent = RunContext(
        organization_id="org-1",
        user_id="user-1",
        agent_id="coordinator",
        session_id="session-1",
        run_id="parent-run",
        permissions=("tools:read",),
        credential_scope="organization:org-1",
        isolation_profile="worker",
    )
    team = Team(task="test", coordinator_llm=mock_llm, members=[], run_context=parent)

    child = team._child_run_context("researcher")

    assert child.organization_id == parent.organization_id
    assert child.user_id == parent.user_id
    assert child.session_id == parent.session_id
    assert child.permissions == parent.permissions
    assert child.agent_id == "researcher"
    assert child.run_id != parent.run_id
    assert dict(child.metadata)["parent_run_id"] == parent.run_id


def test_agent_rejects_memory_from_a_different_scope(mock_llm, tool_registry):
    context = RunContext(
        organization_id="org-1",
        user_id="alice",
        agent_id="agent-1",
        session_id="session-1",
        run_id="run-1",
    )
    wrong_memory = AgentMemory(
        organization_id="org-2",
        user_id="bob",
        agent_id="agent-1",
        session_id="session-1",
    )

    with pytest.raises(ValueError, match="exactly match"):
        Agent(
            task="test",
            llm=mock_llm,
            tools_registry=tool_registry,
            memory=wrong_memory,
            run_context=context,
        )


async def test_orchestrator_builds_memory_from_explicit_run_context(orchestrator):
    context = RunContext(
        organization_id="org-1",
        user_id="alice",
        agent_id="agent-1",
        session_id="session-1",
        run_id="run-1",
    )

    result = await orchestrator.execute_task(
        "test",
        agent_settings=AgentSettings(max_steps=1),
        run_context=context,
    )

    agent = result["agent"]
    assert agent.memory.organization_id == "org-1"
    assert agent.memory.user_id == "alice"
    assert agent.memory.agent_id == "agent-1"
    assert agent.memory.session_id == "session-1"
