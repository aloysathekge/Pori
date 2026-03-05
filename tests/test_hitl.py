"""
Tests for the Human-in-the-Loop (HITL) module.

Covers: AutoApproveHandler, approval gates in Agent.execute_actions,
rejection feedback, edit (modify params/tool), safe tool bypass, and
auto_approve_duplicates.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from pori.agent import Agent, AgentOutput, AgentSettings
from pori.evaluation import ActionResult
from pori.memory import AgentMemory
from pori.tools.registry import ToolRegistry
from pori.hitl import (
    HITLConfig,
    HITLHandler,
    AutoApproveHandler,
    InterruptConfig,
    ApprovalRequest,
    ApprovalResponse,
    ActionRequest,
    ReviewConfig,
    Decision,
    EditedAction,
    resolve_interrupt_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLMForHITL:
    """Minimal mock LLM that returns a single tool call."""

    def __init__(self, tool_name: str = "test_tool", params: dict = None):
        self._tool_name = tool_name
        self._params = params or {"param1": "val1"}

    def with_structured_output(self, output_model, include_raw=True):
        self._output_model = output_model
        return self

    async def ainvoke(self, messages):
        from pori.agent import PlanOutput, ReflectOutput

        if hasattr(self._output_model, "plan_steps"):
            return MagicMock(
                get=lambda k, d=None: PlanOutput(
                    plan_steps=["step1"], rationale="test"
                )
                if k == "parsed"
                else d
            )
        if hasattr(self._output_model, "critique"):
            return MagicMock(
                get=lambda k, d=None: ReflectOutput(
                    critique="ok", update_plan=None
                )
                if k == "parsed"
                else d
            )
        output = AgentOutput(
            current_state={"next_goal": "test"},
            action=[{self._tool_name: self._params}],
        )
        return MagicMock(get=lambda k, d=None: output if k == "parsed" else d)


class RejectHandler(HITLHandler):
    def __init__(self, message="Not allowed"):
        self.requests = []
        self.message = message

    async def request_approval(self, request):
        self.requests.append(request)
        return ApprovalResponse(
            decisions=[Decision(type="reject", message=self.message)]
        )


class EditHandler(HITLHandler):
    def __init__(self, new_name=None, new_args=None):
        self.requests = []
        self.new_name = new_name
        self.new_args = new_args or {}

    async def request_approval(self, request):
        self.requests.append(request)
        orig = request.action_requests[0]
        return ApprovalResponse(
            decisions=[
                Decision(
                    type="edit",
                    edited_action=EditedAction(
                        name=self.new_name or orig.name, args=self.new_args
                    ),
                )
            ]
        )


class ApproveHandler(HITLHandler):
    def __init__(self):
        self.requests = []

    async def request_approval(self, request):
        self.requests.append(request)
        return ApprovalResponse(decisions=[Decision(type="approve")])


def _make_registry():
    registry = ToolRegistry()

    class TestParams(BaseModel):
        param1: str = Field(default="val1")

    def test_fn(params, context):
        return f"result:{params.param1}"

    registry.register_tool(
        name="test_tool", param_model=TestParams,
        function=test_fn, description="A test tool",
    )

    class SafeParams(BaseModel):
        query: str = Field(default="hello")

    def safe_fn(params, context):
        return f"safe:{params.query}"

    registry.register_tool(
        name="safe_tool", param_model=SafeParams,
        function=safe_fn, description="A safe tool",
    )

    class AnswerParams(BaseModel):
        final_answer: str = Field(default="")
        reasoning: str = Field(default="")

    def answer_fn(params, context):
        if context and "memory" in context:
            context["memory"].update_state(
                "final_answer",
                {"final_answer": params.final_answer, "reasoning": params.reasoning},
            )
        return {"final_answer": params.final_answer}

    registry.register_tool(
        name="answer", param_model=AnswerParams,
        function=answer_fn, description="answer",
    )

    class DoneParams(BaseModel):
        success: bool = True
        message: str = "done"

    def done_fn(params, context):
        return {"success": params.success}

    registry.register_tool(
        name="done", param_model=DoneParams,
        function=done_fn, description="done",
    )
    return registry


def _make_hitl_config(tool_names=None, **kwargs):
    interrupt_on = {}
    if tool_names:
        for name in tool_names:
            interrupt_on[name] = True
    return HITLConfig(enabled=True, interrupt_on=interrupt_on, **kwargs)


def _make_agent(llm=None, hitl_handler=None, hitl_config=None,
                tool_name="test_tool", params=None):
    return Agent(
        task="Test HITL task",
        llm=llm or MockLLMForHITL(tool_name=tool_name, params=params),
        tools_registry=_make_registry(),
        settings=AgentSettings(max_steps=3, max_failures=2, retry_delay=0),
        memory=AgentMemory(),
        hitl_handler=hitl_handler,
        hitl_config=hitl_config,
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# resolve_interrupt_config
# ---------------------------------------------------------------------------


class TestResolveInterruptConfig:
    def test_true_returns_default_config(self):
        cfg = HITLConfig(enabled=True, interrupt_on={"bash": True})
        result = resolve_interrupt_config("bash", cfg)
        assert result is not None
        assert set(result.allowed_decisions) == {"approve", "edit", "reject"}

    def test_false_returns_none(self):
        cfg = HITLConfig(enabled=True, interrupt_on={"bash": False})
        assert resolve_interrupt_config("bash", cfg) is None

    def test_not_listed_returns_none(self):
        cfg = HITLConfig(enabled=True, interrupt_on={"bash": True})
        assert resolve_interrupt_config("read_file", cfg) is None

    def test_interrupt_config_object(self):
        cfg = HITLConfig(
            enabled=True,
            interrupt_on={"sql": InterruptConfig(allowed_decisions=["approve", "reject"])},
        )
        result = resolve_interrupt_config("sql", cfg)
        assert result is not None
        assert "edit" not in result.allowed_decisions

    def test_dict_from_yaml(self):
        cfg = HITLConfig(
            enabled=True,
            interrupt_on={"bash": {"allowed_decisions": ["approve", "reject"]}},
        )
        result = resolve_interrupt_config("bash", cfg)
        assert result is not None
        assert result.allowed_decisions == ["approve", "reject"]


# ---------------------------------------------------------------------------
# AutoApproveHandler
# ---------------------------------------------------------------------------


def test_auto_approve_handler(event_loop):
    handler = AutoApproveHandler()
    request = ApprovalRequest(
        action_requests=[
            ActionRequest(name="bash", arguments={"cmd": "ls"}, description="test"),
            ActionRequest(name="write", arguments={"f": "x"}, description="test"),
        ],
        review_configs=[
            ReviewConfig(action_name="bash", allowed_decisions=["approve", "reject"]),
            ReviewConfig(action_name="write", allowed_decisions=["approve", "reject"]),
        ],
        task_id="t1",
        step_number=1,
    )
    response = event_loop.run_until_complete(handler.request_approval(request))
    assert len(response.decisions) == 2
    assert all(d.type == "approve" for d in response.decisions)


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------


def test_rejection_blocks_tool(event_loop):
    """Rejected tool should not execute; error captured in memory."""
    handler = RejectHandler(message="Dangerous")
    config = _make_hitl_config(tool_names=["test_tool"])
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())

    assert len(handler.requests) == 1
    assert handler.requests[0].action_requests[0].name == "test_tool"
    msgs = [m.content for m in agent.memory.messages]
    assert any("rejected" in m.lower() and "test_tool" in m for m in msgs)


def test_rejection_message_in_memory(event_loop):
    """Rejection feedback text should appear in agent memory."""
    handler = RejectHandler(message="Use a safer approach")
    config = _make_hitl_config(tool_names=["test_tool"])
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())

    msgs = [m.content for m in agent.memory.messages]
    assert any("Use a safer approach" in m for m in msgs)


def test_edit_changes_params(event_loop):
    """Edit decision should run the tool with modified params."""
    handler = EditHandler(new_args={"param1": "edited_value"})
    config = _make_hitl_config(tool_names=["test_tool"])
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())

    calls = [tc for tc in agent.memory.tool_call_history if tc.tool_name == "test_tool"]
    assert len(calls) >= 1
    assert calls[0].parameters.get("param1") == "edited_value"


def test_safe_tool_skips_gate(event_loop):
    """Tool not in interrupt_on should not trigger approval gate."""
    handler = ApproveHandler()
    config = _make_hitl_config(tool_names=["bash"])
    agent = _make_agent(
        hitl_handler=handler, hitl_config=config,
        tool_name="safe_tool", params={"query": "hello"},
    )

    event_loop.run_until_complete(agent.step())

    assert len(handler.requests) == 0


def test_approve_allows_execution(event_loop):
    """Approved tool should execute normally."""
    handler = ApproveHandler()
    config = _make_hitl_config(tool_names=["test_tool"])
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())

    assert len(handler.requests) == 1
    calls = [tc for tc in agent.memory.tool_call_history if tc.tool_name == "test_tool"]
    assert len(calls) >= 1
    assert calls[0].success is True


def test_auto_approve_duplicates(event_loop):
    """Once approved, identical tool+params should auto-approve on next step."""
    handler = ApproveHandler()
    config = _make_hitl_config(tool_names=["test_tool"], auto_approve_duplicates=True)
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())
    assert len(handler.requests) == 1

    event_loop.run_until_complete(agent.step())
    assert len(handler.requests) == 1  # Not prompted again


def test_no_hitl_when_disabled(event_loop):
    """HITL disabled in config → no approval requested."""
    handler = ApproveHandler()
    config = HITLConfig(enabled=False, interrupt_on={"test_tool": True})
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())
    assert len(handler.requests) == 0


def test_no_hitl_without_handler(event_loop):
    """No handler provided → tools execute normally."""
    config = _make_hitl_config(tool_names=["test_tool"])
    agent = _make_agent(hitl_handler=None, hitl_config=config)

    event_loop.run_until_complete(agent.step())
    calls = [tc for tc in agent.memory.tool_call_history if tc.tool_name == "test_tool"]
    assert len(calls) >= 1


def test_per_tool_allowed_decisions(event_loop):
    """ReviewConfig should reflect per-tool allowed_decisions."""
    handler = ApproveHandler()
    config = HITLConfig(
        enabled=True,
        interrupt_on={"test_tool": InterruptConfig(allowed_decisions=["approve", "reject"])},
    )
    agent = _make_agent(hitl_handler=handler, hitl_config=config)

    event_loop.run_until_complete(agent.step())

    assert len(handler.requests) == 1
    review = handler.requests[0].review_configs[0]
    assert "edit" not in review.allowed_decisions
    assert "approve" in review.allowed_decisions
