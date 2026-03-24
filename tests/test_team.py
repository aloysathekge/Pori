"""Tests for the multi-agent Team system."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from pori.agent import AgentOutput, AgentSettings
from pori.memory import AgentMemory
from pori.team import (
    BroadcastSummary,
    DelegationPlan,
    MemberConfig,
    MemberRunResult,
    RoutingDecision,
    Team,
    TeamConfig,
    TeamMode,
)
from pori.team.models import DelegationStep
from pori.tools.registry import ToolRegistry

# ========== Helper: Mock LLM that returns structured output objects ==========


class StructuredMockLLM:
    """Mock LLM for coordinator calls that returns Pydantic model instances.

    Uses a shared mutable list ``[index]`` so that clones created by
    ``with_structured_output`` advance the same counter.
    """

    model = "mock-coordinator"

    def __init__(self, responses: Optional[List] = None, _shared_index=None):
        self.responses = responses or []
        # Shared mutable index so clones stay in sync
        self._shared_index = _shared_index if _shared_index is not None else [0]
        self.ainvoke_calls = []

    def with_structured_output(self, output_model, include_raw=False):
        clone = StructuredMockLLM(self.responses, _shared_index=self._shared_index)
        clone.ainvoke_calls = self.ainvoke_calls
        return clone

    async def ainvoke(self, messages, output_format=None):
        self.ainvoke_calls.append(messages)
        if not self.responses:
            return ""
        idx = self._shared_index[0]
        response = self.responses[idx % len(self.responses)]
        self._shared_index[0] = idx + 1
        return response


class AgentMockLLM:
    """Mock LLM for member agents that produces tool-call responses (answer + done)."""

    model = "mock-agent"

    def __init__(
        self, final_answer: str = "mock answer", reasoning: str = "mock reasoning"
    ):
        self._final_answer = final_answer
        self._reasoning = reasoning
        self._call_count = 0

    def with_structured_output(self, output_model, include_raw=True):
        self._output_model = output_model
        return self

    async def ainvoke(self, messages, output_format=None):
        self._call_count += 1
        if self._call_count == 1:
            return _make_agent_output(
                goal="Provide final answer",
                action=[
                    {
                        "answer": {
                            "final_answer": self._final_answer,
                            "reasoning": self._reasoning,
                        }
                    }
                ],
            )
        return _make_agent_output(
            goal="Mark done",
            action=[{"done": {"success": True, "message": "done"}}],
        )


def _make_agent_output(goal: str, action: list) -> AgentOutput:
    return AgentOutput(
        current_state={
            "evaluation_previous_goal": goal,
            "memory": "",
            "next_goal": goal,
        },
        action=action,
    )


# ========== Fixtures ==========


def _make_registry():
    """Create a tool registry with answer and done tools."""
    registry = ToolRegistry()

    class AnswerParams(BaseModel):
        final_answer: str = Field(..., description="The final answer")
        reasoning: str = Field("", description="Reasoning")

    def answer_tool(params: AnswerParams, context: Dict[str, Any]):
        if context and "memory" in context:
            context["memory"].update_state(
                "final_answer",
                {"final_answer": params.final_answer, "reasoning": params.reasoning},
            )
        return {"final_answer": params.final_answer, "reasoning": params.reasoning}

    registry.register_tool(
        name="answer",
        param_model=AnswerParams,
        function=answer_tool,
        description="Provide a final answer",
    )

    class DoneParams(BaseModel):
        success: bool = Field(True)
        message: str = Field("Task completed")

    def done_tool(params: DoneParams, context: Dict[str, Any]):
        return {"success": params.success, "message": params.message}

    registry.register_tool(
        name="done",
        param_model=DoneParams,
        function=done_tool,
        description="Mark as done",
    )

    return registry


@pytest.fixture
def registry():
    return _make_registry()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ========== Model Tests ==========


class TestModels:
    def test_team_mode_values(self):
        assert TeamMode.ROUTER == "router"
        assert TeamMode.BROADCAST == "broadcast"
        assert TeamMode.DELEGATE == "delegate"

    def test_member_config_minimal(self):
        mc = MemberConfig(name="researcher", description="does research")
        assert mc.name == "researcher"
        assert mc.tools is None
        assert mc.team_config is None

    def test_member_config_with_tools(self):
        mc = MemberConfig(name="a", description="b", tools=["search", "answer"])
        assert "search" in mc.tools
        assert "answer" in mc.tools

    def test_routing_decision(self):
        rd = RoutingDecision(
            chosen_member="researcher",
            reasoning="best fit",
            rewritten_task="Search for X",
        )
        assert rd.chosen_member == "researcher"
        assert rd.rewritten_task == "Search for X"

    def test_delegation_plan(self):
        plan = DelegationPlan(
            rationale="split the work",
            steps=[
                DelegationStep(step_number=1, member_name="a", subtask="do X"),
                DelegationStep(
                    step_number=2, member_name="b", subtask="do Y", depends_on=[1]
                ),
            ],
        )
        assert len(plan.steps) == 2
        assert plan.steps[1].depends_on == [1]

    def test_broadcast_summary(self):
        bs = BroadcastSummary(combined_answer="combined", reasoning="merged results")
        assert bs.combined_answer == "combined"

    def test_member_run_result_defaults(self):
        r = MemberRunResult(member_name="a", task="t")
        assert r.completed is False
        assert r.steps_taken == 0
        assert r.error is None

    def test_team_config(self):
        tc = TeamConfig(
            name="my-team",
            mode=TeamMode.BROADCAST,
            members=[MemberConfig(name="a", description="d")],
        )
        assert tc.mode == TeamMode.BROADCAST
        assert len(tc.members) == 1

    def test_team_config_nesting(self):
        inner = TeamConfig(
            name="inner",
            members=[MemberConfig(name="x", description="d")],
        )
        mc = MemberConfig(name="nested", description="nested team", team_config=inner)
        assert mc.team_config.name == "inner"


# ========== Router Mode Tests ==========


class TestRouterMode:
    def test_router_routes_to_correct_member(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[
                RoutingDecision(
                    chosen_member="analyst",
                    reasoning="analysis task",
                    rewritten_task="Analyse the data",
                )
            ]
        )

        agent_llm = AgentMockLLM(final_answer="analysis result")

        members = [
            MemberConfig(name="researcher", description="searches"),
            MemberConfig(name="analyst", description="analyses"),
        ]

        # Patch Agent to use our agent mock LLM
        team = Team(
            task="Analyse this data",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.ROUTER,
            tools_registry=registry,
        )

        # Replace _run_agent_member to avoid full Agent lifecycle
        async def mock_run_agent(config, task):
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=2,
                final_answer="analysis result",
                reasoning="done",
            )

        team._run_agent_member = mock_run_agent

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is True
        assert result["final_state"]["chosen_member"] == "analyst"
        assert result["final_state"]["final_answer"] == "analysis result"

    def test_router_unknown_member_returns_error(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[
                RoutingDecision(
                    chosen_member="nonexistent",
                    reasoning="oops",
                )
            ]
        )

        team = Team(
            task="test",
            coordinator_llm=coordinator,
            members=[MemberConfig(name="a", description="d")],
            mode=TeamMode.ROUTER,
            tools_registry=registry,
        )

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is False
        assert "nonexistent" in result["final_state"]["error"]


# ========== Broadcast Mode Tests ==========


class TestBroadcastMode:
    def test_broadcast_runs_all_members(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[
                BroadcastSummary(
                    combined_answer="combined result",
                    reasoning="merged all",
                )
            ]
        )

        members = [
            MemberConfig(name="a", description="first"),
            MemberConfig(name="b", description="second"),
        ]

        team = Team(
            task="Broadcast task",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.BROADCAST,
            tools_registry=registry,
        )

        ran_members = []

        async def mock_run_agent(config, task):
            ran_members.append(config.name)
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=1,
                final_answer=f"result from {config.name}",
            )

        team._run_agent_member = mock_run_agent

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is True
        assert set(ran_members) == {"a", "b"}
        assert result["final_state"]["final_answer"] == "combined result"
        assert result["steps_taken"] == 2  # 1 per member

    def test_broadcast_handles_member_exception(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[
                BroadcastSummary(
                    combined_answer="partial",
                    reasoning="one failed",
                )
            ]
        )

        members = [
            MemberConfig(name="ok", description="works"),
            MemberConfig(name="bad", description="fails"),
        ]

        team = Team(
            task="test",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.BROADCAST,
            tools_registry=registry,
        )

        async def mock_run_member(config, task):
            if config.name == "bad":
                raise RuntimeError("boom")
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=1,
                final_answer="ok",
            )

        team._run_agent_member = mock_run_member

        result = event_loop.run_until_complete(team.run())
        # Should still complete (coordinator synthesises partial results)
        assert result["completed"] is True
        member_results = result["final_state"]["member_results"]
        errors = [r for r in member_results if r.get("error")]
        assert len(errors) == 1
        assert "boom" in errors[0]["error"]


# ========== Delegate Mode Tests ==========


class TestDelegateMode:
    def test_delegate_executes_steps_in_order(self, event_loop, registry):
        plan = DelegationPlan(
            rationale="sequential work",
            steps=[
                DelegationStep(step_number=1, member_name="a", subtask="step 1"),
                DelegationStep(
                    step_number=2, member_name="b", subtask="step 2", depends_on=[1]
                ),
            ],
        )

        coordinator = StructuredMockLLM(
            responses=[
                plan,
                "Final synthesised answer",  # for synthesize call
            ]
        )

        members = [
            MemberConfig(name="a", description="first"),
            MemberConfig(name="b", description="second"),
        ]

        team = Team(
            task="Complex task",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.DELEGATE,
            tools_registry=registry,
        )

        execution_order = []

        async def mock_run_agent(config, task):
            execution_order.append(config.name)
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=1,
                final_answer=f"result from {config.name}",
            )

        team._run_agent_member = mock_run_agent

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is True
        # Step 2 depends on step 1, so 'a' must run before 'b'
        assert execution_order == ["a", "b"]
        assert result["final_state"]["final_answer"] == "Final synthesised answer"

    def test_delegate_parallel_independent_steps(self, event_loop, registry):
        plan = DelegationPlan(
            rationale="parallel work",
            steps=[
                DelegationStep(step_number=1, member_name="a", subtask="independent 1"),
                DelegationStep(step_number=2, member_name="b", subtask="independent 2"),
                DelegationStep(
                    step_number=3,
                    member_name="a",
                    subtask="combine",
                    depends_on=[1, 2],
                ),
            ],
        )

        coordinator = StructuredMockLLM(responses=[plan, "Final answer"])

        members = [
            MemberConfig(name="a", description="d"),
            MemberConfig(name="b", description="d"),
        ]

        team = Team(
            task="test",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.DELEGATE,
            tools_registry=registry,
        )

        async def mock_run_agent(config, task):
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=1,
                final_answer=f"result-{config.name}",
            )

        team._run_agent_member = mock_run_agent

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is True
        assert result["steps_taken"] == 3

    def test_delegate_max_steps_exceeded(self, event_loop, registry):
        steps = [
            DelegationStep(step_number=i, member_name="a", subtask=f"step {i}")
            for i in range(20)
        ]
        plan = DelegationPlan(rationale="too many", steps=steps)

        coordinator = StructuredMockLLM(responses=[plan])

        team = Team(
            task="test",
            coordinator_llm=coordinator,
            members=[MemberConfig(name="a", description="d")],
            mode=TeamMode.DELEGATE,
            tools_registry=registry,
            max_delegation_steps=10,
        )

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is False
        assert "exceeding" in result["final_state"]["error"]

    def test_delegate_injects_prior_results(self, event_loop, registry):
        plan = DelegationPlan(
            rationale="chained",
            steps=[
                DelegationStep(step_number=1, member_name="a", subtask="gather data"),
                DelegationStep(
                    step_number=2,
                    member_name="b",
                    subtask="analyse data",
                    depends_on=[1],
                ),
            ],
        )

        coordinator = StructuredMockLLM(responses=[plan, "done"])

        members = [
            MemberConfig(name="a", description="d"),
            MemberConfig(name="b", description="d"),
        ]

        team = Team(
            task="test",
            coordinator_llm=coordinator,
            members=members,
            mode=TeamMode.DELEGATE,
            tools_registry=registry,
        )

        received_tasks = {}

        async def mock_run_agent(config, task):
            received_tasks[config.name] = task
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=1,
                final_answer=f"data from {config.name}",
            )

        team._run_agent_member = mock_run_agent

        event_loop.run_until_complete(team.run())
        # Member 'b' should receive enriched task with step 1 result
        assert "data from a" in received_tasks["b"]


# ========== Tool Registry Filtering Tests ==========


class TestToolFiltering:
    def test_filtered_registry_always_includes_answer_and_done(self, registry):
        # Add an extra tool
        class P(BaseModel):
            q: str = ""

        registry.register_tool("search", P, lambda p, c: "ok", "search tool")

        config = MemberConfig(name="a", description="d", tools=["search"])
        team = Team(
            task="t",
            coordinator_llm=StructuredMockLLM(),
            members=[config],
            tools_registry=registry,
        )

        filtered = team._build_registry(config)
        assert "search" in filtered.tools
        assert "answer" in filtered.tools
        assert "done" in filtered.tools

    def test_no_filter_returns_full_registry(self, registry):
        config = MemberConfig(name="a", description="d", tools=None)
        team = Team(
            task="t",
            coordinator_llm=StructuredMockLLM(),
            members=[config],
            tools_registry=registry,
        )

        result = team._build_registry(config)
        assert result is registry


# ========== Nesting Tests ==========


class TestNesting:
    def test_nested_team_member(self, event_loop, registry):
        inner_config = TeamConfig(
            name="inner-team",
            mode=TeamMode.ROUTER,
            members=[MemberConfig(name="inner-agent", description="inner")],
        )

        outer_members = [
            MemberConfig(
                name="nested",
                description="a nested team",
                team_config=inner_config,
            ),
        ]

        # Coordinator routes to nested member
        coordinator = StructuredMockLLM(
            responses=[
                RoutingDecision(
                    chosen_member="nested", reasoning="delegate to sub-team"
                ),
            ]
        )

        team = Team(
            task="Complex nested task",
            coordinator_llm=coordinator,
            members=outer_members,
            mode=TeamMode.ROUTER,
            tools_registry=registry,
        )

        # Mock the nested team run
        async def mock_nested_team(config, task):
            return MemberRunResult(
                member_name=config.name,
                task=task,
                completed=True,
                steps_taken=3,
                final_answer="nested result",
            )

        team._run_nested_team = mock_nested_team

        result = event_loop.run_until_complete(team.run())
        assert result["completed"] is True
        assert result["final_state"]["final_answer"] == "nested result"


# ========== Config Integration Tests ==========


class TestConfigIntegration:
    def test_config_accepts_team_field(self):
        from pori.config import Config, LLMConfig

        tc = TeamConfig(
            name="test-team",
            mode=TeamMode.BROADCAST,
            members=[MemberConfig(name="m", description="d")],
        )

        config = Config(
            llm=LLMConfig(provider="anthropic", model="test"),
            team=tc,
        )
        assert config.team is not None
        assert config.team.name == "test-team"
        assert config.team.mode == TeamMode.BROADCAST

    def test_config_team_none_by_default(self):
        from pori.config import Config, LLMConfig

        config = Config(llm=LLMConfig(provider="anthropic", model="test"))
        assert config.team is None


# ========== Return Shape Tests ==========


class TestReturnShape:
    """Verify all modes return the same dict shape as Agent.run()."""

    REQUIRED_KEYS = {"task", "completed", "steps_taken", "final_state", "metrics"}

    def test_router_return_shape(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[RoutingDecision(chosen_member="a", reasoning="ok")]
        )
        team = Team(
            task="t",
            coordinator_llm=coordinator,
            members=[MemberConfig(name="a", description="d")],
            mode=TeamMode.ROUTER,
            tools_registry=registry,
        )

        async def mock_run(config, task):
            return MemberRunResult(
                member_name="a",
                task=task,
                completed=True,
                steps_taken=1,
                final_answer="ok",
            )

        team._run_agent_member = mock_run
        result = event_loop.run_until_complete(team.run())
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_broadcast_return_shape(self, event_loop, registry):
        coordinator = StructuredMockLLM(
            responses=[BroadcastSummary(combined_answer="c", reasoning="r")]
        )
        team = Team(
            task="t",
            coordinator_llm=coordinator,
            members=[MemberConfig(name="a", description="d")],
            mode=TeamMode.BROADCAST,
            tools_registry=registry,
        )

        async def mock_run(config, task):
            return MemberRunResult(
                member_name="a",
                task=task,
                completed=True,
                steps_taken=1,
                final_answer="ok",
            )

        team._run_agent_member = mock_run
        result = event_loop.run_until_complete(team.run())
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_delegate_return_shape(self, event_loop, registry):
        plan = DelegationPlan(
            rationale="r",
            steps=[DelegationStep(step_number=1, member_name="a", subtask="s")],
        )
        coordinator = StructuredMockLLM(responses=[plan, "answer"])
        team = Team(
            task="t",
            coordinator_llm=coordinator,
            members=[MemberConfig(name="a", description="d")],
            mode=TeamMode.DELEGATE,
            tools_registry=registry,
        )

        async def mock_run(config, task):
            return MemberRunResult(
                member_name="a",
                task=task,
                completed=True,
                steps_taken=1,
                final_answer="ok",
            )

        team._run_agent_member = mock_run
        result = event_loop.run_until_complete(team.run())
        assert self.REQUIRED_KEYS.issubset(result.keys())
