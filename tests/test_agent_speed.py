"""Tests for speed/cost controls in the agent loop."""

import asyncio

import pytest

from pori.agent import Agent, AgentOutput, AgentSettings, PlanOutput, ReflectOutput
from pori.memory import AgentMemory


class MockLLMResponse:
    def __init__(self, parsed):
        self.parsed = parsed

    def get(self, key, default=None):
        if key == "parsed":
            return self.parsed
        return default


def make_action_response(action):
    return MockLLMResponse(
        AgentOutput(
            current_state={
                "evaluation_previous_goal": "ok",
                "memory": "none",
                "next_goal": "answer",
            },
            action=action,
        )
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def basic_registry():
    from pori.tools.registry import tool_registry

    return tool_registry()


def test_auto_mode_skips_planning_for_simple_tasks(basic_registry, event_loop):
    """A simple answer should use one action LLM call and no plan/reflect calls."""

    class CountingLLM:
        def __init__(self):
            self.output_model = None
            self.action_calls = 0
            self.plan_calls = 0
            self.reflect_calls = 0

        def with_structured_output(self, output_model, include_raw=True):
            self.output_model = output_model
            return self

        async def ainvoke(self, messages):
            if self.output_model == PlanOutput:
                self.plan_calls += 1
                return PlanOutput(plan_steps=["answer"], rationale="test")
            if self.output_model == ReflectOutput:
                self.reflect_calls += 1
                return ReflectOutput(critique="ok", update_plan=None)

            self.action_calls += 1
            return make_action_response(
                [{"answer": {"final_answer": "Done", "reasoning": "Complete"}}]
            )

    llm = CountingLLM()
    agent = Agent(
        task="Simple task",
        llm=llm,
        tools_registry=basic_registry,
        settings=AgentSettings(max_steps=5),
        memory=AgentMemory(),
    )

    result = event_loop.run_until_complete(agent.run())

    assert result["completed"] is True
    assert llm.action_calls == 1
    assert llm.plan_calls == 0
    assert llm.reflect_calls == 0


def test_auto_mode_plans_for_complex_tasks(basic_registry, event_loop):
    """Complex tasks should use a planning call in auto mode."""

    class CountingLLM:
        def __init__(self):
            self.output_model = None
            self.action_calls = 0
            self.plan_calls = 0
            self.reflect_calls = 0

        def with_structured_output(self, output_model, include_raw=True):
            self.output_model = output_model
            return self

        async def ainvoke(self, messages):
            if self.output_model == PlanOutput:
                self.plan_calls += 1
                return PlanOutput(
                    plan_steps=["inspect", "patch", "test"], rationale="test"
                )
            if self.output_model == ReflectOutput:
                self.reflect_calls += 1
                return ReflectOutput(critique="ok", update_plan=None)

            self.action_calls += 1
            return make_action_response(
                [{"answer": {"final_answer": "Done", "reasoning": "Complete"}}]
            )

    llm = CountingLLM()
    agent = Agent(
        task="Investigate and refactor the database layer across multiple modules",
        llm=llm,
        tools_registry=basic_registry,
        settings=AgentSettings(max_steps=5),
        memory=AgentMemory(),
    )

    result = event_loop.run_until_complete(agent.run())

    assert result["completed"] is True
    assert llm.action_calls == 1
    assert llm.plan_calls == 1
    assert llm.reflect_calls == 0


def test_auto_mode_reflects_after_failed_progress(basic_registry, event_loop):
    """Auto reflection should run for recovery after failed tool results."""

    class CountingLLM:
        def __init__(self):
            self.output_model = None
            self.action_calls = 0
            self.plan_calls = 0
            self.reflect_calls = 0

        def with_structured_output(self, output_model, include_raw=True):
            self.output_model = output_model
            return self

        async def ainvoke(self, messages):
            if self.output_model == PlanOutput:
                self.plan_calls += 1
                return PlanOutput(plan_steps=["answer"], rationale="test")
            if self.output_model == ReflectOutput:
                self.reflect_calls += 1
                return ReflectOutput(critique="recover", update_plan=["answer"])

            self.action_calls += 1
            if self.action_calls == 1:
                return make_action_response([{"missing_tool": {"value": "bad"}}])
            return make_action_response(
                [{"answer": {"final_answer": "Recovered", "reasoning": "Complete"}}]
            )

    llm = CountingLLM()
    agent = Agent(
        task="Simple task",
        llm=llm,
        tools_registry=basic_registry,
        settings=AgentSettings(max_steps=5),
        memory=AgentMemory(),
    )

    result = event_loop.run_until_complete(agent.run())

    assert result["completed"] is True
    assert llm.action_calls == 2
    assert llm.plan_calls == 0
    assert llm.reflect_calls == 1


def test_planning_and_reflection_can_be_forced(basic_registry, event_loop):
    """The old plan/reflect loop remains available when explicitly forced."""

    class CountingLLM:
        def __init__(self):
            self.output_model = None
            self.action_calls = 0
            self.plan_calls = 0
            self.reflect_calls = 0

        def with_structured_output(self, output_model, include_raw=True):
            self.output_model = output_model
            return self

        async def ainvoke(self, messages):
            if self.output_model == PlanOutput:
                self.plan_calls += 1
                return PlanOutput(plan_steps=["answer"], rationale="test")
            if self.output_model == ReflectOutput:
                self.reflect_calls += 1
                return ReflectOutput(critique="ok", update_plan=None)

            self.action_calls += 1
            if self.action_calls == 1:
                return make_action_response(
                    [{"think": {"thoughts": "need answer", "next_action": "answer"}}]
                )
            return make_action_response(
                [{"answer": {"final_answer": "Done", "reasoning": "Complete"}}]
            )

    llm = CountingLLM()
    agent = Agent(
        task="Simple task",
        llm=llm,
        tools_registry=basic_registry,
        settings=AgentSettings(
            max_steps=5,
            planning_mode="always",
            reflection_mode="always",
        ),
        memory=AgentMemory(),
    )

    result = event_loop.run_until_complete(agent.run())

    assert result["completed"] is True
    assert llm.action_calls == 2
    assert llm.plan_calls == 1
    # Reflection runs after think, then skips after answer.
    assert llm.reflect_calls == 1
