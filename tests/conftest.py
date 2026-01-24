"""
Pytest configuration and fixtures for Pori tests.

This file contains reusable fixtures for testing the Pori framework.
"""

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pori.llm import SystemMessage, UserMessage, AssistantMessage
from pydantic import BaseModel, Field

from pori.agent import Agent, AgentOutput, AgentSettings
from pori.evaluation import ActionResult, Evaluator
from pori import AgentMemory, TaskState
from pori.orchestrator import Orchestrator
from pori.tools.registry import ToolInfo, ToolRegistry


# ========== Mock LLM Fixtures ==========


class MockLLMResponse:
    """Configurable mock LLM response for testing."""

    def __init__(
        self, content: Union[str, Dict[str, Any]], parsed: Optional[Any] = None
    ):
        self.content = content
        self.parsed = parsed

    def get(self, key, default=None):
        if key == "raw":
            return self
        if key == "parsed" and self.parsed is not None:
            return self.parsed
        return default


class MockLLM:
    """Mock LLM for testing that returns predefined responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.response_index = 0
        self.invoke_calls = []
        self.ainvoke_calls = []

    def with_structured_output(self, output_model, include_raw=True):
        """Returns self to mimic LangChain's structured output method."""
        self.output_model = output_model
        return self

    def invoke(self, messages):
        """Synchronous invoke method."""
        self.invoke_calls.append(messages)
        if not self.responses:
            # Default empty response
            return MockLLMResponse({"current_state": {}, "action": []})

        response = self.responses[self.response_index]
        self.response_index = (self.response_index + 1) % len(self.responses)
        return response

    async def ainvoke(self, messages):
        """Asynchronous invoke method."""
        self.ainvoke_calls.append(messages)
        if not self.responses:
            # Default empty response
            return MockLLMResponse({"current_state": {}, "action": []})

        response = self.responses[self.response_index]
        self.response_index = (self.response_index + 1) % len(self.responses)
        return response


@pytest.fixture
def mock_llm():
    """Returns a configurable mock LLM."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_tool_calls():
    """Returns a mock LLM that generates tool calls."""
    responses = [
        MockLLMResponse(
            content=json.dumps(
                {
                    "current_state": {
                        "evaluation_previous_goal": "Unknown - First step",
                        "memory": "Need to gather information",
                        "next_goal": "Use the test_tool to get data",
                    },
                    "action": [{"test_tool": {"param1": "test_value", "param2": 42}}],
                }
            ),
            parsed=AgentOutput(
                current_state={
                    "evaluation_previous_goal": "Unknown - First step",
                    "memory": "Need to gather information",
                    "next_goal": "Use the test_tool to get data",
                },
                action=[{"test_tool": {"param1": "test_value", "param2": 42}}],
            ),
        ),
        MockLLMResponse(
            content=json.dumps(
                {
                    "current_state": {
                        "evaluation_previous_goal": "Success - Got test data",
                        "memory": "Received test data: Test result",
                        "next_goal": "Provide final answer",
                    },
                    "action": [
                        {
                            "answer": {
                                "final_answer": "The test was successful",
                                "reasoning": "Based on the test data received",
                            }
                        }
                    ],
                }
            ),
            parsed=AgentOutput(
                current_state={
                    "evaluation_previous_goal": "Success - Got test data",
                    "memory": "Received test data: Test result",
                    "next_goal": "Provide final answer",
                },
                action=[
                    {
                        "answer": {
                            "final_answer": "The test was successful",
                            "reasoning": "Based on the test data received",
                        }
                    }
                ],
            ),
        ),
        MockLLMResponse(
            content=json.dumps(
                {
                    "current_state": {
                        "evaluation_previous_goal": "Success - Provided answer",
                        "memory": "Task completed successfully",
                        "next_goal": "Mark task as complete",
                    },
                    "action": [
                        {
                            "done": {
                                "success": True,
                                "message": "Task completed successfully",
                            }
                        }
                    ],
                }
            ),
            parsed=AgentOutput(
                current_state={
                    "evaluation_previous_goal": "Success - Provided answer",
                    "memory": "Task completed successfully",
                    "next_goal": "Mark task as complete",
                },
                action=[
                    {
                        "done": {
                            "success": True,
                            "message": "Task completed successfully",
                        }
                    }
                ],
            ),
        ),
    ]
    return MockLLM(responses=responses)


@pytest.fixture
def mock_plan_llm():
    """Returns a mock LLM that generates plans."""
    from pori.agent import PlanOutput

    responses = [
        MockLLMResponse(
            content=json.dumps(
                {
                    "plan_steps": [
                        "Use test_tool to gather information",
                        "Process the results",
                        "Provide final answer",
                    ],
                    "rationale": "This plan will help accomplish the task efficiently",
                }
            ),
            parsed=PlanOutput(
                plan_steps=[
                    "Use test_tool to gather information",
                    "Process the results",
                    "Provide final answer",
                ],
                rationale="This plan will help accomplish the task efficiently",
            ),
        )
    ]
    return MockLLM(responses=responses)


@pytest.fixture
def mock_reflect_llm():
    """Returns a mock LLM that generates reflections."""
    from pori.agent import ReflectOutput

    responses = [
        MockLLMResponse(
            content=json.dumps(
                {"critique": "The current plan is working well", "update_plan": None}
            ),
            parsed=ReflectOutput(
                critique="The current plan is working well", update_plan=None
            ),
        )
    ]
    return MockLLM(responses=responses)


# ========== Memory Fixtures ==========


@pytest.fixture
def legacy_memory():
    """Returns a simple session memory instance for testing."""
    return AgentMemory()


# ========== Tool Fixtures ==========


class TestToolParams(BaseModel):
    """Parameters for a test tool."""

    param1: str = Field(..., description="Test parameter 1")
    param2: int = Field(42, description="Test parameter 2")


def test_tool_function(params: TestToolParams, context: Dict[str, Any]):
    """Test tool implementation that returns predictable results."""
    return f"Test result: {params.param1}, {params.param2}"


@pytest.fixture
def tool_registry():
    """Returns a tool registry with test tools."""
    registry = ToolRegistry()

    # Register a test tool
    registry.register_tool(
        name="test_tool",
        param_model=TestToolParams,
        function=test_tool_function,
        description="A test tool for testing",
    )

    # Register a mock answer tool
    class AnswerParams(BaseModel):
        final_answer: str = Field(..., description="The final answer")
        reasoning: str = Field("", description="Reasoning for the answer")

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

    # Register a mock done tool
    class DoneParams(BaseModel):
        success: bool = Field(True, description="Whether the task succeeded")
        message: str = Field("Task completed", description="Completion message")

    def done_tool(params: DoneParams, context: Dict[str, Any]):
        return {"success": params.success, "message": params.message}

    registry.register_tool(
        name="done",
        param_model=DoneParams,
        function=done_tool,
        description="Mark the task as done",
    )

    return registry


# ========== Agent Fixtures ==========


@pytest.fixture
def agent_settings():
    """Returns agent settings for testing."""
    return AgentSettings(
        max_steps=5,
        max_failures=2,
        retry_delay=0,
        summary_interval=3,
        validate_output=False,
    )


@pytest.fixture
def test_agent(mock_llm, tool_registry, legacy_memory, agent_settings):
    """Returns a configured agent for testing."""
    return Agent(
        task="Test task",
        llm=mock_llm,
        tools_registry=tool_registry,
        settings=agent_settings,
        memory=legacy_memory,
    )


@pytest.fixture
def test_agent_with_tool_calls(
    mock_llm_with_tool_calls, tool_registry, legacy_memory, agent_settings
):
    """Returns an agent configured with a mock LLM that makes tool calls."""
    return Agent(
        task="Test task with tool calls",
        llm=mock_llm_with_tool_calls,
        tools_registry=tool_registry,
        settings=agent_settings,
        memory=legacy_memory,
    )


# ========== Orchestrator Fixtures ==========


@pytest.fixture
def orchestrator(mock_llm, tool_registry):
    """Returns an orchestrator for testing."""
    return Orchestrator(llm=mock_llm, tools_registry=tool_registry)


@pytest.fixture
def orchestrator_with_tool_calls(mock_llm_with_tool_calls, tool_registry):
    """Returns an orchestrator with a mock LLM that makes tool calls."""
    return Orchestrator(llm=mock_llm_with_tool_calls, tools_registry=tool_registry)


# ========== Evaluation Fixtures ==========


@pytest.fixture
def evaluator():
    """Returns an evaluator for testing."""
    return Evaluator(max_retries=2)


@pytest.fixture
def action_result_success():
    """Returns a successful action result."""
    return ActionResult(
        success=True, value="Test success result", error=None, include_in_memory=True
    )


@pytest.fixture
def action_result_failure():
    """Returns a failed action result."""
    return ActionResult(
        success=False, value=None, error="Test error message", include_in_memory=True
    )


# ========== Utility Fixtures ==========


@pytest.fixture
def test_task_id():
    """Returns a consistent task ID for testing."""
    return "test_task_123"


@pytest.fixture
def async_context():
    """Creates and tears down an event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_prompt_loader():
    """Mocks the prompt loader to return a test prompt."""
    with patch("pori.utils.prompt_loader.load_prompt") as mock:
        mock.return_value = """
        You are a test agent.
        
        Available tools:
        {tool_descriptions}
        
        Complete the task step by step.
        """
        yield mock
