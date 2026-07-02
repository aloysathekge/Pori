"""Reliability evaluation — deterministic tool call verification."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .base import BaseEval, EvalResult

if TYPE_CHECKING:
    from pori.agent import Agent


@dataclass
class ReliabilityResult(EvalResult):
    """Result of a reliability evaluation."""

    passed_tool_calls: List[str] = field(default_factory=list)
    failed_tool_calls: List[str] = field(default_factory=list)


class ReliabilityEval(BaseEval):
    """Test whether the agent calls the expected tools for a given input.

    No LLM judge needed — purely deterministic comparison.

    Usage:
        eval = ReliabilityEval(
            agent=my_agent,
            expected_tool_calls=["web_search", "answer"],
        )
        result = await eval.run()
        result.assert_passed()
    """

    def __init__(
        self,
        agent: "Agent",
        expected_tool_calls: List[str],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agent = agent
        self.expected_tool_calls = expected_tool_calls

    async def run(self, **kwargs) -> ReliabilityResult:
        await self.agent.run()

        actual_tools = [tc.tool_name for tc in self.agent.memory.tool_call_history]

        passed = []
        failed = []
        for expected in self.expected_tool_calls:
            if expected in actual_tools:
                passed.append(expected)
            else:
                failed.append(expected)

        return ReliabilityResult(
            eval_id=self.eval_id,
            eval_type="reliability",
            passed=len(failed) == 0,
            data={
                "expected": self.expected_tool_calls,
                "actual": actual_tools,
            },
            passed_tool_calls=passed,
            failed_tool_calls=failed,
        )
