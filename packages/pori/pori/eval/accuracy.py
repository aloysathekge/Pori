"""Accuracy evaluation — LLM-judged answer quality scoring."""

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import BaseEval, EvalResult

if TYPE_CHECKING:
    from pori.agent import Agent
    from pori.llm.base import BaseChatModel


class AccuracyScore(BaseModel):
    """Structured output from the evaluator agent."""

    score: int = Field(..., ge=1, le=10, description="Accuracy score 1-10")
    reason: str = Field(..., description="Reasoning for the score")


@dataclass
class AccuracyResult(EvalResult):
    """Result of an accuracy evaluation."""

    score: float = 0.0
    reason: str = ""
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0


class AccuracyEval(BaseEval):
    """Test whether the agent's output matches an expected answer.

    Uses a separate LLM as judge to score accuracy 1-10.

    Usage:
        eval = AccuracyEval(
            agent=my_agent,
            expected_output="4",
            evaluator_llm=my_evaluator_model,
        )
        result = await eval.run()
        assert result.avg_score >= 7
    """

    def __init__(
        self,
        agent: "Agent",
        expected_output: str,
        evaluator_llm: "BaseChatModel",
        num_iterations: int = 1,
        threshold: float = 7.0,
        additional_guidelines: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agent = agent
        self.expected_output = expected_output
        self.evaluator_llm = evaluator_llm
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.additional_guidelines = additional_guidelines or ""

    async def run(self, **kwargs) -> AccuracyResult:
        from pori.llm.messages import SystemMessage, UserMessage

        iterations = []

        for i in range(self.num_iterations):
            run_result = await self.agent.run()
            agent_output = str(
                self.agent.memory.get_final_answer() or "No answer provided"
            )

            eval_prompt = f"""Compare the agent's output to the expected output.

<agent_input>{self.agent.task}</agent_input>
<expected_output>{self.expected_output}</expected_output>
<agent_output>{agent_output}</agent_output>

{self.additional_guidelines}

Score the accuracy from 1-10:
1-2: Completely incorrect
3-4: Major inaccuracies
5-6: Partially correct with significant issues
7-8: Mostly accurate with minor issues
9-10: Highly accurate, matches expected output closely

You must assume the expected_output is correct."""

            structured = self.evaluator_llm.with_structured_output(
                AccuracyScore, include_raw=True
            )
            response = await structured.ainvoke(
                [
                    SystemMessage(
                        content="You are an expert evaluator. Score objectively."
                    ),
                    UserMessage(content=eval_prompt),
                ]
            )

            parsed = response.get("parsed") if isinstance(response, dict) else response
            if parsed and hasattr(parsed, "score"):
                iterations.append(
                    {
                        "iteration": i + 1,
                        "agent_output": agent_output,
                        "score": parsed.score,
                        "reason": parsed.reason,
                    }
                )

        scores = [it["score"] for it in iterations]
        avg = statistics.mean(scores) if scores else 0
        mn = min(scores) if scores else 0
        mx = max(scores) if scores else 0

        return AccuracyResult(
            eval_id=self.eval_id,
            eval_type="accuracy",
            passed=avg >= self.threshold,
            data={"iterations": iterations},
            score=avg,
            reason=iterations[-1]["reason"] if iterations else "",
            iterations=iterations,
            avg_score=avg,
            min_score=mn,
            max_score=mx,
        )
