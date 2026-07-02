"""Agent-as-Judge evaluation — custom criteria scoring via LLM judge.

Also implements pre_check/post_check for use as a runtime guardrail.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import BaseEval, EvalResult

if TYPE_CHECKING:
    from pori.llm.base import BaseChatModel


class BinaryJudgment(BaseModel):
    """Binary pass/fail judgment."""

    passed: bool = Field(..., description="Whether the output passes the criteria")
    reason: str = Field(..., description="Reasoning for the judgment")


class NumericJudgment(BaseModel):
    """Numeric 1-10 judgment."""

    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    reason: str = Field(..., description="Reasoning for the score")


@dataclass
class AgentJudgeResult(EvalResult):
    """Result of an agent-as-judge evaluation."""

    score: Optional[int] = None
    reason: str = ""
    pass_rate: float = 0.0
    evaluations: List[Dict] = field(default_factory=list)


class AgentJudgeEval(BaseEval):
    """Evaluate agent output against custom criteria using an LLM judge.

    Supports two scoring modes:
    - "binary": PASS / FAIL
    - "numeric": Score 1-10 with configurable threshold

    Also implements pre_check / post_check for use as a runtime guardrail.

    Usage as eval:
        eval = AgentJudgeEval(
            criteria="Response must be professional and cite sources",
            judge_llm=my_model,
            scoring="binary",
        )
        result = await eval.run(input="...", output="...")

    Usage as guardrail (attach to agent):
        agent = Agent(
            ...,
            guardrails=[
                AgentJudgeEval(
                    criteria="No PII in responses",
                    judge_llm=my_model,
                    scoring="binary",
                )
            ],
        )
    """

    def __init__(
        self,
        criteria: str,
        judge_llm: "BaseChatModel",
        scoring: Literal["binary", "numeric"] = "binary",
        threshold: int = 7,
        on_fail: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.criteria = criteria
        self.judge_llm = judge_llm
        self.scoring = scoring
        self.threshold = threshold
        self.on_fail = on_fail

    async def run(
        self,
        input: str = "",
        output: str = "",
        cases: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AgentJudgeResult:
        """Evaluate one or multiple input/output pairs."""
        from pori.llm.messages import SystemMessage, UserMessage

        pairs = cases or [{"input": input, "output": output}]
        evaluations = []

        schema = NumericJudgment if self.scoring == "numeric" else BinaryJudgment
        structured = self.judge_llm.with_structured_output(schema, include_raw=True)

        for pair in pairs:
            prompt = f"""Evaluate the following output against the criteria.

<criteria>{self.criteria}</criteria>
<input>{pair['input']}</input>
<output>{pair['output']}</output>

Be objective and thorough."""

            response = await structured.ainvoke(
                [
                    SystemMessage(content="You are an expert evaluator."),
                    UserMessage(content=prompt),
                ]
            )

            parsed = response.get("parsed") if isinstance(response, dict) else response
            if parsed and hasattr(parsed, "reason"):
                if self.scoring == "numeric":
                    passed = parsed.score >= self.threshold
                    evaluations.append(
                        {
                            "input": pair["input"],
                            "output": pair["output"],
                            "score": parsed.score,
                            "reason": parsed.reason,
                            "passed": passed,
                        }
                    )
                else:
                    evaluations.append(
                        {
                            "input": pair["input"],
                            "output": pair["output"],
                            "passed": parsed.passed,
                            "reason": parsed.reason,
                        }
                    )

                if not evaluations[-1]["passed"] and self.on_fail:
                    self.on_fail(evaluations[-1])

        pass_count = sum(1 for e in evaluations if e["passed"])
        pass_rate = pass_count / len(evaluations) if evaluations else 0.0

        return AgentJudgeResult(
            eval_id=self.eval_id,
            eval_type="agent_judge",
            passed=pass_rate == 1.0,
            data={"criteria": self.criteria, "scoring": self.scoring},
            score=evaluations[-1].get("score") if evaluations else None,
            reason=evaluations[-1].get("reason", "") if evaluations else "",
            pass_rate=pass_rate,
            evaluations=evaluations,
        )

    # --- Guardrail hooks ---

    async def pre_check(self, input_text: str) -> None:
        """Use as input guardrail: validate input against criteria."""
        result = await self.run(input=input_text, output="(pre-check: input only)")
        if not result.passed:
            raise ValueError(f"Input guardrail failed: {result.reason}")

    async def post_check(self, input_text: str, output_text: str) -> None:
        """Use as output guardrail: validate output against criteria."""
        result = await self.run(input=input_text, output=output_text)
        if not result.passed:
            raise ValueError(f"Output guardrail failed: {result.reason}")
