"""Base evaluation interface — also serves as the guardrail interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4


@dataclass
class EvalResult:
    """Base result from any evaluation."""

    eval_id: str
    eval_type: str
    passed: bool
    data: Dict[str, Any] = field(default_factory=dict)

    def assert_passed(self):
        assert self.passed, f"Evaluation {self.eval_type} FAILED: {self.data}"


class BaseEval(ABC):
    """Abstract base for all evaluation types.

    Also serves as the guardrail interface:
    - pre_check(): input guardrail (before agent runs)
    - post_check(): output guardrail (after agent runs)
    """

    def __init__(self, name: Optional[str] = None):
        self.eval_id = str(uuid4())
        self.name = name

    @abstractmethod
    async def run(self, **kwargs) -> EvalResult:
        """Run the evaluation and return a result."""
        ...

    async def pre_check(self, input_text: str) -> None:
        """Validate input before the agent processes it.
        Raise an exception to block the run.
        """
        pass

    async def post_check(self, input_text: str, output_text: str) -> None:
        """Validate output before returning to the user.
        Raise an exception to block the response.
        """
        pass
