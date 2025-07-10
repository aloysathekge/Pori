from typing import Any, Dict, List, Optional, Tuple


class ActionResult:
    """Result of an action taken by the agent."""

    def __init__(
        self,
        success: bool,
        value: Any = None,
        error: Optional[str] = None,
        include_in_memory: bool = True,
    ):
        self.success = success
        self.value = value
        self.error = error
        self.include_in_memory = include_in_memory

    def __repr__(self):
        if self.success:
            return f"Success: {self.value}"
        else:
            return f"Error: {self.error}"


class Evaluator:
    """Evaluates the success or failure of agent actions."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}

    def evaluate_tool_result(
        self, tool_name: str, result: Dict[str, Any]
    ) -> ActionResult:
        """Evaluate the result of a tool execution."""
        if result.get("success", False):
            # Reset retry count for successful tool
            self.retry_counts[tool_name] = 0
            return ActionResult(success=True, value=result.get("result"))
        else:
            # Increment retry count
            self.retry_counts[tool_name] = self.retry_counts.get(tool_name, 0) + 1

            # Check if we've exceeded max retries
            if self.retry_counts[tool_name] > self.max_retries:
                return ActionResult(
                    success=False,
                    error=f"Failed after {self.max_retries} attempts: {result.get('error')}",
                    include_in_memory=True,
                )
            else:
                return ActionResult(
                    success=False,
                    error=f"Attempt {self.retry_counts[tool_name]}/{self.max_retries} failed: {result.get('error')}",
                    include_in_memory=True,
                )

    def evaluate_task_completion(
        self, task_description: str, memory: Any
    ) -> Tuple[bool, str]:
        """
        Determine if the overall task is complete.
        """
        # Check if the agent has provided a final answer
        has_final_answer = memory.get_state("final_answer") is not None

        # Look for answer tool call in history
        answer_provided = False
        for tool_call in memory.tool_call_history:
            if tool_call.tool_name == "answer":
                answer_provided = True
                break

        # A task is complete only if a final answer has been provided
        if has_final_answer or answer_provided:
            for tool_call in memory.tool_call_history:
                if tool_call.tool_name == "done":
                    return True, "Task marked as complete with final answer provided"

            # Allow completion even if done isn't explicitly called
            return True, "Final answer provided to user's question"

        # Not complete if we haven't provided an answer yet
        return False, "Task is not complete - no final answer has been provided"

    def should_retry(self, tool_name: str) -> bool:
        """Determine if a failed tool should be retried."""
        return self.retry_counts.get(tool_name, 0) < self.max_retries
