"""Optional planning/reflection phases + their gating heuristics. `Agent` methods,
grouped here for readability and bound onto the class in `core`."""

import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

from pori.llm import (
    AssistantMessage,
    BaseChatModel,
    BaseMessage,
    SystemMessage,
    UserMessage,
    normalize_usage,
)

from ..evaluation import ActionResult, Evaluator
from ..runtime import BudgetExceeded
from ..utils.logging_config import ensure_logger_configured
from .schemas import (
    AgentOutput,
    AgentSettings,
    AgentState,
    CompletionValidation,
    FatalAgentError,
    PlanOutput,
    ReflectOutput,
    _format_memory_context,
)

logger = ensure_logger_configured("pori.agent")


def _should_plan(self) -> bool:
    """Decide whether this task should pay for a separate planning call."""
    mode = self.settings.planning_mode
    if mode == "never":
        return False
    if mode == "always":
        return True
    if self.state.current_plan:
        return False
    return self._task_looks_complex()


def _should_reflect(self, tool_results: List[ActionResult]) -> bool:
    """Decide whether this step should pay for a separate reflection call."""
    mode = self.settings.reflection_mode
    if mode == "never":
        return False
    if self._current_task_terminal():
        return False
    if self.memory.get_state("final_answer") is not None:
        return False
    if mode == "always":
        return True
    # Auto reflection is for recovery, not routine progress updates.
    return any(not result.success for result in tool_results)


def _task_looks_complex(self) -> bool:
    """Heuristic classifier for adaptive planning.

    The goal is conservative: skip the planner for direct/simple tasks, but
    use it when the request likely spans multiple files, debugging,
    architecture, memory mutation, or a longer investigation.
    """
    task_lower = self.task.lower()
    words = re.findall(r"\w+", task_lower)

    complex_keywords = {
        "architecture",
        "architect",
        "audit",
        "debug",
        "deep-dive",
        "design",
        "diagnose",
        "implement",
        "investigate",
        "migration",
        "migrate",
        "multi-step",
        "optimize",
        "plan",
        "refactor",
        "review",
        "root cause",
        "tests",
        "update",
    }
    memory_mutation_keywords = {
        "forget",
        "delete",
        "erase",
        "remove",
        "memory",
        "persona",
        "instruction",
        "role",
    }
    multi_file_keywords = {"files", "modules", "codebase", "repo", "project"}
    simple_patterns = (
        "write and run",
        "write me a",
        "what is",
        "calculate",
        "prints",
        "print",
    )

    has_complex_keyword = any(keyword in task_lower for keyword in complex_keywords)
    has_memory_mutation = any(
        keyword in task_lower for keyword in memory_mutation_keywords
    )
    has_multi_file_hint = any(keyword in task_lower for keyword in multi_file_keywords)
    has_multiple_clauses = task_lower.count(" and ") >= 2 or task_lower.count(",") >= 2
    is_long_request = len(words) >= 35
    is_simple_direct = any(pattern in task_lower for pattern in simple_patterns)

    if has_memory_mutation or has_multi_file_hint:
        return True
    if has_complex_keyword and not is_simple_direct:
        return True
    if is_long_request or has_multiple_clauses:
        return True
    return False


async def _plan_if_needed(self) -> None:
    """Create a brief plan if none exists."""
    if self.state.current_plan:
        return

    plan_prompt = (
        "You are planning how to accomplish the user's task. "
        "Return 1-3 short steps. Each step should map directly to a tool call or the final 'answer' action. "
        "IMPORTANT: If the task requires information the user has not provided, "
        "the FIRST step must be to use `ask_user` to gather the missing details. "
        "To inspect core memory, use `core_memory_read`. "
        "For core-memory rewrites, use `memory_rethink` (or `core_memory_replace` for targeted edits); "
        "never use rewrite tools just to view or summarize memory. "
        "Also, if you do not have the tools needed to deliver the requested output, "
        "the plan should be to inform the user via `answer` immediately. "
        "Do NOT describe internal operations that a tool already abstracts (e.g., loops or data structures). "
        "Reference tools by their exact names and only the essential arguments."
    )

    messages = [
        SystemMessage(content=self.system_message),
        UserMessage(content=f"Task: {self.task}\n{plan_prompt}"),
    ]

    try:
        structured_llm = self.llm.with_structured_output(PlanOutput)
        plan: PlanOutput = await structured_llm.ainvoke(messages)
        steps = [s.strip() for s in (plan.plan_steps or []) if s and s.strip()]
        # Cap plan length to 5 steps
        self.state.current_plan = steps[:5]
        self.state.last_reflection = None
        if self.state.current_plan:
            self.memory.add_message(
                "system",
                f"Plan established:\n"
                + "\n".join([f"- {s}" for s in self.state.current_plan]),
            )
            logger.info(
                f"Plan created with {len(self.state.current_plan)} steps",
                extra={"task_id": self.task_id},
            )
            # Also log the concrete steps for visibility in console
            try:
                logger.info(
                    "Plan steps:\n"
                    + "\n".join([f"- {s}" for s in self.state.current_plan]),
                    extra={"task_id": self.task_id},
                )
            except Exception:
                pass
    except BudgetExceeded:
        raise
    except Exception as e:
        logger.debug(f"Plan generation failed: {e}", extra={"task_id": self.task_id})


async def _reflect_and_update_plan(self, tool_results: List[ActionResult]) -> None:
    """Reflect on recent actions and update the plan if necessary."""
    # Build a compact summary of the last step's tool outcomes
    results_summary = []
    for r in tool_results[-3:]:
        if isinstance(r, ActionResult):
            if r.success:
                results_summary.append("success")
            else:
                results_summary.append(f"fail: {r.error}")
        else:
            results_summary.append(str(r))
    results_text = (
        ", ".join(results_summary) if results_summary else "(no tool results)"
    )

    reflect_prompt = (
        "Briefly critique progress vs the plan. If a step is done or wrong, propose an updated 1-5 step plan. "
        "Only update the plan if it clearly improves progress."
    )

    plan_text = (
        "\n".join([f"- {s}" for s in self.state.current_plan])
        if self.state.current_plan
        else "(no plan)"
    )
    messages = [
        SystemMessage(content="Reflect succinctly. Be pragmatic."),
        UserMessage(
            content=(
                f"Task: {self.task}\n"
                f"Current plan:\n{plan_text}\n"
                f"Recent results: {results_text}\n"
                f"{reflect_prompt}"
            )
        ),
    ]

    try:
        structured_llm = self.llm.with_structured_output(ReflectOutput)
        reflection: ReflectOutput = await structured_llm.ainvoke(messages)
        self.state.last_reflection = reflection.critique
        self.memory.add_message("system", f"Reflection: {reflection.critique}")

        if reflection.update_plan:
            new_steps = [s.strip() for s in reflection.update_plan if s and s.strip()]
            if new_steps:
                self.state.current_plan = new_steps[:5]
                self.memory.add_message(
                    "system",
                    "Plan updated:\n"
                    + "\n".join([f"- {s}" for s in self.state.current_plan]),
                )
                logger.info(
                    f"Plan updated with {len(self.state.current_plan)} steps",
                    extra={"task_id": self.task_id},
                )
                try:
                    logger.info(
                        "Updated plan steps:\n"
                        + "\n".join([f"- {s}" for s in self.state.current_plan]),
                        extra={"task_id": self.task_id},
                    )
                except Exception:
                    pass
    except BudgetExceeded:
        raise
    except Exception as e:
        logger.debug(f"Reflection failed: {e}", extra={"task_id": self.task_id})
