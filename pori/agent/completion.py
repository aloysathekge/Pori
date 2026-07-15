"""Answer/completion gates — may we accept this answer, is the task done.
These are `Agent` methods, grouped here for readability and bound onto the
class in `core` (they take `self`).
"""

import re
from typing import Optional, Tuple

from pori.llm import SystemMessage, UserMessage

from ..utils.logging_config import ensure_logger_configured
from .schemas import CompletionValidation

logger = ensure_logger_configured("pori.agent")


_COMMITTED_ACTION_CLAIM = re.compile(
    r"\b(?:i|we)\s+(?:have\s+)?(?:sent|booked|published|scheduled|deleted|"
    r"created|updated)\b|\b(?:email|message|booking|event|post|record)\s+"
    r"(?:was\s+|has\s+been\s+)?(?:sent|booked|published|scheduled|deleted|"
    r"created|updated)\b",
    re.IGNORECASE,
)


def _staged_outcome_claim_error(self, answer_text: str) -> Optional[str]:
    """Reject claims that a merely staged external consequence happened."""
    staged = [
        receipt
        for receipt in self.execution_receipts
        if receipt.status.value == "staged"
    ]
    if not staged or not _COMMITTED_ACTION_CLAIM.search(answer_text):
        return None
    lowered = answer_text.lower()
    if any(
        marker in lowered
        for marker in (
            "not sent",
            "not been sent",
            "not executed",
            "awaiting approval",
            "pending approval",
            "staged for approval",
        )
    ):
        return None
    return (
        "Cannot claim an external action completed when its receipt is only "
        "staged. Say that it is awaiting approval and has not executed yet."
    )


def _current_task_status(self) -> str:
    """Return the status of the current task."""
    task = self.memory.tasks.get(self.task_id)
    return task.status if task else "unknown"


def _current_task_terminal(self) -> bool:
    """Return True if the current task is in a terminal state (completed or failed)."""
    status = self._current_task_status()
    return status in ("completed", "failed")


def _task_requests_memory_mutation(self) -> bool:
    """Return True if the current task explicitly requests memory deletion/removal."""
    task_lower = self.task.lower()
    deletion_keywords = ["forget", "delete", "remove", "erase"]
    memory_keywords = ["memory", "role", "instruction", "persona", "identity"]
    has_deletion = any(kw in task_lower for kw in deletion_keywords)
    has_memory = any(kw in task_lower for kw in memory_keywords)
    return has_deletion and has_memory


def _current_task_has_core_memory_mutation(self) -> bool:
    """Return True if a core memory mutation tool succeeded in the current task.

    Only replacement/rewrite tools count (core_memory_replace, memory_rethink,
    core_memory_rethink). Append/insert tools do not satisfy deletion requests.
    """
    mutation_tools = {
        "core_memory_replace",
        "memory_rethink",
        "core_memory_rethink",
    }
    for tc in self.memory.tool_call_history:
        if (
            getattr(tc, "task_id", None) == self.task_id
            and getattr(tc, "tool_name", None) in mutation_tools
            and getattr(tc, "success", False)
        ):
            return True
    return False


def _current_task_has_loaded_skill(self) -> bool:
    """Return True if this task loaded a skill at runtime or explicitly."""
    if self.selected_skills:
        return True
    for tc in self.memory.tool_call_history:
        if (
            getattr(tc, "task_id", None) == self.task_id
            and getattr(tc, "tool_name", None) == "skill_view"
            and getattr(tc, "success", False)
        ):
            return True
    return False


# A skill is only nudged when the task references the skill's own declared
# identity strongly enough to clear the catalog search score (slug/name/
# command boosts), rather than matching a hardcoded list of English phrases.
_SKILL_NUDGE_MIN_SCORE = 6


def _required_skill_view_before_answer(self) -> Optional[str]:
    """Return a skill id when this task should load a skill before answering.

    Catalog-driven: the trigger is a high-confidence match against a skill's
    own metadata, not a fixed set of workflow phrases.
    """
    if self.skill_catalog is None or self._current_task_has_loaded_skill():
        return None
    hits = self.skill_catalog.search(
        self.task,
        self.capability_snapshot,
        model_capabilities=self.model_capabilities,
        limit=5,
        min_score=self._SKILL_NUDGE_MIN_SCORE,
    )
    for hit in hits:
        entry = hit.entry
        # Honor `disable-model-invocation`: such skills are user-invoked only
        # (slash command / explicit selection), never auto-loaded by the model.
        if entry.eligible and not entry.model_invocation_disabled:
            return entry.skill_id
    return None


async def _validate_answer_text(self, answer_text: str) -> Tuple[bool, str]:
    """LLM-judge whether a proposed final answer adequately addresses the task.

    Fails open: if the validator itself errors, the answer is accepted so a
    validator outage never blocks legitimate completion. Returns
    (is_adequate, reason).
    """
    try:
        messages = [
            SystemMessage(
                content=(
                    "You are a strict reviewer. Decide whether the proposed "
                    "answer adequately and directly addresses the user's task. "
                    "Reject answers that are empty, evasive, off-topic, refuse "
                    "without cause, or leave the core request unaddressed. "
                    "Do not reward verbosity; a short correct answer is adequate."
                )
            ),
            UserMessage(
                content=(
                    f"Task:\n{self.task}\n\n"
                    f"Proposed answer:\n{answer_text}\n\n"
                    "Is this answer adequate? Set adequate=true/false and give a "
                    "brief reason. If false, the reason should say what is missing."
                )
            ),
        ]
        structured_llm = self.llm.with_structured_output(CompletionValidation)
        result = await structured_llm.ainvoke(messages)
        adequate = bool(getattr(result, "adequate", True))
        reason = str(getattr(result, "reason", "") or "")
        logger.info(
            f"Output validation: adequate={adequate}"
            + (f" — {reason}" if reason else ""),
            extra={"task_id": self.task_id},
        )
        return adequate, reason
    except Exception as e:
        logger.debug(
            f"Output validation failed open (accepting answer): {e}",
            extra={"task_id": self.task_id},
        )
        return True, ""
