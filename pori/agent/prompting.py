"""System-prompt, message-window, and context/plan rendering. `Agent` methods,
grouped here for readability and bound onto the class in `core` (they take `self`)."""

import json
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
    TextBlock,
    UserMessage,
    normalize_usage,
)

from ..prompts import (
    SystemPromptTiers,
    build_system_prompt,
    discover_project_context,
    resolve_identity,
)
from ..retrieval import RetrievalEvidence
from ..runtime import (
    BudgetExceeded,
    BudgetLedger,
    CancellationToken,
    ReceiptStatus,
    RunContext,
    ToolExecutionReceipt,
    stable_fingerprint,
    utc_now,
)
from ..skills import (
    SelectedSkill,
    SkillCatalog,
    SkillIndexEntry,
    SkillSummary,
    render_selected_skills,
)
from ..utils.logging_config import ensure_logger_configured
from ..utils.prompt_loader import load_prompt
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


def _setup_system_message(self):
    """Set up the system message for the agent."""
    logger.debug("Setting up system message", extra={"task_id": self.task_id})

    tool_count = len(self.tools_registry.tools)
    logger.info(f"Available tools: {tool_count}", extra={"task_id": self.task_id})

    # Assemble the system prompt from cache-ordered tiers:
    #   stable   -> default identity + core operating rules
    #   context  -> caller-supplied custom prompt
    #   volatile -> available-skills index + selected-skill instructions
    # Tool schemas are sent natively via the provider API, so the prompt
    # carries no JSON envelope or textual tool descriptions.
    core_instructions = load_prompt("system/agent_core.md")
    # Identity: a user SOUL.md persona if present (re-read per task), else
    # the neutral default identity.
    tiers = SystemPromptTiers()
    tiers.stable.append(resolve_identity(self._soul_path, soul_text=self._soul_text))
    tiers.stable.append(core_instructions)

    if self._custom_system_prompt:
        tiers.context.append(self._custom_system_prompt)

    if self._load_project_context:
        tiers.context.extend(discover_project_context())

    available_skill_prompt = self._render_available_skills_prompt()
    if available_skill_prompt:
        tiers.volatile.append(available_skill_prompt)

    selected_skill_prompt = render_selected_skills(self.selected_skills)
    if selected_skill_prompt:
        tiers.volatile.append("# Selected Skills\n" + selected_skill_prompt)

    self.system_message = build_system_prompt(tiers)

    self.prompt_fingerprint = stable_fingerprint(
        {
            "system_message": self.system_message,
            "frozen_core_memory": self._frozen_core_memory,
            "selected_skills": [
                {
                    "skill_id": skill.manifest.skill_id,
                    "fingerprint": skill.fingerprint,
                }
                for skill in self.selected_skills
            ],
        }
    )

    # Add system message to memory
    self.memory.add_message("system", self.system_message)

    # Add task message to memory (working memory resets per task via begin_task)
    self.memory.add_message("user", f"Task: {self.task}")
    # Task is already tracked in memory.tasks — no need to duplicate as an experience.
    # Only user facts (via core_memory/remember tools) should be stored as experiences.

    logger.debug("System message setup complete", extra={"task_id": self.task_id})


def _render_available_skills_prompt(self) -> str:
    """Render Hermes-style skill metadata without loading skill instructions."""
    if self.skill_catalog is None:
        return ""
    tool_names = self.capability_snapshot.tool_names
    if not {"skills_list", "skill_view"}.issubset(tool_names):
        return ""
    entries = tuple(
        entry
        for entry in self.skill_catalog.index(
            self.capability_snapshot,
            model_capabilities=self.model_capabilities,
        )
        if not entry.model_invocation_disabled
    )
    if not entries:
        return ""

    def _entry_line(entry: SkillIndexEntry) -> str:
        tags = f" tags={', '.join(entry.tags)}" if entry.tags else ""
        commands = f" commands=/{', /'.join(entry.commands)}" if entry.commands else ""
        readiness = (
            f" readiness={entry.readiness}" if entry.readiness != "ready" else ""
        )
        eligibility = (
            " eligible"
            if entry.eligible
            else f" unavailable={', '.join(entry.reasons)}"
        )
        return (
            f"- {entry.skill_id}: {entry.summary}"
            f"{tags}{commands}{readiness} ({eligibility})"
        )

    lines = [
        "# Available Skills",
        "Skills are available on demand through the skills_list and skill_view tools.",
        (
            "Before performing an actionable task, if a listed skill is relevant, "
            "load it with skill_view and follow its instructions."
        ),
        (
            "For explicit learning/workflow requests such as 'teach me', "
            "'step by step', 'as a course', 'lesson', or 'workshop', load the "
            "matching skill before answering when one is available."
        ),
        (
            "Do not load a skill for small talk, meta questions about whether to "
            "use a skill, or follow-up answers unless the current task actually "
            "needs that skill."
        ),
        "Explicitly selected skills, if any, are injected separately under Selected Skills.",
    ]
    lines.extend(_entry_line(entry) for entry in entries[:20])
    if len(entries) > 20:
        lines.append(
            f"- ... {len(entries) - 20} more skill(s); call skills_list to search."
        )
    return "\n".join(lines)


def _build_messages(self) -> List[BaseMessage]:
    """Build the list of messages for the LLM."""
    messages: List[BaseMessage] = []

    # Add system message. Memory is appended later as fenced background
    # context so it cannot outrank the current task.
    system_content = self.system_message
    messages.append(SystemMessage(content=system_content))

    recent_structured: List[Dict[str, Any]] = []
    try:
        if hasattr(self.memory, "get_token_limited_messages"):
            context_window = self.context_engine.build(
                self.memory,
                max_tokens=self.settings.context_window_tokens,
                reserve_tokens=self.settings.context_window_reserve_tokens,
            )
            recent_structured = list(context_window.messages)
            self.context_diagnostics = context_window.diagnostics
        else:
            recent_structured = self.memory.get_recent_messages_structured(10)
    except Exception as msg_err:
        logger.debug(
            f"Token-limited message fetch failed, using recent fallback: {msg_err}",
            extra={"task_id": self.task_id},
        )
        recent_structured = self.memory.get_recent_messages_structured(10)

    for msg in recent_structured:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(UserMessage(content=content))
        elif role == "assistant":
            messages.append(AssistantMessage(content=content))
        elif role == "system":
            messages.append(UserMessage(content=f"Context summary:\n{content}"))

    if self._frozen_retrieval_evidence:
        facts = "\n".join(
            f"- [{item.source_type}:{item.source_id}] {item.content}"
            for item in self._frozen_retrieval_evidence[:3]
        )
        messages.append(
            UserMessage(
                content=(
                    "Retrieved Knowledge (frozen for this run):\n"
                    f"{facts}\n\nUse only relevant evidence for the current task; "
                    "preserve its source identity and verify dates and context."
                )
            )
        )

    if self._frozen_core_memory:
        messages.append(
            UserMessage(content=_format_memory_context(self._frozen_core_memory))
        )

    # Volatile per-step state (Runtime Facts, recent actions) goes as late as
    # possible WITHOUT displacing the task from last position: everything
    # before it (system + history + frozen context) then forms a stable,
    # cacheable prefix (AC-1b), while CURRENT TASK stays the final,
    # highest-priority message. The Anthropic wrapper's sliding-window cache
    # markers keep that stable prefix warm across steps.
    messages.append(UserMessage(content=self._get_current_context()))

    task_text = (
        "CURRENT TASK (highest priority):\n"
        f"{self.task}\n\n"
        "Answer this task directly. Use memory only when it is clearly "
        "relevant to this exact task; ignore unrelated remembered facts."
    )
    task_attachments = getattr(self, "task_attachments", None)
    if task_attachments:
        # Multimodal turn: the user's images ride WITH the task as one message
        # (image blocks first, task text last), so every provider adapter maps
        # them natively (Anthropic image blocks / OpenAI image_url / Gemini).
        messages.append(
            UserMessage(content=[*task_attachments, TextBlock(text=task_text)])
        )
    else:
        messages.append(UserMessage(content=task_text))

    return messages


def _capture_retrieval_evidence(self, query: str) -> List[RetrievalEvidence]:
    """Freeze durable recall so writes become visible on the next run."""
    try:
        retrieved = self.memory.recall(query=query, k=5, min_score=0.35)
    except Exception as exc:
        logger.debug(
            "Semantic recall unavailable during run snapshot: %s",
            exc,
            extra={"task_id": self.task_id},
        )
        return []
    evidence = []
    for record_id, content, score in retrieved:
        text = str(content)
        if '"final_answer"' in text or "Final answer" in text:
            continue
        evidence.append(
            RetrievalEvidence(
                source_type="memory",
                source_id=record_id,
                session_id=self.run_context.session_id,
                content=text,
                score=max(0.0, float(score)),
                provenance={"scope": self.memory.scope.namespace},
            )
        )
    return evidence


def _get_current_context(self) -> str:
    """Get the current context for the LLM.

    Scoped to the CURRENT task/turn only. Prior tasks and prior-turn tool
    calls are excluded so the model never sees an earlier answer to a similar
    question and wrongly concludes it "already answered" the current one.
    """
    current = self.memory.tasks.get(self.task_id)
    tasks_status = (
        f"Task '{current.description}': {current.status}"
        if current is not None
        else f"Task '{self.task}': in_progress"
    )

    current_calls = [
        t
        for t in self.memory.tool_call_history
        if getattr(t, "task_id", None) == self.task_id
    ]
    recent_tools = (
        "\n".join(
            [
                f"Tool '{t.tool_name}' called with {t.parameters} → {'Success' if t.success else 'Failed'}\n  Result: {t.result}"
                for t in current_calls[-5:]  # last 5 for THIS task
            ]
        )
        or "(no actions yet for this task)"
    )
    runtime_facts = json.dumps(
        self._runtime_fact_summary(),
        ensure_ascii=True,
        default=str,
        indent=2,
    )

    # Check if we have tool results but no final answer yet
    has_tool_results = len(self.memory.tool_call_history) > 0
    has_done_call = any(t.tool_name == "done" for t in self.memory.tool_call_history)

    # Include the current plan. Prefer the model-owned todo list (update_plan);
    # fall back to the legacy side-call plan when planning_mode is forced on.
    plan_lines = self._render_plan_for_prompt()

    context_prompt = f"""
Current Status:
{tasks_status}

Current Plan (follow these steps; revise only if needed):
{plan_lines}

Recent Actions:
{recent_tools}

Runtime Facts (source of truth for UI-visible outputs):
{runtime_facts}

When using the answer tool, put any UI-visible created files in
artifact_references using paths/receipt_ids from Runtime Facts. If Runtime Facts
has no matching artifact, do not add an artifact reference; explain limitations
in final_answer as normal text.

Please decide on the next action to take to accomplish the task."""

    if has_tool_results and not has_done_call:
        context_prompt += """

REMINDER: You have gathered information using tools. Now analyze the results and use the "answer" tool to provide your final answer. Only call "done" in a later step after "answer" succeeds."""

    return context_prompt


def _render_plan_for_prompt(self) -> str:
    """Render the active plan for the step prompt.

    Prefers the model-owned todo list (update_plan); falls back to the legacy
    side-call plan only when planning_mode/reflection_mode are forced on.
    """
    rendered = self.plan_store.format_for_prompt()
    if rendered:
        return rendered
    if self.state.current_plan:
        return "\n".join([f"- {s}" for s in self.state.current_plan[:5]])
    return "(no plan yet)"
