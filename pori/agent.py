import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from pori.llm import AssistantMessage, BaseChatModel, SystemMessage, UserMessage

from .evaluation import ActionResult, Evaluator
from .hitl import (
    ActionRequest,
    ApprovalRequest,
    ApprovalResponse,
    HITLConfig,
    HITLHandler,
    ReviewConfig,
    resolve_interrupt_config,
)
from .memory import AgentMemory
from .metrics import (
    LLMCallMetrics,
    RunMetrics,
    StepMetrics,
    TokenUsage,
    ToolCallMetrics,
    estimate_llm_call_cost,
)
from .observability.trace import Span, SpanStatus, SpanType, Trace
from .tools.registry import ToolExecutor, ToolRegistry
from .utils.logging_config import ensure_logger_configured
from .utils.prompt_loader import load_prompt

# Set up logger for this module - this will work regardless of import order
logger = ensure_logger_configured("pori.agent")


class AgentState(BaseModel):
    """The current state of the agent."""

    n_steps: int = 0
    consecutive_failures: int = 0
    paused: bool = False
    stopped: bool = False
    # Planning/Reflection state
    current_plan: List[str] = Field(default_factory=list)
    last_reflection: Optional[str] = None


class AgentSettings(BaseModel):
    """Settings for the agent."""

    max_steps: int = 50
    max_failures: int = 3
    retry_delay: int = 2
    summary_interval: int = 5
    validate_output: bool = False
    context_window_tokens: int = 3000
    context_window_reserve_tokens: int = 1200


class AgentOutput(BaseModel):
    """Output from the agent's decision process."""

    current_state: Dict[str, str]
    action: List[Dict[str, Any]]


class PlanOutput(BaseModel):
    plan_steps: List[str]
    rationale: str


class ReflectOutput(BaseModel):
    critique: str
    update_plan: Optional[List[str]] = None


class Agent:
    """
    A general-purpose agent that can perform tasks using tools and have memory.

    """

    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        tools_registry: ToolRegistry,
        settings: AgentSettings = AgentSettings(),
        memory: Optional[Any] = None,
        sandbox_base_dir: Optional[str] = None,
        hitl_handler: Optional[HITLHandler] = None,
        hitl_config: Optional[HITLConfig] = None,
        guardrails: Optional[List] = None,
    ):
        # Generate unique task ID for tracking (also used as thread_id for sandbox)
        self.task_id = str(uuid.uuid4())[:8]  # Short ID for logging
        self.sandbox_base_dir = sandbox_base_dir

        # Human-in-the-loop
        self.hitl_handler = hitl_handler
        self.hitl_config = hitl_config
        self._approved_signatures: set = set()  # for auto_approve_duplicates

        # Guardrails (BaseEval instances with pre_check/post_check)
        self.guardrails = guardrails or []

        logger.info(f"Initializing new agent", extra={"task_id": self.task_id})
        logger.info(f"Task: {task}", extra={"task_id": self.task_id})
        if hitl_handler and hitl_config and hitl_config.enabled:
            logger.info("HITL approval gates enabled", extra={"task_id": self.task_id})

        self.task = task
        self.llm = llm
        self.tools_registry = tools_registry
        self.tool_executor = ToolExecutor(tools_registry)
        self.settings = settings

        logger.info(
            f"Agent settings: max_steps={settings.max_steps}, max_failures={settings.max_failures}",
            extra={"task_id": self.task_id},
        )

        # Initialize state components
        self.state = AgentState()
        self.memory = memory if memory is not None else AgentMemory()
        self.evaluator = Evaluator(max_retries=settings.max_failures)

        # Metrics and trace for this agent run (initialized in run())
        self._run_metrics: Optional[RunMetrics] = None
        self._trace: Optional[Trace] = None
        self._current_step_span: Optional[Span] = None

        # Create task record
        self.memory.create_task(self.task_id, task)
        logger.info(f"Created task record in memory", extra={"task_id": self.task_id})

        # Set up system message
        self._setup_system_message()

    def _setup_system_message(self):
        """Set up the system message for the agent."""
        logger.debug("Setting up system message", extra={"task_id": self.task_id})

        # Get tool descriptions for the prompt
        tool_descriptions = self.tools_registry.get_tool_descriptions()
        tool_count = len(self.tools_registry.tools)
        logger.info(f"Available tools: {tool_count}", extra={"task_id": self.task_id})

        # Load prompt template from file and fill in dynamic values
        prompt_template = load_prompt("system/agent_core.md")
        self.system_message = prompt_template.replace(
            "{tool_descriptions}", tool_descriptions
        )

        # Add system message to memory
        self.memory.add_message("system", self.system_message)

        # Add task message to memory (working memory resets per task via begin_task)
        self.memory.add_message("user", f"Task: {self.task}")
        # Also store task text as an experience for recall
        try:
            self.memory.add_experience(
                f"Task stated: {self.task}", importance=1, meta={"type": "task"}
            )
        except Exception:
            pass

        logger.debug("System message setup complete", extra={"task_id": self.task_id})

    async def step(self) -> None:
        """Execute one step of the task."""
        step_number = self.state.n_steps + 1
        logger.info(
            f"Starting step {step_number}",
            extra={"task_id": self.task_id, "step": step_number},
        )

        tool_results = []
        step_start_time = datetime.now()
        step_metrics = StepMetrics(step_number=step_number)

        # Start a step span in the trace
        step_span = None
        if self._trace:
            step_span = self._trace.start_span(
                name=f"step_{step_number}",
                span_type=SpanType.AGENT,
            )
            self._current_step_span = step_span

        try:
            # Create summaries at regular intervals to avoid context overflow
            if (
                self.state.n_steps % self.settings.summary_interval == 0
                and self.state.n_steps > 0
            ):
                logger.info(
                    f"Creating memory summary at step {self.state.n_steps}",
                    extra={"task_id": self.task_id, "step": step_number},
                )
                summary = self.memory.create_summary(self.state.n_steps)
                self.memory.add_message("system", f"Memory summary: {summary}")

            # Ensure we have a minimal plan before acting
            await self._plan_if_needed()

            # Get the next action from the LLM
            logger.debug(
                "Getting next action from LLM",
                extra={"task_id": self.task_id, "step": step_number},
            )
            llm_span = None
            if self._trace and step_span:
                llm_span = self._trace.start_span(
                    name=f"{getattr(self.llm, 'model', 'llm')}.invoke",
                    span_type=SpanType.LLM,
                    parent_span_id=step_span.span_id,
                    attributes={"model": getattr(self.llm, "model", "")},
                )
            llm_start_time = datetime.now()
            model_output = await self.get_next_action()
            llm_duration = (datetime.now() - llm_start_time).total_seconds()
            if llm_span:
                llm_span.attributes["duration_seconds"] = llm_duration
                llm_span.finish()

            # Record LLM call metrics for this step
            try:
                # Optional token usage from provider-specific metadata
                tokens = TokenUsage()
                usage = getattr(self.llm, "last_usage", None)
                if isinstance(usage, dict):
                    # Anthropic-style keys
                    if "input_tokens" in usage or "output_tokens" in usage:
                        tokens.input_tokens = int(usage.get("input_tokens", 0) or 0)
                        tokens.output_tokens = int(usage.get("output_tokens", 0) or 0)
                        tokens.total_tokens = tokens.input_tokens + tokens.output_tokens
                        tokens.cache_read_tokens = int(
                            usage.get("cache_read_input_tokens", 0) or 0
                        )
                        tokens.cache_write_tokens = int(
                            usage.get("cache_creation_input_tokens", 0) or 0
                        )
                    # OpenAI-style keys
                    elif "prompt_tokens" in usage or "completion_tokens" in usage:
                        tokens.input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        tokens.output_tokens = int(
                            usage.get("completion_tokens", 0) or 0
                        )
                        tokens.total_tokens = int(usage.get("total_tokens", 0) or 0)

                model_id = getattr(self.llm, "model", "")
                cost = (
                    estimate_llm_call_cost(model_id, tokens)
                    if tokens.total_tokens
                    else None
                )

                llm_metrics = LLMCallMetrics(
                    model_id=model_id,
                    model_provider=self.llm.__class__.__name__,
                    tokens=tokens,
                    cost=cost,
                    duration_seconds=llm_duration,
                )
                step_metrics.llm_calls.append(llm_metrics)
            except Exception:
                pass

            action_count = len(model_output.action) if model_output.action else 0
            if action_count > 0:
                logger.info(
                    f"LLM suggested {action_count} actions",
                    extra={"task_id": self.task_id, "step": step_number},
                )
            else:
                logger.warning(
                    "LLM provided no actions",
                    extra={"task_id": self.task_id, "step": step_number},
                )

            # Execute actions
            if model_output.action:
                tool_results = await self.execute_actions(
                    model_output.action,
                    agent_reasoning=model_output.current_state,
                    step_metrics=step_metrics,
                )

            # Update state
            self.state.n_steps += 1
            self.state.consecutive_failures = 0

            logger.info(
                f"Step {step_number} completed successfully",
                extra={"task_id": self.task_id, "step": step_number},
            )

            # Reflect briefly and revise plan if needed
            try:
                await self._reflect_and_update_plan(tool_results)
            except Exception as reflect_err:
                logger.debug(
                    f"Reflection skipped/failed: {reflect_err}",
                    extra={"task_id": self.task_id, "step": step_number},
                )

        except Exception as e:
            logger.error(
                f"Error during step {step_number}: {str(e)}",
                extra={"task_id": self.task_id, "step": step_number},
                exc_info=True,
            )
            if step_span:
                step_span.finish(SpanStatus.ERROR, error=str(e))
            error_msg = f"Error during step: {str(e)}"
            tool_results = [ActionResult(success=False, error=error_msg)]
            self.state.consecutive_failures += 1
            logger.warning(
                f"Consecutive failures: {self.state.consecutive_failures}/{self.settings.max_failures}",
                extra={"task_id": self.task_id, "step": step_number},
            )

        # Record step metadata
        step_duration = (datetime.now() - step_start_time).total_seconds()
        step_metadata = {
            "step_number": self.state.n_steps,
            "duration_seconds": step_duration,
        }
        logger.info(
            f"Step duration: {step_duration:.2f}s",
            extra={"task_id": self.task_id, "step": step_number},
        )

        # Record step duration in metrics
        try:
            step_metrics.duration_seconds = step_duration
            if self._run_metrics is not None:
                self._run_metrics.steps.append(step_metrics)
        except Exception:
            pass

        # Finish step span if not already finished (error case finishes early)
        if step_span and step_span.end_time is None:
            step_span.finish()

        # Add step results to memory
        for result in tool_results:
            if result.include_in_memory:
                self.memory.add_message("system", str(result))
                # Capture final answer in state for this task
                try:
                    if (
                        isinstance(result.value, dict)
                        and "final_answer" in result.value
                    ):
                        self.memory.update_state("final_answer", result.value)
                except Exception:
                    pass
                # Index non-final results
                try:
                    self.memory.add_experience(
                        f"Step result: {str(result)}",
                        importance=1,
                        meta={"type": "step_result"},
                    )
                except Exception:
                    pass

    async def get_next_action(self) -> AgentOutput:
        """Get next action from LLM based on current state."""
        logger.debug("Building messages for LLM", extra={"task_id": self.task_id})

        # Build messages for the LLM
        messages = self._build_messages()
        message_count = len(messages)
        logger.debug(
            f"Built {message_count} messages for LLM", extra={"task_id": self.task_id}
        )

        # Create dynamic model for the LLM response
        output_model = self._create_output_model()

        # Get response from LLM
        try:
            logger.debug("Calling LLM for next action", extra={"task_id": self.task_id})
            structured_llm = self.llm.with_structured_output(
                output_model, include_raw=True
            )
            response = await structured_llm.ainvoke(messages)

            # Parse the response
            parsed_output = response.get("parsed")
            if not parsed_output:
                logger.warning(
                    "Structured output failed, attempting to parse raw response",
                    extra={"task_id": self.task_id},
                )
                # Attempt to parse raw response if structured output failed
                raw = response.get("raw")
                raw_content = getattr(raw, "content", raw)

                def _coerce_to_output_json(obj: Any) -> Dict[str, Any]:
                    """Coerce various raw shapes into AgentOutput JSON shape."""
                    default = {"current_state": {}, "action": []}
                    # Already a dict
                    if isinstance(obj, dict):
                        # Ensure required keys exist
                        if "current_state" not in obj:
                            obj["current_state"] = obj.get("state", {}) or {}
                        if "action" not in obj:
                            obj["action"] = obj.get("actions", []) or []
                        return obj
                    # String → try JSON
                    if isinstance(obj, str):
                        try:
                            parsed = json.loads(obj)
                            return _coerce_to_output_json(parsed)
                        except Exception:
                            return default
                    # List handling
                    if isinstance(obj, list):
                        if not obj:
                            return default
                        # Common LC message.content shape: list of {type, text}
                        if all(isinstance(x, dict) and "text" in x for x in obj):
                            combined = "\n".join(str(x.get("text", "")) for x in obj)
                            try:
                                parsed = json.loads(combined)
                                return _coerce_to_output_json(parsed)
                            except Exception:
                                return default
                        # If it's a list of action dicts, wrap
                        if all(isinstance(x, dict) for x in obj):
                            first = obj[0]
                            if "current_state" in first or "action" in first:
                                return _coerce_to_output_json(first)
                            # Heuristic: list of tool calls like [{tool: {...}}, ...]
                            if all(
                                len(x.keys()) == 1
                                and isinstance(list(x.values())[0], (dict, list))
                                for x in obj
                            ):
                                return {"current_state": {}, "action": obj}
                            return default
                        # List of strings → try parse first
                        if all(isinstance(x, str) for x in obj):
                            try:
                                parsed = json.loads(obj[0])
                                return _coerce_to_output_json(parsed)
                            except Exception:
                                return default
                        return default
                    # Fallback: try str-JSON
                    try:
                        parsed = json.loads(str(obj))
                        return _coerce_to_output_json(parsed)
                    except Exception:
                        return default

                parsed_json = _coerce_to_output_json(raw_content)
                parsed_output = output_model(**parsed_json)

            logger.debug(
                "Successfully parsed LLM response", extra={"task_id": self.task_id}
            )
            return parsed_output

        except Exception as e:
            logger.error(
                f"Failed to get action from LLM: {str(e)}",
                extra={"task_id": self.task_id},
                exc_info=True,
            )
            raise ValueError(f"Failed to get action from LLM: {str(e)}")

    def _build_messages(self) -> List[Any]:
        """Build the list of messages for the LLM."""
        messages = []

        # Add system message (include compiled core memory if present)
        system_content = self.system_message
        if getattr(self.memory, "core_memory", None):
            compiled = self.memory.core_memory.compile()
            if compiled:
                system_content = system_content + "\n\n" + compiled
        messages.append(SystemMessage(content=system_content))

        recent_structured = []
        try:
            if hasattr(self.memory, "get_token_limited_messages"):
                recent_structured = self.memory.get_token_limited_messages(
                    max_tokens=self.settings.context_window_tokens,
                    reserve_tokens=self.settings.context_window_reserve_tokens,
                    include_summary_message=True,
                )
            else:
                recent_structured = self.memory.get_recent_messages_structured(10)
        except Exception:
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

        # Add current state information
        context = self._get_current_context()
        messages.append(UserMessage(content=context))

        # Inject retrieved long-term knowledge via semantic recall
        try:
            # Prefer last user message as recall query if present
            try:
                recent_msgs = self.memory.get_recent_messages_structured(5)
                last_user = next(
                    (
                        m["content"]
                        for m in reversed(recent_msgs)
                        if m.get("role") == "user"
                    ),
                    self.task,
                )
                recall_query = last_user or self.task
            except Exception:
                recall_query = self.task

            retrieved = self.memory.recall(query=recall_query, k=5, min_score=0.35)
            if retrieved:
                # Filter out items that look like prior final answers
                filtered = []
                for _, text, _score in retrieved:
                    tx = str(text)
                    if '"final_answer"' in tx or "Final answer" in tx:
                        continue
                    filtered.append(tx)
                if filtered:
                    top = filtered[:3]
                    facts = "\n".join([f"- {t}" for t in top])
                    guidance = (
                        "Use relevant background facts below to answer the CURRENT question. "
                        "Do not copy prior final answers verbatim; verify dates/contexts."
                    )
                    try:
                        logger.info(
                            "Retrieved knowledge (top):\n" + "\n".join(top),
                            extra={"task_id": self.task_id},
                        )
                    except Exception:
                        pass
                    messages.append(
                        UserMessage(
                            content=f"Retrieved Knowledge (for reference):\n{facts}\n\n{guidance}"
                        )
                    )
        except Exception:
            # If memory backend lacks recall, skip silently
            pass

        return messages

    def _get_current_context(self) -> str:
        """Get the current context for the LLM."""
        # In a real implementation, this would include details about:
        # - Current state in the target domain
        # - Recent tool calls and results
        # - Progress toward the goal

        # For this minimal example, we'll just provide task status
        tasks_status = "\n".join(
            [
                f"Task '{task.description}': {task.status}"
                for task in self.memory.tasks.values()
            ]
        )

        recent_tools = "\n".join(
            [
                f"Tool '{t.tool_name}' called with {t.parameters} → {'Success' if t.success else 'Failed'}\n  Result: {t.result}"
                for t in self.memory.tool_call_history[-5:]  # Last 5 tool calls
            ]
        )

        # Check if we have tool results but no final answer yet
        has_tool_results = len(self.memory.tool_call_history) > 0
        has_done_call = any(
            t.tool_name == "done" for t in self.memory.tool_call_history
        )

        # Include current plan (trim to first 5 steps for brevity)
        plan_lines = (
            "\n".join([f"- {s}" for s in self.state.current_plan[:5]])
            if self.state.current_plan
            else "(no plan yet)"
        )

        context_prompt = f"""
Current Status:
{tasks_status}

Current Plan (follow these steps; revise only if needed):
{plan_lines}

Recent Actions:
{recent_tools}

Please decide on the next action to take to accomplish the task."""

        if has_tool_results and not has_done_call:
            context_prompt += """

REMINDER: You have gathered information using tools. Now analyze the results and use the "done" tool to provide your final answer to the user's question."""

        return context_prompt

    def _create_output_model(self) -> Type[BaseModel]:
        """Create the output model for the LLM's response."""
        return AgentOutput

    def _get_interrupt_config(self, tool_name: str) -> Optional[Any]:
        """Check if a tool requires HITL approval."""
        if not self.hitl_config or not self.hitl_config.enabled:
            return None

        cfg = resolve_interrupt_config(tool_name, self.hitl_config)
        if cfg is None:
            return None

        # Check if auto_approve_duplicates applies
        if self.hitl_config.auto_approve_duplicates:
            # We don't have the exact params here, so the actual check happens
            # in execute_actions where we know the params. But we can check
            # if we previously approved *this exact tool+params signature*.
            # The logic in execute_actions uses this helper just to see if the
            # tool generally requires approval, and then handles duplicates directly.
            pass

        return cfg

    async def _plan_if_needed(self) -> None:
        """Create a brief plan if none exists."""
        if self.state.current_plan:
            return

        plan_prompt = (
            "You are planning how to accomplish the user's task. "
            "Return 1-3 short steps. Each step should map directly to a tool call or the final 'answer' action. "
            "Do NOT describe internal operations that a tool already abstracts (e.g., loops or data structures). "
            "Reference tools by name with essentia."
        )

        messages = [
            SystemMessage(content="Plan the task succinctly."),
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
        except Exception as e:
            logger.debug(
                f"Plan generation failed: {e}", extra={"task_id": self.task_id}
            )

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
                new_steps = [
                    s.strip() for s in reflection.update_plan if s and s.strip()
                ]
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
        except Exception as e:
            logger.debug(f"Reflection failed: {e}", extra={"task_id": self.task_id})

    async def execute_actions(
        self,
        actions: List[Dict[str, Any]],
        agent_reasoning: Optional[Dict[str, str]] = None,
        step_metrics: Optional[StepMetrics] = None,
    ) -> List[ActionResult]:
        """Execute a list of actions."""
        logger.info(
            f"Executing {len(actions)} actions", extra={"task_id": self.task_id}
        )
        results = []
        # Track duplicates within the same step
        seen_signatures_this_step: set[str] = set()
        results_by_signature_this_step: Dict[str, Dict[str, Any]] = {}

        def _make_signature(tool: str, p: Dict[str, Any]) -> str:
            try:
                return f"{tool}:{json.dumps(p, sort_keys=True)}"
            except Exception:
                # Fallback to string repr if params not JSON-serializable
                return f"{tool}:{str(p)}"

        def _is_duplicate_this_step(tool: str, p: Dict[str, Any]) -> bool:
            sig = _make_signature(tool, p)
            return sig in seen_signatures_this_step

        def _find_recent_same_call_result(
            tool: str, p: Dict[str, Any], lookback: int = 5
        ) -> Optional[Dict[str, Any]]:
            """Return the most recent recorded result for the same tool+params within this task."""
            sig = _make_signature(tool, p)
            # Prefer same-step prior result if present
            if sig in results_by_signature_this_step:
                return results_by_signature_this_step[sig]
            # Search recent history across steps
            try:
                recent = self.memory.tool_call_history[-lookback:]
            except Exception:
                recent = []
            for rec in reversed(recent):
                try:
                    if (
                        getattr(rec, "tool_name", None) == tool
                        and _make_signature(tool, getattr(rec, "parameters", {})) == sig
                        and getattr(rec, "success", False)
                    ):
                        return getattr(rec, "result", None)
                except Exception:
                    continue
            return None

        for i, action_dict in enumerate(actions, 1):
            if not action_dict:
                logger.warning(
                    f"Empty action {i}, skipping", extra={"task_id": self.task_id}
                )
                continue

            # Extract tool name and parameters
            tool_name = list(action_dict.keys())[0]
            params = action_dict[tool_name]

            logger.info(
                f"Executing action {i}: {tool_name}", extra={"task_id": self.task_id}
            )
            logger.debug(f"Tool parameters: {params}", extra={"task_id": self.task_id})

            # Duplicate handling: reuse last identical result instead of re-running
            try:
                sig = _make_signature(tool_name, params)
                seen_before_this_step = _is_duplicate_this_step(tool_name, params)
                prior_result = _find_recent_same_call_result(
                    tool_name, params, lookback=5
                )
                if seen_before_this_step or prior_result is not None:
                    reused = prior_result or results_by_signature_this_step.get(sig)
                    if reused is not None:
                        logger.info(
                            f"Reusing previous result for duplicate action {i}: {tool_name}",
                            extra={"task_id": self.task_id},
                        )
                        # Record the tool call for traceability
                        try:
                            self.memory.add_tool_call(
                                tool_name=tool_name,
                                parameters=params,
                                result=reused,
                                success=True,
                            )
                        except Exception:
                            pass
                        # Evaluate and append reused result
                        action_result = self.evaluator.evaluate_tool_result(
                            tool_name=tool_name, result=reused
                        )
                        results.append(action_result)
                        # Ensure signature recorded for this step
                        seen_signatures_this_step.add(sig)
                        results_by_signature_this_step[sig] = reused
                        continue
                # First time seeing this signature this step
                seen_signatures_this_step.add(sig)
            except Exception:
                # If duplicate detection fails, proceed normally
                pass

            # Special handling for "done" tool
            if tool_name == "done":
                is_successful = params.get("success", True)
                message = params.get("message", "Task completed")

                logger.info(
                    f"Task marked as done - Success: {is_successful}",
                    extra={"task_id": self.task_id},
                )

                # Mark all tasks as complete
                for task in self.memory.tasks.values():
                    task.complete(success=is_successful)

                # Record the tool call so the evaluator knows the task is complete.
                self.memory.add_tool_call(
                    tool_name=tool_name,
                    parameters=params,
                    result={"message": message},
                    success=is_successful,
                )

                results.append(
                    ActionResult(
                        success=is_successful, value=message, include_in_memory=True
                    )
                )
                continue

            # For regular tools, execute and evaluate
            try:
                # --- HITL approval gate ---
                if self.hitl_handler and self.hitl_config and self.hitl_config.enabled:
                    interrupt_cfg = self._get_interrupt_config(tool_name)
                    if interrupt_cfg is not None:
                        # Auto-approve duplicates: skip if same tool+params was already approved
                        if self.hitl_config.auto_approve_duplicates:
                            sig = _make_signature(tool_name, params)
                            if sig in self._approved_signatures:
                                logger.info(
                                    f"HITL: auto-approved duplicate {tool_name}",
                                    extra={"task_id": self.task_id},
                                )
                                # Fall through to normal execution
                                interrupt_cfg = None

                    if interrupt_cfg is not None:
                        request = ApprovalRequest(
                            action_requests=[
                                ActionRequest(
                                    name=tool_name,
                                    arguments=params,
                                    description=(
                                        f"{interrupt_cfg.description or self.hitl_config.description_prefix}"
                                        + (
                                            f"\n\nAgent reasoning:"
                                            f"\n  Goal: {agent_reasoning.get('next_goal', 'N/A')}"
                                            f"\n  Context: {agent_reasoning.get('memory', 'N/A')}"
                                            if agent_reasoning
                                            else ""
                                        )
                                    ),
                                )
                            ],
                            review_configs=[
                                ReviewConfig(
                                    action_name=tool_name,
                                    allowed_decisions=interrupt_cfg.allowed_decisions,
                                )
                            ],
                            task_id=self.task_id,
                            step_number=self.state.n_steps,
                        )
                        response = await self.hitl_handler.request_approval(request)
                        decision = response.decisions[0]

                        if decision.type == "reject":
                            msg = decision.message or "Action rejected by human."
                            logger.info(
                                f"HITL: {tool_name} rejected — {msg}",
                                extra={"task_id": self.task_id},
                            )
                            self.memory.add_message(
                                "system", f"Human rejected {tool_name}: {msg}"
                            )
                            self.memory.add_tool_call(
                                tool_name=tool_name,
                                parameters=params,
                                result={"rejected": True, "message": msg},
                                success=False,
                            )
                            results.append(
                                ActionResult(
                                    success=False,
                                    error=f"Rejected by human: {msg}",
                                    include_in_memory=True,
                                )
                            )
                            continue

                        if decision.type == "edit" and decision.edited_action:
                            old_name = tool_name
                            tool_name = decision.edited_action.name
                            params = decision.edited_action.args
                            logger.info(
                                f"HITL: edited {old_name} → {tool_name} with new params",
                                extra={"task_id": self.task_id},
                            )

                        # Track approved signature for auto_approve_duplicates
                        try:
                            self._approved_signatures.add(
                                _make_signature(tool_name, params)
                            )
                        except Exception:
                            pass
                # --- end HITL gate ---

                start_time = datetime.now()

                # Execute the tool
                tool_context = {
                    "memory": self.memory,
                    "state": self.state,
                    "thread_id": self.task_id,
                    "sandbox_base_dir": self.sandbox_base_dir,
                }
                tool_result = self.tool_executor.execute_tool(
                    tool_name=tool_name,
                    params=params,
                    context=tool_context,
                )

                execution_time = (datetime.now() - start_time).total_seconds()
                success = tool_result.get("success", False)

                logger.info(
                    f"Tool {tool_name} executed in {execution_time:.2f}s - {'Success' if success else 'Failed'}",
                    extra={"task_id": self.task_id},
                )

                if not success:
                    logger.warning(
                        f"Tool {tool_name} failed: {tool_result.get('error', 'Unknown error')}",
                        extra={"task_id": self.task_id},
                    )

                # Record tool metrics for this step
                try:
                    if step_metrics is not None:
                        step_metrics.tool_calls.append(
                            ToolCallMetrics(
                                tool_name=tool_name,
                                duration_seconds=execution_time,
                                success=success,
                            )
                        )
                except Exception:
                    pass

                # Record the tool call
                self.memory.add_tool_call(
                    tool_name=tool_name,
                    parameters=params,
                    result=tool_result,
                    success=success,
                )

                # Cache the result for potential reuse within this step
                try:
                    results_by_signature_this_step[
                        _make_signature(tool_name, params)
                    ] = tool_result
                except Exception:
                    pass

                # Evaluate the result
                action_result = self.evaluator.evaluate_tool_result(
                    tool_name=tool_name, result=tool_result
                )

                results.append(action_result)

                # If failed and should retry, add info to memory
                if not action_result.success and self.evaluator.should_retry(tool_name):
                    retry_count = self.evaluator.retry_counts[tool_name]
                    logger.info(
                        f"Tool {tool_name} will be retried ({retry_count}/{self.settings.max_failures})",
                        extra={"task_id": self.task_id},
                    )
                    self.memory.add_message(
                        "system",
                        f"Tool '{tool_name}' failed. Retrying ({retry_count}/{self.settings.max_failures}).",
                    )

            except Exception as e:
                logger.error(
                    f"Error executing tool {tool_name}: {str(e)}",
                    extra={"task_id": self.task_id},
                    exc_info=True,
                )
                results.append(
                    ActionResult(
                        success=False,
                        error=f"Error executing tool '{tool_name}': {str(e)}",
                        include_in_memory=True,
                    )
                )

        logger.info(
            f"Completed executing {len(actions)} actions",
            extra={"task_id": self.task_id},
        )
        return results

    async def run(self) -> Dict[str, Any]:
        """Run the agent until the task is complete or max steps is reached."""
        logger.info(f"Starting agent run", extra={"task_id": self.task_id})
        print(f"Starting task: {self.task}")

        # Initialize run metrics
        self._run_metrics = RunMetrics(
            run_id=self.task_id,
            agent_id=self.task_id,
            agent_name=None,
            model_id=getattr(self.llm, "model", ""),
            model_provider=self.llm.__class__.__name__,
            start_time=datetime.now(),
        )

        # Initialize trace
        self._trace = Trace(
            name=f"Agent.run",
            run_id=self.task_id,
            agent_id=self.task_id,
            start_time=datetime.now(),
            input=self.task,
        )

        # Pre-check guardrails (input validation)
        for guardrail in self.guardrails:
            try:
                await guardrail.pre_check(self.task)
            except ValueError as e:
                logger.warning(
                    f"Input guardrail blocked: {e}", extra={"task_id": self.task_id}
                )
                return {
                    "task": self.task,
                    "completed": False,
                    "blocked_by": "input_guardrail",
                    "reason": str(e),
                    "steps_taken": 0,
                    "metrics": None,
                }

        for step_count in range(self.settings.max_steps):
            # Check control flags
            if self.state.stopped:
                logger.info("Agent stopped by request", extra={"task_id": self.task_id})
                print("Agent stopped")
                break

            if self.state.paused:
                logger.info("Agent paused", extra={"task_id": self.task_id})
                print("Agent paused")
                while self.state.paused and not self.state.stopped:
                    await asyncio.sleep(0.5)
                if self.state.stopped:
                    break
                logger.info("Agent resumed", extra={"task_id": self.task_id})

            # Check for too many consecutive failures
            if self.state.consecutive_failures >= self.settings.max_failures:
                logger.error(
                    f"Stopping due to {self.settings.max_failures} consecutive failures",
                    extra={"task_id": self.task_id},
                )
                print(
                    f"Stopping due to {self.settings.max_failures} consecutive failures"
                )
                break

            # Execute step
            await self.step()

            # Check if task is complete
            is_complete, completion_message = self.evaluator.evaluate_task_completion(
                self.task, self.memory
            )

            if is_complete:
                logger.info(
                    f"Task completed: {completion_message}",
                    extra={"task_id": self.task_id},
                )
                print(f"Task complete: {completion_message}")
                # Ensure tasks are marked complete so final result.completed is True
                for task in self.memory.tasks.values():
                    task.complete(success=True)
                break

            # Optional: validate task completion
            if self.settings.validate_output and is_complete:
                logger.debug(
                    "Validating task completion", extra={"task_id": self.task_id}
                )
                # In a real implementation, you would use the LLM to validate task completion
                pass

        # Check if we hit the step limit
        if self.state.n_steps >= self.settings.max_steps:
            logger.warning(
                f"Reached maximum steps ({self.settings.max_steps}) without completing task",
                extra={"task_id": self.task_id},
            )
            print(
                f"Reached maximum steps ({self.settings.max_steps}) without completing task"
            )

        completed = any(
            task.status == "completed" for task in self.memory.tasks.values()
        )

        # Finalize metrics
        if self._run_metrics is not None:
            try:
                self._run_metrics.end_time = datetime.now()
                self._run_metrics.finalize()
                summary = self._run_metrics.summary()
                logger.info(
                    "Run metrics summary: " + f"duration={summary['duration']}, "
                    f"steps={summary['steps']}, "
                    f"llm_calls={summary['llm_calls']}, "
                    f"tool_calls={summary['tool_calls']}",
                    extra={"task_id": self.task_id},
                )
            except Exception:
                pass

        # Post-check guardrails (output validation)
        if completed and self.guardrails:
            final_answer = self.memory.get_final_answer()
            if final_answer:
                output_text = str(final_answer.get("final_answer", ""))
                for guardrail in self.guardrails:
                    try:
                        await guardrail.post_check(self.task, output_text)
                    except ValueError as e:
                        logger.warning(
                            f"Output guardrail blocked: {e}",
                            extra={"task_id": self.task_id},
                        )
                        return {
                            "task": self.task,
                            "completed": False,
                            "blocked_by": "output_guardrail",
                            "reason": str(e),
                            "steps_taken": self.state.n_steps,
                            "metrics": (
                                self._run_metrics.summary()
                                if self._run_metrics is not None
                                else None
                            ),
                        }

        # Finalize trace
        if self._trace:
            final = self.memory.get_final_answer()
            self._trace.output = str(final.get("final_answer", "")) if final else None
            self._trace.finish()

        logger.info(
            f"Agent run finished - Completed: {completed}, Steps: {self.state.n_steps}",
            extra={"task_id": self.task_id},
        )

        return {
            "task": self.task,
            "completed": completed,
            "steps_taken": self.state.n_steps,
            "final_state": self.state.dict(),
            "metrics": (
                self._run_metrics.summary() if self._run_metrics is not None else None
            ),
            "trace": self._trace.to_dict() if self._trace else None,
        }

    def pause(self) -> None:
        """Pause the agent."""
        logger.info("Agent paused", extra={"task_id": self.task_id})
        self.state.paused = True

    def resume(self) -> None:
        """Resume the agent."""
        logger.info("Agent resumed", extra={"task_id": self.task_id})
        self.state.paused = False

    def stop(self) -> None:
        """Stop the agent."""
        logger.info("Agent stopped", extra={"task_id": self.task_id})
        self.state.stopped = True
