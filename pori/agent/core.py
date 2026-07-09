import asyncio
import inspect
import json
import logging
import re
import uuid
from datetime import datetime
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

from pydantic import BaseModel, Field

from pori.llm import (
    AssistantMessage,
    BaseChatModel,
    BaseMessage,
    DocumentBlock,
    ImageBlock,
    SystemMessage,
    UserMessage,
    normalize_usage,
)
from pori.llm.error_classifier import classify_error
from pori.llm.model_context import get_model_context_length

from ..compression import compress_context
from ..context import ContextDiagnostics, ContextEngine, DefaultContextEngine
from ..evaluation import ActionResult, Evaluator
from ..evolution import EvolutionRepository
from ..hitl import (
    ActionRequest,
    ApprovalRequest,
    ApprovalResponse,
    HITLConfig,
    HITLHandler,
    ReviewConfig,
    resolve_interrupt_config,
)
from ..memory import AgentMemory
from ..metrics import (
    LLMCallMetrics,
    RunMetrics,
    StepMetrics,
    TokenUsage,
    ToolCallMetrics,
    estimate_llm_call_cost,
)
from ..observability import (
    RUN_END,
    RUN_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    PoriEvent,
)
from ..observability.trace import Span, SpanStatus, SpanType, Trace
from ..planning import PlanStore
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
from ..tool_guardrails import ToolCallGuardrailController
from ..tools.policy import AuthorizationDecision, ToolAuthorizationPolicy
from ..tools.registry import ToolExecutor, ToolRegistry
from ..utils.action_decode import decode_action_envelope
from ..utils.logging_config import ensure_logger_configured
from ..utils.prompt_loader import load_prompt

# Set up logger for this module - this will work regardless of import order
logger = ensure_logger_configured("pori.agent")

from . import artifacts as _artifacts
from . import authorization as _authz
from . import planning as _planning
from . import prompting as _prompting
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


class Agent:
    """
    A general-purpose agent that can perform tasks using tools and have memory.

    """

    # Cohesive method groups extracted to sibling modules for readability and
    # bound back onto the class (they take `self`); see agent/artifacts.py,
    # agent/authorization.py.
    _record_tool_receipt = _artifacts._record_tool_receipt
    _extract_tool_artifacts = _artifacts._extract_tool_artifacts
    _run_artifacts = _artifacts._run_artifacts
    _runtime_fact_summary = _artifacts._runtime_fact_summary
    _artifact_reference_errors = _artifacts._artifact_reference_errors
    _tool_side_effects = _authz._tool_side_effects
    _authorize_side_effects = _authz._authorize_side_effects
    _memory_deletion_forbidden_terms = _authz._memory_deletion_forbidden_terms
    _params_write_forbidden_memory_terms = _authz._params_write_forbidden_memory_terms
    _get_interrupt_config = _authz._get_interrupt_config
    _setup_system_message = _prompting._setup_system_message
    _render_available_skills_prompt = _prompting._render_available_skills_prompt
    _build_messages = _prompting._build_messages
    _capture_retrieval_evidence = _prompting._capture_retrieval_evidence
    _get_current_context = _prompting._get_current_context
    _render_plan_for_prompt = _prompting._render_plan_for_prompt
    _should_plan = _planning._should_plan
    _should_reflect = _planning._should_reflect
    _task_looks_complex = _planning._task_looks_complex
    _plan_if_needed = _planning._plan_if_needed
    _reflect_and_update_plan = _planning._reflect_and_update_plan

    # Lazily set in _build_messages (now prompting.py); declared here so it's a
    # known Agent attribute after that method was extracted.
    prompt_fingerprint: str = ""

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
        system_prompt: Optional[str] = None,
        run_context: Optional[RunContext] = None,
        context_engine: Optional[ContextEngine] = None,
        skill_catalog: Optional[SkillCatalog] = None,
        skill_limit: int = 3,
        selected_skill_ids: Optional[List[str]] = None,
        model_capabilities: Optional[frozenset[str]] = None,
        budget_ledger: Optional[BudgetLedger] = None,
        cancellation_token: Optional[CancellationToken] = None,
        evolution_repository: Optional[EvolutionRepository] = None,
        tool_authorization_policy: Optional[ToolAuthorizationPolicy] = None,
        soul_path: Optional[str] = None,
        soul_text: Optional[str] = None,
        load_project_context: bool = False,
        tool_context_extra: Optional[Dict[str, Any]] = None,
        resume_task_id: Optional[str] = None,
        task_attachments: Optional[List[Union["ImageBlock", "DocumentBlock"]]] = None,
    ):
        # Generate unique task ID for tracking (also used as thread_id for sandbox).
        # A caller may pass resume_task_id to adopt an existing (interrupted)
        # task instead: the loop position, plan, and history are restored from
        # the memory store's per-step checkpoint and the run continues from
        # where it stopped rather than from step 0.
        self.task_id = resume_task_id or str(uuid.uuid4())[:8]
        self._resuming = resume_task_id is not None
        # Attachments for THIS task (images/PDF documents) — they ride with
        # the CURRENT TASK message so every provider adapter maps them natively.
        self.task_attachments = list(task_attachments or [])
        # Extra keys merged into every tool's context (e.g. a clarify_handler that
        # renders options as gateway buttons instead of a CLI menu).
        self._tool_context_extra = dict(tool_context_extra or {})
        if run_context is not None:
            self.run_context = run_context
        elif isinstance(memory, AgentMemory):
            self.run_context = RunContext(
                organization_id=memory.organization_id,
                user_id=memory.user_id,
                agent_id=memory.agent_id,
                session_id=memory.session_id,
                run_id=self.task_id,
            )
        else:
            self.run_context = RunContext.local(
                run_id=self.task_id,
                agent_id=self.task_id,
            )
        self.execution_receipts: List[ToolExecutionReceipt] = []
        self.sandbox_base_dir = sandbox_base_dir

        # Human-in-the-loop
        self.hitl_handler = hitl_handler
        self.hitl_config = hitl_config
        self._approved_signatures: set = set()  # for auto_approve_duplicates
        # Bounded counter for opt-in output validation (see settings.validate_output)
        self._completion_validation_attempts = 0

        # Guardrails (BaseEval instances with pre_check/post_check)
        self.guardrails = guardrails or []

        logger.info(f"Initializing new agent", extra={"task_id": self.task_id})
        logger.info(f"Task: {task}", extra={"task_id": self.task_id})
        if hitl_handler and hitl_config and hitl_config.enabled:
            logger.info("HITL approval gates enabled", extra={"task_id": self.task_id})

        self.task = task
        self.llm = llm
        self.capability_snapshot = tools_registry.snapshot()
        self.tools_registry = self.capability_snapshot.to_registry()
        self._custom_system_prompt = system_prompt
        self._soul_path = soul_path
        self._soul_text = soul_text
        self._load_project_context = load_project_context
        # Optional event consumer + whether to ask the provider to stream; both
        # set per-run via run(). Decoupled: you can consume events (e.g. for a
        # JSONL audit trail) without forcing token streaming.
        self._on_event: Optional[Callable[[Any], None]] = None
        self._stream: bool = False
        self.tool_executor = ToolExecutor(self.tools_registry)
        self.tool_guardrails = ToolCallGuardrailController()
        self.tool_surface_fingerprint = self.capability_snapshot.fingerprint
        self.settings = settings

        # Model-aware context sizing: unless disabled, size the history budget to
        # the model's real context length so large-context models use their
        # capacity and compression only fires on genuine overflow. Work on a copy
        # so a settings object shared across agents (e.g. a Team) isn't mutated.
        if getattr(self.settings, "context_window_auto", False):
            self.settings = self.settings.model_copy()
            model_ctx = get_model_context_length(getattr(self.llm, "model", "") or "")
            output_reserve = getattr(self.llm, "max_tokens", 4096) or 4096
            self.settings.context_window_tokens = model_ctx
            # Reserve room for the reply plus headroom for the system prompt and
            # tool schemas, which sit outside the counted history window.
            self.settings.context_window_reserve_tokens = max(
                self.settings.context_window_reserve_tokens, int(output_reserve) + 8000
            )

        logger.info(
            f"Agent settings: max_steps={settings.max_steps}, max_failures={settings.max_failures}",
            extra={"task_id": self.task_id},
        )

        # Initialize state components
        self.state = AgentState()
        if isinstance(memory, AgentMemory):
            expected_scope = (
                self.run_context.organization_id,
                self.run_context.user_id,
                self.run_context.agent_id,
                self.run_context.session_id,
            )
            actual_scope = (
                memory.organization_id,
                memory.user_id,
                memory.agent_id,
                memory.session_id,
            )
            if actual_scope != expected_scope:
                raise ValueError(
                    "AgentMemory scope must exactly match RunContext scope"
                )
            self.memory = memory
        elif memory is not None:
            self.memory = memory
        else:
            self.memory = AgentMemory(
                organization_id=self.run_context.organization_id,
                user_id=self.run_context.user_id,
                agent_id=self.run_context.agent_id,
                session_id=self.run_context.session_id,
            )
        self.context_engine = context_engine or DefaultContextEngine()
        self.context_diagnostics: Optional[ContextDiagnostics] = None
        self.skill_catalog = skill_catalog
        self.skill_limit = skill_limit
        self.model_capabilities = model_capabilities or frozenset()
        self.budget_ledger = budget_ledger or BudgetLedger(self.run_context.budget)
        self.cancellation_token = cancellation_token or CancellationToken()
        self.evolution_repository = evolution_repository
        self.tool_authorization_policy = (
            tool_authorization_policy or ToolAuthorizationPolicy()
        )
        # Model-owned, run-scoped todo list (maintained via the update_plan tool).
        self.plan_store = PlanStore()
        self.selected_skill_summaries: Tuple[SkillSummary, ...] = ()
        self.selected_skills: Tuple[SelectedSkill, ...] = ()
        if self.skill_catalog is not None and selected_skill_ids:
            self.selected_skill_summaries = self.skill_catalog.select(
                task,
                self.capability_snapshot,
                model_capabilities=self.model_capabilities,
                limit=skill_limit,
                explicit_skill_ids=selected_skill_ids,
            )
            self.selected_skills = self.skill_catalog.load_selected(
                self.selected_skill_summaries
            )
        self._frozen_core_memory = (
            self.memory.core_memory.compile()
            if getattr(self.memory, "core_memory", None)
            else ""
        )
        self._frozen_retrieval_evidence = self._capture_retrieval_evidence(task)
        self.evaluator = Evaluator(max_retries=settings.max_failures)

        # Metrics and trace for this agent run (initialized in run())
        self._run_metrics: Optional[RunMetrics] = None
        self._trace: Optional[Trace] = None
        self._current_step_span: Optional[Span] = None

        # Create (or, when resuming, re-adopt) the task record
        resumed_task = self.memory.tasks.get(self.task_id) if self._resuming else None
        if resumed_task is not None:
            if resumed_task.status != "in_progress":
                raise ValueError(
                    f"Cannot resume task {self.task_id}: "
                    f"status is '{resumed_task.status}', expected 'in_progress'"
                )
            # Restore the loop position and plan from the last step checkpoint.
            self.state.n_steps = resumed_task.n_steps
            self.state.consecutive_failures = resumed_task.consecutive_failures
            self.state.current_activity = resumed_task.current_activity
            if resumed_task.plan:
                self.plan_store.write(resumed_task.plan)
            self.memory.begin_task(self.task_id)
            # Surface write-ahead journal entries that never completed: the
            # previous process died mid-tool, so those side effects are unknown.
            interrupted = self.memory.pending_dispatches(self.task_id)
            if interrupted:
                names = ", ".join(sorted({r.tool_name for r in interrupted}))
                self.memory.add_message(
                    "system",
                    f"[resume] The previous run was interrupted while executing: "
                    f"{names}. Those side effects may or may not have taken "
                    f"place — verify their outcome before re-running them.",
                )
            logger.info(
                f"Resuming task from step {self.state.n_steps}",
                extra={"task_id": self.task_id},
            )
        else:
            self.memory.create_task(self.task_id, task)
            logger.info(
                f"Created task record in memory", extra={"task_id": self.task_id}
            )

        # Set up system message
        self._setup_system_message()

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

            # Planning adds an LLM call before acting. In auto mode it is used
            # only for tasks that look complex enough to benefit from it.
            if self._should_plan():
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
            # AC-3: summarize context that would overflow the window before it is
            # dropped, so long tasks don't silently lose information (fail-open).
            if self.settings.compress_context:
                await compress_context(
                    self.memory,
                    self.llm,
                    max_tokens=self.settings.context_window_tokens,
                    reserve_tokens=self.settings.context_window_reserve_tokens,
                )
            llm_start_time = datetime.now()
            model_output = await self.get_next_action()
            llm_duration = (datetime.now() - llm_start_time).total_seconds()

            # Capture the model's intent for this step as the live activity line.
            next_goal = (model_output.current_state or {}).get("next_goal", "")
            if next_goal and next_goal.strip():
                self.state.current_activity = next_goal.strip()
            if llm_span:
                llm_span.attributes["duration_seconds"] = llm_duration
                llm_span.finish()

            # Record LLM call metrics for this step
            try:
                # Token usage: normalize the provider-shaped dict in the llm
                # layer (AC-4) so the loop never branches on Anthropic vs
                # OpenAI/Google token keys.
                usage = normalize_usage(getattr(self.llm, "last_usage", None))
                tokens = TokenUsage()
                tokens.input_tokens = usage.input_tokens
                tokens.output_tokens = usage.output_tokens
                tokens.total_tokens = usage.total_tokens
                tokens.cache_read_tokens = usage.cache_read_tokens
                tokens.cache_write_tokens = usage.cache_write_tokens

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
                self.budget_ledger.consume_tokens(tokens.total_tokens)
                if cost is not None:
                    self.budget_ledger.consume_cost(cost)
            except BudgetExceeded:
                raise
            except Exception as metrics_err:
                logger.debug(
                    f"Failed to record LLM call metrics: {metrics_err}",
                    extra={"task_id": self.task_id, "step": step_number},
                )

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

            # Reflection adds an LLM call after acting. In auto mode it is only
            # used for failed/unclear progress, and never after a final answer.
            if self._should_reflect(tool_results):
                try:
                    await self._reflect_and_update_plan(tool_results)
                except Exception as reflect_err:
                    logger.debug(
                        f"Reflection skipped/failed: {reflect_err}",
                        extra={"task_id": self.task_id, "step": step_number},
                    )
            else:
                logger.debug(
                    "Skipping reflection: disabled or task is complete",
                    extra={"task_id": self.task_id, "step": step_number},
                )

        except FatalAgentError as fatal:
            logger.error(
                f"Fatal error during step {step_number}: {fatal}",
                extra={"task_id": self.task_id, "step": step_number},
                exc_info=True,
            )
            if step_span:
                step_span.finish(SpanStatus.ERROR, error=str(fatal))
            tool_results = [ActionResult(success=False, error=f"Fatal error: {fatal}")]
            # Unrecoverable (auth/billing): halt promptly instead of retrying an
            # identical hopeless call for max_failures steps (AC-2).
            self.state.consecutive_failures = self.settings.max_failures
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
        except Exception as metrics_err:
            logger.debug(
                f"Failed to record step metrics: {metrics_err}",
                extra={"task_id": self.task_id, "step": step_number},
            )

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
                except Exception as state_err:
                    logger.debug(
                        f"Failed to update final_answer state: {state_err}",
                        extra={"task_id": self.task_id},
                    )
                # Skip indexing intermediate step results — they add noise.
                # Only task descriptions and final answers are stored as experiences.

        # Checkpoint the loop position on the task record so an interrupted run
        # can resume from here instead of step 0 (fail-open).
        try:
            self.memory.checkpoint_task_progress(
                self.task_id,
                n_steps=self.state.n_steps,
                consecutive_failures=self.state.consecutive_failures,
                current_activity=self.state.current_activity,
                plan=self._plan_snapshot(),
            )
        except Exception as checkpoint_err:
            logger.debug(
                f"Failed to checkpoint task progress: {checkpoint_err}",
                extra={"task_id": self.task_id},
            )

    async def _invoke_for_action(self) -> AgentOutput:
        """Build messages, call the model, and parse one action decision."""
        messages = self._build_messages()
        logger.debug(
            f"Built {len(messages)} messages for LLM",
            extra={"task_id": self.task_id},
        )
        tools = self.tools_registry.tool_schemas()
        # Only ask the provider to stream (and pass on_event to it) when
        # streaming is on, so the default call signature is unchanged for
        # non-streaming callers/mocks — even if an event consumer is attached.
        if self._stream and self._on_event is not None:
            turn = await self.llm.ainvoke_tools(
                messages, tools, on_event=self._on_event
            )
        else:
            turn = await self.llm.ainvoke_tools(messages, tools)
        action: List[Dict[str, Any]] = [
            {call.name: dict(call.arguments)} for call in turn.tool_calls if call.name
        ]
        text = turn.text.strip()
        if not action and text:
            # No native tool call. The model may have emitted its action as a
            # JSON envelope of text (some providers/turns do this) — decode and
            # run it. Only a genuine prose reply falls through to `answer`, so
            # the run reaches a real answer instead of freezing raw JSON.
            decoded = decode_action_envelope(text)
            if decoded is not None:
                action = decoded
            else:
                action = [{"answer": {"final_answer": text, "reasoning": ""}}]
        current_state: Dict[str, str] = {"next_goal": turn.text} if turn.text else {}
        return AgentOutput(current_state=current_state, action=action)

    async def get_next_action(self) -> AgentOutput:
        """Get the next action from the LLM via native tool-calling.

        The provider returns real tool calls plus assistant text; each tool call
        maps to the internal {name: arguments} action dict, and the text becomes
        the activity line (next_goal). Provider errors are classified (AC-2): a
        context-overflow triggers compress-and-retry-once; an auth/billing
        failure raises ``FatalAgentError`` to stop the run promptly; anything
        else re-raises as before.
        """
        try:
            return await self._invoke_for_action()
        except Exception as e:
            classified = classify_error(e)
            if classified.should_compress:
                # Request too big to send unchanged: compress dropped history
                # (AC-3) and retry once before giving up.
                try:
                    await compress_context(
                        self.memory,
                        self.llm,
                        max_tokens=self.settings.context_window_tokens,
                        reserve_tokens=self.settings.context_window_reserve_tokens,
                    )
                    return await self._invoke_for_action()
                except Exception:
                    pass  # fall through to the normal failure handling
            if classified.should_fail_fast:
                logger.error(
                    f"Fatal LLM error ({classified.reason.value}); stopping run: {e}",
                    extra={"task_id": self.task_id},
                )
                raise FatalAgentError(classified.reason.value, str(e))
            logger.error(
                f"Failed to get action from LLM: {str(e)}",
                extra={"task_id": self.task_id},
                exc_info=True,
            )
            raise ValueError(f"Failed to get action from LLM: {str(e)}")

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

    def _reject_action(
        self,
        tool_name: str,
        params: Any,
        results: List[ActionResult],
        reject_msg: str,
        *,
        log_level: str = "warning",
    ) -> None:
        """Record a rejected tool action and append a failed ActionResult.

        Logs the rejection, persists a tool_call record for traceability, and
        appends a failed ActionResult. The caller is responsible for
        ``continue``-ing the action loop after invoking this.
        """
        log = getattr(logger, log_level, logger.warning)
        log(reject_msg, extra={"task_id": self.task_id})
        self.memory.add_tool_call(
            tool_name=tool_name,
            parameters=params,
            result={"rejected": True, "message": reject_msg},
            success=False,
        )
        self._record_tool_receipt(
            tool_name,
            params,
            ReceiptStatus.REJECTED,
            error=reject_msg,
        )
        results.append(
            ActionResult(success=False, error=reject_msg, include_in_memory=True)
        )

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
            """Return the most recent recorded result for the same tool+params within this task.

            Terminal tools (answer, done) are never eligible for reuse.
            Only results from the current task are considered.
            """
            # Never reuse terminal tools
            if tool in ("answer", "done"):
                return None
            sig = _make_signature(tool, p)
            # Prefer same-step prior result if present
            if sig in results_by_signature_this_step:
                return results_by_signature_this_step[sig]
            # Search recent history across steps, but only within the current task
            try:
                recent = [
                    tc
                    for tc in self.memory.tool_call_history[-lookback:]
                    if getattr(tc, "task_id", None) == self.task_id
                ]
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
            # Write-ahead journal id for this action (set just before execution)
            dispatch_id: Optional[str] = None

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
                        except Exception as e:
                            logger.debug(
                                f"Failed to record reused tool call for {tool_name}: {e}",
                                extra={"task_id": self.task_id},
                            )
                        # Evaluate and append reused result
                        action_result = self.evaluator.evaluate_tool_result(
                            tool_name=tool_name, result=reused
                        )
                        results.append(action_result)
                        # Ensure signature recorded for this step
                        seen_signatures_this_step.add(sig)
                        results_by_signature_this_step[sig] = reused
                        self._record_tool_receipt(
                            tool_name,
                            params,
                            ReceiptStatus.REUSED,
                            metadata={"reason": "duplicate"},
                        )
                        continue
                # First time seeing this signature this step
                seen_signatures_this_step.add(sig)
            except Exception as e:
                # If duplicate detection fails, proceed normally
                logger.debug(
                    f"Duplicate detection failed for {tool_name}, proceeding: {e}",
                    extra={"task_id": self.task_id},
                )

            # Special handling for "done" tool
            if tool_name == "done":
                is_successful = params.get("success", True)
                message = params.get("message", "Task completed")

                # Reject done(success=True) if no final_answer exists yet
                if is_successful and self.memory.get_state("final_answer") is None:
                    reject_msg = (
                        "Cannot mark task done with success=True before providing "
                        "a final answer via the 'answer' tool."
                    )
                    logger.warning(reject_msg, extra={"task_id": self.task_id})
                    self.memory.add_tool_call(
                        tool_name=tool_name,
                        parameters=params,
                        result={"rejected": True, "message": reject_msg},
                        success=False,
                    )
                    results.append(
                        ActionResult(
                            success=False, error=reject_msg, include_in_memory=True
                        )
                    )
                    self._record_tool_receipt(
                        tool_name,
                        params,
                        ReceiptStatus.REJECTED,
                        error=reject_msg,
                    )
                    continue

                logger.info(
                    f"Task marked as done - Success: {is_successful}",
                    extra={"task_id": self.task_id},
                )

                # Mark only the current task as complete, not all tasks
                current_task = self.memory.tasks.get(self.task_id)
                if current_task:
                    current_task.complete(success=is_successful)

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
                self._record_tool_receipt(
                    tool_name,
                    params,
                    ReceiptStatus.SUCCEEDED if is_successful else ReceiptStatus.FAILED,
                    error=None if is_successful else message,
                )
                continue

            # For regular tools, execute and evaluate
            try:
                authorization = self._authorize_side_effects(tool_name)
                if not authorization.allowed:
                    self._reject_action(
                        tool_name, params, results, authorization.reason
                    )
                    continue

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
                            self._record_tool_receipt(
                                tool_name,
                                params,
                                ReceiptStatus.REJECTED,
                                error=f"Rejected by human: {msg}",
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
                        except Exception as e:
                            logger.debug(
                                f"Failed to track approved signature for {tool_name}: {e}",
                                extra={"task_id": self.task_id},
                            )
                # --- end HITL gate ---

                # --- Memory deletion guards ---
                # For deletion tasks, reject append/insert tools that cannot satisfy deletion
                if self._task_requests_memory_mutation():
                    if tool_name in ("core_memory_append", "memory_insert"):
                        reject_msg = (
                            f"Tool '{tool_name}' cannot satisfy a memory deletion request. "
                            "Use core_memory_replace, memory_rethink, or core_memory_rethink instead."
                        )
                        self._reject_action(tool_name, params, results, reject_msg)
                        continue

                # Reject memory tools that attempt to write forbidden terms back
                memory_tools = {
                    "core_memory_append",
                    "core_memory_replace",
                    "memory_insert",
                    "memory_rethink",
                    "core_memory_rethink",
                }
                if tool_name in memory_tools:
                    if self._params_write_forbidden_memory_terms(tool_name, params):
                        reject_msg = f"Tool '{tool_name}' would reintroduce terms the user asked to remove."
                        self._reject_action(tool_name, params, results, reject_msg)
                        continue

                # Guard for answer: reject false deletion confirmations
                if tool_name == "answer":
                    if self._task_requests_memory_mutation():
                        if not self._current_task_has_core_memory_mutation():
                            reject_msg = (
                                "Cannot confirm memory deletion without first successfully "
                                "using a core memory mutation tool (core_memory_replace, "
                                "memory_rethink, or core_memory_rethink)."
                            )
                            self._reject_action(tool_name, params, results, reject_msg)
                            continue
                # --- end Memory deletion guards ---

                # --- Completion quality gate (answer tool) ---
                if tool_name == "answer":
                    answer_text = ""
                    if isinstance(params, dict):
                        raw = params.get("final_answer")
                        if isinstance(raw, str):
                            answer_text = raw.strip()

                    # 1) Deterministic (always on): never accept an empty answer.
                    if not answer_text:
                        reject_msg = (
                            "The 'answer' tool requires a non-empty final_answer. "
                            "Provide a substantive answer to the user's question."
                        )
                        self._reject_action(tool_name, params, results, reject_msg)
                        continue

                    required_skill_id = self._required_skill_view_before_answer()
                    if required_skill_id:
                        reject_msg = (
                            "This task matches an available skill workflow. Load "
                            f"{required_skill_id} with skill_view before answering, "
                            "then follow the loaded skill instructions."
                        )
                        self._reject_action(tool_name, params, results, reject_msg)
                        continue

                    artifact_reference_errors = self._artifact_reference_errors(
                        params.get("artifact_references")
                        if isinstance(params, dict)
                        else None
                    )
                    if artifact_reference_errors:
                        reject_msg = (
                            "Invalid artifact_references in final answer. "
                            "Only reference artifacts present in Runtime Facts: "
                            + "; ".join(artifact_reference_errors)
                        )
                        self._reject_action(tool_name, params, results, reject_msg)
                        continue

                    # 2) Semantic (opt-in via validate_output): LLM judges adequacy,
                    #    bounded by max_validation_retries to avoid loops.
                    if (
                        self.settings.validate_output
                        and self._completion_validation_attempts
                        < self.settings.max_validation_retries
                    ):
                        adequate, reason = await self._validate_answer_text(answer_text)
                        if not adequate:
                            self._completion_validation_attempts += 1
                            reject_msg = (
                                "Answer rejected by output validation "
                                f"({self._completion_validation_attempts}/"
                                f"{self.settings.max_validation_retries}): "
                                f"{reason or 'inadequate'}. Revise and call 'answer' "
                                "again with a corrected response."
                            )
                            self._reject_action(
                                tool_name, params, results, reject_msg, log_level="info"
                            )
                            continue
                # --- end Completion quality gate ---

                start_time = datetime.now()
                # Announce the tool with its full args, so the renderer can show a
                # specific label ("Writing age_calculator.py", not "Writing a file").
                self._emit(
                    TOOL_CALL_START,
                    {
                        "name": tool_name,
                        "args": params if isinstance(params, dict) else {},
                    },
                )

                # Execute the tool
                tool_context = {
                    "memory": self.memory,
                    "state": self.state,
                    "thread_id": self._sandbox_thread_id(),
                    "sandbox_base_dir": self.sandbox_base_dir,
                    "run_context": self.run_context,
                    "evolution_repository": self.evolution_repository,
                    "skill_catalog": self.skill_catalog,
                    "capability_snapshot": self.capability_snapshot,
                    "tools_registry": self.tools_registry,
                    "plan_store": self.plan_store,
                    "task_id": self.task_id,
                    **self._tool_context_extra,
                }
                # Write-ahead journal: persist the dispatch BEFORE side effects
                # run, so a crash mid-tool is distinguishable on resume from a
                # call that never happened (fail-open).
                try:
                    dispatch_id = self.memory.record_tool_dispatch(
                        tool_name, params if isinstance(params, dict) else {}
                    )
                except Exception as dispatch_err:
                    logger.debug(
                        f"Failed to journal dispatch for {tool_name}: {dispatch_err}",
                        extra={"task_id": self.task_id},
                    )

                tool_result = self.tool_executor.execute_tool(
                    tool_name=tool_name,
                    params=params,
                    context=tool_context,
                )

                execution_time = (datetime.now() - start_time).total_seconds()
                success = tool_result.get("success", False)
                # Emit tool_call_end so a renderer can close the "» ..." line
                # with a "✓ / ✗" as soon as the tool finishes.
                self._emit(TOOL_CALL_END, {"name": tool_name, "success": bool(success)})

                # AC-5: cross-step loop guard. Surface a recovery hint on the tool
                # output; halt the run on a detected loop instead of spinning to
                # max_steps.
                if self.settings.tool_loop_guardrail:
                    guard = self.tool_guardrails.after_call(
                        tool_name, params, bool(success), tool_result
                    )
                    if guard is not None:
                        note = f"\n\n[loop-guard] {guard.guidance}"
                        if isinstance(tool_result.get("error"), str):
                            tool_result["error"] += note
                        elif isinstance(tool_result.get("output"), str):
                            tool_result["output"] += note
                        else:
                            tool_result["guardrail"] = guard.guidance
                        if guard.action == "halt":
                            logger.warning(
                                f"Tool loop-guard halt ({guard.reason}) on {tool_name}",
                                extra={"task_id": self.task_id},
                            )
                            self.state.consecutive_failures = self.settings.max_failures

                artifacts = self._extract_tool_artifacts(tool_name, params, tool_result)
                self._record_tool_receipt(
                    tool_name,
                    params,
                    ReceiptStatus.SUCCEEDED if success else ReceiptStatus.FAILED,
                    started_at=start_time,
                    duration_seconds=execution_time,
                    error=tool_result.get("error") if not success else None,
                    artifacts=artifacts,
                    metadata={"backend": self.tool_executor.__class__.__name__},
                )

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
                except Exception as e:
                    logger.debug(
                        f"Failed to record tool metrics for {tool_name}: {e}",
                        extra={"task_id": self.task_id},
                    )

                # Complete the write-ahead journal entry with the real result
                if dispatch_id is not None:
                    self.memory.complete_tool_dispatch(
                        dispatch_id, tool_result, success
                    )
                else:
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
                except Exception as e:
                    logger.debug(
                        f"Failed to cache result for {tool_name}: {e}",
                        extra={"task_id": self.task_id},
                    )

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
                # Close the write-ahead journal entry: the process survived, so
                # the outcome is known (failed) — only a real crash may leave a
                # record in "dispatched".
                if dispatch_id is not None:
                    try:
                        self.memory.complete_tool_dispatch(
                            dispatch_id, {"error": str(e)}, success=False
                        )
                    except Exception as journal_err:
                        logger.debug(
                            f"Failed to close dispatch journal: {journal_err}",
                            extra={"task_id": self.task_id},
                        )
                results.append(
                    ActionResult(
                        success=False,
                        error=f"Error executing tool '{tool_name}': {str(e)}",
                        include_in_memory=True,
                    )
                )
                self._record_tool_receipt(
                    tool_name,
                    params,
                    ReceiptStatus.FAILED,
                    error=str(e),
                )

        logger.info(
            f"Completed executing {len(actions)} actions",
            extra={"task_id": self.task_id},
        )
        return results

    async def _invoke_step_callback(
        self,
        callback: Optional[Callable[["Agent"], Union[Any, Awaitable[Any]]]],
    ) -> None:
        """Invoke a step lifecycle callback, tolerating sync or async callables.

        The callback receives this Agent instance so observers (e.g. SSE
        streaming in Pori Cloud, the CLI, custom monitors) can read live state:
        `agent.state`, `agent.memory.tool_call_history`, the latest
        `agent._run_metrics.steps[-1]`, current plan, reflection, etc. A failing
        callback must never crash the agent run, so errors are logged and
        swallowed.
        """
        if callback is None:
            return
        try:
            result = callback(self)
            if inspect.isawaitable(result):
                await result
        except Exception as cb_err:
            logger.debug(
                f"Step callback raised, ignoring: {cb_err}",
                extra={"task_id": self.task_id},
            )

    def _emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit a normalized PoriEvent to the subscribed consumer, if any."""
        if self._on_event is None:
            return
        try:
            self._on_event(
                PoriEvent(
                    event_type,
                    payload or {},
                    step=getattr(self.state, "n_steps", 0),
                )
            )
        except Exception:
            # A consumer error must never break the run.
            pass

    async def run(
        self,
        on_step_start: Optional[Callable[["Agent"], Union[Any, Awaitable[Any]]]] = None,
        on_step_end: Optional[Callable[["Agent"], Union[Any, Awaitable[Any]]]] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Run the agent until the task is complete or max steps is reached.

        Args:
            on_step_start: Optional callback invoked immediately before each
                step executes. Receives this Agent instance. May be sync or
                async. Exceptions are logged and ignored.
            on_step_end: Optional callback invoked immediately after each step
                executes (including any planning/reflection within the step).
                Receives this Agent instance. May be sync or async. Exceptions
                are logged and ignored.
            on_event: Optional sink for normalized ``PoriEvent``s (text_delta,
                tool_call_start, ...). When provided, the agent asks the provider
                to stream and forwards events here; when None the LLM call is
                non-streaming (default).
        """
        self._on_event = on_event
        self._stream = stream
        # Arm the wall-clock budget (max_duration_seconds); idempotent, so a
        # shared ledger measures from the first run that starts.
        self.budget_ledger.start_clock()
        self._emit(RUN_START, {"task": self.task, "task_id": self.task_id})
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
            run_id=self.run_context.run_id,
            session_id=self.run_context.session_id,
            agent_id=self.run_context.agent_id,
            run_context=self.run_context.model_dump(mode="json"),
            prompt_fingerprint=self.prompt_fingerprint,
            tool_surface_fingerprint=self.tool_surface_fingerprint,
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

        # Budget the loop by remaining steps so a resumed run continues from its
        # checkpoint instead of getting max_steps all over again. Fresh runs
        # (n_steps == 0) behave exactly as before.
        remaining_steps = max(0, self.settings.max_steps - self.state.n_steps)
        for _ in range(remaining_steps):
            # Check control flags
            if self.cancellation_token.cancelled:
                logger.info("Agent cancelled", extra={"task_id": self.task_id})
                self.state.stopped = True
                break

            try:
                self.budget_ledger.consume_step()
            except BudgetExceeded as exc:
                logger.warning(
                    "Stopping because budget was exhausted: %s",
                    exc,
                    extra={"task_id": self.task_id},
                )
                current_task = self.memory.tasks.get(self.task_id)
                if current_task:
                    current_task.status = "failed"
                break

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

            # Stop immediately if the current task is already terminal
            if self._current_task_terminal():
                logger.info(
                    "Current task already terminal, stopping run loop",
                    extra={"task_id": self.task_id},
                )
                break

            # Notify observers that a step is about to start
            await self._invoke_step_callback(on_step_start)

            # Execute step
            try:
                await self.step()
            except BudgetExceeded as exc:
                logger.warning(
                    "Stopping because budget was exhausted during step: %s",
                    exc,
                    extra={"task_id": self.task_id},
                )
                current_task = self.memory.tasks.get(self.task_id)
                if current_task:
                    current_task.status = "failed"
                break

            # Notify observers that the step finished (after plan/reflect)
            await self._invoke_step_callback(on_step_end)

            # Stop immediately if current task became terminal during the step
            if self._current_task_terminal():
                logger.info(
                    f"Task {self.task_id} reached terminal state, stopping",
                    extra={"task_id": self.task_id},
                )
                break

            # Check if task is complete via evaluator
            is_complete, completion_message = self.evaluator.evaluate_task_completion(
                self.task, self.memory
            )

            if is_complete:
                logger.info(
                    f"Task completed: {completion_message}",
                    extra={"task_id": self.task_id},
                )
                print(f"Task complete: {completion_message}")
                # Mark only the current task complete
                current_task = self.memory.tasks.get(self.task_id)
                if current_task:
                    current_task.complete(success=True)
                break

            # Note: output validation (settings.validate_output) is enforced at
            # the 'answer' tool gate in execute_actions, before a final answer is
            # ever recorded — so by the time a task reads as complete here, the
            # answer has already passed the empty-check and any LLM adequacy check.

        # Check if we hit the step limit
        if self.state.n_steps >= self.settings.max_steps:
            logger.warning(
                f"Reached maximum steps ({self.settings.max_steps}) without completing task",
                extra={"task_id": self.task_id},
            )
            print(
                f"Reached maximum steps ({self.settings.max_steps}) without completing task"
            )

        # Completion is based solely on the current task
        current_task = self.memory.tasks.get(self.task_id)
        completed = current_task is not None and current_task.status == "completed"

        # Salvage: a run that dies without an answer still delivers a handoff.
        partial_result = None
        if (
            not completed
            and self.settings.salvage_summary
            and self.state.n_steps > 0
            and not self.cancellation_token.cancelled
            and not self.state.stopped
            and self.memory.get_final_answer() is None
        ):
            partial_result = await self._salvage_best_effort_summary()

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
            except Exception as e:
                logger.debug(
                    f"Failed to log run metrics summary: {e}",
                    extra={"task_id": self.task_id},
                )

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
            self._trace.execution_receipts = [
                receipt.model_dump(mode="json") for receipt in self.execution_receipts
            ]
            self._trace.context_diagnostics = (
                self.context_diagnostics.model_dump(mode="json")
                if self.context_diagnostics
                else None
            )
            self._trace.finish()

        logger.info(
            f"Agent run finished - Completed: {completed}, Steps: {self.state.n_steps}",
            extra={"task_id": self.task_id},
        )
        self._emit(RUN_END, {"completed": completed, "steps": self.state.n_steps})

        return {
            "task": self.task,
            "completed": completed,
            "steps_taken": self.state.n_steps,
            "selected_skills": [
                skill.manifest.skill_id for skill in self.selected_skills
            ],
            "final_state": self.state.dict(),
            "metrics": (
                self._run_metrics.summary() if self._run_metrics is not None else None
            ),
            "trace": self._trace.to_dict() if self._trace else None,
            "run_context": self.run_context.model_dump(mode="json"),
            "execution_receipts": [
                receipt.model_dump(mode="json") for receipt in self.execution_receipts
            ],
            "artifacts": self._run_artifacts(),
            "plan": self._plan_snapshot(),
            "budget_usage": self.budget_ledger.snapshot(),
            "partial_result": partial_result,
        }

    async def _salvage_best_effort_summary(self) -> Optional[Dict[str, Any]]:
        """One tools-stripped LLM call that turns a dead run into a handoff.

        When the loop stops without an answer (step limit, failure limit,
        budget), the work already paid for is summarized — what was done, what
        was found, what remains — instead of discarded. Fails open: any error
        returns None and the run result is unchanged.
        """
        try:
            if self.state.consecutive_failures >= self.settings.max_failures:
                reason = "failure_limit"
            elif self.state.n_steps >= self.settings.max_steps:
                reason = "step_limit"
            else:
                reason = "budget_exhausted"
            messages = self._build_messages()
            messages.append(
                UserMessage(
                    content=(
                        f"The run has stopped before completing the task "
                        f"(reason: {reason}). Do not attempt any further "
                        "actions or tool calls. In plain text, briefly "
                        "summarize: (1) what was accomplished, (2) key "
                        "findings or artifacts produced so far, (3) what "
                        "remains to be done and the best next step."
                    )
                )
            )
            response = await self.llm.ainvoke(messages)
            text = (response if isinstance(response, str) else str(response)).strip()
            if not text:
                return None
            partial = {
                "summary": text,
                "reason": reason,
                "steps_taken": self.state.n_steps,
            }
            self.memory.add_message("assistant", f"[partial-result] {text}")
            self.memory.update_state("partial_result", partial)
            logger.info(
                f"Salvaged best-effort summary after {reason}",
                extra={"task_id": self.task_id},
            )
            return partial
        except Exception as salvage_err:
            logger.debug(
                f"Salvage summary failed (fail-open): {salvage_err}",
                extra={"task_id": self.task_id},
            )
            return None

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

    def _plan_snapshot(self) -> List[Dict[str, Any]]:
        """Return the model-owned plan (update_plan) items as serializable dicts."""
        return [item.model_dump() for item in self.plan_store.items()]

    def _sandbox_thread_id(self) -> str:
        """Sandbox workspace key — the session, so files persist across tasks.

        Keying on the session (not the per-task id) means files created in one
        task are visible to follow-up tasks in the same session (e.g. a CLI
        conversation). Falls back to the task id when no session is set.
        """
        return self.run_context.session_id or self.task_id

    def result_summary(self) -> Dict[str, Any]:
        """Return the public run result fields consumers should depend on."""
        final = self.memory.get_final_answer() or {}
        return {
            "task": self.task,
            "completed": self._current_task_terminal(),
            "steps_taken": self.state.n_steps,
            "final_answer": final.get("final_answer"),
            "reasoning": final.get("reasoning"),
            "selected_skills": [
                skill.manifest.skill_id for skill in self.selected_skills
            ],
            "artifacts": self._run_artifacts(),
            "plan": self._plan_snapshot(),
            "metrics": (
                self._run_metrics.summary() if self._run_metrics is not None else None
            ),
            "trace": self._trace.to_dict() if self._trace else None,
        }
