"""The ``Agent`` class and its Plan → Act → Reflect → Evaluate loop.
``core`` keeps the loop whole; cohesive method groups (prompting, planning,
dispatch, artifacts, authorization) live in sibling modules and are bound onto
the class here. ``Agent.run()`` is bounded by ``AgentSettings.max_steps`` /
``max_failures``, terminates when the model calls a terminal tool (``answer``
/ ``done``), and always returns its trace + metrics.
"""

import asyncio
import inspect
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
    UserMessage,
    ensure_budgeted_chat_model,
    normalize_usage,
)
from pori.llm.error_classifier import classify_error
from pori.llm.model_context import get_model_context_length

from ..compression import compress_context
from ..context import ContextDiagnostics, ContextEngine, DefaultContextEngine
from ..eval.base import BaseEval
from ..evaluation import ActionResult, Evaluator
from ..evolution import EvolutionRepository
from ..hitl import ApprovalResponse, HITLConfig, HITLHandler, resolve_interrupt_config
from ..memory import AgentMemory
from ..metrics import (
    LLMCallMetrics,
    RunMetrics,
    StepMetrics,
    TokenUsage,
    estimate_llm_call_cost,
)
from ..observability import (
    ACTIVITY_CHANGED,
    RUN_END,
    RUN_START,
    STEP_END,
    STEP_START,
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
    RunCancelled,
    RunContext,
    ToolExecutionReceipt,
    fail_open,
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
from . import completion as _completion
from . import dispatch as _dispatch
from . import planning as _planning
from . import prompting as _prompting
from .schemas import (
    AgentOutput,
    AgentSettings,
    AgentState,
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
    # agent/authorization.py, agent/prompting.py, agent/planning.py,
    # agent/completion.py, agent/dispatch.py.
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
    _current_task_status = _completion._current_task_status
    _current_task_terminal = _completion._current_task_terminal
    _task_requests_memory_mutation = _completion._task_requests_memory_mutation
    _current_task_has_core_memory_mutation = (
        _completion._current_task_has_core_memory_mutation
    )
    _current_task_has_loaded_skill = _completion._current_task_has_loaded_skill
    _SKILL_NUDGE_MIN_SCORE = _completion._SKILL_NUDGE_MIN_SCORE
    _required_skill_view_before_answer = _completion._required_skill_view_before_answer
    _staged_outcome_claim_error = _completion._staged_outcome_claim_error
    _validate_answer_text = _completion._validate_answer_text
    _reject_action = _dispatch._reject_action
    execute_actions = _dispatch.execute_actions

    # Lazily set in _build_messages (now prompting.py); declared here so it's a
    # known Agent attribute after that method was extracted.
    prompt_fingerprint: str = ""

    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        tools_registry: ToolRegistry,
        settings: AgentSettings = AgentSettings(),
        memory: Optional[AgentMemory] = None,
        sandbox_base_dir: Optional[str] = None,
        hitl_handler: Optional[HITLHandler] = None,
        hitl_config: Optional[HITLConfig] = None,
        guardrails: Optional[List[BaseEval]] = None,
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
        elif memory is not None:
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
        self.guardrails: List[BaseEval] = guardrails or []

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
        if memory is not None:
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
        else:
            self.memory = AgentMemory(
                organization_id=self.run_context.organization_id,
                user_id=self.run_context.user_id,
                agent_id=self.run_context.agent_id,
                session_id=self.run_context.session_id,
            )
        self._frozen_trusted_context = str(
            getattr(self.memory, "trusted_context", "") or ""
        )
        self._trusted_context_fingerprint = str(
            getattr(self.memory, "trusted_context_fingerprint", "") or ""
        )
        self._trusted_context_cacheable = bool(
            getattr(self.memory, "trusted_context_cacheable", True)
        )
        self.context_engine = context_engine or DefaultContextEngine()
        self.context_diagnostics: Optional[ContextDiagnostics] = None
        self.skill_catalog = skill_catalog
        self.skill_limit = skill_limit
        self.model_capabilities = model_capabilities or frozenset()
        self.budget_ledger = budget_ledger or BudgetLedger(self.run_context.budget)
        self.llm = ensure_budgeted_chat_model(self.llm, self.budget_ledger)
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

    def _history_context_limits(self) -> tuple[int, int]:
        """Return total/reserve limits for the transcript-only window.

        ``context_window_tokens`` is provider capacity. Products may choose a
        smaller stable history allowance so opening an old conversation does
        not grow prompt latency forever. Adding the reserve back converts that
        allowance to the existing ``max_tokens - reserve_tokens`` contract.
        """
        reserve = int(self.settings.context_window_reserve_tokens)
        history_budget = self.settings.history_window_tokens
        if history_budget is None:
            return int(self.settings.context_window_tokens), reserve
        return (
            min(
                int(self.settings.context_window_tokens),
                int(history_budget) + reserve,
            ),
            reserve,
        )

    async def step(self) -> None:
        """Execute one step of the task."""
        step_number = self.state.n_steps + 1
        logger.info(
            f"Starting step {step_number}",
            extra={"task_id": self.task_id, "step": step_number},
        )
        # Live progress for stream consumers ("step 2 of 15" in a UI).
        self._emit(
            STEP_START,
            {"step": step_number, "max_steps": self.settings.max_steps},
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
                history_tokens, history_reserve = self._history_context_limits()
                await compress_context(
                    self.memory,
                    self.llm,
                    max_tokens=history_tokens,
                    reserve_tokens=history_reserve,
                )
            llm_start_time = datetime.now()
            model_output = await self.get_next_action()
            llm_duration = (datetime.now() - llm_start_time).total_seconds()

            # Capture the model's intent for this step as the live activity line.
            next_goal = (model_output.current_state or {}).get("next_goal", "")
            if next_goal and next_goal.strip():
                activity = next_goal.strip()
                if activity != self.state.current_activity:
                    self.state.current_activity = activity
                    self._emit(ACTIVITY_CHANGED, {"activity": activity})
            if llm_span:
                llm_span.attributes["duration_seconds"] = llm_duration
                llm_span.finish()

            # Record LLM call metrics for this step
            self._record_llm_call_metrics(step_metrics, llm_duration)

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

            # A stop that landed after the LLM call still means: don't act on
            # the returned tool calls.
            if self.cancellation_token.cancelled:
                raise RunCancelled()

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
                await self._reflect_and_update_plan(tool_results)
            else:
                logger.debug(
                    "Skipping reflection: disabled or task is complete",
                    extra={"task_id": self.task_id, "step": step_number},
                )

        except RunCancelled:
            # Not a failure: the user stopped the run. Close the span and let
            # the run loop wind down.
            if step_span:
                step_span.finish(SpanStatus.OK)
            raise
        except BudgetExceeded as exhausted:
            # A budget stop belongs to the whole Run. It is not a retryable
            # model/tool failure and must reach Agent.run() unchanged.
            if step_span:
                step_span.finish(SpanStatus.ERROR, error=str(exhausted))
            raise
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
        with fail_open("recording step metrics", logger):
            step_metrics.duration_seconds = step_duration
            if self._run_metrics is not None:
                self._run_metrics.steps.append(step_metrics)

        # Finish step span if not already finished (error case finishes early)
        if step_span and step_span.end_time is None:
            step_span.finish()

        # Add step results to memory
        for result in tool_results:
            if result.include_in_memory:
                self.memory.add_message("system", str(result))
                # Capture final answer in state for this task
                with fail_open("updating final_answer state", logger):
                    if (
                        isinstance(result.value, dict)
                        and "final_answer" in result.value
                    ):
                        self.memory.update_state("final_answer", result.value)
                # Skip indexing intermediate step results — they add noise.
                # Only task descriptions and final answers are stored as experiences.

        # Checkpoint the loop position on the task record so an interrupted run
        # can resume from here instead of step 0 (fail-open).
        with fail_open("checkpointing task progress", logger):
            self.memory.checkpoint_task_progress(
                self.task_id,
                n_steps=self.state.n_steps,
                consecutive_failures=self.state.consecutive_failures,
                current_activity=self.state.current_activity,
                plan=self._plan_snapshot(),
            )

        self._emit(
            STEP_END,
            {
                "step": step_number,
                "duration_seconds": step_duration,
                "success": all(result.success for result in tool_results),
            },
        )

    def _record_llm_call_metrics(
        self, step_metrics: StepMetrics, llm_duration: float
    ) -> None:
        """Record step metrics for the primary LLM call.

        The model wrapper charges every provider call centrally, including
        planning, reflection, compression, validation, and Team coordination.
        This method only preserves the primary step's detailed telemetry.
        """
        step_number = self.state.n_steps + 1
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
        except Exception as metrics_err:
            logger.debug(
                f"Failed to record LLM call metrics: {metrics_err}",
                extra={"task_id": self.task_id, "step": step_number},
            )

    async def _await_cancellable(self, coro: Any) -> Any:
        """Await ``coro``, aborting promptly when the cancellation token fires.

        The token is a ``threading.Event`` set from another thread (a product's
        stop endpoint, a CLI signal handler), so it can't wake this loop by
        itself — poll it alongside the task. Without this, a stop request
        waits out the whole in-flight LLM call before anything notices."""
        task = asyncio.ensure_future(coro)
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if done:
                return task.result()
            if self.cancellation_token.cancelled:
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
                raise RunCancelled()

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
            turn = await self._await_cancellable(
                self.llm.ainvoke_tools(messages, tools, on_event=self._on_event)
            )
        else:
            turn = await self._await_cancellable(
                self.llm.ainvoke_tools(messages, tools)
            )
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
        except RunCancelled:
            raise
        except BudgetExceeded:
            # Host-owned stop signal: never classify or wrap it as a retryable
            # provider failure.
            raise
        except Exception as e:
            classified = classify_error(e)
            if classified.should_compress:
                # Request too big to send unchanged: compress dropped history
                # (AC-3) and retry once before giving up.
                try:
                    history_tokens, history_reserve = self._history_context_limits()
                    await compress_context(
                        self.memory,
                        self.llm,
                        max_tokens=history_tokens,
                        reserve_tokens=history_reserve,
                    )
                    return await self._invoke_for_action()
                except BudgetExceeded:
                    raise
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
        with fail_open("step callback", logger):
            result = callback(self)
            if inspect.isawaitable(result):
                await result

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
        budget_exhaustion: Optional[BudgetExceeded] = None
        # Arm the wall-clock budget (max_duration_seconds); idempotent, so a
        # shared ledger measures from the first run that starts.
        self.budget_ledger.start_clock()
        self._emit(RUN_START, {"task": self.task, "task_id": self.task_id})
        logger.info(f"Starting agent run", extra={"task_id": self.task_id})

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
                budget_exhaustion = exc
                break

            if self.state.stopped:
                logger.info("Agent stopped by request", extra={"task_id": self.task_id})
                break

            if self.state.paused:
                logger.info("Agent paused", extra={"task_id": self.task_id})
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
            except RunCancelled:
                logger.info("Agent cancelled mid-step", extra={"task_id": self.task_id})
                self.state.stopped = True
                break
            except BudgetExceeded as exc:
                logger.warning(
                    "Stopping because budget was exhausted during step: %s",
                    exc,
                    extra={"task_id": self.task_id},
                )
                current_task = self.memory.tasks.get(self.task_id)
                if current_task:
                    current_task.status = "failed"
                budget_exhaustion = exc
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

        # Completion is based solely on the current task
        current_task = self.memory.tasks.get(self.task_id)
        completed = current_task is not None and current_task.status == "completed"

        if budget_exhaustion is not None:
            stop_reason = budget_exhaustion.code
            budget_error = str(budget_exhaustion)
        elif not completed and self.state.n_steps >= self.settings.max_steps:
            stop_reason = "max_steps"
            budget_error = (
                f"Step budget exhausted after {self.settings.max_steps} steps"
            )
        elif (
            not completed
            and self.state.consecutive_failures >= self.settings.max_failures
        ):
            stop_reason = "failure_limit"
            budget_error = None
        elif self.cancellation_token.cancelled or self.state.stopped:
            stop_reason = "cancelled"
            budget_error = None
        else:
            stop_reason = None
            budget_error = None

        # Salvage: a run that dies without an answer still delivers a handoff.
        # A step ceiling may still produce the established no-tool partial
        # handoff. Token, cost, duration, and tool-call exhaustion cannot spend
        # another provider call; the model wrapper enforces those ceilings if
        # salvage itself would cross one.
        partial_result = None
        if (
            not completed
            and budget_exhaustion is None
            and self.settings.salvage_summary
            and self.state.n_steps > 0
            and not self.cancellation_token.cancelled
            and not self.state.stopped
            and self.memory.get_final_answer() is None
        ):
            try:
                partial_result = await self._salvage_best_effort_summary()
            except BudgetExceeded as exc:
                budget_exhaustion = exc
                stop_reason = exc.code
                budget_error = str(exc)

        # Finalize metrics
        if self._run_metrics is not None:
            with fail_open("run metrics summary", logger):
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

        budget_usage = self.budget_ledger.snapshot()
        metrics_summary = (
            self._run_metrics.summary() if self._run_metrics is not None else {}
        )
        metrics_summary["budget_usage"] = budget_usage

        return {
            "task": self.task,
            "completed": completed,
            "steps_taken": self.state.n_steps,
            "selected_skills": [
                skill.manifest.skill_id for skill in self.selected_skills
            ],
            "final_state": self.state.dict(),
            "metrics": metrics_summary,
            "trace": self._trace.to_dict() if self._trace else None,
            "run_context": self.run_context.model_dump(mode="json"),
            "execution_receipts": [
                receipt.model_dump(mode="json") for receipt in self.execution_receipts
            ],
            "artifacts": self._run_artifacts(),
            "plan": self._plan_snapshot(),
            "budget_usage": budget_usage,
            "stop_reason": stop_reason,
            "budget_error": budget_error,
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
        except BudgetExceeded:
            raise
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
        """Return the durable workspace key for sandbox acquisition.

        Event-aware products provide ``workspace_id`` so Sessions share durable
        files. Legacy callers retain Session-scoped behavior, with task id as
        the final fallback.
        """
        workspace_id = self.run_context.workspace_id
        return workspace_id or self.run_context.session_id or self.task_id

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
