"""Tool dispatch pipeline — how a model-proposed action becomes an executed tool
call. Contract, in order per action: duplicate reuse (same tool+params this step
or recently this task), terminal `done` handling, side-effect authorization,
HITL approval gate, memory-deletion guards, completion-quality gate for
`answer`, then write-ahead journal + execute + receipt + metrics recording.
These are `Agent` methods, grouped here for readability and bound onto the
class in `core` (they take `self`).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..evaluation import ActionResult
from ..hitl import ActionRequest, ApprovalRequest, ReviewConfig
from ..metrics import StepMetrics, ToolCallMetrics
from ..observability import (
    PLAN_CHANGED,
    TOOL_CALL_END,
    TOOL_CALL_START,
    build_tool_preview,
    build_tool_result_preview,
)
from ..runtime import BudgetExceeded, ReceiptStatus
from ..utils.logging_config import ensure_logger_configured

logger = ensure_logger_configured("pori.agent")


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
    logger.info(f"Executing {len(actions)} actions", extra={"task_id": self.task_id})
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
        event_call_id: Optional[str] = None
        event_started = False
        event_finished = False

        logger.info(
            f"Executing action {i}: {tool_name}", extra={"task_id": self.task_id}
        )
        logger.debug(f"Tool parameters: {params}", extra={"task_id": self.task_id})
        # Count every model-requested tool action, including terminal, rejected,
        # staged, and duplicate calls. Reserve before any dispatch so crossing
        # the ceiling can never execute one extra action.
        self.budget_ledger.consume_tool_call()

        # Duplicate handling: reuse last identical result instead of re-running
        try:
            sig = _make_signature(tool_name, params)
            seen_before_this_step = _is_duplicate_this_step(tool_name, params)
            prior_result = _find_recent_same_call_result(tool_name, params, lookback=5)
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
        except BudgetExceeded:
            # This is a host-owned stop signal, not a failed tool result. Let
            # Agent.run() terminalize the run without dispatching more actions.
            raise
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
                self._reject_action(tool_name, params, results, authorization.reason)
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

                    if decision.type == "defer":
                        staged = decision.result or {"status": "staged"}
                        self.memory.add_message(
                            "system",
                            f"{tool_name} was staged and has not executed: {staged}",
                        )
                        self.memory.add_tool_call(
                            tool_name=tool_name,
                            parameters=params,
                            result=staged,
                            success=False,
                        )
                        results.append(
                            ActionResult(
                                success=True,
                                value=staged,
                                include_in_memory=True,
                            )
                        )
                        self._record_tool_receipt(
                            tool_name,
                            params,
                            ReceiptStatus.STAGED,
                            metadata={
                                "proposal_id": staged.get("proposal_id"),
                                "executed": False,
                            },
                        )
                        continue

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

                staged_claim_error = self._staged_outcome_claim_error(answer_text)
                if staged_claim_error:
                    self._reject_action(
                        tool_name, params, results, staged_claim_error, log_level="info"
                    )
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
            normalized_params = params if isinstance(params, dict) else {}
            plan_before = self._plan_snapshot()

            # Write-ahead journal before announcing execution. Its durable id
            # also correlates the public start/end lifecycle without exposing
            # raw tool output.
            try:
                dispatch_id = self.memory.record_tool_dispatch(
                    tool_name, normalized_params
                )
            except Exception as dispatch_err:
                logger.debug(
                    f"Failed to journal dispatch for {tool_name}: {dispatch_err}",
                    extra={"task_id": self.task_id},
                )
            event_call_id = dispatch_id or (
                f"{self.task_id}:{self.state.n_steps + 1}:{i}"
            )
            self._emit(
                TOOL_CALL_START,
                {
                    "name": tool_name,
                    "call_id": event_call_id,
                    "label": build_tool_preview(tool_name, normalized_params),
                    "args": normalized_params,
                },
            )
            event_started = True

            # Execute the tool
            thread_data = None
            if self.sandbox_base_dir and self.run_context.workspace_id:
                from ..sandbox.path_resolution import get_workspace_data

                thread_data = get_workspace_data(
                    self.run_context.workspace_id,
                    self.run_context.run_id,
                    self.sandbox_base_dir,
                )
            tool_context = {
                "memory": self.memory,
                "state": self.state,
                "thread_id": self._sandbox_thread_id(),
                "sandbox_base_dir": self.sandbox_base_dir,
                "thread_data": thread_data,
                "event_id": self.run_context.event_id,
                "workspace_id": self.run_context.workspace_id,
                "run_id": self.run_context.run_id,
                "run_context": self.run_context,
                "evolution_repository": self.evolution_repository,
                "skill_catalog": self.skill_catalog,
                "capability_snapshot": self.capability_snapshot,
                "tools_registry": self.tools_registry,
                "plan_store": self.plan_store,
                "task_id": self.task_id,
                **self._tool_context_extra,
            }
            tool_result = await self.tool_executor.execute_tool_async(
                tool_name=tool_name,
                params=params,
                context=tool_context,
            )

            execution_time = (datetime.now() - start_time).total_seconds()
            success = tool_result.get("success", False)
            # Emit tool_call_end so a renderer can close the "» ..." line
            # with a "✓ / ✗" as soon as the tool finishes.
            self._emit(
                TOOL_CALL_END,
                {
                    "name": tool_name,
                    "call_id": event_call_id,
                    "label": build_tool_result_preview(tool_name, normalized_params),
                    "success": bool(success),
                    "duration_seconds": execution_time,
                },
            )
            event_finished = True

            plan_after = self._plan_snapshot()
            if plan_after != plan_before:
                self._emit(
                    PLAN_CHANGED,
                    {"plan": plan_after, "summary": self.plan_store.summary()},
                )

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
                self.memory.complete_tool_dispatch(dispatch_id, tool_result, success)
            else:
                self.memory.add_tool_call(
                    tool_name=tool_name,
                    parameters=params,
                    result=tool_result,
                    success=success,
                )

            # Cache the result for potential reuse within this step
            try:
                results_by_signature_this_step[_make_signature(tool_name, params)] = (
                    tool_result
                )
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

        except BudgetExceeded:
            raise
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
            if event_started and not event_finished:
                self._emit(
                    TOOL_CALL_END,
                    {
                        "name": tool_name,
                        "call_id": event_call_id,
                        "label": build_tool_result_preview(
                            tool_name,
                            params if isinstance(params, dict) else {},
                        ),
                        "success": False,
                    },
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
