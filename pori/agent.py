import asyncio
from datetime import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from .memory_v2 import EnhancedAgentMemory
from .tools import ToolRegistry, ToolExecutor
from .evaluation import ActionResult, Evaluator
from .utils.prompt_loader import load_prompt
from .utils.logging_config import ensure_logger_configured

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
    ):
        # Generate unique task ID for tracking
        self.task_id = str(uuid.uuid4())[:8]  # Short ID for logging

        logger.info(f"Initializing new agent", extra={"task_id": self.task_id})
        logger.info(f"Task: {task}", extra={"task_id": self.task_id})

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
        self.memory = (
            memory
            if memory is not None
            else EnhancedAgentMemory(
                persistent=False, vector=True, vector_config={"backend": "local"}
            )
        )
        self.evaluator = Evaluator(max_retries=settings.max_failures)

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
            model_output = await self.get_next_action()

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
                tool_results = await self.execute_actions(model_output.action)

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

        # Add system message
        messages.append(SystemMessage(content=self.system_message))

        # Add conversation history (truncated if needed)
        max_history = 10  # Simplified - in a real system, use token counting
        # Use EnhancedAgentMemory working buffer if available, else fallback
        try:
            recent_structured = self.memory.get_recent_messages_structured(max_history)
            for msg in recent_structured:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        except Exception:
            # Backward-compatibility with legacy AgentMemory
            for msg in getattr(self.memory, "conversation_history", [])[-max_history:]:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

        # Add current state information
        context = self._get_current_context()
        messages.append(HumanMessage(content=context))

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
                        HumanMessage(
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
            HumanMessage(content=f"Task: {self.task}\n{plan_prompt}"),
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
            HumanMessage(
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
        self, actions: List[Dict[str, Any]]
    ) -> List[ActionResult]:
        """Execute a list of actions."""
        logger.info(
            f"Executing {len(actions)} actions", extra={"task_id": self.task_id}
        )
        results = []

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
                start_time = datetime.now()

                # Execute the tool
                tool_result = self.tool_executor.execute_tool(
                    tool_name=tool_name,
                    params=params,
                    context={"memory": self.memory, "state": self.state},
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

                # Record the tool call
                self.memory.add_tool_call(
                    tool_name=tool_name,
                    parameters=params,
                    result=tool_result,
                    success=success,
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
        print(f"🚀 Starting task: {self.task}")

        for step_count in range(self.settings.max_steps):
            # Check control flags
            if self.state.stopped:
                logger.info("Agent stopped by request", extra={"task_id": self.task_id})
                print("🛑 Agent stopped")
                break

            if self.state.paused:
                logger.info("Agent paused", extra={"task_id": self.task_id})
                print("⏸ Agent paused")
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
                    f"❌ Stopping due to {self.settings.max_failures} consecutive failures"
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
                print(f"✅ Task complete: {completion_message}")
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
                f"⚠️ Reached maximum steps ({self.settings.max_steps}) without completing task"
            )

        completed = any(
            task.status == "completed" for task in self.memory.tasks.values()
        )

        logger.info(
            f"Agent run finished - Completed: {completed}, Steps: {self.state.n_steps}",
            extra={"task_id": self.task_id},
        )

        return {
            "task": self.task,
            "completed": completed,
            "steps_taken": self.state.n_steps,
            "final_state": self.state.dict(),
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
