import asyncio
from datetime import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from memory import AgentMemory
from tools import ToolRegistry, ToolExecutor
from evaluation import ActionResult, Evaluator
from utils.prompt_loader import load_prompt
from utils.logging_config import ensure_logger_configured

# Set up logger for this module - this will work regardless of import order
logger = ensure_logger_configured("pori.agent")


class AgentState(BaseModel):
    """The current state of the agent."""

    n_steps: int = 0
    consecutive_failures: int = 0
    paused: bool = False
    stopped: bool = False


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
        self.memory = AgentMemory()
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

        # Add task message to memory
        self.memory.add_message("user", f"Task: {self.task}")

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
                raw_content = response.get("raw", {}).content
                parsed_json = json.loads(raw_content)
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
        for msg in self.memory.conversation_history[-max_history:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Add current state information
        context = self._get_current_context()
        messages.append(HumanMessage(content=context))

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

        context_prompt = f"""
Current Status:
{tasks_status}

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
