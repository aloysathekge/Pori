import asyncio
import uuid
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, AgentSettings
from .memory import AgentMemory
from .tools import ToolRegistry
from langchain.chat_models.base import BaseChatModel


class Orchestrator:
    """
    Orchestrates one or more agents to complete tasks.

    This layer handles:
    1. Task management and delegation
    2. Agent lifecycle (creation, monitoring)
    3. Parallel execution if needed
    4. Callbacks for monitoring agent progress
    """

    def __init__(self, llm: BaseChatModel, tools_registry: ToolRegistry):
        self.llm = llm
        self.tools_registry = tools_registry
        self.agents: Dict[str, Agent] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Shared memory - simple per-session memory
        self.shared_memory = AgentMemory()

    async def execute_task(
        self,
        task: str,
        agent_settings: Optional[AgentSettings] = None,
        on_step_start: Optional[Callable[[Agent], Any]] = None,
        on_step_end: Optional[Callable[[Agent], Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task with a new agent."""
        # Create a unique ID for this task
        task_id = str(uuid.uuid4())

        # Create agent with default settings if none provided
        settings = agent_settings or AgentSettings()

        # Create and register the agent
        agent = Agent(
            task=task,
            llm=self.llm,
            tools_registry=self.tools_registry,
            settings=settings,
            memory=self.shared_memory,
        )
        self.agents[task_id] = agent

        # Create hooks for monitoring if provided
        async def step_start_hook(agent: Agent):
            if on_step_start:
                on_step_start(agent)

        async def step_end_hook(agent: Agent):
            if on_step_end:
                on_step_end(agent)

        # Execute the task
        try:
            result = await agent.run()
            return {
                "task_id": task_id,
                "success": result["completed"],
                "steps_taken": result["steps_taken"],
                "result": result,
                "agent": agent,  # Keep agent reference for accessing final answer
            }
        finally:
            # Clean up from agents dict (but keep reference in result)
            if task_id in self.agents:
                del self.agents[task_id]

    async def execute_tasks_parallel(
        self,
        tasks: List[str],
        max_concurrent: int = 5,
        agent_settings: Optional[AgentSettings] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel with limits."""
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.execute_task(task, agent_settings)

        # Launch all tasks
        task_futures = [execute_with_semaphore(task) for task in tasks]

        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures, return_exceptions=True)

        # Process results, handling exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "task": tasks[i],
                        "success": False,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_agent(self, task_id: str) -> Optional[Agent]:
        """Get an agent by its task ID."""
        return self.agents.get(task_id)
