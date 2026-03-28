"""Team — a coordinator that uses an LLM to route, broadcast, or delegate tasks across member agents."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pori.agent import Agent, AgentSettings
from pori.hitl import HITLConfig, HITLHandler
from pori.llm.base import BaseChatModel
from pori.llm.messages import SystemMessage, UserMessage
from pori.memory import AgentMemory
from pori.tools.registry import ToolRegistry

from .models import (
    BroadcastSummary,
    DelegationPlan,
    MemberConfig,
    MemberRunResult,
    RoutingDecision,
    TeamMode,
)

logger = logging.getLogger(__name__)


class Team:
    """Coordinates multiple member agents via an LLM-powered coordinator.

    Members are *blueprints* (``MemberConfig``), not live agents.
    A fresh ``Agent`` (or nested ``Team``) is created for every execution,
    matching the pattern used by ``Orchestrator.execute_task()``.
    """

    def __init__(
        self,
        task: str,
        coordinator_llm: BaseChatModel,
        members: List[MemberConfig],
        mode: TeamMode = TeamMode.ROUTER,
        tools_registry: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        hitl_handler: Optional[HITLHandler] = None,
        hitl_config: Optional[HITLConfig] = None,
        agent_defaults: Optional[AgentSettings] = None,
        max_delegation_steps: int = 10,
        max_concurrent_members: int = 5,
        name: str = "team",
    ):
        self.task = task
        self.coordinator_llm = coordinator_llm
        self.members = {m.name: m for m in members}
        self.mode = mode
        self.tools_registry = tools_registry or ToolRegistry()
        self.memory = memory
        self.hitl_handler = hitl_handler
        self.hitl_config = hitl_config
        self.agent_defaults = agent_defaults or AgentSettings()
        self.max_delegation_steps = max_delegation_steps
        self.max_concurrent_members = max_concurrent_members
        self.name = name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Run the team. Returns the same shape as ``Agent.run()``."""
        logger.info(f"[{self.name}] Starting team run in {self.mode.value} mode")
        logger.info(f"\n[Team:{self.name}] Starting in {self.mode.value} mode")
        logger.info(f"[Team:{self.name}] Task: {self.task}")
        logger.info(f"[Team:{self.name}] Members: {', '.join(self.members.keys())}")

        if self.mode == TeamMode.ROUTER:
            return await self._run_router()
        elif self.mode == TeamMode.BROADCAST:
            return await self._run_broadcast()
        elif self.mode == TeamMode.DELEGATE:
            return await self._run_delegate()
        else:
            raise ValueError(f"Unknown team mode: {self.mode}")

    # ------------------------------------------------------------------
    # Router mode
    # ------------------------------------------------------------------

    async def _run_router(self) -> Dict[str, Any]:
        logger.info(f"\n[Team:{self.name}] Coordinator is choosing a member...")
        decision = await self._coordinator_route()
        logger.info(
            f"[{self.name}] Router chose '{decision.chosen_member}': {decision.reasoning}"
        )
        logger.info(f"[Team:{self.name}] Routed to: '{decision.chosen_member}'")
        logger.info(f"[Team:{self.name}] Reasoning: {decision.reasoning}")
        if decision.rewritten_task:
            logger.info(f"[Team:{self.name}] Rewritten task: {decision.rewritten_task}")

        config = self.members.get(decision.chosen_member)
        if config is None:
            return self._error_result(
                f"Coordinator chose unknown member '{decision.chosen_member}'"
            )

        task_text = decision.rewritten_task or self.task
        result = await self._run_member(config, task_text)

        return {
            "task": self.task,
            "completed": result.completed,
            "steps_taken": result.steps_taken,
            "final_state": {
                "mode": self.mode.value,
                "chosen_member": decision.chosen_member,
                "reasoning": decision.reasoning,
                "final_answer": result.final_answer,
            },
            "metrics": result.metrics,
        }

    async def _coordinator_route(self) -> RoutingDecision:
        member_descriptions = "\n".join(
            f"- {name}: {cfg.description}" for name, cfg in self.members.items()
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a task router. Given a task and a list of specialist members, "
                    "choose the single best member to handle the task. "
                    "You may optionally rewrite the task to better suit the chosen member.\n\n"
                    f"Available members:\n{member_descriptions}"
                )
            ),
            UserMessage(content=self.task),
        ]
        structured_llm = self.coordinator_llm.with_structured_output(RoutingDecision)
        return await structured_llm.ainvoke(messages)

    # ------------------------------------------------------------------
    # Broadcast mode
    # ------------------------------------------------------------------

    async def _run_broadcast(self) -> Dict[str, Any]:
        logger.info(
            f"\n[Team:{self.name}] Broadcasting task to all {len(self.members)} members in parallel..."
        )
        semaphore = asyncio.Semaphore(self.max_concurrent_members)
        member_configs = list(self.members.values())

        async def _run_with_sem(cfg: MemberConfig) -> MemberRunResult:
            async with semaphore:
                return await self._run_member(cfg, self.task)

        results = await asyncio.gather(
            *[_run_with_sem(cfg) for cfg in member_configs],
            return_exceptions=True,
        )

        # Normalise exceptions
        member_results: List[MemberRunResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                member_results.append(
                    MemberRunResult(
                        member_name=member_configs[i].name,
                        task=self.task,
                        error=str(r),
                    )
                )
            else:
                member_results.append(r)

        logger.info(
            f"\n[Team:{self.name}] All members finished. Coordinator is combining results..."
        )
        summary = await self._coordinator_combine(member_results)
        total_steps = sum(r.steps_taken for r in member_results)

        return {
            "task": self.task,
            "completed": True,
            "steps_taken": total_steps,
            "final_state": {
                "mode": self.mode.value,
                "member_results": [r.model_dump() for r in member_results],
                "final_answer": summary.combined_answer,
                "reasoning": summary.reasoning,
            },
            "metrics": None,
        }

    async def _coordinator_combine(
        self, results: List[MemberRunResult]
    ) -> BroadcastSummary:
        results_text = "\n\n".join(
            f"### {r.member_name}\nCompleted: {r.completed}\n"
            f"Answer: {r.final_answer or '(no answer)'}\n"
            f"Error: {r.error or 'none'}"
            for r in results
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a result synthesiser. Multiple specialists were given the same task. "
                    "Combine their outputs into a single coherent answer."
                )
            ),
            UserMessage(
                content=f"Task: {self.task}\n\nMember results:\n{results_text}"
            ),
        ]
        structured_llm = self.coordinator_llm.with_structured_output(BroadcastSummary)
        return await structured_llm.ainvoke(messages)

    # ------------------------------------------------------------------
    # Delegate mode
    # ------------------------------------------------------------------

    async def _run_delegate(self) -> Dict[str, Any]:
        logger.info(
            f"\n[Team:{self.name}] Coordinator is creating a delegation plan..."
        )
        plan = await self._coordinator_plan()
        logger.info(
            f"[{self.name}] Delegation plan ({len(plan.steps)} steps): {plan.rationale}"
        )
        logger.info(
            f"[Team:{self.name}] Plan ({len(plan.steps)} steps): {plan.rationale}"
        )
        for s in plan.steps:
            deps = f" (depends on: {s.depends_on})" if s.depends_on else ""
            logger.info(f"Step {s.step_number}: [{s.member_name}] {s.subtask}{deps}")

        if len(plan.steps) > self.max_delegation_steps:
            return self._error_result(
                f"Delegation plan has {len(plan.steps)} steps, "
                f"exceeding max of {self.max_delegation_steps}"
            )

        # Execute steps respecting dependencies
        step_results: Dict[int, MemberRunResult] = {}
        completed_steps: set = set()

        # Build adjacency: which steps are ready once all deps are satisfied
        remaining = {s.step_number: s for s in plan.steps}
        total_steps = 0

        while remaining:
            # Find steps whose dependencies are all met
            ready = [
                s
                for s in remaining.values()
                if all(d in completed_steps for d in s.depends_on)
            ]
            if not ready:
                return self._error_result(
                    "Delegation plan has unresolvable dependencies"
                )

            # Run ready steps in parallel (with semaphore)
            semaphore = asyncio.Semaphore(self.max_concurrent_members)

            async def _exec_step(step):
                async with semaphore:
                    config = self.members.get(step.member_name)
                    if config is None:
                        return MemberRunResult(
                            member_name=step.member_name,
                            task=step.subtask,
                            error=f"Unknown member '{step.member_name}'",
                        )

                    # Inject prior step results into the subtask text
                    enriched_task = step.subtask
                    if step.depends_on:
                        prior_context_parts = []
                        for dep in step.depends_on:
                            dep_result = step_results.get(dep)
                            if dep_result and dep_result.final_answer:
                                prior_context_parts.append(
                                    f"[Result from step {dep} ({dep_result.member_name})]: "
                                    f"{dep_result.final_answer}"
                                )
                        if prior_context_parts:
                            enriched_task = (
                                enriched_task
                                + "\n\nContext from prior steps:\n"
                                + "\n".join(prior_context_parts)
                            )

                    return await self._run_member(config, enriched_task)

            batch_results = await asyncio.gather(
                *[_exec_step(s) for s in ready], return_exceptions=True
            )

            for step, result in zip(ready, batch_results):
                if isinstance(result, Exception):
                    result = MemberRunResult(
                        member_name=step.member_name,
                        task=step.subtask,
                        error=str(result),
                    )
                step_results[step.step_number] = result
                total_steps += result.steps_taken
                completed_steps.add(step.step_number)
                del remaining[step.step_number]

        # Synthesise final answer
        final_answer = await self._coordinator_synthesize(plan, step_results)

        return {
            "task": self.task,
            "completed": all(r.completed for r in step_results.values()),
            "steps_taken": total_steps,
            "final_state": {
                "mode": self.mode.value,
                "plan_steps": len(plan.steps),
                "agent_steps": total_steps,
                "plan": plan.model_dump(),
                "step_results": {k: v.model_dump() for k, v in step_results.items()},
                "final_answer": final_answer,
            },
            "metrics": None,
        }

    async def _coordinator_plan(self) -> DelegationPlan:
        member_descriptions = "\n".join(
            f"- {name}: {cfg.description}" for name, cfg in self.members.items()
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a task planner. Break the task into steps and assign each "
                    "step to the most suitable member. Steps can depend on prior steps. "
                    "Use depends_on to express ordering constraints.\n\n"
                    f"Available members:\n{member_descriptions}"
                )
            ),
            UserMessage(content=self.task),
        ]
        structured_llm = self.coordinator_llm.with_structured_output(DelegationPlan)
        return await structured_llm.ainvoke(messages)

    async def _coordinator_synthesize(
        self, plan: DelegationPlan, step_results: Dict[int, MemberRunResult]
    ) -> str:
        results_text = "\n\n".join(
            f"Step {num} ({r.member_name}): {r.final_answer or r.error or '(no output)'}"
            for num, r in sorted(step_results.items())
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a result synthesiser. A multi-step plan was executed. "
                    "Combine the step results into a single coherent final answer."
                )
            ),
            UserMessage(
                content=(
                    f"Original task: {self.task}\n\n"
                    f"Plan rationale: {plan.rationale}\n\n"
                    f"Step results:\n{results_text}"
                )
            ),
        ]
        return await self.coordinator_llm.ainvoke(messages)

    # ------------------------------------------------------------------
    # Member execution
    # ------------------------------------------------------------------

    async def _run_member(self, config: MemberConfig, task: str) -> MemberRunResult:
        """Run a single member (Agent or nested Team)."""
        member_type = "Team" if config.team_config else "Agent"
        logger.info(
            f"[{self.name}] Running member '{config.name}' with task: {task[:80]}..."
        )
        logger.info(f"\n{'='*50}")
        logger.info(f"[Member:{config.name}] Starting ({member_type})")
        logger.info(f"[Member:{config.name}] Task: {task[:120]}")
        tools = config.tools or list(self.tools_registry.tools.keys())
        logger.info(f"[Member:{config.name}] Tools: {len(tools)} available")
        try:
            if config.team_config is not None:
                result = await self._run_nested_team(config, task)
            else:
                result = await self._run_agent_member(config, task)
            logger.info(
                f"[Member:{config.name}] Completed: {result.completed} | Steps: {result.steps_taken}"
            )
            if result.final_answer:
                preview = result.final_answer[:150]
                logger.info(
                    f"[Member:{config.name}] Answer: {preview}{'...' if len(result.final_answer) > 150 else ''}"
                )
            if result.error:
                logger.info(f"[Member:{config.name}] Error: {result.error}")
            logger.info(f"{'='*50}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Member '{config.name}' failed: {e}")
            logger.info(f"[Member:{config.name}] FAILED: {e}")
            logger.info(f"{'='*50}")
            return MemberRunResult(
                member_name=config.name,
                task=task,
                error=str(e),
            )

    async def _run_agent_member(
        self, config: MemberConfig, task: str
    ) -> MemberRunResult:
        """Create a fresh Agent for this member and run it."""
        # Determine LLM
        if config.llm_config is not None:
            from pori.config import create_llm

            llm = create_llm(config.llm_config)
        else:
            llm = self.coordinator_llm

        # Build tool registry (filtered if needed)
        registry = self._build_registry(config)

        # Build agent settings
        settings_kwargs = self.agent_defaults.model_dump()
        if config.agent_settings:
            settings_kwargs.update(config.agent_settings)
        settings = AgentSettings(**settings_kwargs)

        # HITL
        hitl_config = None
        if config.hitl_config:
            hitl_config = HITLConfig(**config.hitl_config)
        elif self.hitl_config:
            hitl_config = self.hitl_config

        # Fresh memory per member, seeded with user context (read-only)
        member_memory = AgentMemory()
        if self.memory and self.memory.core_memory:
            member_memory.core_memory = self.memory.core_memory.clone_read_only()

        agent = Agent(
            task=task,
            llm=llm,
            tools_registry=registry,
            settings=settings,
            memory=member_memory,
            sandbox_base_dir=config.sandbox_base_dir,
            hitl_handler=self.hitl_handler,
            hitl_config=hitl_config,
        )

        result = await agent.run()

        # Extract final answer
        answer_data = member_memory.get_final_answer()
        final_answer = None
        reasoning = None
        if answer_data:
            final_answer = answer_data.get("final_answer")
            reasoning = answer_data.get("reasoning")

        return MemberRunResult(
            member_name=config.name,
            task=task,
            completed=result["completed"],
            steps_taken=result["steps_taken"],
            final_answer=final_answer,
            reasoning=reasoning,
            metrics=result.get("metrics"),
        )

    async def _run_nested_team(
        self, config: MemberConfig, task: str
    ) -> MemberRunResult:
        """Create a nested Team for this member and run it."""
        tc = config.team_config

        # Determine coordinator LLM for nested team
        if tc.coordinator_llm is not None:
            from pori.config import create_llm

            nested_llm = create_llm(tc.coordinator_llm)
        elif config.llm_config is not None:
            from pori.config import create_llm

            nested_llm = create_llm(config.llm_config)
        else:
            nested_llm = self.coordinator_llm

        nested_team = Team(
            task=task,
            coordinator_llm=nested_llm,
            members=tc.members,
            mode=tc.mode,
            tools_registry=self.tools_registry,
            hitl_handler=self.hitl_handler,
            hitl_config=self.hitl_config,
            agent_defaults=(
                AgentSettings(**tc.agent_defaults)
                if tc.agent_defaults
                else self.agent_defaults
            ),
            max_delegation_steps=tc.max_delegation_steps,
            max_concurrent_members=tc.max_concurrent_members,
            name=tc.name,
        )

        result = await nested_team.run()

        final_answer = None
        if result.get("final_state"):
            final_answer = result["final_state"].get("final_answer")

        return MemberRunResult(
            member_name=config.name,
            task=task,
            completed=result["completed"],
            steps_taken=result["steps_taken"],
            final_answer=final_answer,
            metrics=result.get("metrics"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_registry(self, config: MemberConfig) -> ToolRegistry:
        """Build a ToolRegistry for a member, filtering tools if specified."""
        if config.tools is None:
            return self.tools_registry

        # Always include answer and done tools
        required = {"answer", "done"}
        allowed = set(config.tools) | required

        filtered = ToolRegistry()
        for name, tool_info in self.tools_registry.tools.items():
            if name in allowed:
                filtered.register_tool(
                    name=tool_info.name,
                    param_model=tool_info.param_model,
                    function=tool_info.function,
                    description=tool_info.description,
                )
        return filtered

    @staticmethod
    def _error_result(message: str) -> Dict[str, Any]:
        """Return a standard error result dict."""
        return {
            "task": "",
            "completed": False,
            "steps_taken": 0,
            "final_state": {"error": message},
            "metrics": None,
        }
