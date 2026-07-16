"""``Orchestrator`` — the main programmatic entrypoint for single-task runs.
``execute_task`` builds an ``Agent`` (threading shared memory, sandbox, HITL
handlers, skills, and sub-agent runners) and runs it; per-``session_key``
in-flight tracking rejects or queues duplicate runs (``ConversationBusy``).
Also home of ``run_subagent``, the child-agent spawner behind the
``delegate_task`` tool.
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from pori.llm import BaseChatModel

if TYPE_CHECKING:
    from pori.llm import ImageBlock
    from pori.mcp import McpServerConfig

from ..agent import Agent, AgentSettings
from ..evolution import EvolutionRepository
from ..file_backends import FILE_BACKEND_CONTEXT_KEY, FileBackend
from ..hitl import HITLConfig, HITLHandler
from ..memory import AgentMemory
from ..profiles import RunProfile, RunProfileResolutionError
from ..runtime import CancellationToken, RunContext
from ..skill_provenance import use_write_origin
from ..skills import SkillCatalog
from ..skills_learn import build_background_review_prompt
from ..tools.registry import CapabilityResolutionError, ToolRegistry
from ..utils.context import use_identity

logger = logging.getLogger("pori.orchestrator")


class ConversationBusy(RuntimeError):
    """Raised when a task is submitted for a ``session_key`` that already has a
    run in flight and ``on_busy='reject'`` (GW-3: duplicate-run prevention)."""

    def __init__(self, session_key: str):
        self.session_key = session_key
        super().__init__(f"A run is already active for session_key={session_key!r}")


class Orchestrator:
    """
    Orchestrates one or more agents to complete tasks.

    This layer handles:
    1. Task management and delegation
    2. Agent lifecycle (creation, monitoring)
    3. Parallel execution if needed
    4. Callbacks for monitoring agent progress
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools_registry: ToolRegistry,
        shared_memory: Optional[AgentMemory] = None,
        skill_catalog: Optional[SkillCatalog] = None,
        skill_limit: int = 3,
        evolution_repository: Optional[EvolutionRepository] = None,
        soul_path: Optional[str] = None,
        soul_text: Optional[str] = None,
        load_project_context: bool = False,
        mcp_connection_factory: Optional[Callable[..., Any]] = None,
        system_prompt: Optional[str] = None,
        model_capabilities: Optional[frozenset[str]] = None,
        run_profile: Optional[RunProfile] = None,
        file_backend: Optional[FileBackend] = None,
    ):
        self.llm = llm
        self.run_profile = run_profile
        self.model_capabilities = model_capabilities or frozenset()
        self.system_prompt = self._compose_system_prompt(system_prompt, run_profile)
        self.file_backend = file_backend
        self.tools_registry = self._resolve_profile(
            tools_registry,
            skill_catalog=skill_catalog,
        )
        # Optional injectable MCP connection factory (tests / custom transports);
        # None -> the session set uses the default SDK-backed factory.
        self._mcp_factory = mcp_connection_factory
        self.agents: Dict[str, Agent] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # GW-3: at most one in-flight run per session_key (duplicate-run guard).
        self._active_sessions: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}
        # SK-1 layer 2: strong refs to fire-and-forget background-review tasks.
        self._review_tasks: Set["asyncio.Task[None]"] = set()
        self.shared_memory = shared_memory
        self.skill_catalog = skill_catalog
        self.skill_limit = skill_limit
        self.evolution_repository = evolution_repository
        self.soul_path = soul_path
        self.soul_text = soul_text
        self.load_project_context = load_project_context

    @staticmethod
    def _compose_system_prompt(
        system_prompt: Optional[str], run_profile: Optional[RunProfile]
    ) -> Optional[str]:
        blocks = [
            block.strip()
            for block in (
                system_prompt or "",
                run_profile.system_prompt if run_profile else "",
            )
            if block.strip()
        ]
        return "\n\n".join(blocks) or None

    def _resolve_profile(
        self,
        tools_registry: ToolRegistry,
        *,
        skill_catalog: Optional[SkillCatalog],
    ) -> ToolRegistry:
        profile = self.run_profile
        if profile is None:
            return tools_registry

        missing_capabilities = profile.required_model_capabilities.difference(
            self.model_capabilities
        )
        if missing_capabilities:
            raise RunProfileResolutionError(
                f"Run profile '{profile.profile_id}' requires unavailable model "
                "capabilities: " + ", ".join(sorted(missing_capabilities))
            )

        try:
            resolved = tools_registry.filtered(
                include_tools=profile.allowed_tools,
                exclude_tools=profile.denied_tools,
            )
        except CapabilityResolutionError as exc:
            raise RunProfileResolutionError(
                f"Run profile '{profile.profile_id}' tool surface could not be resolved: {exc}"
            ) from exc

        missing_tools = profile.required_tools.difference(resolved.tools)
        if missing_tools:
            raise RunProfileResolutionError(
                f"Run profile '{profile.profile_id}' requires unavailable tools: "
                + ", ".join(sorted(missing_tools))
            )

        available_skills = (
            {manifest.skill_id for manifest in skill_catalog.manifests()}
            if skill_catalog is not None
            else set()
        )
        missing_skills = profile.required_skill_ids.difference(available_skills)
        if missing_skills:
            raise RunProfileResolutionError(
                f"Run profile '{profile.profile_id}' requires unavailable skills: "
                + ", ".join(sorted(missing_skills))
            )
        return resolved

    async def execute_task(
        self,
        task: str,
        agent_settings: Optional[AgentSettings] = None,
        memory: Optional[AgentMemory] = None,
        on_step_start: Optional[Callable[[Agent], Any]] = None,
        on_step_end: Optional[Callable[[Agent], Any]] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        stream: bool = False,
        sandbox_base_dir: Optional[str] = None,
        hitl_handler: Optional[HITLHandler] = None,
        hitl_config: Optional[HITLConfig] = None,
        run_context: Optional[RunContext] = None,
        selected_skill_ids: Optional[List[str]] = None,
        tool_context_extra: Optional[Dict[str, Any]] = None,
        session_key: Optional[str] = None,
        on_busy: str = "reject",
        resume_task_id: Optional[str] = None,
        mcp_servers: Optional[List["McpServerConfig"]] = None,
        task_attachments: Optional[List[Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Execute a task with a new agent.

        When ``session_key`` is given, a duplicate-run guard (GW-3) allows at most
        one run in flight per key: a concurrent submit either raises
        ``ConversationBusy`` (``on_busy='reject'``, the default) or awaits the
        in-flight run (``on_busy='coalesce'``), so a double-submit (retry,
        double-click) can't run one conversation twice into the same memory.

        ``resume_task_id`` is forwarded to the Agent so an interrupted run can
        continue from its per-step checkpoint instead of restarting from step 0
        (see docs/long-running.md Phase 2).

        ``cancellation_token`` is a thread-safe cooperative stop signal: the
        agent loop checks it between steps, so a caller (e.g. a product's
        stop-generation endpoint) can end the run early with a clean result.
        """

        async def _run() -> Dict[str, Any]:
            # GW-5: bind per-turn identity so tools/guardrails/tracing can read
            # the current session without it being threaded through signatures.
            with use_identity(session_id=session_key):
                return await self._execute_task_inner(
                    task,
                    agent_settings=agent_settings,
                    memory=memory,
                    on_step_start=on_step_start,
                    on_step_end=on_step_end,
                    on_event=on_event,
                    stream=stream,
                    sandbox_base_dir=sandbox_base_dir,
                    hitl_handler=hitl_handler,
                    hitl_config=hitl_config,
                    run_context=run_context,
                    selected_skill_ids=selected_skill_ids,
                    tool_context_extra=tool_context_extra,
                    resume_task_id=resume_task_id,
                    mcp_servers=mcp_servers,
                    task_attachments=task_attachments,
                    cancellation_token=cancellation_token,
                )

        if session_key is None:
            return await _run()

        existing = self._active_sessions.get(session_key)
        if existing is not None:
            if on_busy == "coalesce":
                return await existing
            raise ConversationBusy(session_key)

        # Claim the slot BEFORE the first await so a concurrent submit racing in
        # between still sees it. ensure_future schedules the run; we register the
        # future synchronously, then await it.
        fut: "asyncio.Future[Dict[str, Any]]" = asyncio.ensure_future(_run())
        self._active_sessions[session_key] = fut
        try:
            return await fut
        finally:
            self._active_sessions.pop(session_key, None)  # idempotent

    async def _execute_task_inner(
        self,
        task: str,
        agent_settings: Optional[AgentSettings] = None,
        memory: Optional[AgentMemory] = None,
        on_step_start: Optional[Callable[[Agent], Any]] = None,
        on_step_end: Optional[Callable[[Agent], Any]] = None,
        on_event: Optional[Callable[[Any], None]] = None,
        stream: bool = False,
        sandbox_base_dir: Optional[str] = None,
        hitl_handler: Optional[HITLHandler] = None,
        hitl_config: Optional[HITLConfig] = None,
        run_context: Optional[RunContext] = None,
        selected_skill_ids: Optional[List[str]] = None,
        tool_context_extra: Optional[Dict[str, Any]] = None,
        resume_task_id: Optional[str] = None,
        mcp_servers: Optional[List["McpServerConfig"]] = None,
        task_attachments: Optional[List[Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Execute a task with a new agent."""
        # Create a unique ID for this task
        task_id = str(uuid.uuid4())

        # Create agent with default settings if none provided
        settings = agent_settings or AgentSettings()
        # A per-call memory (e.g. a per-request AgentMemory from the API) takes
        # precedence over the orchestrator's shared_memory, so concurrent tasks
        # never share one transcript. Falls back to shared_memory when omitted.
        memory = memory or self.shared_memory
        if memory is None and run_context is None:
            memory = AgentMemory()
            self.shared_memory = memory

        # MCP (session-scoped): connect this run's servers and register their
        # tools into a PER-RUN registry copy — never the shared one, so nothing
        # leaks across concurrent runs/tenants. Connect off the event loop.
        run_registry = self.tools_registry
        mcp_set = None
        if mcp_servers:
            from pori.mcp import McpSessionSet

            run_registry = self.tools_registry.filtered()
            mcp_set = McpSessionSet(mcp_servers, connection_factory=self._mcp_factory)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, mcp_set.connect_and_register, run_registry)
            # MCP tools arrive after construction. Re-resolve the profile so a
            # server cannot reintroduce a denied tool or escape an allowlist.
            run_registry = self._resolve_profile(
                run_registry,
                skill_catalog=self.skill_catalog,
            )

        # Create and register the agent
        resolved_tool_context_extra = dict(tool_context_extra or {})
        if self.file_backend is not None:
            supplied_backend = resolved_tool_context_extra.get(FILE_BACKEND_CONTEXT_KEY)
            if (
                supplied_backend is not None
                and supplied_backend is not self.file_backend
            ):
                raise ValueError(
                    "tool_context_extra cannot replace the orchestrator's file backend"
                )
            resolved_tool_context_extra[FILE_BACKEND_CONTEXT_KEY] = self.file_backend
        resolved_skill_ids = list(
            dict.fromkeys(
                [
                    *sorted(
                        self.run_profile.required_skill_ids if self.run_profile else ()
                    ),
                    *(selected_skill_ids or []),
                ]
            )
        )
        agent = Agent(
            task=task,
            llm=self.llm,
            tools_registry=run_registry,
            settings=settings,
            memory=memory,
            sandbox_base_dir=sandbox_base_dir,
            hitl_handler=hitl_handler,
            hitl_config=hitl_config,
            run_context=run_context,
            skill_catalog=self.skill_catalog,
            skill_limit=max(self.skill_limit, len(resolved_skill_ids)),
            selected_skill_ids=resolved_skill_ids or None,
            system_prompt=self.system_prompt,
            model_capabilities=self.model_capabilities,
            evolution_repository=self.evolution_repository,
            soul_path=self.soul_path,
            soul_text=self.soul_text,
            load_project_context=self.load_project_context,
            tool_context_extra=resolved_tool_context_extra,
            resume_task_id=resume_task_id,
            task_attachments=task_attachments,
            cancellation_token=cancellation_token,
        )
        self.agents[task_id] = agent

        # Execute the task, forwarding step lifecycle callbacks so observers
        # (SSE streaming, CLI, monitors) receive real per-step events instead
        # of having to poll agent state.
        try:
            result = await agent.run(
                on_step_start=on_step_start,
                on_step_end=on_step_end,
                on_event=on_event,
                stream=stream,
            )
            summary = agent.result_summary()
            # SK-1 layer 2: mine the finished session for a reusable skill —
            # autonomously, non-blocking, and without mutating this run's memory.
            if getattr(settings, "background_review", False) and memory is not None:
                self._spawn_background_review(memory)
            return {
                "task_id": task_id,
                "success": result["completed"],
                "steps_taken": result["steps_taken"],
                "final_answer": summary.get("final_answer"),
                "reasoning": summary.get("reasoning"),
                "selected_skills": summary.get("selected_skills", []),
                "run_profile": (
                    self.run_profile.descriptor() if self.run_profile else None
                ),
                "artifacts": summary.get("artifacts", []),
                "metrics": summary.get("metrics"),
                "trace": summary.get("trace"),
                "summary": summary,
                "result": result,
                "agent": agent,  # Keep agent reference for accessing final answer
            }
        finally:
            # Tear down this run's MCP connections (session-scoped).
            if mcp_set is not None:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, mcp_set.close)
                except Exception:
                    logger.debug("MCP session-set close failed", exc_info=True)
            # Clean up from agents dict (but keep reference in result)
            if task_id in self.agents:
                del self.agents[task_id]

    def _spawn_background_review(self, memory: AgentMemory) -> None:
        """Fire-and-forget a cheap review agent over the finished session (SK-1
        layer 2). Never blocks the caller or mutates the live memory."""
        try:
            digest = self._session_digest(memory)
        except Exception:
            return
        if not digest.strip():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        future = loop.create_task(self._run_background_review(digest))
        self._review_tasks.add(future)
        future.add_done_callback(self._review_tasks.discard)

    async def _run_background_review(self, digest: str) -> None:
        """Run the isolated review agent; author a skill iff it finds one."""
        try:
            agent = Agent(
                task=build_background_review_prompt(digest),
                llm=self.llm,
                tools_registry=self.tools_registry,
                # Cheap and non-recursive: small step budget, and background review
                # disabled so a review can never spawn another review.
                settings=AgentSettings(max_steps=6, background_review=False),
                memory=AgentMemory(),  # isolated — cannot corrupt the real session
                skill_catalog=self.skill_catalog,
                skill_limit=self.skill_limit,
            )
            # Mark the write-origin so any skill authored here is recorded
            # agent-created (SK-2) and thus curatable — never a user's skill.
            with use_write_origin("background_review"):
                await agent.run()
        except Exception as exc:  # a background task must never surface an error
            logger.debug("Background review skipped: %s", exc)

    @staticmethod
    def _session_digest(memory: AgentMemory) -> str:
        try:
            return memory.get_recent_messages(12) or ""
        except Exception:
            return ""

    async def run_subagent(
        self,
        task: str,
        *,
        system_prompt: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        max_steps: int = 15,
        allow_delegation: bool = False,
        hitl_config: Any = None,
        hitl_handler: Optional[HITLHandler] = None,
        child_tool_context: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None,
    ) -> str:
        """Run an isolated sub-agent to a single result (the delegation primitive).

        The child gets a FRESH memory (its working transcript never touches the
        caller's context — only the returned answer does), a restricted tool surface,
        and a non-interactive HITL policy. Unless ``allow_delegation`` is set,
        ``delegate_task`` is stripped so the child cannot spawn its own sub-agents.
        """
        all_names = set(self.tools_registry.tools.keys())
        protected = set(self.tools_registry.protected_tools)
        exclude: set = set()
        if tool_names is not None:
            allowed = {t.strip() for t in tool_names if t.strip()} | protected
            exclude |= all_names - allowed
        if not allow_delegation:
            exclude |= {"delegate_task"}
        exclude -= protected  # never exclude protected/kernel tools

        registry = self.tools_registry
        if exclude:
            try:
                registry = self.tools_registry.filtered(exclude_tools=exclude)
            except Exception:  # never fail the run on a tool-restriction issue
                registry = self.tools_registry

        agent = Agent(
            task=task,
            llm=llm
            or self.llm,  # model-per-agent override (tier-resolved) else inherit
            tools_registry=registry,
            settings=AgentSettings(max_steps=max_steps, background_review=False),
            memory=AgentMemory(),  # isolated context — the point of a sub-agent
            skill_catalog=self.skill_catalog,
            skill_limit=self.skill_limit,
            soul_text=system_prompt,
            hitl_config=hitl_config,
            hitl_handler=hitl_handler,
            tool_context_extra={
                **(child_tool_context or {}),
                **(
                    {FILE_BACKEND_CONTEXT_KEY: self.file_backend}
                    if self.file_backend is not None
                    else {}
                ),
            },
        )
        await agent.run()
        summary = agent.result_summary()
        return (
            summary.get("final_answer")
            or summary.get("reasoning")
            or "(the sub-agent returned no answer)"
        )

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
        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
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
