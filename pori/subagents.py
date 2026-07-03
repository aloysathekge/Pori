"""Sub-agent delegation, Hermes-style.

`delegate_task` spawns focused child agents with an **isolated context** and a
**restricted toolset**, built from the delegated GOAL + CONTEXT (not a preset
persona). One task runs alone; several run as a **parallel batch**; the parent
blocks for the shaped summaries. A child is a `leaf` by default (cannot delegate
further); `role='orchestrator'` lets it decompose its own work, bounded by a
spawn-depth cap. Children run non-interactively, so risky HITL-gated tools are
**auto-denied** unless the caller opts into auto-approve.

The parent's context only ever sees the delegation call and the returned summary,
never a child's (potentially huge) working transcript.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

LEAF = "leaf"
ORCHESTRATOR = "orchestrator"
MAX_CONCURRENT_CHILDREN = 8  # cap on one batch (each child costs API tokens)
MAX_SPAWN_DEPTH = 2  # parent(0) -> child(1) -> grandchild(2, forced leaf)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")


@dataclass(frozen=True)
class AgentDefinition:
    """A curated specialist (Claude Code style): a tuned prompt + tool allowlist."""

    name: str
    description: str  # when to use this specialist
    prompt: str  # the specialist's system prompt (the "how")
    tools: Optional[Tuple[str, ...]] = None  # allowlist; None = all tools
    model: Optional[str] = None  # reserved — per-agent model is a future step


def parse_agent_markdown(text: str, *, fallback_name: str) -> Optional[AgentDefinition]:
    """Parse a specialist ``.md`` (--- YAML frontmatter --- then prompt body)."""
    text = text.strip()
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    name = str(meta.get("name") or fallback_name).strip().lower()
    description = str(meta.get("description") or "").strip()
    prompt = parts[2].strip()
    if not _SLUG_RE.match(name) or not description or not prompt:
        return None

    tools_raw = meta.get("tools")
    tools: Optional[Tuple[str, ...]] = None
    if isinstance(tools_raw, str):
        tools = tuple(t.strip() for t in tools_raw.split(",") if t.strip())
    elif isinstance(tools_raw, (list, tuple)):
        tools = tuple(str(t).strip() for t in tools_raw if str(t).strip())
    model = meta.get("model")
    return AgentDefinition(
        name=name,
        description=description,
        prompt=prompt,
        tools=tools,
        model=str(model).strip() if model else None,
    )


class AgentCatalog:
    """Optional curated specialists loaded from a directory (Claude Code style).

    A layer *on* the Hermes base: a ``delegate_task`` item may name an ``agent`` to
    get a tuned prompt + tool allowlist; without one, delegation stays goal-driven.
    """

    def __init__(self, definitions: Optional[Dict[str, AgentDefinition]] = None):
        self._by_name: Dict[str, AgentDefinition] = dict(definitions or {})

    @classmethod
    def load(cls, directory: Optional[Path]) -> "AgentCatalog":
        found: Dict[str, AgentDefinition] = {}
        if directory is not None and Path(directory).is_dir():
            for path in sorted(Path(directory).glob("*.md")):
                try:
                    definition = parse_agent_markdown(
                        path.read_text(encoding="utf-8"), fallback_name=path.stem
                    )
                except OSError:
                    continue
                if definition is not None:
                    found[definition.name] = definition
        return cls(found)

    def resolve(self, name: str) -> Optional[AgentDefinition]:
        return self._by_name.get((name or "").strip().lower())

    def names(self) -> List[str]:
        return sorted(self._by_name)


def build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    role: str = LEAF,
    child_depth: int = 1,
    max_depth: int = MAX_SPAWN_DEPTH,
    specialist_prompt: Optional[str] = None,
) -> str:
    """Build a focused child system prompt from the goal + context (Hermes style).

    If ``specialist_prompt`` is given (a curated agent from the catalog), it frames
    *how* the child works; the goal + context say *what* to do this time.
    """
    header = (
        specialist_prompt.strip()
        if specialist_prompt and specialist_prompt.strip()
        else "You are a focused sub-agent working on a specific delegated task."
    )
    parts = [
        header,
        "",
        f"YOUR TASK:\n{goal.strip()}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context.strip()}")
    parts.append(
        "\nComplete this task using the tools available to you. When finished, "
        "return a clear, concise summary of what you did, what you found or "
        "accomplished, any files you created or modified, and any issues — your "
        "response is returned to the parent agent as the result, so include exactly "
        "what it needs and nothing else."
    )
    if role == ORCHESTRATOR and child_depth < max_depth:
        parts.append(
            "\n## Delegation\nYou may call `delegate_task` to spawn your OWN "
            "sub-agents for independent, reasoning-heavy subtasks that would flood "
            "your context. Do trivial or single-step work yourself, and synthesize "
            f"your workers' results into your summary. You are at depth {child_depth}; "
            f"the tree is capped at depth {max_depth}."
        )
    return "\n".join(parts)


def _resolve_specialist(
    catalog: Optional["AgentCatalog"], agent_name: str
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return (specialist_prompt, tool_names) for a named agent, or raise ValueError
    listing the available specialists. Empty name -> (None, None) = goal-driven."""
    if not agent_name:
        return None, None
    definition = catalog.resolve(agent_name) if catalog else None
    if definition is None:
        available = (
            ", ".join(catalog.names())
            if catalog and catalog.names()
            else "(none defined)"
        )
        raise ValueError(f"unknown agent {agent_name!r}; available: {available}")
    return definition.prompt, (list(definition.tools) if definition.tools else None)


def _auto_deny_handler() -> Any:
    """A non-interactive HITL handler that rejects every gated action — children
    have no user to prompt, so deny is the safe default."""
    from pori.hitl import ApprovalResponse, Decision, HITLHandler

    class _AutoDeny(HITLHandler):
        async def request_approval(self, request: Any) -> Any:
            return ApprovalResponse(
                decisions=[
                    Decision(
                        type="reject",
                        message="Auto-denied: a sub-agent cannot request approval.",
                    )
                    for _ in request.action_requests
                ]
            )

    return _AutoDeny()


def _run_coro_in_thread(coro_factory: Callable[[], Any]) -> Any:
    """Run a coroutine to completion on its own loop in a worker thread, blocking
    the caller — lets the sync `delegate_task` tool run async children without
    needing (or deadlocking) the caller's event loop."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(coro_factory())
        except Exception as exc:  # surfaced to the tool, not swallowed
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def make_delegate_runner(
    orchestrator: Any,
    *,
    catalog: Optional[AgentCatalog] = None,
    hitl_config: Any = None,
    subagent_auto_approve: bool = False,
    max_steps: int = 15,
    child_depth: int = 1,
    max_depth: int = MAX_SPAWN_DEPTH,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build the sync ``delegate_runner`` for the tool context.

    Runs one task alone or several as a concurrent batch; each child gets an
    isolated context, a role-restricted toolset, and a non-interactive HITL policy
    (auto-deny by default, so a child can't run a risky gated tool unapproved).
    Recurses to give an ``orchestrator`` child a depth+1 runner so it can delegate
    too — a ``leaf`` child gets no runner, so it cannot (bounded by ``max_depth``).
    """

    async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(item.get("goal") or "").strip()
        if not goal:
            return {"success": False, "error": "each task needs a non-empty goal"}
        role = str(item.get("role") or LEAF).strip().lower()
        agent_name = str(item.get("agent") or "").strip()
        try:
            specialist_prompt, tool_names = _resolve_specialist(catalog, agent_name)
        except ValueError as exc:
            return {"success": False, "agent": agent_name, "error": str(exc)}

        can_delegate = role == ORCHESTRATOR and child_depth < max_depth
        child_context: Optional[Dict[str, Any]] = None
        if can_delegate:
            child_context = {
                "delegate_runner": make_delegate_runner(
                    orchestrator,
                    catalog=catalog,
                    hitl_config=hitl_config,
                    subagent_auto_approve=subagent_auto_approve,
                    max_steps=max_steps,
                    child_depth=child_depth + 1,
                    max_depth=max_depth,
                )
            }
        try:
            result = await orchestrator.run_subagent(
                goal,
                system_prompt=build_child_system_prompt(
                    goal,
                    item.get("context"),
                    role=role,
                    child_depth=child_depth,
                    max_depth=max_depth,
                    specialist_prompt=specialist_prompt,
                ),
                tool_names=tool_names,
                max_steps=max_steps,
                allow_delegation=can_delegate,
                hitl_config=None if subagent_auto_approve else hitl_config,
                hitl_handler=None if subagent_auto_approve else _auto_deny_handler(),
                child_tool_context=child_context,
            )
            return {
                "success": True,
                "role": role,
                "agent": agent_name or None,
                "result": result,
            }
        except Exception as exc:
            return {
                "success": False,
                "role": role,
                "agent": agent_name or None,
                "error": str(exc),
            }

    def run(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async def run_all() -> List[Dict[str, Any]]:
            return list(await asyncio.gather(*[_one(item) for item in items]))

        return _run_coro_in_thread(run_all)

    return run


def make_background_delegate(
    orchestrator: Any,
    registry: Any,
    *,
    catalog: Optional[AgentCatalog] = None,
    hitl_config: Any = None,
    subagent_auto_approve: bool = False,
    max_steps: int = 15,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Build the ``background_delegate`` callable: dispatch each task to run in the
    background (as a leaf child) via ``registry``, returning handles immediately so
    the parent keeps working. Completed children surface later, drained by the CLI.
    """

    def dispatch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in items:
            goal = str(item.get("goal") or "").strip()
            if not goal:
                out.append(
                    {"success": False, "error": "each task needs a non-empty goal"}
                )
                continue
            agent_name = str(item.get("agent") or "").strip()
            try:
                specialist_prompt, tool_names = _resolve_specialist(catalog, agent_name)
            except ValueError as exc:
                out.append({"success": False, "agent": agent_name, "error": str(exc)})
                continue
            prompt = build_child_system_prompt(
                goal,
                item.get("context"),
                role=LEAF,
                specialist_prompt=specialist_prompt,
            )

            def factory(
                goal: str = goal,
                prompt: str = prompt,
                tool_names: Optional[List[str]] = tool_names,
            ) -> Any:
                return orchestrator.run_subagent(
                    goal,
                    system_prompt=prompt,
                    tool_names=tool_names,
                    max_steps=max_steps,
                    allow_delegation=False,
                    hitl_config=None if subagent_auto_approve else hitl_config,
                    hitl_handler=(
                        None if subagent_auto_approve else _auto_deny_handler()
                    ),
                )

            out.append(
                {
                    "success": True,
                    "handle": registry.dispatch(goal, factory),
                    "goal": goal,
                    "agent": agent_name or None,
                }
            )
        return out

    return dispatch
