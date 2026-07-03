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
import threading
from typing import Any, Callable, Dict, List, Optional

LEAF = "leaf"
ORCHESTRATOR = "orchestrator"
MAX_CONCURRENT_CHILDREN = 8  # cap on one batch (each child costs API tokens)
MAX_SPAWN_DEPTH = 2  # parent(0) -> child(1) -> grandchild(2, forced leaf)


def build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    role: str = LEAF,
    child_depth: int = 1,
    max_depth: int = MAX_SPAWN_DEPTH,
) -> str:
    """Build a focused child system prompt from the goal + context (Hermes style)."""
    parts = [
        "You are a focused sub-agent working on a specific delegated task.",
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
        can_delegate = role == ORCHESTRATOR and child_depth < max_depth

        child_context: Optional[Dict[str, Any]] = None
        if can_delegate:
            child_context = {
                "delegate_runner": make_delegate_runner(
                    orchestrator,
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
                ),
                max_steps=max_steps,
                allow_delegation=can_delegate,
                hitl_config=None if subagent_auto_approve else hitl_config,
                hitl_handler=None if subagent_auto_approve else _auto_deny_handler(),
                child_tool_context=child_context,
            )
            return {"success": True, "role": role, "result": result}
        except Exception as exc:
            return {"success": False, "role": role, "error": str(exc)}

    def run(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async def run_all() -> List[Dict[str, Any]]:
            return list(await asyncio.gather(*[_one(item) for item in items]))

        return _run_coro_in_thread(run_all)

    return run


def make_background_delegate(
    orchestrator: Any,
    registry: Any,
    *,
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
            prompt = build_child_system_prompt(goal, item.get("context"), role=LEAF)

            def factory(goal: str = goal, prompt: str = prompt) -> Any:
                return orchestrator.run_subagent(
                    goal,
                    system_prompt=prompt,
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
                }
            )
        return out

    return dispatch
