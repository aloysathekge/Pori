"""Sub-agent definitions + task delegation (the Claude Code / deepagents pattern).

A sub-agent is a markdown file (YAML frontmatter + a system-prompt body) describing
a focused specialist the main agent can delegate a subtask to. The ``task`` tool
spawns one in an ISOLATED context — its own memory and a restricted tool surface —
runs it to a single result, and returns just that. The subagent's (potentially huge)
working transcript never enters the parent's context, so the parent stays clean.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

GENERAL_PURPOSE = "general-purpose"

_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for researching complex questions, searching code/files, "
    "and multi-step subtasks. Use when a subtask deserves its own focused context."
)
_GENERAL_PURPOSE_PROMPT = (
    "You are a focused sub-agent handling one delegated task autonomously. Complete "
    "the task fully, then return a single concise result containing exactly what the "
    "caller asked for — they see only your final answer, nothing of your working."
)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str  # tells the main agent WHEN to delegate here (drives routing)
    prompt: str  # the sub-agent's system prompt
    tools: Optional[Tuple[str, ...]] = None  # None = all tools; else a restricted set
    model: Optional[str] = None  # None/"inherit" = the main model (per-model: future)


def _general_purpose() -> AgentDefinition:
    return AgentDefinition(
        name=GENERAL_PURPOSE,
        description=_GENERAL_PURPOSE_DESCRIPTION,
        prompt=_GENERAL_PURPOSE_PROMPT,
    )


def parse_agent_markdown(text: str, *, fallback_name: str) -> Optional[AgentDefinition]:
    """Parse an agent ``.md`` (--- YAML frontmatter --- then the system-prompt body)."""
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
    model = str(model).strip() if model else None
    return AgentDefinition(
        name=name, description=description, prompt=prompt, tools=tools, model=model
    )


class AgentCatalog:
    """Sub-agent definitions from a directory plus the built-in general-purpose one."""

    def __init__(self, definitions: Optional[Dict[str, AgentDefinition]] = None):
        self._by_name: Dict[str, AgentDefinition] = dict(definitions or {})
        self._by_name.setdefault(GENERAL_PURPOSE, _general_purpose())

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "AgentCatalog":
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

    def types(self) -> List[AgentDefinition]:
        return [self._by_name[name] for name in sorted(self._by_name)]

    def describe_types(self) -> str:
        return "\n".join(f"- {d.name}: {d.description}" for d in self.types())


MAX_PARALLEL_SUBAGENTS = 8


def _run_coro_in_thread(coro_factory: Callable[[], Any]) -> Any:
    """Run a coroutine to completion on its own loop in a worker thread, blocking
    the caller. Lets the sync ``task`` tools run async sub-agents without needing
    (or deadlocking) the caller's event loop."""
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


def make_subagent_runner(
    orchestrator: Any, catalog: AgentCatalog, *, max_steps: int = 15
) -> Callable[[str, str], str]:
    """Build the sync ``subagent_runner`` injected into the tool context.

    It resolves a sub-agent definition and runs it to completion (in a worker
    thread), returning the final answer. An unknown type raises with the list of
    available types so the model can correct itself.
    """

    def run(subagent_type: str, task: str) -> str:
        definition = catalog.resolve(subagent_type)
        if definition is None:
            available = ", ".join(d.name for d in catalog.types())
            raise ValueError(
                f"Unknown subagent_type {subagent_type!r}. Available: {available}"
            )
        return _run_coro_in_thread(
            lambda: orchestrator.run_subagent(
                task,
                system_prompt=definition.prompt,
                tool_names=list(definition.tools) if definition.tools else None,
                max_steps=max_steps,
            )
        )

    return run


def make_parallel_subagent_runner(
    orchestrator: Any, catalog: AgentCatalog, *, max_steps: int = 15
) -> Callable[[List[Tuple[str, str]]], List[Dict[str, Any]]]:
    """Build the sync ``parallel_subagent_runner``: run several INDEPENDENT subtasks
    concurrently, each in its own isolated sub-agent, on one loop in one worker
    thread. Collects a per-task result — one failing does not fail the rest."""

    async def _one(subagent_type: str, task: str) -> Dict[str, Any]:
        definition = catalog.resolve(subagent_type)
        if definition is None:
            available = ", ".join(d.name for d in catalog.types())
            return {
                "subagent_type": subagent_type,
                "success": False,
                "error": f"Unknown subagent_type {subagent_type!r}. Available: {available}",
            }
        try:
            result = await orchestrator.run_subagent(
                task,
                system_prompt=definition.prompt,
                tool_names=list(definition.tools) if definition.tools else None,
                max_steps=max_steps,
            )
            return {"subagent_type": subagent_type, "success": True, "result": result}
        except Exception as exc:
            return {"subagent_type": subagent_type, "success": False, "error": str(exc)}

    def run(items: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        async def run_all() -> List[Dict[str, Any]]:
            return list(await asyncio.gather(*[_one(st, task) for st, task in items]))

        return _run_coro_in_thread(run_all)

    return run
