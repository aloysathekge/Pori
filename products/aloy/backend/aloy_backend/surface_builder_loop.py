"""Bounded Builder loop over a provider-neutral Surface workspace.

Models receive a small authoring tool vocabulary instead of returning an
entire application in one schema-bound response.  Native provider tool calls
are preferred when available; the same calls may be returned as a compact JSON
action envelope by text-only models.  The host validates every argument,
executes no model-provided command, and accepts ``finish_candidate`` only after
the exact current source compiles successfully.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pori import AssistantMessage, BudgetLedger, ToolResultMessage, UserMessage

from .surface_development_workspace import (
    FinishedSurfaceWorkspace,
    SurfaceDevelopmentWorkspace,
    SurfaceWorkspaceEdit,
)
from .surface_pipeline import (
    SurfaceCandidate,
    bind_surface_manifest_primary_jobs,
    surface_primary_job_contract_diagnostics,
)

MAX_SURFACE_BUILDER_TURNS = 20
MAX_SURFACE_ACTIONS_PER_TURN = 12
MAX_SURFACE_TOOL_RESULT_CHARS = 60_000

SurfaceBuilderToolName = Literal[
    "list_files",
    "read_file",
    "search_source",
    "write_file",
    "replace_text",
    "delete_file",
    "run_typecheck",
    "run_preview_check",
    "read_diagnostics",
    "finish_candidate",
]


class SurfaceBuilderAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default="", max_length=200)
    name: SurfaceBuilderToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class SurfaceBuilderActionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: list[SurfaceBuilderAction] = Field(
        min_length=1,
        max_length=MAX_SURFACE_ACTIONS_PER_TURN,
    )


class SurfaceBuilderLoopError(RuntimeError):
    """The model failed to complete the bounded workspace protocol."""


class SurfaceBuilderLoopResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    finished: FinishedSurfaceWorkspace
    turns: int
    tool_calls: int
    protocol: Literal["native_tools", "action_json", "mixed"]
    transcript: list[dict[str, Any]] = Field(default_factory=list)


SURFACE_WORKSPACE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List every editable Surface project file.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a bounded line range from one UTF-8 Surface source file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "line_count": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_source",
        "description": "Find text in the current Surface source.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Create or replace one complete Surface source file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_text",
        "description": "Replace one exact, unique source fragment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "match": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "required": ["path", "match", "replacement"],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_file",
        "description": "Delete one Surface source file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_typecheck",
        "description": "Run the fixed host validation and TypeScript compiler.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "run_preview_check",
        "description": "Build the exact current source with the trusted preview toolchain.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "read_diagnostics",
        "description": "Read the most recent trusted check result.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "finish_candidate",
        "description": "Submit the exact checked source to Aloy's full publication gate.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


def surface_workspace_protocol_instructions(*, native_tools: bool) -> str:
    common = (
        "Work against the persistent Surface project using only the host operations below. "
        "Inspect before editing, make the smallest coherent change, run a trusted check, "
        "repair every diagnostic, and call finish_candidate only when the current source is "
        "ready. Never invent shell commands, package dependencies, URLs, Event facts, or "
        "publication success. The host preserves canonical Event state and the last-good live "
        "Surface."
    )
    if native_tools:
        return common
    return (
        common
        + "\n\nAvailable operations and their exact input contracts:\n"
        + json.dumps(SURFACE_WORKSPACE_TOOL_SCHEMAS, ensure_ascii=False)
        + "\n\nReturn only one JSON object shaped as "
        + '{"actions":[{"name":"list_files","arguments":{}}]}. '
        + "Use only the listed operation names and no Markdown fences. The host "
        + "will return each operation result in the next user message."
    )


def _json_action_envelope(value: str) -> SurfaceBuilderActionEnvelope:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    decoder = json.JSONDecoder()
    candidates = [index for index, character in enumerate(text) if character == "{"]
    for index in candidates:
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return SurfaceBuilderActionEnvelope.model_validate(parsed)
        except (json.JSONDecodeError, ValueError):
            continue
    raise SurfaceBuilderLoopError("Builder returned no valid Surface action envelope")


def _required_string(arguments: dict[str, Any], name: str, *, maximum: int) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SurfaceBuilderLoopError(f"Surface tool argument {name!r} is invalid")
    return value


def _no_extra(arguments: dict[str, Any], allowed: set[str]) -> None:
    extras = set(arguments) - allowed
    if extras:
        raise SurfaceBuilderLoopError(
            "Surface tool received unsupported arguments: " + ", ".join(sorted(extras))
        )


def _optional_int(
    arguments: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise SurfaceBuilderLoopError(f"Surface tool argument {name!r} is invalid")
    return value


def _bounded_result(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) <= MAX_SURFACE_TOOL_RESULT_CHARS:
        return rendered
    return json.dumps(
        {
            "truncated": True,
            "characters": len(rendered),
            "head": rendered[: MAX_SURFACE_TOOL_RESULT_CHARS // 2],
            "tail": rendered[-MAX_SURFACE_TOOL_RESULT_CHARS // 2 :],
        },
        ensure_ascii=False,
    )


async def _contract_diagnostics(
    workspace: SurfaceDevelopmentWorkspace,
    *,
    summary: str,
    required_primary_jobs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Evaluate the host publication contract against the current workspace.

    Uses the exact bind + validation functions the paid host gate runs later,
    so a green ``finish_candidate`` can never be deterministically rejected by
    the primary-job contract.
    """
    files = {
        path: await workspace.read_file(path) for path in await workspace.list_files()
    }
    candidate = SurfaceCandidate.model_validate(
        {
            "summary": summary,
            "primary_jobs": [item["description"] for item in required_primary_jobs],
            "files": [
                {"path": f"/workspace{path}", "content": content}
                for path, content in sorted(files.items())
            ],
        }
    )
    bound = bind_surface_manifest_primary_jobs(
        candidate,
        required_primary_jobs=required_primary_jobs,
    )
    return surface_primary_job_contract_diagnostics(
        bound,
        required_primary_jobs=required_primary_jobs,
    )


async def _execute_action(
    workspace: SurfaceDevelopmentWorkspace,
    action: SurfaceBuilderAction,
    *,
    primary_jobs: list[str],
    required_primary_jobs: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], FinishedSurfaceWorkspace | None]:
    arguments = action.arguments
    if action.name == "list_files":
        _no_extra(arguments, set())
        return {"files": await workspace.list_files()}, None
    if action.name == "read_file":
        _no_extra(arguments, {"path", "start_line", "line_count"})
        path = _required_string(arguments, "path", maximum=500)
        start_line = _optional_int(
            arguments,
            "start_line",
            default=1,
            minimum=1,
            maximum=1_000_000,
        )
        line_count = _optional_int(
            arguments,
            "line_count",
            default=400,
            minimum=1,
            maximum=1_000,
        )
        content = await workspace.read_file(path)
        lines = content.splitlines(keepends=True)
        start_index = min(len(lines), start_line - 1)
        end_index = min(len(lines), start_index + line_count)
        return {
            "path": path,
            "start_line": start_index + 1,
            "end_line": end_index,
            "total_lines": len(lines),
            "truncated": end_index < len(lines),
            "content": "".join(lines[start_index:end_index]),
        }, None
    if action.name == "search_source":
        _no_extra(arguments, {"query"})
        query = _required_string(arguments, "query", maximum=500)
        return {"matches": await workspace.search_source(query)}, None
    if action.name == "write_file":
        _no_extra(arguments, {"path", "content"})
        edit = SurfaceWorkspaceEdit(
            operation="write",
            path=_required_string(arguments, "path", maximum=500),
            content=_required_string(arguments, "content", maximum=256 * 1024),
        )
        return await workspace.apply([edit]), None
    if action.name == "replace_text":
        _no_extra(arguments, {"path", "match", "replacement"})
        replacement = arguments.get("replacement")
        if not isinstance(replacement, str) or len(replacement) > 256 * 1024:
            raise SurfaceBuilderLoopError(
                "Surface tool argument 'replacement' is invalid"
            )
        edit = SurfaceWorkspaceEdit(
            operation="replace_text",
            path=_required_string(arguments, "path", maximum=500),
            match=_required_string(arguments, "match", maximum=256 * 1024),
            replacement=replacement,
        )
        return await workspace.apply([edit]), None
    if action.name == "delete_file":
        _no_extra(arguments, {"path"})
        edit = SurfaceWorkspaceEdit(
            operation="delete",
            path=_required_string(arguments, "path", maximum=500),
        )
        return await workspace.apply([edit]), None
    if action.name in {"run_typecheck", "run_preview_check"}:
        _no_extra(arguments, set())
        check = await workspace.check()
        return check.model_dump(mode="json"), None
    if action.name == "read_diagnostics":
        _no_extra(arguments, set())
        diagnostics_check = await workspace.diagnostics()
        return (
            diagnostics_check.model_dump(mode="json")
            if diagnostics_check is not None
            else {"status": "not_run"}
        ), None
    if action.name == "finish_candidate":
        _no_extra(arguments, {"summary"})
        summary = _required_string(arguments, "summary", maximum=2_000)
        # Reuse a successful check only while the workspace still recognizes
        # it as current. Every edit clears this result and ``finish`` verifies
        # the exact source fingerprint again, avoiding a duplicate compile for
        # the common check -> finish sequence.
        finish_check = await workspace.diagnostics()
        if finish_check is None:
            finish_check = await workspace.check()
        if finish_check.status != "succeeded":
            return {
                "accepted": False,
                "reason": "trusted_check_failed",
                "check": finish_check.model_dump(mode="json"),
            }, None
        if required_primary_jobs:
            contract = await _contract_diagnostics(
                workspace,
                summary=summary,
                required_primary_jobs=required_primary_jobs,
            )
            if contract:
                return {
                    "accepted": False,
                    "reason": "primary_job_contract_failed",
                    "required_primary_jobs": required_primary_jobs,
                    "diagnostics": contract,
                }, None
        finished = await workspace.finish(summary=summary, primary_jobs=primary_jobs)
        return {
            "accepted": True,
            "source_fingerprint": finished.receipt.source_fingerprint,
            "candidate_commit": finished.receipt.candidate_commit,
        }, finished
    raise SurfaceBuilderLoopError(f"Unsupported Surface workspace tool: {action.name}")


async def _provider_turn(
    invoke: Callable[[], Awaitable[Any]],
    *,
    timeout_seconds: float | None,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None,
    turn_number: int,
    tool_calls: int,
) -> Any:
    """Bound one non-streaming provider call and retry a single hang.

    Providers occasionally accept a request and never answer; without a client
    timeout that silence rides until the run-level stall watchdog terminally
    fails the whole paid run. One bounded retry isolates a transport hang from
    the Builder's repair budget. The retry emits progress so the watchdog's
    idle clock restarts before the second attempt.
    """
    attempts = 2 if timeout_seconds is not None else 1
    for attempt in range(1, attempts + 1):
        try:
            if timeout_seconds is None:
                return await invoke()
            return await asyncio.wait_for(invoke(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            if on_progress is not None:
                await on_progress(
                    {
                        "stage": "retrying_provider_call",
                        "turn": turn_number,
                        "tool_calls": tool_calls,
                        "attempt": attempt,
                    }
                )
            if attempt == attempts:
                raise SurfaceBuilderLoopError(
                    "The Builder provider produced no response within "
                    f"{timeout_seconds:g} seconds twice in a row"
                )


async def run_surface_builder_loop(
    *,
    llm: Any,
    workspace: SurfaceDevelopmentWorkspace,
    messages: list[Any],
    primary_jobs: list[str],
    capabilities: set[str] | frozenset[str],
    required_primary_jobs: list[dict[str, str]] | None = None,
    budget_ledger: BudgetLedger | None = None,
    max_turns: int = MAX_SURFACE_BUILDER_TURNS,
    provider_call_timeout_seconds: float | None = None,
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> SurfaceBuilderLoopResult:
    """Run one model against a persistent workspace until checked source finishes."""
    native_available = "tools" in capabilities and callable(
        getattr(llm, "ainvoke_tools", None)
    )
    history = list(messages)
    history[0].content += "\n\n" + surface_workspace_protocol_instructions(
        native_tools=native_available
    )
    transcript: list[dict[str, Any]] = []
    used_native = False
    used_json = False
    tool_calls = 0

    for turn_number in range(1, max_turns + 1):
        if budget_ledger is not None:
            budget_ledger.consume_step()
        if on_progress is not None:
            await on_progress(
                {
                    "stage": "editing_workspace",
                    "turn": turn_number,
                    "tool_calls": tool_calls,
                }
            )
        turn_used_native = False
        if native_available:
            turn = await _provider_turn(
                lambda: llm.ainvoke_tools(history, SURFACE_WORKSPACE_TOOL_SCHEMAS),
                timeout_seconds=provider_call_timeout_seconds,
                on_progress=on_progress,
                turn_number=turn_number,
                tool_calls=tool_calls,
            )
            actions = [
                SurfaceBuilderAction(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                for call in turn.tool_calls
            ]
            if actions:
                used_native = True
                turn_used_native = True
            elif turn.text.strip():
                actions = list(_json_action_envelope(turn.text).actions)
                used_json = True
            else:
                raise SurfaceBuilderLoopError("Builder returned no workspace action")
            history.append(
                AssistantMessage(
                    content=turn.text,
                    tool_calls=turn.tool_calls,
                )
            )
        else:
            raw = await _provider_turn(
                lambda: llm.ainvoke(history),
                timeout_seconds=provider_call_timeout_seconds,
                on_progress=on_progress,
                turn_number=turn_number,
                tool_calls=tool_calls,
            )
            text = raw if isinstance(raw, str) else str(raw)
            actions = list(_json_action_envelope(text).actions)
            used_json = True
            history.append(AssistantMessage(content=text))

        if len(actions) > MAX_SURFACE_ACTIONS_PER_TURN:
            raise SurfaceBuilderLoopError("Builder returned too many workspace actions")
        if any(action.name == "finish_candidate" for action in actions[:-1]):
            raise SurfaceBuilderLoopError(
                "finish_candidate must be the final action in a Builder turn"
            )
        results: list[dict[str, Any]] = []
        for index, action in enumerate(actions, start=1):
            if budget_ledger is not None:
                budget_ledger.consume_tool_call()
            tool_calls += 1
            action_id = action.id or f"surface-action-{turn_number}-{index}"
            result: dict[str, Any] | None = None
            payload: dict[str, Any]
            try:
                result, finished = await _execute_action(
                    workspace,
                    action,
                    primary_jobs=primary_jobs,
                    required_primary_jobs=required_primary_jobs,
                )
                payload = {"ok": True, "name": action.name, "result": result}
            except Exception as exc:
                finished = None
                payload = {
                    "ok": False,
                    "name": action.name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            transcript.append(
                {
                    "turn": turn_number,
                    "action": action.name,
                    "ok": payload["ok"],
                    "result": result if payload["ok"] else None,
                    "error": payload.get("error"),
                }
            )
            results.append(payload)
            if turn_used_native:
                history.append(
                    ToolResultMessage(
                        tool_call_id=action_id,
                        content=_bounded_result(payload),
                    )
                )
            if finished is not None:
                protocol: Literal["native_tools", "action_json", "mixed"] = (
                    "mixed"
                    if used_native and used_json
                    else "native_tools" if used_native else "action_json"
                )
                return SurfaceBuilderLoopResult(
                    finished=finished,
                    turns=turn_number,
                    tool_calls=tool_calls,
                    protocol=protocol,
                    transcript=transcript,
                )
        if not turn_used_native:
            history.append(
                UserMessage(
                    content=(
                        "Trusted workspace results for the previous action envelope:\n"
                        + _bounded_result(results)
                        + "\nContinue with the next JSON action envelope."
                    )
                )
            )

    raise SurfaceBuilderLoopError(
        f"Builder did not finish checked Surface source within {max_turns} turns"
    )


__all__ = [
    "MAX_SURFACE_BUILDER_TURNS",
    "SURFACE_WORKSPACE_TOOL_SCHEMAS",
    "SurfaceBuilderAction",
    "SurfaceBuilderActionEnvelope",
    "SurfaceBuilderLoopError",
    "SurfaceBuilderLoopResult",
    "run_surface_builder_loop",
    "surface_workspace_protocol_instructions",
]
