"""Human-readable status text for tool calls (activity descriptors).

`build_tool_preview` turns a raw tool call into a short, user-facing line — the
*detail* layer of an activity descriptor (e.g. "Running: npm run build"). It is
purely mechanical: it only sees the tool name and arguments, so it cannot infer
intent. The *intent* layer ("Rebuilding and validating the CV") is model-authored
and surfaced separately from the agent's ``next_goal``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_MAX_PREVIEW_CHARS = 80

# Generic fallback: tool name -> the single most meaningful argument to show.
_PRIMARY_ARG = {
    "think": "thoughts",
    "ask_user": "question",
    "answer": "final_answer",
    "web_search": "query",
}


def _oneline(value: Any) -> str:
    return " ".join(str(value).split())


def _truncate(text: str, max_len: int = _MAX_PREVIEW_CHARS) -> str:
    text = _oneline(text)
    if max_len > 3 and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_tool_preview(tool_name: str, params: Optional[Dict[str, Any]]) -> str:
    """Return a short, user-facing status line for a tool call.

    Never raw JSON. Falls back to the tool name when no useful argument exists.
    """
    p = params if isinstance(params, dict) else {}

    def first(*keys: str) -> Any:
        for key in keys:
            value = p.get(key)
            if value:
                return value
        return None

    # Verb-phrase special cases.
    if tool_name in ("write_file", "sandbox_write_file"):
        return _truncate(f"Writing {first('file_path', 'path') or 'a file'}")
    if tool_name == "read_file":
        return _truncate(f"Reading {first('file_path', 'path') or 'a file'}")
    if tool_name in ("create_directory", "sandbox_create_dir"):
        return _truncate(f"Creating folder {first('directory_path', 'path') or ''}")
    if tool_name in ("list_directory", "sandbox_list_dir"):
        return _truncate(f"Listing {first('directory_path', 'path') or ''}")
    if tool_name == "search_files":
        return _truncate(f"Searching for {first('content_search', 'pattern') or ''}")
    if tool_name == "bash":
        return _truncate(f"Running: {first('command') or ''}")
    if tool_name == "web_search":
        return _truncate(f"Searching the web: {first('query') or ''}")
    if tool_name == "update_plan":
        todos = p.get("todos") or []
        count = len(todos) if isinstance(todos, list) else 0
        return f"Updating the plan ({count} step(s))"
    if tool_name == "answer":
        return "Writing the answer"
    if tool_name == "done":
        return "Finishing up"

    # Generic fallback: name + primary argument.
    arg = first(_PRIMARY_ARG.get(tool_name, ""))
    return _truncate(f"{tool_name}: {arg}") if arg else tool_name


__all__ = ["build_tool_preview"]
