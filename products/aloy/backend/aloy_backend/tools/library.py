"""fetch_my_file — bring a file from the user's library into the workspace.

The library manifest rides in the tool context (injected at run setup, no DB
access from tool code); the tool materializes the blob into the current
Event's shared sandbox uploads dir (hash-skipped) and returns the virtual
path. Gated per-run: when the user has no library files, the tool is
excluded from the surface entirely (the Google-tools pattern), so it costs
zero context until it's usable.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

LIBRARY_TOOL_NAMES = frozenset({"fetch_my_file"})


class FetchMyFileParams(BaseModel):
    name: str = Field(
        ...,
        description=("The file's name (as saved in the user's library) or its file id"),
    )


def fetch_my_file_tool(
    params: FetchMyFileParams, context: Dict[str, Any]
) -> Dict[str, Any]:
    files = context.get("library_files") or []
    if not files:
        return {"error": "The user's file library is empty."}

    query = params.name.strip()
    q = query.lower()
    match = next(
        (f for f in files if f["name"].lower() == q or f["file_id"] == query),
        None,
    )
    if match is None:
        partial = [f for f in files if q in f["name"].lower()]
        if len(partial) == 1:
            match = partial[0]
    if match is None:
        return {
            "error": f"No library file matches {query!r}.",
            "available_files": [f["name"] for f in files],
        }

    workspace_id = context.get("workspace_id") or context.get("thread_id")
    run_id = context.get("run_id") or context.get("task_id")
    if not workspace_id or not run_id:
        return {"error": "No workspace is available in this run."}

    from ..provisioning import provision_manifest_entry

    provisioned = provision_manifest_entry(str(workspace_id), str(run_id), match)
    if provisioned is None:
        return {
            "error": f"{match['name']!r} is in the library but its content "
            "is no longer available."
        }
    result = {
        "path": f"/mnt/user-data/uploads/{provisioned['name']}",
        "name": match["name"],
        "size_bytes": match["size_bytes"],
        "content_type": match["content_type"],
        "note": "The file is now in the workspace at the path above.",
    }
    if provisioned.get("extracted_text"):
        result["extracted_text_path"] = (
            f"/mnt/user-data/uploads/{provisioned['extracted_text']}"
        )
        result["note"] = (
            "The file is in the workspace. It is a binary document — read the "
            "plain-text copy at extracted_text_path instead."
        )
    return result
