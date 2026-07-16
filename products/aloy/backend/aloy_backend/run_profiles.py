"""Purpose-scoped run profiles owned by the Aloy product."""

from pori import RunProfile

from .skills import SURFACE_BUILDER_SKILL_ID
from .tools.surfaces import SURFACE_AUTHORING_TOOL_NAMES

SURFACE_BUILDER_ALLOWED_TOOLS = frozenset(
    {
        "edit_file",
        "list_directory",
        "read_file",
        "write_file",
        *SURFACE_AUTHORING_TOOL_NAMES,
    }
)

SURFACE_BUILDER_RUN_PROFILE = RunProfile(
    profile_id="aloy.surface-builder",
    version="1",
    system_prompt=(
        "You are building or revising one Event Surface. Treat generated UI "
        "source as a proposed artifact: use only the Surface SDK and declared "
        "intents, preserve canonical Event data and the last-good revision, "
        "and never bypass validation, preview, approval, or publication rails."
    ),
    allowed_tools=SURFACE_BUILDER_ALLOWED_TOOLS,
    required_tools=SURFACE_AUTHORING_TOOL_NAMES,
    required_skill_ids=frozenset({SURFACE_BUILDER_SKILL_ID}),
    required_model_capabilities=frozenset({"tools"}),
)


__all__ = ["SURFACE_BUILDER_ALLOWED_TOOLS", "SURFACE_BUILDER_RUN_PROFILE"]
