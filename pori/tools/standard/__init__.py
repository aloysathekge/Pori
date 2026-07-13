"""Assembles the built-in toolset: ``register_all_tools`` installs the
standard domains (core, filesystem, internet, planning, skills) plus external
plugins from the ``pori.tools`` entry-point group. ``STANDARD_KERNEL_TOOLS``
names the always-on kernel tools — the terminal ``answer`` / ``done`` among
them must never be removed.
"""

from importlib import metadata

from pori.capabilities import CapabilityGroup, CapabilityPrerequisites

from ..registry import CollisionPolicy, tool_registry
from .core_tools import register_core_tools
from .filesystem_tools import register_filesystem_tools
from .internet_tools import register_internet_tools
from .planning_tools import register_planning_tools
from .skills_tools import register_skill_tools

STANDARD_KERNEL_TOOLS = frozenset(
    {"answer", "done", "ask_user", "think", "skills_list", "skill_view", "update_plan"}
)


def _load_tool_plugins(registry) -> None:
    """Load external tool plugins via entrypoints (pori.tools group)."""
    try:
        eps = metadata.entry_points()
    except Exception:
        return

    # Python 3.10+ supports .select; older versions use dict-style access
    if hasattr(eps, "select"):
        selected = eps.select(group="pori.tools")
    else:  # pragma: no cover - legacy importlib_metadata API
        # Access .get dynamically: on modern Python EntryPoints has no .get,
        # and mypy's error code for this differs across versions.
        selected = getattr(eps, "get")("pori.tools", [])

    for ep in selected:
        # Avoid recursively loading this package's own standard tools via entrypoint
        if ep.module and ep.module.startswith("pori.tools.standard"):
            continue
        try:
            fn = ep.load()
        except Exception:
            continue
        try:
            fn(registry)
        except Exception:
            # Plugin misbehaviour should not break core tool registration
            continue


def register_all_tools(registry):
    """Register built-in tools and then load any plugin tools via entrypoints."""
    register_core_tools(registry)
    register_filesystem_tools(registry)
    register_internet_tools(registry)
    register_skill_tools(registry)
    register_planning_tools(registry)
    canonical = tool_registry()
    if registry is not canonical:
        for info in canonical.tools.values():
            registry.register_tool(
                info.name,
                info.param_model,
                info.function,
                info.description,
                side_effects=info.side_effects,
                collision_policy=CollisionPolicy.KEEP,
            )
    _define_standard_groups(registry)
    _load_tool_plugins(registry)


def _define_standard_groups(registry) -> None:
    definitions = (
        CapabilityGroup(
            name="kernel",
            description="Always-available control and completion tools.",
            tool_names=STANDARD_KERNEL_TOOLS,
            protected=True,
        ),
        CapabilityGroup(
            name="memory",
            description="Conversation, archival, and core-memory operations.",
            tool_names=frozenset(
                {
                    "remember",
                    "conversation_search",
                    "archival_memory_insert",
                    "archival_memory_search",
                    "core_memory_read",
                    "core_memory_append",
                    "core_memory_replace",
                    "core_memory_rethink",
                    "memory_insert",
                    "memory_rethink",
                }
            ),
        ),
        CapabilityGroup(
            name="filesystem",
            description="Bounded file and directory operations.",
            tool_names=frozenset(
                {
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_directory",
                    "search_files",
                    "file_info",
                    "create_directory",
                    "copy_file",
                    "move_file",
                    "delete_file",
                }
            ),
            max_output_chars=50_000,
        ),
        CapabilityGroup(
            name="internet",
            description="Public web retrieval.",
            tool_names=frozenset({"web_search"}),
            # Web search accepts either backend: Tavily (tavily-python) or
            # Google via Serper (a plain REST call, no extra package). The tool
            # picks whichever key is present and reports a per-backend error at
            # call time, so gate on "at least one key set" — not a hard AND on
            # Tavily, which would hide the tool from Google-only deployments.
            prerequisites=CapabilityPrerequisites(
                environment_any=("TAVILY_API_KEY", "SERPER_API_KEY")
            ),
            max_output_chars=50_000,
        ),
        CapabilityGroup(
            name="evolution",
            description="Governed self-evolution proposal drafting.",
            tool_names=frozenset({"propose_evolution"}),
        ),
    )
    for group in definitions:
        if group.name not in registry.groups:
            registry.define_group(group)
