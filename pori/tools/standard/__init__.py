from importlib import metadata

from .core_tools import register_core_tools
from .filesystem_tools import register_filesystem_tools
from .internet_tools import register_internet_tools
from .math_tools import register_math_tools
from .number_tools import register_number_tool
from .spotify_tools import register_spotify_tools


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
        selected = eps.get("pori.tools", [])

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
    register_math_tools(registry)
    register_core_tools(registry)
    register_number_tool(registry)
    register_spotify_tools(registry)
    register_filesystem_tools(registry)
    register_internet_tools(registry)
    _load_tool_plugins(registry)
