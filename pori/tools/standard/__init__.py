from .math_tools import register_math_tools
from .core_tools import register_core_tools
from .number_tools import register_number_tool
from .spotify_tools import register_spotify_tools
from .filesystem_tools import register_filesystem_tools


def register_all_tools(registry):
    register_math_tools(registry)
    register_core_tools(registry)
    register_number_tool(registry)
    register_spotify_tools(registry)
    register_filesystem_tools(registry)
