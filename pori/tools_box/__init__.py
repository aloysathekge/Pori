from .math_tools import register_math_tools
from .core_tools import register_core_tools
from .number_tools import register_number_tool


def register_all_tools(registry):
    register_math_tools(registry)
    register_core_tools(registry)
    register_number_tool(registry)
