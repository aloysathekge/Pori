from pydantic import BaseModel, Field
from ..registry import tool_registry

Registry = tool_registry()


class CalculateParams(BaseModel):
    expression: str = Field(..., description="Math expression to calculate")


@Registry.tool(description="Calculate the result of a mathematical expression")
def calculate_tool(params: CalculateParams, context: dict):
    """Calculate the result of a math expression."""
    try:
        # Warning: eval is unsafe for production use, this is just for demonstration
        result = eval(params.expression)
        return result
    except Exception as e:
        return {"error": str(e)}


def register_math_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
