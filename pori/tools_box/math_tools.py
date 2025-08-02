from pydantic import BaseModel, Field


class CalculateParams(BaseModel):
    expression: str = Field(..., description="Math expression to calculate")


def calculate_tool(params: CalculateParams, context: dict):
    """Calculate the result of a math expression."""
    try:
        # Warning: eval is unsafe for production use, this is just for demonstration
        result = eval(params.expression)
        return result
    except Exception as e:
        return {"error": str(e)}


def register_math_tools(registry):
    """Register math-related tools with the given registry."""
    registry.register_tool(
        name="calculate",
        param_model=CalculateParams,
        function=calculate_tool,
        description="Calculate the result of a mathematical expression",
    )
