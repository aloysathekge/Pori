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
