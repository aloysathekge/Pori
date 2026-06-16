import ast
import operator

from pydantic import BaseModel, Field

from ..registry import tool_registry

Registry = tool_registry()

# Allowed operators for the safe arithmetic evaluator. Anything not listed here
# (calls, names, attribute access, etc.) is rejected, so the tool cannot be used
# to execute arbitrary code the way a bare eval() could.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guard against cheap denial-of-service via huge exponents (e.g. 10**10**10).
_MAX_EXPONENT = 1000


def _safe_eval(node: ast.AST):
    """Recursively evaluate an arithmetic AST, rejecting anything unsafe."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str):
    """Safely evaluate a basic arithmetic expression without using eval()."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


class CalculateParams(BaseModel):
    expression: str = Field(..., description="Math expression to calculate")


@Registry.tool(description="Calculate the result of a mathematical expression")
def calculate_tool(params: CalculateParams, context: dict):
    """Calculate the result of a math expression using a safe evaluator."""
    try:
        result = evaluate_expression(params.expression)
        return result
    except (ValueError, SyntaxError, ArithmeticError) as e:
        return {"error": str(e)}


def register_math_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
