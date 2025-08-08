from pydantic import BaseModel, Field
from typing import Dict
import random
from ..tools import tool_registry

Registry = tool_registry()


class RandomParams(BaseModel):
    min_val: int = Field(1, description="Minimum value")
    max_val: int = Field(1000, description="Maximum value")
    count: int = Field(1, description="How many numbers to generate")


class FibonacciParams(BaseModel):
    count: int = Field(
        10, description="How many Fibonacci numbers to sum (must be >= 0)"
    )


@Registry.tool(
    name="fibonacci_generator",
    description="Calculate the sum of the first N Fibonacci numbers",
)
def generate_fibonacci_tool(params: FibonacciParams, context: Dict):
    """Calculate the sum of the first *n* Fibonacci numbers.

    Returns the sum as a single value.
    """

    n = params.count

    if n < 0:
        return {"error": "Count must be non-negative"}

    if n == 0:
        return {"sum": 0, "count": 0}

    total = 0
    a, b = 1, 1

    for _ in range(n):
        total += a
        a, b = b, a + b

    return {"sum": total, "count": n}


@Registry.tool(name="random_generator", description="Generate a list of random numbers")
def generate_random_tool(params: RandomParams, context: Dict):
    min_val = params.min_val
    max_val = params.max_val
    count = params.count

    random_list = []
    if min_val > max_val:
        return {"error": "Maximum value is less the minimum value"}
    else:
        for i in range(count):
            random_num = random.randint(min_val, max_val)
            random_list.append(random_num)

        return {"numbers": random_list, "count": count}


def register_number_tool(registry=None):
    """Tools are auto-registered on import via decorator; kept for compatibility."""
    return None
