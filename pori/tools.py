from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints
from functools import wraps
from pydantic import BaseModel, create_model


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    param_model: Type[BaseModel]
    function: Callable
    description: str


class ToolRegistry:
    """Registry for tools that the agent can use."""

    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}

    def register_tool(
        self,
        name: str,
        param_model: Type[BaseModel],
        function: Callable,
        description: str,
    ) -> None:
        """Register a tool implementation in the registry."""
        self.tools[name] = ToolInfo(
            name=name,
            param_model=param_model,
            function=function,
            description=description,
        )

    def tool(
        self,
        name: Optional[str] = None,
        param_model: Optional[Type[BaseModel]] = None,
        description: str = "",
    ):
        """Decorator for registering tools.

        Args:
            name: Name of the tool (defaults to function name if not provided)
            param_model: Pydantic model for parameter validation (if omitted, inferred from the 'params' argument type annotation)
            description: Description of what the tool does
        """

        def decorator(func: Callable):
            # Infer defaults when not provided
            derived_name = name or func.__name__
            derived_param_model = param_model
            if derived_param_model is None:
                hints = get_type_hints(func)
                model_type = hints.get("params")
                if (
                    model_type is None
                    or not isinstance(model_type, type)
                    or not issubclass(model_type, BaseModel)
                ):
                    raise ValueError(
                        f"Cannot infer param_model for tool '{derived_name}'. "
                        "Provide 'param_model' or annotate the 'params' argument with a Pydantic BaseModel subclass."
                    )
                derived_param_model = model_type

            # Register the function immediately and return the original function
            self.register_tool(
                name=derived_name,
                param_model=derived_param_model,
                function=func,
                description=description,
            )
            return func

        return decorator

    def get_tool(self, name: str) -> ToolInfo:
        """Get a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self.tools[name]

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all registered tools for prompting."""
        descriptions = []
        for name, tool in self.tools.items():
            params = [
                f"{field_name}: {field_info.annotation.__name__}"
                for field_name, field_info in tool.param_model.__fields__.items()
            ]
            param_str = ", ".join(params)
            descriptions.append(f"{name}({param_str}): {tool.description}")

        return "\n".join(descriptions)

    def create_tools_model(
        self, include_tools: Optional[List[str]] = None
    ) -> Type[BaseModel]:
        """Create a Pydantic model for all tools or specified tools."""
        # Filter tools if needed
        available_tools = {
            name: info
            for name, info in self.tools.items()
            if include_tools is None or name in include_tools
        }

        # Create fields for the dynamic model
        model_fields = {}
        for name, info in available_tools.items():
            model_fields[name] = (Optional[info.param_model], None)

        # Create and return the model
        return create_model("ToolsModel", **model_fields)


class ToolExecutor:
    """Executes tools registered in the registry."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_tool(
        self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with the given parameters and context."""
        tool = self.registry.get_tool(tool_name)

        # Validate parameters using the tool's Pydantic model
        validated_params = tool.param_model(**params)

        # Execute the tool function with validated parameters
        try:
            result = tool.function(validated_params, context)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global singleton registry for shared tool registration/access across the app
TOOL_REGISTRY = ToolRegistry()


def tool_registry() -> ToolRegistry:
    """Return the module-level global tool registry singleton."""
    return TOOL_REGISTRY


# Example of defining and registering a tool
if __name__ == "__main__":
    from pydantic import BaseModel, Field

    registry = ToolRegistry()

    # Define parameter models for tools
    class SearchParams(BaseModel):
        query: str = Field(..., description="Search query")
        max_results: int = Field(5, description="Maximum number of results to return")

    # Define tool functions
    @registry.tool(
        name="search",
        param_model=SearchParams,
        description="Search for information on web",
    )
    def search_tool(params: SearchParams, context: Dict[str, Any]) -> List[str]:
        """Simple mock search tool."""
        # In a real implementation, this would connect to a search API
        query = params.query.lower()
        results = [
            f"Result 1 for {query}",
            f"Result 2 for {query}",
            f"Result 3 for {query}",
        ]
        return results[: params.max_results]

    # Create and populate registry
    registry = ToolRegistry()
    registry.register_tool(
        name="search",
        param_model=SearchParams,
        function=search_tool,
        description="Search for information on a given query",
    )

    # Create executor
    executor = ToolExecutor(registry)

    # Execute a tool
    result = executor.execute_tool(
        tool_name="search",
        params={"query": "python agent", "max_results": 2},
        context={},
    )

    print(result)
