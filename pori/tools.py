from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type
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

    def tool(
        self,
        name: str,
        param_model: Type[BaseModel],
        
        description: str,
    ) :
       

        """Decorator for registering tools.
        
        Args:
            name: Name of the tool
            param_model: Pydantic model for parameter validation
            description: Description of what the tool does
        """

        def decorator(func:Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            self.tools(
                name=name,
                param_model=param_model,
                function=func,
                description=description
                )
            return wrapper
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


# Example of defining and registering a tool
if __name__ == "__main__":
    from pydantic import BaseModel, Field

    # Define parameter models for tools
    class SearchParams(BaseModel):
        query: str = Field(..., description="Search query")
        max_results: int = Field(5, description="Maximum number of results to return")

    # Define tool functions
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
