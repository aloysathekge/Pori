"""Tool registration and execution. ``ToolRegistry`` holds Pydantic-validated
tool definitions with capability gating (``CapabilityGroup`` prerequisites and
per-tool ``check_fn``); ``ToolExecutor`` validates params and runs a tool.
Every registered tool's schema ships on every LLM call, so additions are a
permanent context-budget tax — see the Footprint Ladder in CLAUDE.md.
``tool_registry()`` returns the process-wide default registry.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, create_model

from ..capabilities import CapabilityGroup
from ..runtime import stable_fingerprint


class CollisionPolicy(str, Enum):
    ERROR = "error"
    KEEP = "keep"
    REPLACE = "replace"


class SideEffect(str, Enum):
    """Declarative classes of externally observable tool side effects.

    Tools advertise these so authorization and policy code can reason about a
    call's blast radius without hardcoding tool-name sets at the call site.
    """

    FILESYSTEM_WRITE = "filesystem:write"


class ReconciliationStatus(str, Enum):
    """Read-only conclusions available after an uncertain provider call."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolReconciliation:
    """Typed evidence returned by a tool's read-only provider reconciler.

    ``UNKNOWN`` is deliberately non-terminal: absence or an inspection error
    must never be treated as proof that it is safe to repeat a consequence.
    """

    status: ReconciliationStatus
    provider_operation_id: Optional[str] = None
    result: Mapping[str, Any] = field(default_factory=dict)
    evidence: Tuple[Mapping[str, Any], ...] = ()
    error: Optional[str] = None


class CapabilityResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ToolInfo:
    """Information about a registered tool."""

    name: str
    param_model: Type[BaseModel]
    function: Callable
    description: str
    side_effects: Tuple[SideEffect, ...] = ()
    # Optional read-only lookup for a provider call whose local commit was
    # interrupted. It receives the original validated params plus the same
    # trusted context (including ``execution_attempt_id``) and MUST NOT repeat
    # the consequential action.
    reconcile_fn: Optional[
        Callable[[BaseModel, Dict[str, Any]], ToolReconciliation]
    ] = None
    # SK-6: optional runtime predicate; when it returns False the tool is dropped
    # from the model-visible surface (a per-tool generalization of group gating).
    check_fn: Optional[Callable[[], bool]] = None


@dataclass(frozen=True)
class CapabilitySnapshot:
    """The immutable model-visible tool surface for one run."""

    tool_items: Tuple[Tuple[str, ToolInfo], ...]
    groups: Tuple[CapabilityGroup, ...]
    excluded: Tuple[Tuple[str, str], ...]
    fingerprint: str

    @property
    def tools(self) -> Mapping[str, ToolInfo]:
        return MappingProxyType(dict(self.tool_items))

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.tool_items)

    def get_tool(self, name: str) -> ToolInfo:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise ValueError(f"Tool '{name}' not found in capability snapshot") from exc

    def to_registry(self) -> "ToolRegistry":
        registry = ToolRegistry()
        for _, info in self.tool_items:
            registry.register_tool(
                info.name,
                info.param_model,
                info.function,
                info.description,
                side_effects=info.side_effects,
                reconcile_fn=info.reconcile_fn,
            )
        for group in self.groups:
            included = group.tool_names.intersection(self.tool_names)
            if included:
                registry.define_group(
                    CapabilityGroup(
                        name=group.name,
                        description=group.description,
                        tool_names=frozenset(included),
                        prerequisites=group.prerequisites,
                        protected=group.protected,
                        max_output_chars=group.max_output_chars,
                    )
                )
        return registry


class ToolRegistry:
    """Mutable registration catalog that resolves immutable run snapshots."""

    def __init__(self, *, collision_policy: CollisionPolicy = CollisionPolicy.REPLACE):
        self.tools: Dict[str, ToolInfo] = {}
        self._groups: Dict[str, CapabilityGroup] = {}
        self.collision_policy = collision_policy

    @property
    def groups(self) -> Mapping[str, CapabilityGroup]:
        return MappingProxyType(self._groups)

    @property
    def protected_tools(self) -> frozenset[str]:
        return frozenset(
            tool_name
            for group in self._groups.values()
            if group.protected
            for tool_name in group.tool_names
        )

    def register_tool(
        self,
        name: str,
        param_model: Type[BaseModel],
        function: Callable,
        description: str,
        *,
        side_effects: Iterable[SideEffect] = (),
        reconcile_fn: Optional[
            Callable[[BaseModel, Dict[str, Any]], ToolReconciliation]
        ] = None,
        collision_policy: Optional[CollisionPolicy] = None,
        check_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Register a tool implementation using an explicit collision policy."""
        policy = collision_policy or self.collision_policy
        if name in self.tools:
            if policy is CollisionPolicy.ERROR:
                raise ValueError(f"Tool '{name}' is already registered")
            if policy is CollisionPolicy.KEEP:
                return
        self.tools[name] = ToolInfo(
            name=name,
            param_model=param_model,
            function=function,
            description=description,
            side_effects=tuple(side_effects),
            reconcile_fn=reconcile_fn,
            check_fn=check_fn,
        )

    def define_group(self, group: CapabilityGroup) -> None:
        unknown = group.tool_names.difference(self.tools)
        if unknown:
            raise ValueError(
                f"Capability group '{group.name}' contains unknown tools: "
                f"{', '.join(sorted(unknown))}"
            )
        if group.name in self._groups and self._groups[group.name] != group:
            raise ValueError(f"Capability group '{group.name}' is already defined")
        self._groups[group.name] = group

    def tool(
        self,
        name: Optional[str] = None,
        param_model: Optional[Type[BaseModel]] = None,
        description: str = "",
        *,
        side_effects: Iterable[SideEffect] = (),
        reconcile_fn: Optional[
            Callable[[BaseModel, Dict[str, Any]], ToolReconciliation]
        ] = None,
        check_fn: Optional[Callable[[], bool]] = None,
    ):
        """Decorator for registering tools."""

        def decorator(func: Callable):
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
                        "Provide 'param_model' or annotate the 'params' argument "
                        "with a Pydantic BaseModel subclass."
                    )
                derived_param_model = model_type
            self.register_tool(
                name=derived_name,
                param_model=derived_param_model,
                function=func,
                description=description,
                side_effects=side_effects,
                reconcile_fn=reconcile_fn,
                check_fn=check_fn,
            )
            return func

        return decorator

    def get_tool(self, name: str) -> ToolInfo:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self.tools[name]

    def snapshot(
        self,
        *,
        include_tools: Optional[Iterable[str]] = None,
        exclude_tools: Iterable[str] = (),
        include_groups: Optional[Iterable[str]] = None,
        protect_kernel: bool = True,
    ) -> CapabilitySnapshot:
        """Resolve one immutable, availability-aware capability surface."""
        requested = set(self.tools if include_tools is None else include_tools)
        excluded_names = set(exclude_tools)
        unknown_tools = requested.difference(self.tools)
        if unknown_tools:
            raise CapabilityResolutionError(
                f"Unknown tools requested: {', '.join(sorted(unknown_tools))}"
            )
        selected_groups = None if include_groups is None else set(include_groups)
        unknown_groups = (selected_groups or set()).difference(self._groups)
        if unknown_groups:
            raise CapabilityResolutionError(
                f"Unknown capability groups: {', '.join(sorted(unknown_groups))}"
            )
        if selected_groups is not None:
            requested.intersection_update(
                tool_name
                for name, group in self._groups.items()
                if name in selected_groups or group.protected
                for tool_name in group.tool_names
            )

        protected = self.protected_tools if protect_kernel else frozenset()
        forbidden_kernel = excluded_names.intersection(protected)
        if forbidden_kernel:
            raise CapabilityResolutionError(
                "Protected kernel tools cannot be excluded: "
                + ", ".join(sorted(forbidden_kernel))
            )
        requested.update(protected)
        requested.difference_update(excluded_names)

        exclusion_reasons: Dict[str, str] = {
            name: "policy:excluded" for name in excluded_names if name in self.tools
        }
        explicitly_requested = set(include_tools or ())
        for group in self._groups.values():
            missing = group.prerequisites.missing()
            if not missing:
                continue
            unavailable = requested.intersection(group.tool_names)
            if explicitly_requested.intersection(unavailable):
                raise CapabilityResolutionError(
                    f"Requested capability '{group.name}' is unavailable: "
                    + ", ".join(missing)
                )
            requested.difference_update(unavailable)
            for name in unavailable:
                exclusion_reasons[name] = ",".join(missing)

        # SK-6: per-tool check_fn gate — drop tools whose runtime predicate is
        # False, with a reason, unless the caller explicitly requested them.
        check_failed = set()
        for name in requested:
            check = self.tools[name].check_fn
            if check is None:
                continue
            try:
                available = bool(check())
            except Exception:
                available = False
            if not available:
                if name in explicitly_requested:
                    raise CapabilityResolutionError(
                        f"Requested tool '{name}' is unavailable (check_fn returned False)"
                    )
                check_failed.add(name)
                exclusion_reasons[name] = "check_fn:false"
        requested.difference_update(check_failed)

        items = tuple((name, self.tools[name]) for name in sorted(requested))
        groups = tuple(
            group
            for _, group in sorted(self._groups.items())
            if group.tool_names.intersection(requested)
        )
        surface = [
            {
                "name": name,
                "description": info.description,
                "parameters": info.param_model.model_json_schema(),
            }
            for name, info in items
        ]
        return CapabilitySnapshot(
            tool_items=items,
            groups=groups,
            excluded=tuple(sorted(exclusion_reasons.items())),
            fingerprint=stable_fingerprint(surface),
        )

    def filtered(self, **kwargs: Any) -> "ToolRegistry":
        """Return a public filtered registry backed by an immutable snapshot."""
        return self.snapshot(**kwargs).to_registry()

    def get_tool_descriptions(self) -> str:
        def _type_name(tp: Any) -> str:
            if tp is None:
                return "None"
            origin = get_origin(tp)
            if origin is None and isinstance(tp, type):
                return getattr(tp, "__name__", str(tp))
            if origin in (Union,) or str(tp).startswith("types.UnionType"):
                args = get_args(tp) or ()
                return " | ".join(_type_name(a) for a in args) if args else "Union"
            if origin is list:
                (arg,) = get_args(tp) or (Any,)
                return f"list[{_type_name(arg)}]"
            if origin is dict:
                key, value = get_args(tp) or (Any, Any)
                return f"dict[{_type_name(key)}, {_type_name(value)}]"
            if origin is tuple:
                args = get_args(tp) or ()
                inner = ", ".join(_type_name(a) for a in args) if args else "Any"
                return f"tuple[{inner}]"
            return str(tp).replace("typing.", "")

        descriptions = []
        for name, tool in self.tools.items():
            params = [
                f"{field_name}: {_type_name(field_info.annotation)}"
                for field_name, field_info in tool.param_model.model_fields.items()
            ]
            descriptions.append(f"{name}({', '.join(params)}): {tool.description}")
        return "\n".join(descriptions)

    def surface_fingerprint(self) -> str:
        return self.snapshot(protect_kernel=False).fingerprint

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Provider-agnostic tool schemas for native tool-calling.

        Returns ``[{name, description, input_schema}]`` where ``input_schema`` is
        each tool's JSON Schema. Provider wrappers translate these into their own
        tool format (Anthropic ``tools``, OpenAI ``tools``, Google
        ``function_declarations``).
        """
        return [
            {
                "name": name,
                "description": info.description,
                "input_schema": info.param_model.model_json_schema(),
            }
            for name, info in self.tools.items()
        ]

    def create_tools_model(
        self, include_tools: Optional[List[str]] = None
    ) -> Type[BaseModel]:
        available_tools = {
            name: info
            for name, info in self.tools.items()
            if include_tools is None or name in include_tools
        }
        model_fields: Dict[str, Any] = {
            name: (Optional[info.param_model], None)
            for name, info in available_tools.items()
        }
        return create_model("ToolsModel", **model_fields)

    def output_limit_for(self, tool_name: str) -> Optional[int]:
        limits = [
            group.max_output_chars
            for group in self._groups.values()
            if tool_name in group.tool_names and group.max_output_chars is not None
        ]
        return min(limits) if limits else None


# Global fallback ceiling so a tool with no group-level cap still can't flood the
# context window with an unbounded result (Hermes' max_result_size_chars idea).
DEFAULT_MAX_OUTPUT_CHARS = 100_000


class ToolExecutor:
    """Executes tools registered in the registry."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_tool(
        self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool = self.registry.get_tool(tool_name)
        validated_params = tool.param_model(**params)
        try:
            result = tool.function(validated_params, context)
            limit = (
                self.registry.output_limit_for(tool_name) or DEFAULT_MAX_OUTPUT_CHARS
            )
            if limit is not None:
                rendered = json.dumps(result, default=str, sort_keys=True)
                if len(rendered) > limit:
                    result = {
                        "truncated": True,
                        "max_output_chars": limit,
                        "preview": rendered[:limit],
                    }
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def execute_tool_async(
        self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute sync or async tools without blocking the agent loop.

        Existing synchronous tools keep the exact ``execute_tool`` path. Product
        integrations that own async resources (for example an async database)
        may register a coroutine function and are awaited here.
        """
        tool = self.registry.get_tool(tool_name)
        if not inspect.iscoroutinefunction(tool.function):
            return self.execute_tool(tool_name, params, context)

        validated_params = tool.param_model(**params)
        try:
            result = await tool.function(validated_params, context)
            limit = (
                self.registry.output_limit_for(tool_name) or DEFAULT_MAX_OUTPUT_CHARS
            )
            if limit is not None:
                rendered = json.dumps(result, default=str, sort_keys=True)
                if len(rendered) > limit:
                    result = {
                        "truncated": True,
                        "max_output_chars": limit,
                        "preview": rendered[:limit],
                    }
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


TOOL_REGISTRY = ToolRegistry()


def tool_registry() -> ToolRegistry:
    return TOOL_REGISTRY


__all__ = [
    "CapabilityResolutionError",
    "CapabilitySnapshot",
    "CollisionPolicy",
    "ReconciliationStatus",
    "SideEffect",
    "ToolExecutor",
    "ToolInfo",
    "ToolReconciliation",
    "ToolRegistry",
    "tool_registry",
]
