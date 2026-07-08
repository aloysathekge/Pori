"""Translate an MCP tool's JSON Schema into one every provider accepts.

Harvested in spirit from Hermes' _normalize_mcp_input_schema — the single most
reusable, provider-portability-critical piece. MCP servers emit arbitrary JSON
Schema; Anthropic/OpenAI/Google each reject different corners of it. This makes
a server's tool schemas safe to ship to any provider.
"""

from __future__ import annotations

import copy
from typing import Any, Dict


def normalize_mcp_input_schema(schema: Any) -> Dict[str, Any]:
    """Return a provider-safe copy of an MCP tool input schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = copy.deepcopy(schema)
    _rewrite_defs_refs(out)
    _coerce_object(out)
    _prune_required(out)
    _collapse_nullable_unions(out)
    return out


def _rewrite_defs_refs(node: Any) -> None:
    """`#/definitions/...` -> `#/$defs/...` and rename the container."""
    if isinstance(node, dict):
        if "definitions" in node and "$defs" not in node:
            node["$defs"] = node.pop("definitions")
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/definitions/"):
            node["$ref"] = ref.replace("#/definitions/", "#/$defs/", 1)
        for value in node.values():
            _rewrite_defs_refs(value)
    elif isinstance(node, list):
        for item in node:
            _rewrite_defs_refs(item)


def _coerce_object(schema: Dict[str, Any]) -> None:
    """The top level must be an object with a properties map."""
    if schema.get("type") is None and "properties" in schema:
        schema["type"] = "object"
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}


def _prune_required(schema: Dict[str, Any]) -> None:
    """Drop `required` entries that name no declared property (providers 400)."""
    props = schema.get("properties")
    required = schema.get("required")
    if isinstance(props, dict) and isinstance(required, list):
        kept = [name for name in required if name in props]
        if kept:
            schema["required"] = kept
        else:
            schema.pop("required", None)
    if isinstance(props, dict):
        for value in props.values():
            if isinstance(value, dict):
                _prune_required(value)


def _collapse_nullable_unions(node: Any) -> None:
    """`anyOf: [{...}, {type: null}]` -> the non-null branch (broad support)."""
    if isinstance(node, dict):
        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [
                b
                for b in any_of
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            if len(non_null) == 1 and len(non_null) < len(any_of):
                merged = non_null[0]
                node.pop("anyOf")
                if isinstance(merged, dict):
                    for k, v in merged.items():
                        node.setdefault(k, v)
        for value in node.values():
            _collapse_nullable_unions(value)
    elif isinstance(node, list):
        for item in node:
            _collapse_nullable_unions(item)
