"""Provider contracts for schema-bound LLM responses.

OpenAI-compatible transports do not imply identical JSON Schema support.
This module keeps those differences explicit and provider-owned so product
code can request a Pydantic result without knowing a provider's dialect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
)


def _without_schema_keywords(value: Any, unsupported: frozenset[str]) -> Any:
    """Copy a schema without constraints while preserving user field names."""
    if isinstance(value, dict):
        adapted: dict[str, Any] = {}
        for key, item in value.items():
            if key in unsupported:
                continue
            if key in SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
                adapted[key] = {
                    name: _without_schema_keywords(child_schema, unsupported)
                    for name, child_schema in item.items()
                }
            else:
                adapted[key] = _without_schema_keywords(item, unsupported)
        return adapted
    if isinstance(value, list):
        return [_without_schema_keywords(item, unsupported) for item in value]
    return value


@dataclass(frozen=True)
class StructuredOutputPolicy:
    """Capabilities and request behavior for one provider/model family."""

    unsupported_schema_keywords: frozenset[str] = frozenset()
    include_schema_in_prompt: bool = False
    strict: bool = True
    request_options: dict[str, Any] = field(default_factory=dict)

    def adapt_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Translate an application schema into the provider's accepted dialect."""
        return _without_schema_keywords(schema, self.unsupported_schema_keywords)

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply provider-required prompt instructions without mutating callers."""
        prepared = [dict(message) for message in messages]
        if not self.include_schema_in_prompt:
            return prepared
        schema_instruction = (
            "\n\nReturn only JSON matching this required schema:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        for message in reversed(prepared):
            if message.get("role") == "user" and isinstance(
                message.get("content"), str
            ):
                message["content"] += schema_instruction
                return prepared
        prepared.append({"role": "user", "content": schema_instruction.lstrip()})
        return prepared

    def response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Build the OpenAI-compatible response-format envelope."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": self.strict,
                "schema": schema,
            },
        }


__all__ = ["StructuredOutputPolicy"]
