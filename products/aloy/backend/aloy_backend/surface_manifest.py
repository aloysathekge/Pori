"""Versioned, fail-closed manifest and payload contract for generated Surfaces."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SURFACE_PROTOCOL_VERSION: Literal["1"] = "1"
SURFACE_SDK_VERSION: Literal["1"] = "1"
MAX_INTENTS = 100
MAX_SCHEMA_DEPTH = 8
MAX_INTERACTION_CHECKS = 100
MAX_INTERACTION_CHECK_STEPS = 20

_CAPABILITY = re.compile(
    r"^(?:event|tasks|files|proposals|receipts|trail|ask_aloy|(?:data|records):[a-z][a-z0-9_.-]{0,63})$"
)
_INTENT_NAME = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
_NAMESPACE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class SurfaceDataWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str
    operation: Literal["create", "replace", "merge", "delete"] = "replace"
    key: str | None = Field(default=None, min_length=1, max_length=200)
    key_field: str | None = Field(default=None, min_length=1, max_length=100)
    posture: Literal["user_reported"] = "user_reported"

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        if not _NAMESPACE.fullmatch(value):
            raise ValueError("Invalid Surface data namespace")
        return value

    @model_validator(mode="after")
    def validate_key_source(self) -> "SurfaceDataWrite":
        if (self.key is None) == (self.key_field is None):
            raise ValueError("Exactly one of key or key_field is required")
        return self


class SurfaceIntentDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    interaction_class: Literal[
        "local",
        "state",
        "reasoning",
        "automation",
        "source_change",
        "durable_selection",
        "reasoning_request",
        "external_action",
        "application_change",
    ] = Field(alias="class")
    schema_: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object"}, alias="schema"
    )
    write: SurfaceDataWrite | None = None
    tool: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_route(self) -> "SurfaceIntentDeclaration":
        _validate_schema_shape(self.schema_)
        state_classes = {"state", "durable_selection"}
        if self.interaction_class in state_classes and self.write is None:
            raise ValueError("State intents require an explicit data mutation")
        if self.interaction_class not in state_classes and self.write is not None:
            raise ValueError("Only state intents may mutate Surface data")
        if self.interaction_class == "external_action" and not self.tool:
            raise ValueError("external_action intents require a host tool")
        if self.interaction_class != "external_action" and self.tool is not None:
            raise ValueError("Only external_action intents may declare a host tool")
        return self


class SurfaceInteractionCheckStep(BaseModel):
    """One accessible user action executed by Aloy's browser publication gate."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["click", "fill", "select"]
    role: Literal["button", "textbox", "combobox"]
    name: str = Field(min_length=1, max_length=200)
    value: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_value(self) -> "SurfaceInteractionCheckStep":
        if self.action in {"fill", "select"} and self.value is None:
            raise ValueError(f"{self.action} interaction steps require a value")
        if self.action == "click" and self.value is not None:
            raise ValueError("click interaction steps cannot declare a value")
        if self.action == "fill" and self.role != "textbox":
            raise ValueError("fill interaction steps require role textbox")
        if self.action == "select" and self.role != "combobox":
            raise ValueError("select interaction steps require role combobox")
        return self


class SurfaceInteractionCheckExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["command", "dispatch", "askAloy", "requestAction"]
    name: str = Field(min_length=1, max_length=128)


class SurfaceInteractionCheck(BaseModel):
    """A semantic UI path that must reach the host SDK before publication."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    steps: list[SurfaceInteractionCheckStep] = Field(
        min_length=1, max_length=MAX_INTERACTION_CHECK_STEPS
    )
    expect: SurfaceInteractionCheckExpectation


class SurfaceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["aloy-react-surface"] = "aloy-react-surface"
    entrypoint: Literal["/src/App.tsx"] = "/src/App.tsx"
    sdk_version: Literal["1"] = SURFACE_SDK_VERSION
    capabilities: list[str] = Field(default_factory=list, max_length=50)
    intents: dict[str, SurfaceIntentDeclaration] = Field(default_factory=dict)
    interaction_checks: list[SurfaceInteractionCheck] = Field(
        default_factory=list, max_length=MAX_INTERACTION_CHECKS
    )
    widgets: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Surface capabilities must be unique")
        for value in values:
            if not _CAPABILITY.fullmatch(value):
                raise ValueError(f"Unsupported Surface capability: {value}")
        return sorted(values)

    @field_validator("intents")
    @classmethod
    def validate_intents(
        cls, values: dict[str, SurfaceIntentDeclaration]
    ) -> dict[str, SurfaceIntentDeclaration]:
        if len(values) > MAX_INTENTS:
            raise ValueError(f"Surface declares more than {MAX_INTENTS} intents")
        for name in values:
            if not _INTENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid Surface intent name: {name}")
        return values

    @model_validator(mode="after")
    def validate_intent_capabilities(self) -> "SurfaceManifest":
        granted = set(self.capabilities)
        for name, declaration in self.intents.items():
            if declaration.write:
                capability = f"data:{declaration.write.namespace}"
                if capability not in granted:
                    raise ValueError(
                        f"Intent {name} writes {declaration.write.namespace} without {capability}"
                    )
        for check in self.interaction_checks:
            expectation = check.expect
            if expectation.method == "askAloy":
                if expectation.name != "aloy.ask":
                    raise ValueError("askAloy checks must expect aloy.ask")
                if "ask_aloy" not in granted:
                    raise ValueError("askAloy check requires ask_aloy capability")
            else:
                checked_declaration = self.intents.get(expectation.name)
                if checked_declaration is None:
                    raise ValueError(
                        f"Interaction check references undeclared intent {expectation.name}"
                    )
                expected_method = {
                    "durable_selection": "dispatch",
                    "state": "command",
                    "reasoning": "command",
                    "external_action": "requestAction",
                    "automation": "command",
                    "source_change": "command",
                }.get(checked_declaration.interaction_class)
                if expectation.method != expected_method:
                    raise ValueError(
                        f"Interaction check uses the wrong SDK method for {expectation.name}"
                    )
        return self


def parse_surface_manifest(
    files: dict[str, str], *, sdk_version: str = SURFACE_SDK_VERSION
) -> SurfaceManifest:
    raw = files.get("/surface.json")
    if sdk_version != SURFACE_SDK_VERSION:
        raise ValueError(f"Unsupported Surface SDK version: {sdk_version}")
    if raw is None:
        return SurfaceManifest()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"surface.json is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("surface.json must contain an object")
    value.setdefault("format", "aloy-react-surface")
    value.setdefault("entrypoint", "/src/App.tsx")
    value.setdefault("sdk_version", sdk_version)
    return SurfaceManifest.model_validate(value)


def validate_intent_payload(schema: dict[str, Any], payload: Any) -> None:
    """Validate the deliberately small JSON-schema subset accepted in V1."""
    _validate_schema_shape(schema)
    _validate_value(schema, payload, path="payload")


def _validate_schema_shape(schema: Any, *, depth: int = 0) -> None:
    if depth > MAX_SCHEMA_DEPTH:
        raise ValueError("Surface intent schema is too deeply nested")
    if not isinstance(schema, dict):
        raise ValueError("Surface intent schema must be an object")
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    }
    unknown = set(schema) - allowed
    if unknown:
        raise ValueError(f"Unsupported Surface schema keywords: {sorted(unknown)}")
    schema_type = schema.get("type", "object")
    if schema_type not in {
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
    }:
        raise ValueError(f"Unsupported Surface schema type: {schema_type}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Surface schema properties must be an object")
    for child in properties.values():
        _validate_schema_shape(child, depth=depth + 1)
    if "items" in schema:
        _validate_schema_shape(schema["items"], depth=depth + 1)
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValueError("Surface schema required must be a string array")
    if not set(required).issubset(properties):
        raise ValueError("Surface schema required names must exist in properties")


def _validate_value(schema: dict[str, Any], value: Any, *, path: str) -> None:
    expected = schema.get("type", "object")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[expected]
    if not valid:
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path} is too short")
        if len(value) > int(schema.get("maxLength", 100_000)):
            raise ValueError(f"{path} is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above maximum")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path} has too few items")
        if len(value) > int(schema.get("maxItems", 1000)):
            raise ValueError(f"{path} has too many items")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate_value(item_schema, item, path=f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ValueError(f"{path} has undeclared fields: {sorted(unknown)}")
        for name, item in value.items():
            if name in properties:
                _validate_value(properties[name], item, path=f"{path}.{name}")


__all__ = [
    "SURFACE_PROTOCOL_VERSION",
    "SURFACE_SDK_VERSION",
    "SurfaceInteractionCheck",
    "SurfaceIntentDeclaration",
    "SurfaceManifest",
    "parse_surface_manifest",
    "validate_intent_payload",
]
