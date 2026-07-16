"""Immutable, purpose-scoped contracts for assembling an agent run."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime import stable_fingerprint

_PROFILE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")


class RunProfileResolutionError(ValueError):
    """Raised when a run cannot satisfy its declared profile."""


class RunProfile(BaseModel):
    """A frozen contract for one kind of agent run.

    Products use profiles to bind a purpose-specific prompt, tool surface,
    skills, and model requirements before execution. The profile is data, not
    middleware: Pori resolves it once, fails closed on missing requirements,
    and records its fingerprint with the result.
    """

    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(min_length=1, max_length=128)
    version: str = Field(default="1", min_length=1, max_length=64)
    system_prompt: str = ""
    allowed_tools: Optional[frozenset[str]] = None
    denied_tools: frozenset[str] = frozenset()
    required_tools: frozenset[str] = frozenset()
    required_skill_ids: frozenset[str] = frozenset()
    required_model_capabilities: frozenset[str] = frozenset()

    @field_validator("profile_id")
    @classmethod
    def _valid_profile_id(cls, value: str) -> str:
        if not _PROFILE_ID.fullmatch(value):
            raise ValueError(
                "profile_id must contain only lowercase letters, numbers, '.', '_', or '-'"
            )
        return value

    @field_validator(
        "allowed_tools",
        "denied_tools",
        "required_tools",
        "required_skill_ids",
        "required_model_capabilities",
    )
    @classmethod
    def _non_empty_names(
        cls, value: Optional[frozenset[str]]
    ) -> Optional[frozenset[str]]:
        if value is None:
            return None
        invalid = sorted(item for item in value if not item or item != item.strip())
        if invalid:
            raise ValueError("profile names must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _consistent_tool_contract(self) -> "RunProfile":
        forbidden_required = self.required_tools.intersection(self.denied_tools)
        if forbidden_required:
            raise ValueError(
                "required_tools cannot also be denied: "
                + ", ".join(sorted(forbidden_required))
            )
        if self.allowed_tools is not None:
            denied_allowed = self.allowed_tools.intersection(self.denied_tools)
            if denied_allowed:
                raise ValueError(
                    "allowed_tools cannot also be denied: "
                    + ", ".join(sorted(denied_allowed))
                )
            missing = self.required_tools.difference(self.allowed_tools)
            if missing:
                raise ValueError(
                    "required_tools must be included in allowed_tools: "
                    + ", ".join(sorted(missing))
                )
        return self

    @property
    def fingerprint(self) -> str:
        """Stable identity for the exact executable profile contract."""
        return stable_fingerprint(
            {
                "profile_id": self.profile_id,
                "version": self.version,
                "system_prompt": self.system_prompt,
                "allowed_tools": (
                    None if self.allowed_tools is None else sorted(self.allowed_tools)
                ),
                "denied_tools": sorted(self.denied_tools),
                "required_tools": sorted(self.required_tools),
                "required_skill_ids": sorted(self.required_skill_ids),
                "required_model_capabilities": sorted(self.required_model_capabilities),
            }
        )

    def descriptor(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }


__all__ = ["RunProfile", "RunProfileResolutionError"]
