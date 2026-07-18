"""Aloy-owned model-role configuration and immutable Run assignments.

Conversation preferences remain separate from privileged product roles. A
Surface Builder or Critic receives a model only from the developer-controlled
role file, and the exact resolved assignment is frozen onto the Run before it
enters the worker queue.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pori import LLMConfig, diagnose_provider, stable_fingerprint

from .config import settings


class ModelRole(str, Enum):
    SURFACE_BUILDER = "surface_builder"
    SURFACE_CRITIC = "surface_critic"


class ModelQualification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["unqualified", "qualified"] = "unqualified"
    suite: str = Field(default="", max_length=200)
    evidence: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def qualified_roles_require_evidence(self) -> "ModelQualification":
        if self.status == "qualified" and not (
            self.suite.strip() and self.evidence.strip()
        ):
            raise ValueError("qualified model roles require suite and evidence")
        return self


class ModelRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    reasoning_mode: Literal["native", "tagged", "none"] = "none"
    required_capabilities: frozenset[str] = frozenset()
    skill_id: str | None = Field(default=None, max_length=200)
    qualification: ModelQualification = Field(default_factory=ModelQualification)


class AloyModelRolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    roles: dict[ModelRole, ModelRoleConfig] = Field(default_factory=dict)


class ModelAssignment(BaseModel):
    """Credential-free model and skill receipt frozen on a durable Run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: int
    role: ModelRole
    provider: str
    model: str
    temperature: float
    max_tokens: int | None = None
    reasoning_mode: Literal["native", "tagged", "none"] = "none"
    capabilities: tuple[str, ...] = ()
    skill_id: str | None = None
    qualification_status: Literal["qualified"] = "qualified"
    qualification_suite: str = ""
    qualification_evidence: str = ""
    config_fingerprint: str
    resolution_ms: float = 0.0
    resolved_at: datetime

    def fingerprint_payload(self) -> dict:
        return {
            "config_version": self.config_version,
            "role": self.role.value,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_mode": self.reasoning_mode,
            "capabilities": list(self.capabilities),
            "skill_id": self.skill_id,
            "qualification_status": self.qualification_status,
            "qualification_suite": self.qualification_suite,
            "qualification_evidence": self.qualification_evidence,
        }

    def verify_fingerprint(self) -> None:
        if stable_fingerprint(self.fingerprint_payload()) != self.config_fingerprint:
            raise ModelRoleUnavailableError(
                "Run model assignment fingerprint is invalid"
            )

    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_mode=self.reasoning_mode,
        )

    def descriptor(self) -> dict:
        return self.model_dump(mode="json")


class ModelRoleUnavailableError(ValueError):
    """A privileged role is absent, unqualified, or unusable."""


def _configured_path(path: str | Path | None = None) -> Path:
    selected = Path(path or settings.aloy_model_roles_path).expanduser()
    return selected if selected.is_absolute() else (Path.cwd() / selected)


@lru_cache(maxsize=8)
def _load_model_roles_cached(path: str) -> AloyModelRolesConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ModelRoleUnavailableError(
            "Aloy model roles are not configured. Copy aloy.models.example.yaml "
            "to aloy.models.yaml and set ALOY_MODEL_ROLES_PATH when it lives "
            "elsewhere."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return AloyModelRolesConfig.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ModelRoleUnavailableError(
            f"Aloy model role configuration is invalid: {exc}"
        ) from exc


def clear_model_role_cache() -> None:
    """Reload operator configuration after a process restart or in tests."""
    _load_model_roles_cached.cache_clear()


def load_model_roles(
    path: str | Path | None = None,
) -> AloyModelRolesConfig:
    return _load_model_roles_cached(str(_configured_path(path).resolve()))


def resolve_model_assignment(
    role: ModelRole,
    *,
    path: str | Path | None = None,
    required_capabilities: frozenset[str] = frozenset(),
    expected_skill_id: str | None = None,
    allowed_provider_profiles: tuple[str, ...] | None = None,
    allowed_models: tuple[str, ...] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModelAssignment:
    started = perf_counter()
    config = load_model_roles(path)
    selection = config.roles.get(role)
    if selection is None:
        raise ModelRoleUnavailableError(
            f"Aloy model role '{role.value}' is not configured"
        )
    if selection.qualification.status != "qualified":
        raise ModelRoleUnavailableError(
            f"Aloy model role '{role.value}' has not passed its qualification suite"
        )
    diagnostic = diagnose_provider(
        selection.provider,
        model=selection.model,
        environ=os.environ if environ is None else environ,
    )
    if not diagnostic.available:
        raise ModelRoleUnavailableError(
            f"Aloy model role '{role.value}' is unavailable: "
            + ", ".join(diagnostic.reasons)
        )
    if (
        allowed_provider_profiles
        and diagnostic.provider not in allowed_provider_profiles
    ):
        raise ModelRoleUnavailableError(
            f"Provider for Aloy model role '{role.value}' is denied by organization policy"
        )
    if allowed_models and selection.model not in allowed_models:
        raise ModelRoleUnavailableError(
            f"Model for Aloy model role '{role.value}' is denied by organization policy"
        )
    capabilities = frozenset(selection.required_capabilities) | required_capabilities
    missing = capabilities - diagnostic.capabilities
    if missing:
        raise ModelRoleUnavailableError(
            f"Aloy model role '{role.value}' lacks required capabilities: "
            + ", ".join(sorted(missing))
        )
    if expected_skill_id and selection.skill_id != expected_skill_id:
        raise ModelRoleUnavailableError(
            f"Aloy model role '{role.value}' must use skill {expected_skill_id}"
        )
    assignment_values = {
        "config_version": config.version,
        "role": role,
        "provider": diagnostic.provider,
        "model": selection.model,
        "temperature": selection.temperature,
        "max_tokens": selection.max_tokens,
        "reasoning_mode": selection.reasoning_mode,
        "capabilities": tuple(sorted(diagnostic.capabilities)),
        "skill_id": selection.skill_id,
        "qualification_status": "qualified",
        "qualification_suite": selection.qualification.suite,
        "qualification_evidence": selection.qualification.evidence,
    }
    fingerprint = stable_fingerprint(
        {
            **assignment_values,
            "role": role.value,
            "capabilities": list(assignment_values["capabilities"]),
        }
    )
    return ModelAssignment.model_validate(
        {
            **assignment_values,
            "config_fingerprint": fingerprint,
            "resolution_ms": round((perf_counter() - started) * 1000, 3),
            "resolved_at": datetime.now(timezone.utc),
        }
    )


__all__ = [
    "AloyModelRolesConfig",
    "ModelAssignment",
    "ModelQualification",
    "ModelRole",
    "ModelRoleConfig",
    "ModelRoleUnavailableError",
    "clear_model_role_cache",
    "load_model_roles",
    "resolve_model_assignment",
]
