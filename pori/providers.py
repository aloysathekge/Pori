"""Declarative metadata for Pori's existing LLM adapters."""

from __future__ import annotations

import importlib.util
import os
from typing import Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class ProviderProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    aliases: Tuple[str, ...] = ()
    adapter: str
    credential_environment: Tuple[str, ...]
    default_model: Optional[str] = None
    models: Tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    required_modules: Tuple[str, ...] = ()
    accepts_custom_models: bool = True
    implemented: bool = True

    def accepts_model(self, model: str) -> bool:
        return self.accepts_custom_models or model in self.models


class ProviderDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: Optional[str]
    available: bool
    credential_configured: bool
    adapter_available: bool
    model_allowed: bool
    reasons: Tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()


_PROFILES = (
    ProviderProfile(
        name="anthropic",
        aliases=("claude",),
        adapter="pori.llm.ChatAnthropic",
        credential_environment=("ANTHROPIC_API_KEY",),
        default_model="claude-sonnet-4-5-20250929",
        models=("claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"),
        capabilities=frozenset({"tools", "structured_output", "vision"}),
        required_modules=("anthropic",),
    ),
    ProviderProfile(
        name="openai",
        aliases=("gpt",),
        adapter="pori.llm.ChatOpenAI",
        credential_environment=("OPENAI_API_KEY",),
        default_model="gpt-4o",
        models=("gpt-4o", "gpt-4o-mini"),
        capabilities=frozenset({"tools", "structured_output", "vision"}),
        required_modules=("openai",),
    ),
    ProviderProfile(
        name="google",
        aliases=("gemini",),
        adapter="pori.llm.ChatGoogle",
        credential_environment=("GOOGLE_API_KEY",),
        default_model="gemini-2.5-flash",
        models=("gemini-2.5-flash", "gemini-2.5-pro"),
        capabilities=frozenset({"tools", "structured_output", "vision"}),
        required_modules=("google.genai",),
    ),
    ProviderProfile(
        name="openrouter",
        adapter="pori.llm.ChatOpenRouter",
        credential_environment=("OPENROUTER_API_KEY",),
        capabilities=frozenset({"tools"}),
        required_modules=("openai",),
    ),
    ProviderProfile(
        name="fireworks",
        adapter="pori.llm.ChatFireworks",
        credential_environment=("FIREWORKS_API_KEY",),
        capabilities=frozenset({"tools", "structured_output"}),
        required_modules=("openai",),
    ),
    ProviderProfile(
        name="azure",
        aliases=("azure-openai",),
        adapter="",
        credential_environment=("AZURE_OPENAI_API_KEY",),
        capabilities=frozenset({"tools"}),
        implemented=False,
    ),
)

PROVIDER_PROFILES: Mapping[str, ProviderProfile] = {
    profile.name: profile for profile in _PROFILES
}
_ALIASES = {alias: profile.name for profile in _PROFILES for alias in profile.aliases}


def provider_profiles(*, implemented_only: bool = True) -> Tuple[ProviderProfile, ...]:
    return tuple(
        profile for profile in _PROFILES if profile.implemented or not implemented_only
    )


def get_provider_profile(name: str) -> ProviderProfile:
    normalized = name.strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    try:
        return PROVIDER_PROFILES[canonical]
    except KeyError as exc:
        supported = ", ".join(profile.name for profile in provider_profiles())
        raise ValueError(
            f"Unsupported provider: {name}. Supported providers: {supported}"
        ) from exc


def diagnose_provider(
    provider: str,
    *,
    model: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> ProviderDiagnostic:
    profile = get_provider_profile(provider)
    values = os.environ if environ is None else environ
    missing_credentials = tuple(
        name for name in profile.credential_environment if not values.get(name)
    )
    missing_modules = tuple(
        name for name in profile.required_modules if not _module_available(name)
    )
    selected_model = model or profile.default_model
    model_allowed = bool(selected_model and profile.accepts_model(selected_model))
    reasons = [f"missing_credential:{name}" for name in missing_credentials]
    reasons.extend(f"missing_module:{name}" for name in missing_modules)
    if not profile.implemented:
        reasons.append("adapter_not_implemented")
    if not selected_model:
        reasons.append("model_required")
    elif not model_allowed:
        reasons.append(f"unsupported_model:{selected_model}")
    return ProviderDiagnostic(
        provider=profile.name,
        model=selected_model,
        available=not reasons,
        credential_configured=not missing_credentials,
        adapter_available=profile.implemented and not missing_modules,
        model_allowed=model_allowed,
        reasons=tuple(reasons),
        capabilities=profile.capabilities,
    )


__all__ = [
    "PROVIDER_PROFILES",
    "ProviderDiagnostic",
    "ProviderProfile",
    "diagnose_provider",
    "get_provider_profile",
    "provider_profiles",
]
