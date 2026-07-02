"""
Configuration management for Pori agent system.

Supports multiple LLM providers (Anthropic, OpenAI, Google, etc.) with
a factory pattern for easy switching between providers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from pori.providers import diagnose_provider, get_provider_profile

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """Base configuration for LLM providers."""

    provider: str = Field(description="LLM provider profile name or alias")
    model: str = Field(description="Model name/identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    streaming: bool = Field(
        default=False,
        description=(
            "Stream the assistant's text live as it generates (native "
            "tool-calling). Currently implemented for OpenAI-compatible "
            "providers (OpenAI/OpenRouter/Fireworks)."
        ),
    )
    reasoning_mode: Literal["native", "tagged", "none"] = Field(
        default="none",
        description=(
            "How this model exposes reasoning while streaming: 'native' (a "
            "separate reasoning channel), 'tagged' (inline <think>...</think> in "
            "the text), or 'none' (no reasoning trace). Controls whether "
            "reasoning is split into a separate thinking block."
        ),
    )

    # Provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return get_provider_profile(value).name


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""

    max_steps: int = Field(default=10, ge=1)
    enable_memory: bool = Field(default=True)
    soul_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a SOUL.md persona file. Overrides the default "
            "identity; project ./SOUL.md takes precedence when present."
        ),
    )
    load_project_context: bool = Field(
        default=True,
        description=(
            "Auto-load project instruction files (AGENTS.md, CLAUDE.md, "
            ".cursorrules) from the working directory into the system prompt."
        ),
    )
    planning_mode: Literal["auto", "always", "never"] = Field(
        default="never",
        description=(
            "Separate planning LLM call: auto, always, or never. Default 'never' — "
            "planning is model-driven via the update_plan tool."
        ),
    )
    reflection_mode: Literal["auto", "always", "never"] = Field(
        default="never",
        description=(
            "Separate reflection LLM calls: auto, always, or never. Default 'never' — "
            "the model revises its own plan via update_plan."
        ),
    )
    # Backward-compatible aliases. Prefer planning_mode/reflection_mode.
    enable_planning: Optional[bool] = Field(default=None, exclude=True)
    enable_reflection: Optional[bool] = Field(default=None, exclude=True)
    context_window_auto: bool = Field(
        default=True,
        description="Size the conversation-history budget to the model's real "
        "context length instead of context_window_tokens below. Set False to use "
        "context_window_tokens as a hard cap.",
    )
    context_window_tokens: int = Field(default=3000, ge=256)
    context_window_reserve_tokens: int = Field(default=1200, ge=0)
    compress_context: bool = Field(
        default=False,
        description="Summarize context that would overflow the window with an aux "
        "LLM call before it is dropped (AC-3), instead of the deterministic stub. "
        "Adds an occasional auxiliary call on overflow.",
    )
    tool_loop_guardrail: bool = Field(
        default=True,
        description="Detect cross-step tool loops (a call failing repeatedly, or an "
        "idempotent read returning the same result) and nudge/halt (AC-5). Only "
        "fires on a detected loop.",
    )
    validate_output: bool = Field(
        default=False,
        description="Run an LLM adequacy check on each proposed final answer; "
        "inadequate answers are rejected and the agent is asked to revise.",
    )
    max_validation_retries: int = Field(
        default=2,
        ge=0,
        description="Max times an answer can be rejected by output validation "
        "before it is accepted anyway (prevents loops).",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.enable_planning is not None:
            self.planning_mode = "always" if self.enable_planning else "never"
        if self.enable_reflection is not None:
            self.reflection_mode = "always" if self.enable_reflection else "never"


class MemoryConfig(BaseModel):
    """Configuration for memory persistence and identity boundaries."""

    backend: str = Field(
        default="memory", description="Memory backend: memory, sqlite, or plugin name"
    )
    sqlite_path: Optional[str] = Field(
        default=None, description="Path to SQLite DB when backend=sqlite"
    )
    organization_id: str = Field(default="default_org")
    user_id: str = Field(default="default_user")
    agent_id: str = Field(default="default_agent")
    session_id: Optional[str] = Field(default=None)


class SandboxConfig(BaseModel):
    """Configuration for sandbox execution (bash, file I/O per thread)."""

    enabled: bool = Field(
        default=False,
        description="Enable sandbox; agent can use bash and per-thread dirs",
    )
    base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for per-thread workspace/uploads/outputs (e.g. .pori_sandbox or /tmp/pori_sandbox)",
    )


from pori.hitl import HITLConfig

if TYPE_CHECKING:
    from pori.team.models import TeamConfig


class PromptsConfig(BaseModel):
    """Configuration for prompt loading."""

    base_dir: Optional[str] = Field(
        default=None,
        description="Optional base directory for prompt templates (overrides packaged prompts)",
    )


class SkillsConfig(BaseModel):
    """Configuration for local progressive skill loading."""

    enabled: bool = Field(default=True)
    default_dir: str = Field(
        default="./.pori/skills",
        description="Default project-local directory scanned for SKILL.md files.",
    )
    directories: List[str] = Field(
        default_factory=list,
        description="Additional directories scanned recursively for local SKILL.md files.",
    )
    external_dirs: List[str] = Field(
        default_factory=list,
        description="External skill directories scanned after local directories.",
    )
    bundles_dir: str = Field(
        default="./.pori/skill-bundles",
        description="Project-local directory scanned for skill bundle YAML files.",
    )
    disabled: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    max_instruction_chars: int = Field(default=50_000, ge=1)
    skill_limit: int = Field(default=3, ge=0)
    background_review: bool = Field(
        default=False,
        description="SK-1 layer 2: after each run, fire a cheap non-blocking review "
        "agent that may autonomously author a skill from the finished session. "
        "Adds one extra (small) agent run per task when on.",
    )


class EvolutionConfig(BaseModel):
    """Configuration for governed self-evolution proposal state."""

    enabled: bool = Field(default=True)
    path: str = Field(
        default="./.pori/evolution.json",
        description="JSON file used to persist governed evolution proposals.",
    )


class Config(BaseModel):
    """Main configuration container."""

    llm: LLMConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sandbox: Optional[SandboxConfig] = Field(default=None)
    hitl: Optional[HITLConfig] = Field(default=None)
    prompts: Optional[PromptsConfig] = Field(default=None)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    team: Optional[TeamConfig] = Field(default=None)


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in config values from the env.

    INF-3: lets a user keep a secret out of ``config.yaml`` by referencing an env
    var (kept in ``.env`` — which is for secrets only) instead of hardcoding it.
    An unset ``${VAR}`` is left verbatim so a miss is visible, not silently blanked.
    """
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), value
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.

    Resolution order:
    1) Explicit config_path argument (if provided)
    2) Environment variable PORI_CONFIG (if set)
    3) ./config.yaml in current working directory
    4) ~/.config/pori/config.yaml (desktop / service-friendly default)
    5) Packaged config.example.yaml next to this module (fallback)

    Returns:
        Config object with loaded settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # 1) Explicit path wins
    if config_path is not None:
        candidate = Path(config_path)
    else:
        # 2) PORI_CONFIG env var
        env_path = os.getenv("PORI_CONFIG")
        if env_path:
            candidate = Path(env_path)
        else:
            # 3) ./config.yaml in CWD
            cwd_candidate = Path.cwd() / "config.yaml"
            if cwd_candidate.exists():
                candidate = cwd_candidate
            else:
                # 4) ~/.config/pori/config.yaml
                home_cfg = Path.home() / ".config" / "pori" / "config.yaml"
                if home_cfg.exists():
                    candidate = home_cfg
                else:
                    # 5) Fallback: packaged example next to this module
                    project_root = Path(__file__).parent.parent
                    example_cfg = project_root / "config.example.yaml"
                    if example_cfg.exists():
                        candidate = example_cfg
                    else:
                        raise FileNotFoundError(
                            "No configuration file found. "
                            "Searched PORI_CONFIG env, ./config.yaml, "
                            "~/.config/pori/config.yaml, and config.example.yaml."
                        )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Config file not found at {candidate}. "
            f"Please create one based on config.example.yaml"
        )

    with open(candidate, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    # INF-3: expand ${VAR} references from the environment so secrets can be
    # referenced (kept in .env) rather than hardcoded in config.yaml.
    config_dict = _expand_env_vars(config_dict)

    return Config(**config_dict)


def create_llm(config: LLMConfig):
    """
    Factory function to create LLM instance based on configuration.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required provider package is not installed
    """
    profile = get_provider_profile(config.provider)
    provider = profile.name
    diagnostic = diagnose_provider(provider, model=config.model)
    if not diagnostic.available:
        missing_credential = next(
            (
                reason.split(":", 1)[1]
                for reason in diagnostic.reasons
                if reason.startswith("missing_credential:")
            ),
            None,
        )
        if missing_credential:
            raise ValueError(f"{missing_credential} environment variable is not set")
        raise ValueError(
            f"Provider '{provider}' is unavailable: {', '.join(diagnostic.reasons)}"
        )

    # Common parameters
    common_params: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "reasoning_mode": config.reasoning_mode,
    }

    if config.max_tokens:
        common_params["max_tokens"] = config.max_tokens

    # Add extra params
    common_params.update(config.extra_params)

    if provider == "anthropic":
        from pori.llm import ChatAnthropic

        api_key = os.getenv(profile.credential_environment[0])

        return ChatAnthropic(api_key=api_key, **common_params)

    elif provider == "openai":
        from pori.llm import ChatOpenAI

        api_key = os.getenv(profile.credential_environment[0])

        return ChatOpenAI(api_key=api_key, **common_params)

    elif provider == "google":
        from pori.llm import ChatGoogle

        api_key = os.getenv(profile.credential_environment[0])

        return ChatGoogle(api_key=api_key, **common_params)

    elif provider == "openrouter":
        from pori.llm import ChatOpenRouter, is_select_sentinel, pick_openrouter_model

        api_key = os.getenv(profile.credential_environment[0])

        # If the config asks us to prompt (model: select / prompt / pick / ?),
        # or PORI_SELECT_MODEL=1 is set, run the interactive picker and
        # override the model slug before constructing the client.
        env_force = os.getenv("PORI_SELECT_MODEL", "").strip() in {"1", "true", "yes"}
        if is_select_sentinel(common_params.get("model")) or env_force:
            chosen = pick_openrouter_model(
                default_slug=None if env_force else None,
            )
            common_params["model"] = chosen

        return ChatOpenRouter(api_key=api_key, **common_params)

    elif provider == "fireworks":
        from pori.llm import ChatFireworks

        api_key = os.getenv(profile.credential_environment[0])

        return ChatFireworks(api_key=api_key, **common_params)

    raise AssertionError(f"No adapter dispatch for validated provider '{provider}'")


def get_configured_llm(config_path: Optional[Union[str, Path]] = None):
    """
    Convenience function to load config and create LLM in one step.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Tuple of (llm_instance, config)
    """
    config = load_config(config_path)
    llm = create_llm(config.llm)
    return llm, config
