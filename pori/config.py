"""
Configuration management for Pori agent system.

Supports multiple LLM providers (Anthropic, OpenAI, Google, etc.) with
a factory pattern for easy switching between providers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """Base configuration for LLM providers."""

    provider: Literal["anthropic", "openai", "google", "azure"] = Field(
        description="LLM provider name"
    )
    model: str = Field(description="Model name/identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""

    max_steps: int = Field(default=10, ge=1)
    enable_memory: bool = Field(default=True)
    context_window_tokens: int = Field(default=3000, ge=256)
    context_window_reserve_tokens: int = Field(default=1200, ge=0)


class MemoryConfig(BaseModel):
    """Configuration for memory persistence and identity boundaries."""

    backend: str = Field(
        default="memory", description="Memory backend: memory, sqlite, or plugin name"
    )
    sqlite_path: Optional[str] = Field(
        default=None, description="Path to SQLite DB when backend=sqlite"
    )
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


class Config(BaseModel):
    """Main configuration container."""

    llm: LLMConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sandbox: Optional[SandboxConfig] = Field(default=None)
    hitl: Optional[HITLConfig] = Field(default=None)
    prompts: Optional[PromptsConfig] = Field(default=None)
    team: Optional[TeamConfig] = Field(default=None)


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
        config_dict = yaml.safe_load(f)

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
    provider = config.provider.lower()

    # Common parameters
    common_params = {
        "model": config.model,
        "temperature": config.temperature,
    }

    if config.max_tokens:
        common_params["max_tokens"] = config.max_tokens

    # Add extra params
    common_params.update(config.extra_params)

    if provider == "anthropic":
        from pori.llm import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        return ChatAnthropic(api_key=api_key, **common_params)

    elif provider == "openai":
        from pori.llm import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return ChatOpenAI(api_key=api_key, **common_params)

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: anthropic, openai, google, azure"
        )


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
