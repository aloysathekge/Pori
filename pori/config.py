"""
Configuration management for Pori agent system.

Supports multiple LLM providers (Anthropic, OpenAI, Google, etc.) with
a factory pattern for easy switching between providers.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

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


class Config(BaseModel):
    """Main configuration container."""
    
    llm: LLMConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for 'config.yaml' 
                    in project root.
    
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if config_path is None:
        # Default to config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Please create one based on config.example.yaml"
        )
    
    with open(config_path, "r") as f:
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
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic provider. "
                "Install it with: pip install langchain-anthropic"
            )
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        return ChatAnthropic(api_key=api_key, **common_params)
    
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install it with: pip install langchain-openai"
            )
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        return ChatOpenAI(api_key=api_key, **common_params)
    
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Google provider. "
                "Install it with: pip install langchain-google-genai"
            )
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        return ChatGoogleGenerativeAI(google_api_key=api_key, **common_params)
    
    elif provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for Azure provider. "
                "Install it with: pip install langchain-openai"
            )
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "environment variables must be set"
            )
        
        return AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            **common_params
        )
    
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
