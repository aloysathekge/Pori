"""LLM client wrappers - lightweight replacements for LangChain."""

from .anthropic import ChatAnthropic
from .base import BaseChatModel
from .fireworks import ChatFireworks
from .google import ChatGoogle
from .messages import AssistantMessage, BaseMessage, SystemMessage, UserMessage
from .openai import ChatOpenAI
from .openrouter import ChatOpenRouter
from .openrouter_models import (
    OPENROUTER_CATALOG,
    OpenRouterModel,
    is_select_sentinel,
    pick_openrouter_model,
    render_catalog,
)

__all__ = [
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "BaseChatModel",
    "ChatAnthropic",
    "ChatFireworks",
    "ChatGoogle",
    "ChatOpenAI",
    "ChatOpenRouter",
    "OPENROUTER_CATALOG",
    "OpenRouterModel",
    "is_select_sentinel",
    "pick_openrouter_model",
    "render_catalog",
]
