"""LLM client wrappers - lightweight replacements for LangChain."""

from .anthropic import ChatAnthropic
from .base import BaseChatModel
from .budgeted import BudgetedChatModel, ensure_budgeted_chat_model
from .fireworks import ChatFireworks
from .google import ChatGoogle
from .messages import (
    AssistantMessage,
    BaseMessage,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    MessageContent,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    ToolTurn,
    Usage,
    UserMessage,
    content_has_images,
    content_text,
    normalize_usage,
)
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
    "ContentBlock",
    "DocumentBlock",
    "ImageBlock",
    "MessageContent",
    "TextBlock",
    "ToolCall",
    "ToolResultMessage",
    "ToolTurn",
    "Usage",
    "content_has_images",
    "content_text",
    "normalize_usage",
    "BaseChatModel",
    "BudgetedChatModel",
    "ensure_budgeted_chat_model",
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
