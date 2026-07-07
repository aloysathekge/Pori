"""Build the adapters whose credentials exist — nothing else loads."""

from __future__ import annotations

import logging
from typing import Dict

from ..config import settings
from .base import BasePlatformAdapter
from .telegram import TelegramAdapter

logger = logging.getLogger("aloy_backend.gateway")


def build_adapters() -> Dict[str, BasePlatformAdapter]:
    """Instantiate every configured platform adapter.

    A platform with no credentials simply doesn't exist at runtime — the same
    disappear-when-unusable discipline as the kernel's gated tools.
    """
    adapters: Dict[str, BasePlatformAdapter] = {}
    if settings.telegram_bot_token:
        adapters["telegram"] = TelegramAdapter(settings.telegram_bot_token)
        logger.info("Gateway adapter enabled: telegram")
    return adapters
