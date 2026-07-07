"""Aloy messaging gateway: platform adapters + delivery routing.

Harvested shape (Hermes `gateway/platforms/base.py` + `gateway/delivery.py`,
see docs/hermes-gap-2026-07.md Tier 1 §2): a small adapter ABC per platform,
a registry that builds only the adapters whose credentials exist, and a
DeliveryRouter that gets any outbound text to the right chat. Telegram is the
first (and currently only) adapter; new platforms implement four methods.
"""

from .base import BasePlatformAdapter
from .delivery import DeliveryRouter
from .registry import build_adapters
from .telegram import TelegramAdapter

__all__ = [
    "BasePlatformAdapter",
    "DeliveryRouter",
    "TelegramAdapter",
    "build_adapters",
]
