"""DeliveryRouter — get outbound text to the right external chat.

The router is the single place that maps "this user should hear about X" to
concrete platform sends. Run/cron completions, notifications, and future
platforms all go through here instead of each caller knowing about adapters.
"""

from __future__ import annotations

import logging
from typing import Dict

from sqlmodel import select

from ..models import GatewayLink
from .base import BasePlatformAdapter

logger = logging.getLogger("aloy_backend.gateway")


class DeliveryRouter:
    def __init__(self, adapters: Dict[str, BasePlatformAdapter]):
        self._adapters = adapters

    async def deliver_to_link(self, link: GatewayLink, text: str) -> bool:
        """Send to one paired chat. Returns False when the platform is down
        or unconfigured — the caller decides whether that matters."""
        adapter = self._adapters.get(link.platform)
        if adapter is None:
            logger.warning(
                "No adapter for platform %r (link %s)", link.platform, link.id
            )
            return False
        try:
            await adapter.send(link.chat_id, text)
            return True
        except Exception:
            logger.exception(
                "Delivery to %s chat %s failed", link.platform, link.chat_id
            )
            return False

    async def deliver_to_user(
        self, session, organization_id: str, user_id: str, text: str
    ) -> int:
        """Send to every chat the user has paired. Returns the send count."""
        result = await session.execute(
            select(GatewayLink).where(
                GatewayLink.organization_id == organization_id,
                GatewayLink.user_id == user_id,
            )
        )
        delivered = 0
        for link in result.scalars().all():
            if await self.deliver_to_link(link, text):
                delivered += 1
        return delivered
