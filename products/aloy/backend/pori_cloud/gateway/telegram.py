"""Telegram adapter — raw Bot API over httpx (no SDK dependency)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from .base import BasePlatformAdapter

logger = logging.getLogger("pori_cloud.gateway")


class TelegramAdapter(BasePlatformAdapter):
    name = "telegram"
    max_message_chars = 4096  # Telegram's hard per-message limit

    def __init__(self, bot_token: str, client: Optional[httpx.AsyncClient] = None):
        if not bot_token:
            raise ValueError("Telegram adapter requires a bot token")
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        self._client = client or httpx.AsyncClient(timeout=35.0)

    async def disconnect(self) -> None:
        await self._client.aclose()

    async def send(self, chat_id: str, text: str) -> None:
        for part in self.chunk(text):
            response = await self._client.post(
                f"{self._base_url}/sendMessage",
                json={"chat_id": chat_id, "text": part},
            )
            response.raise_for_status()

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        response = await self._client.post(
            f"{self._base_url}/getChat", json={"chat_id": chat_id}
        )
        response.raise_for_status()
        return response.json().get("result", {})

    async def get_updates(self, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Long-poll for inbound updates (25s server-side hold)."""
        payload: Dict[str, Any] = {"timeout": 25, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        response = await self._client.post(f"{self._base_url}/getUpdates", json=payload)
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            logger.warning("Telegram getUpdates not ok: %s", body)
            return []
        return list(body.get("result") or [])
