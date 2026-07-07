"""Platform adapter contract for the Aloy gateway."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePlatformAdapter(ABC):
    """One chat platform. Implementations stay thin: connect/disconnect
    lifecycle, one send primitive, and chat metadata. Everything else
    (chunking, retries at the HTTP layer) lives in shared defaults so a new
    platform is four small methods."""

    name: str = "base"
    #: Platform hard limit per outbound message; ``chunk`` respects it.
    max_message_chars: int = 4000

    async def connect(self) -> None:  # pragma: no cover - default no-op
        return None

    async def disconnect(self) -> None:  # pragma: no cover - default no-op
        return None

    @abstractmethod
    async def send(self, chat_id: str, text: str) -> None:
        """Deliver ``text`` to ``chat_id``, splitting to platform limits."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"chat_id": chat_id}

    def chunk(self, text: str) -> List[str]:
        """Split text to the platform limit, preferring newline boundaries."""
        text = text.strip()
        if not text:
            return []
        limit = self.max_message_chars
        chunks: List[str] = []
        remaining = text
        while len(remaining) > limit:
            cut = remaining.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            chunks.append(remaining)
        return chunks
