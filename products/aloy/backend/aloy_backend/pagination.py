"""Opaque keyset cursors shared by long Event and Conversation histories."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone


class CursorError(ValueError):
    """Raised when a client supplies a malformed history cursor."""


@dataclass(frozen=True)
class HistoryCursor:
    created_at: datetime
    row_id: str


def encode_cursor(created_at: datetime, row_id: str) -> str:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    raw = json.dumps(
        {"at": created_at.astimezone(timezone.utc).isoformat(), "id": row_id},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(value: str) -> HistoryCursor:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        created_at = datetime.fromisoformat(str(payload["at"]).replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        row_id = str(payload["id"])
        if not row_id:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CursorError("Invalid history cursor") from exc
    return HistoryCursor(created_at.astimezone(timezone.utc), row_id)


__all__ = ["CursorError", "HistoryCursor", "decode_cursor", "encode_cursor"]
