from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..memory_store import MemoryStore


class InMemoryStore(MemoryStore[str]):
    """Simple in-memory key->string store with naive substring search."""

    def __init__(self) -> None:
        self.data: Dict[str, str] = {}
        self.meta: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def add(self, key: str, value: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.data[key] = value
        if meta:
            self.meta[key] = meta

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not query:
            return []
        q = query.lower()
        hits: List[Tuple[str, str, float]] = []
        for k, v in self.data.items():
            if q in v.lower():
                hits.append((k, v, 1.0))
        return hits[:top_k]

    def forget(self, key: str) -> bool:
        return self.data.pop(key, None) is not None
