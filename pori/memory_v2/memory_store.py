from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class MemoryStore(ABC, Generic[T]):
    """Abstract interface for memory storage backends.

    Implementations can be in-memory, SQLite, Redis, vector DBs, etc.
    """

    @abstractmethod
    def add(self, key: str, value: T, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add or overwrite a value under a key with optional metadata."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Return the value for a key or None if missing."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, T, float]]:
        """Return a ranked list of (key, value, score) results for a query."""

    @abstractmethod
    def forget(self, key: str) -> bool:
        """Delete a key if present. Returns True if removed, False otherwise."""
