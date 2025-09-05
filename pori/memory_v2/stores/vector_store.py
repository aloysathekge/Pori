from __future__ import annotations

from ..memory_store import MemoryStore


class VectorMemoryStore(MemoryStore[str]):
    """Deprecated local vector store. This backend has been removed.

    Any attempt to instantiate or use this class will raise a RuntimeError.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "Local vector store has been removed. Use a remote backend (e.g., 'weaviate')."
        )

    # Keep method signatures to avoid import errors in old call sites, but fail fast
    def add(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Local vector store is removed")

    def get(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Local vector store is removed")

    def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Local vector store is removed")

    def forget(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Local vector store is removed")
