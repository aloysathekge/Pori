"""Provenance-preserving retrieval fusion contracts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: str
    source_id: str
    content: str
    score: float = Field(ge=0.0)
    session_id: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)


def fuse_retrieval(
    *sources: Iterable[RetrievalEvidence],
    limit: int = 10,
) -> List[RetrievalEvidence]:
    """Merge ranked sources without erasing identity or provenance."""
    best: Dict[tuple[str, str], RetrievalEvidence] = {}
    for source in sources:
        for item in source:
            key = (item.source_type, item.source_id)
            current = best.get(key)
            if current is None or item.score > current.score:
                best[key] = item
    return sorted(
        best.values(),
        key=lambda item: (item.score, item.source_type, item.source_id),
        reverse=True,
    )[: max(0, limit)]


__all__ = ["RetrievalEvidence", "fuse_retrieval"]
