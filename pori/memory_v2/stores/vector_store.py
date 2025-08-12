from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..memory_store import MemoryStore


class VectorMemoryStore(MemoryStore[str]):
    """Local vector store using SentenceTransformers and cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.keys: List[str] = []

    def _embed(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, normalize_embeddings=True)
        # Some ST versions return list; ensure ndarray float32
        return np.asarray(emb, dtype=np.float32)

    def add(self, key: str, value: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.keys.append(key)
        self.texts.append(value)
        self.embeddings.append(self._embed(value))

    def get(self, key: str) -> Optional[str]:
        try:
            idx = self.keys.index(key)
        except ValueError:
            return None
        return self.texts[idx]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not self.texts:
            return []
        q = self._embed(query)
        # Stack to (N, D)
        M = np.vstack(self.embeddings)
        sims = (M @ q).astype(np.float32)
        idxs = np.argsort(sims)[::-1][:top_k]
        results: List[Tuple[str, str, float]] = []
        for i in idxs:
            results.append((self.keys[int(i)], self.texts[int(i)], float(sims[int(i)])))
        return results

    def forget(self, key: str) -> bool:
        try:
            idx = self.keys.index(key)
        except ValueError:
            return False
        self.keys.pop(idx)
        self.texts.pop(idx)
        self.embeddings.pop(idx)
        return True
