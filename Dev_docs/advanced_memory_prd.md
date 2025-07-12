# Advanced Memory System Guide

*Enhancing the `agents` framework with robust, production-ready memory*

---

## 1  Why Upgrade the Memory?

1. **Context depth** – Richer memories let the agent reason across long conversations, past tasks, and external documents.  
2. **Persistence** – A restart should not wipe the agent’s knowledge.  
3. **Semantic search** – Vector‐based recall surfaces the most relevant facts instead of the most recent ones.  
4. **Scalability** – Proper back-ends (SQL, Redis, vector DBs) keep performance acceptable as data grows.  
5. **Community extensions** – A clean, pluggable interface invites OSS contributors to add new storage or retrieval strategies.

---

## 2  Architecture Overview

```
┌─────────────────────────┐
│  Agent (agent.py)       │
│  ├─ calls Memory API ─┐ │
└─────────────────────────┘ │
          ▲                 │
          │                 ▼
┌───────────────────────────────────────────────┐
│ EnhancedAgentMemory                           │
│  • WorkingMemory  – short-term buffer         │
│  • LongTermMemory – persisted store           │
│  • VectorMemory   – semantic embeddings       │
│  • APIs: add / search / summary / forget      │
└───────────────────────────────────────────────┘
          ▲
          │ implements
          ▼
┌───────────────────────────────────────────────┐
│ MemoryStore (abstract)                        │
│  ├─ InMemoryStore         (default)           │
│  ├─ SqliteStore           (relational)        │
│  ├─ RedisStore            (cache/TTL)         │
│  └─ VectorMemoryStore     (similarity)        │
└───────────────────────────────────────────────┘
```

Key ideas:

* **Strategy pattern** – `EnhancedAgentMemory` delegates persistence to interchangeable `MemoryStore` back-ends.  
* **Multiple Memory Types** – Working vs long-term enables aggressive pruning of the prompt while still retaining durable knowledge.  
* **Embeddings** – Texts are stored with vectors for similarity search. Any model (OpenAI, SentenceTransformers, etc.) can be plugged in.

---

## 3  Implementation Steps

### 3.1 Create the abstract store

```python
# memory_store.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Generic, TypeVar

T = TypeVar("T")

class MemoryStore(ABC, Generic[T]):
    @abstractmethod
    def add(self, key: str, value: T, meta: Dict[str, Any] | None = None): ...
    @abstractmethod
    def get(self, key: str) -> Optional[T]: ...
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, T, float]]: ...
    @abstractmethod
    def forget(self, key: str) -> bool: ...
```

### 3.2 Baseline in-memory implementation

```python
# stores/in_memory.py
from collections import defaultdict
from memory_store import MemoryStore

class InMemoryStore(MemoryStore[str]):
    def __init__(self):
        self.data: dict[str, str] = {}
        self.meta: dict[str, dict] = defaultdict(dict)

    def add(self, key, value, meta=None):
        self.data[key] = value
        if meta: self.meta[key] = meta

    def get(self, key):
        return self.data.get(key)

    def search(self, query, top_k=5):
        # naive substring ranking
        hits = [(k, v, 1.0) for k, v in self.data.items() if query.lower() in v.lower()]
        return hits[:top_k]

    def forget(self, key):
        return self.data.pop(key, None) is not None
```

### 3.3 Vector memory store

```python
# stores/vector_store.py
import numpy as np
from memory_store import MemoryStore
from sentence_transformers import SentenceTransformer

class VectorMemoryStore(MemoryStore[str]):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts: list[str] = []
        self.embeds: list[np.ndarray] = []
        self.keys: list[str] = []

    def _embed(self, txt): return self.model.encode(txt, normalize_embeddings=True)

    def add(self, key, value, meta=None):
        self.keys.append(key)
        self.texts.append(value)
        self.embeds.append(self._embed(value))

    def search(self, query, top_k=5):
        q = self._embed(query)
        sims = (np.dot(self.embeds, q) ).tolist()
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self.keys[i], self.texts[i], sims[i]) for i in idxs]

    # get / forget similar to InMemoryStore
```

### 3.4 Enhanced memory façade

```python
# enhanced_memory.py
from stores.in_memory import InMemoryStore
from stores.vector_store import VectorMemoryStore

class EnhancedAgentMemory:
    def __init__(self, persistent=False, vector=True):
        self.working = InMemoryStore()
        self.long_term = InMemoryStore()  # swap for SqliteStore if persistent
        self.vector = VectorMemoryStore() if vector else None

    def add_experience(self, text: str, importance: int = 1):
        key = f"exp_{len(self.long_term.data)}"
        self.long_term.add(key, text, meta={"importance": importance})
        if self.vector:
            self.vector.add(key, text)

    def recall(self, query: str, k: int = 5):
        if self.vector:
            return self.vector.search(query, k)
        return self.long_term.search(query, k)

    # Wrapper helpers used by Agent ---------------------------
    def add_message(self, role, content):
        self.working.add(f"msg_{len(self.working.data)}", f"{role}: {content}")

    def get_recent_messages(self, n=10):
        items = list(self.working.data.values())[-n:]
        return "\n".join(items)
```

### 3.5 Wire into the Agent

1. Replace `AgentMemory()` with `EnhancedAgentMemory()` in `agent.py`.
2. Update calls (`add_message`, `tool_call_history`, etc.) to the new API or provide adapter methods for backward compatibility.
3. Use `memory.recall(query)` before tool invocation to supply extra context to the LLM prompt.

---

## 4  Best Practices & Tips

| Area | Recommendation |
|------|----------------|
| **Vector DB** | For production, back `VectorMemoryStore` with Pinecone, Weaviate or Milvus instead of local embeddings. |
| **Token Limits** | Summarize or prune `working` memory every *N* steps (already done via `create_summary`). |
| **Security** | Strip PII before persistence; consider field-level encryption for sensitive data. |
| **Evaluation** | Track retrieval success rate and memory hit ratio in your evaluator to detect drift. |
| **Extensibility** | Encourage plugins: `pip install agents-mem-redis` could inject a `RedisStore` that complies with `MemoryStore`. |
| **Testing** | Create fixtures that simulate thousands of memories to benchmark retrieval latency under load. |

---

## 5  Next Steps

1. **Persistence layer** – Implement `SqliteStore` via SQLAlchemy with automatic migrations.  
2. **Forgetting curve** – Periodically decay importance scores and prune low-value memories.  
3. **Knowledge graphs** – Convert facts to triples and expose graph-based reasoning tools.  
4. **UI dashboard** – Visualize memory timelines and debug retrieval quality.  

Contributions are welcome—open a PR if you build a new store or retrieval strategy!  
Happy hacking 🚀
