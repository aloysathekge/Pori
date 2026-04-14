from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field

DEFAULT_CORE_BLOCK_LIMIT = 2000
LINE_NUMBER_REGEX = re.compile(r"\n\d+→ ")


class MemoryStore(Protocol):
    def load(self, namespace: str) -> Optional[Dict[str, Any]]: ...

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._db: Dict[str, Dict[str, Any]] = {}

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        data = self._db.get(namespace)
        if data is None:
            return None
        return json.loads(json.dumps(data))

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        self._db[namespace] = json.loads(json.dumps(snapshot))


class SQLiteMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    namespace TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load(self, namespace: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM memory_snapshots WHERE namespace = ?",
                (namespace,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(str(row["payload"]))

    def save(self, namespace: str, snapshot: Dict[str, Any]) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_snapshots (namespace, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                  payload = excluded.payload,
                  updated_at = excluded.updated_at
                """,
                (namespace, payload, now),
            )
            conn.commit()


def create_memory_store(
    backend: str = "memory",
    sqlite_path: Optional[str] = None,
) -> MemoryStore:
    normalized = (backend or "memory").strip().lower()
    if normalized in {"memory", "in_memory", "in-memory"}:
        return InMemoryMemoryStore()
    if normalized == "sqlite":
        path = sqlite_path or os.getenv("PORI_MEMORY_SQLITE_PATH") or ".pori/memory.db"
        return SQLiteMemoryStore(path)

    try:
        eps = metadata.entry_points()
        selected = (
            eps.select(group="pori.memory_stores")
            if hasattr(eps, "select")
            else eps.get("pori.memory_stores", [])
        )
    except Exception:
        selected = []

    for ep in selected:
        if ep.name.strip().lower() != normalized:
            continue
        factory = ep.load()
        store = factory(sqlite_path=sqlite_path)
        if not hasattr(store, "load") or not hasattr(store, "save"):
            raise ValueError(
                f"Memory store plugin '{normalized}' does not implement load/save"
            )
        return store
    raise ValueError(f"Unsupported memory backend: {backend}")


def create_in_memory_store(**_: Any) -> MemoryStore:
    return InMemoryMemoryStore()


def create_sqlite_memory_store(
    *,
    sqlite_path: Optional[str] = None,
    **_: Any,
) -> MemoryStore:
    path = sqlite_path or os.getenv("PORI_MEMORY_SQLITE_PATH") or ".pori/memory.db"
    return SQLiteMemoryStore(path)


@dataclass
class Block:
    label: str
    value: str = ""
    limit: int = DEFAULT_CORE_BLOCK_LIMIT
    read_only: bool = False

    def append(self, content: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        new_value = (self.value + "\n" + content).strip() if self.value else content
        self.value = (
            new_value[: self.limit] if len(new_value) > self.limit else new_value
        )

    def replace(self, old_string: str, new_string: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        if old_string not in self.value:
            raise ValueError(f"Text not found in block '{self.label}'")
        self.value = (self.value.replace(old_string, new_string))[: self.limit]

    def set_value(self, value: str) -> None:
        if self.read_only:
            raise ValueError(f"Block '{self.label}' is read-only")
        self.value = value[: self.limit] if len(value) > self.limit else value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "limit": self.limit,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        return cls(
            label=str(data.get("label", "")),
            value=str(data.get("value", "")),
            limit=int(data.get("limit", DEFAULT_CORE_BLOCK_LIMIT)),
            read_only=bool(data.get("read_only", False)),
        )


class CoreMemory:
    def __init__(self, block_limit: int = DEFAULT_CORE_BLOCK_LIMIT):
        self._blocks: Dict[str, Block] = {}
        self._block_limit = block_limit
        for label in ("persona", "human", "notes"):
            self._blocks[label] = Block(label=label, limit=block_limit)

    def get_block(self, label: str) -> Block:
        key = (label or "").strip()
        if not key:
            raise ValueError("Block label is required")
        if key not in self._blocks:
            self._blocks[key] = Block(label=key, limit=self._block_limit)
        return self._blocks[key]

    def update_block_value(self, label: str, value: str) -> None:
        block = self.get_block(label)
        block.set_value(value)

    def memory_insert(self, label: str, new_str: str, insert_line: int = -1) -> None:
        block = self.get_block(label)
        lines = block.value.splitlines()
        if insert_line < 0:
            idx = len(lines)
        else:
            idx = min(max(0, insert_line), len(lines))
        lines.insert(idx, new_str)
        new_value = "\n".join(lines).strip()
        if len(new_value) > block.limit:
            raise ValueError(
                f"Edit failed: New content ({len(new_value)} chars) exceeds block limit ({block.limit})"
            )
        block.set_value(new_value)

    def memory_rethink(self, label: str, new_memory: str) -> None:
        if LINE_NUMBER_REGEX.search(new_memory):
            raise ValueError("new_memory contains line-number prefixes")
        block = self.get_block(label)
        if len(new_memory) > block.limit:
            raise ValueError(
                f"Edit failed: New content ({len(new_memory)} chars) exceeds block limit ({block.limit})"
            )
        block.set_value(new_memory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_limit": self._block_limit,
            "blocks": {label: block.to_dict() for label, block in self._blocks.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreMemory":
        cm = cls(block_limit=int(data.get("block_limit", DEFAULT_CORE_BLOCK_LIMIT)))
        cm._blocks = {}
        for label, b in (data.get("blocks") or {}).items():
            cm._blocks[label] = Block.from_dict(b)
        for label in ("persona", "human", "notes"):
            if label not in cm._blocks:
                cm._blocks[label] = Block(label=label, limit=cm._block_limit)
        return cm

    def clone_read_only(self) -> "CoreMemory":
        """Return a deep copy with all blocks set to read-only."""
        cm = CoreMemory(block_limit=self._block_limit)
        cm._blocks = {}
        for label, block in self._blocks.items():
            cm._blocks[label] = Block(
                label=block.label,
                value=block.value,
                limit=block.limit,
                read_only=True,
            )
        return cm

    def compile(self) -> str:
        parts = []
        for label in ("persona", "human", "notes"):
            block = self._blocks.get(label)
            if block and block.value.strip():
                parts.append(f"<{label}>\n{block.value.strip()}\n</{label}>")
        if not parts:
            return ""
        return "<memory_blocks>\n" + "\n\n".join(parts) + "\n</memory_blocks>"


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolCallRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    task_id: Optional[str] = None


class TaskState(BaseModel):
    task_id: str
    description: str
    status: str = "in_progress"
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def complete(self, success: bool = True):
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()


class SerializableMemoryState(BaseModel):
    namespace: str
    user_id: str
    agent_id: str
    session_id: str
    current_task_id: Optional[str] = None
    message_ids: List[str] = Field(default_factory=list)
    task_ids: List[str] = Field(default_factory=list)
    summary_count: int = 0


class AgentMemory:
    _sentence_transformer_models: Dict[str, Any] = {}

    def __init__(
        self,
        *,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: Optional[str] = None,
        store: Optional[MemoryStore] = None,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:10]}"
        self.namespace = f"{self.user_id}:{self.agent_id}:{self.session_id}"

        if store is None:
            backend = os.getenv("PORI_MEMORY_BACKEND", "memory")
            sqlite_path = os.getenv("PORI_MEMORY_SQLITE_PATH")
            self.store = create_memory_store(backend=backend, sqlite_path=sqlite_path)
        else:
            self.store = store

        self.core_memory = CoreMemory()
        self.messages: List[AgentMessage] = []
        self.tool_call_history: List[ToolCallRecord] = []
        self.tasks: Dict[str, TaskState] = {}
        self.state: Dict[str, Any] = {}
        self.summaries: List[Dict[str, Any]] = []
        self.current_task_id: Optional[str] = None
        self.experiences: List[Dict[str, Any]] = []
        self.archival_passages: List[Dict[str, Any]] = []
        self._summary_message_id: Optional[str] = None
        self._embedding_backend = (
            os.getenv("PORI_MEMORY_EMBEDDING_BACKEND", "hash").strip().lower()
        )
        self._embedding_dim = max(
            64, int(os.getenv("PORI_MEMORY_EMBEDDING_DIM", "384") or 384)
        )
        self._embedding_model_name = os.getenv(
            "PORI_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        self._load_from_store()

    def _serialize_any(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, BaseModel):
            return json.loads(value.model_dump_json())
        if isinstance(value, dict):
            return {str(k): self._serialize_any(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_any(v) for v in value]
        return value

    def _safe_from_iso(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _hydrate_timestamps(self, items: List[Any]) -> None:
        """Parse ISO timestamp strings on each dict item in-place."""
        for item in items:
            if not isinstance(item, dict):
                continue
            ts = self._safe_from_iso(item.get("timestamp"))
            if ts is not None:
                item["timestamp"] = ts

    def _to_snapshot(self) -> Dict[str, Any]:
        return {
            "meta": {
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "current_task_id": self.current_task_id,
                "summary_message_id": self._summary_message_id,
            },
            "core_memory": self.core_memory.to_dict(),
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "tool_call_history": [
                tc.model_dump(mode="json") for tc in self.tool_call_history
            ],
            "tasks": {tid: t.model_dump(mode="json") for tid, t in self.tasks.items()},
            "state": self._serialize_any(self.state),
            "summaries": self._serialize_any(self.summaries),
            "experiences": self._serialize_any(self.experiences),
            "archival_passages": self._serialize_any(self.archival_passages),
        }

    def _load_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        meta = snapshot.get("meta") or {}
        self.current_task_id = meta.get("current_task_id")
        self._summary_message_id = meta.get("summary_message_id")

        self.core_memory = CoreMemory.from_dict(snapshot.get("core_memory") or {})
        self.messages = [
            AgentMessage.model_validate(m) for m in (snapshot.get("messages") or [])
        ]
        self.tool_call_history = [
            ToolCallRecord.model_validate(tc)
            for tc in (snapshot.get("tool_call_history") or [])
        ]
        self.tasks = {
            str(tid): TaskState.model_validate(t)
            for tid, t in (snapshot.get("tasks") or {}).items()
        }
        self.state = snapshot.get("state") or {}
        self.summaries = snapshot.get("summaries") or []
        self.experiences = snapshot.get("experiences") or []
        self.archival_passages = snapshot.get("archival_passages") or []

        for collection in (self.experiences, self.archival_passages, self.summaries):
            self._hydrate_timestamps(collection)

    def _load_from_store(self) -> None:
        snapshot = self.store.load(self.namespace)
        if not snapshot:
            return
        self._load_from_snapshot(snapshot)

    def _persist(self) -> None:
        self.store.save(self.namespace, self._to_snapshot())

    def persist(self) -> None:
        self._persist()

    def export_state(self) -> SerializableMemoryState:
        return SerializableMemoryState(
            namespace=self.namespace,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            current_task_id=self.current_task_id,
            message_ids=[m.id for m in self.messages],
            task_ids=list(self.tasks.keys()),
            summary_count=len(self.summaries),
        )

    @classmethod
    def from_state(
        cls,
        state: SerializableMemoryState | Dict[str, Any],
        *,
        store: Optional[MemoryStore] = None,
    ) -> "AgentMemory":
        parsed = (
            state
            if isinstance(state, SerializableMemoryState)
            else SerializableMemoryState.model_validate(state)
        )
        return cls(
            user_id=parsed.user_id,
            agent_id=parsed.agent_id,
            session_id=parsed.session_id,
            store=store,
        )

    def add_message(self, role: str, content: str) -> str:
        message = AgentMessage(role=role, content=content)
        self.messages.append(message)
        self._persist()
        return message.id

    def get_recent_messages(self, n: int = 10) -> str:
        """Get the last n messages as a string."""
        recent = self.messages[-n:]
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])

    def get_recent_messages_structured(self, n: int = 10) -> List[Dict[str, Any]]:
        recent = self.messages[-n:]
        return [
            {"id": msg.id, "role": msg.role, "content": msg.content} for msg in recent
        ]

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 1
        return max(1, int(math.ceil(len(text) / 4)))

    def _summarize_messages(self, messages: List[AgentMessage]) -> str:
        if not messages:
            return ""
        user_count = sum(1 for m in messages if m.role == "user")
        assistant_count = sum(1 for m in messages if m.role == "assistant")
        tool_count = sum(1 for m in messages if m.role not in {"user", "assistant"})
        snippets = []
        for m in messages[-5:]:
            txt = (m.content or "").strip().replace("\n", " ")
            snippets.append(f"{m.role}: {txt[:120]}")
        body = "\n".join(f"- {s}" for s in snippets)
        return (
            f"Conversation summary of earlier context: "
            f"{len(messages)} messages (user={user_count}, assistant={assistant_count}, other={tool_count}).\n"
            f"Recent highlights:\n{body}"
        )

    def get_token_limited_messages(
        self,
        max_tokens: int = 3000,
        reserve_tokens: int = 1200,
        include_summary_message: bool = True,
    ) -> List[Dict[str, Any]]:
        budget = max(200, int(max_tokens) - int(reserve_tokens))
        chosen: List[AgentMessage] = []
        used = 0
        eligible = [m for m in self.messages if m.role in {"user", "assistant"}]

        for msg in reversed(eligible):
            msg_tokens = self.estimate_tokens(msg.content)
            if chosen and (used + msg_tokens) > budget:
                break
            if not chosen and msg_tokens > budget:
                chosen.append(msg)
                used += msg_tokens
                break
            chosen.append(msg)
            used += msg_tokens

        chosen.reverse()

        dropped_count = max(0, len(eligible) - len(chosen))
        structured = [{"role": m.role, "content": m.content} for m in chosen]

        if include_summary_message and dropped_count > 0:
            dropped_messages = eligible[: -len(chosen)] if chosen else eligible
            dropped_ids = [m.id for m in dropped_messages]
            cached_summary = next(
                (
                    s
                    for s in reversed(self.summaries)
                    if isinstance(s, dict)
                    and s.get("source_message_ids") == dropped_ids
                    and s.get("summary")
                ),
                None,
            )
            summary_text = (
                str(cached_summary.get("summary"))
                if isinstance(cached_summary, dict)
                else self._summarize_messages(dropped_messages)
            )
            if summary_text:
                if not cached_summary:
                    self._summary_message_id = f"summary_{uuid.uuid4().hex[:10]}"
                    self.summaries.append(
                        {
                            "id": self._summary_message_id,
                            "timestamp": datetime.now(),
                            "dropped_count": dropped_count,
                            "source_message_ids": dropped_ids,
                            "summary": summary_text,
                        }
                    )
                    self._persist()
                structured.insert(
                    0,
                    {
                        "role": "system",
                        "content": summary_text,
                    },
                )
        return structured

    def add_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
    ) -> str:
        tool_call = ToolCallRecord(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            task_id=self.current_task_id,
        )
        self.tool_call_history.append(tool_call)
        self._persist()
        return tool_call.id

    def create_task(self, task_id: str, description: str) -> TaskState:
        task = TaskState(task_id=task_id, description=description)
        self.tasks[task_id] = task
        self.begin_task(task_id)
        self._persist()
        return task

    def begin_task(self, task_id: str) -> None:
        self.current_task_id = task_id
        self.state.pop("final_answer", None)
        self._persist()

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value
        self._persist()

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def get_final_answer(self) -> Optional[Dict[str, str]]:
        value = self.get_state("final_answer")
        if isinstance(value, dict):
            return value
        return None

    def create_summary(self, step: int, max_messages: int = 10) -> str:
        summary = {
            "step": step,
            "timestamp": datetime.now(),
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_call_history),
            "current_tasks": {id: task.status for id, task in self.tasks.items()},
        }
        self.summaries.append(summary)
        active_tasks = [t for t in self.tasks.values() if t.status == "in_progress"]
        text = (
            f"Step {step} summary: {len(self.messages)} messages, "
            f"{len(self.tool_call_history)} tool calls, "
            f"{len(active_tasks)} tasks in progress"
        )
        self._persist()
        return text

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower()) if t]

    def _semantic_score(self, query: str, text: str) -> float:
        q = self._tokenize(query)
        t = self._tokenize(text)
        if not q or not t:
            return 0.0
        qset, tset = set(q), set(t)
        overlap = len(qset.intersection(tset))
        if overlap == 0:
            return 0.0
        jaccard = overlap / len(qset.union(tset))
        exact = 1.0 if query.strip().lower() in (text or "").lower() else 0.0
        return min(1.0, (0.7 * jaccard) + (0.3 * exact))

    def _hash_embedding(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._embedding_dim
        vec = [0.0] * self._embedding_dim
        for token in tokens:
            idx_hash = hashlib.sha256(f"{token}:idx".encode("utf-8")).hexdigest()
            sign_hash = hashlib.sha256(f"{token}:sign".encode("utf-8")).hexdigest()
            idx = int(idx_hash, 16) % self._embedding_dim
            sign = 1.0 if (int(sign_hash, 16) % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0:
            return vec
        return [v / norm for v in vec]

    def _sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None
        model = self._sentence_transformer_models.get(self._embedding_model_name)
        if model is None:
            try:
                model = SentenceTransformer(self._embedding_model_name)
            except Exception:
                return None
            self._sentence_transformer_models[self._embedding_model_name] = model
        try:
            embedding = model.encode(text or "", normalize_embeddings=True)
            return [float(v) for v in embedding]
        except Exception:
            return None

    def _embed_text(self, text: str) -> List[float]:
        if self._embedding_backend in {
            "sentence-transformers",
            "sentence_transformers",
        }:
            embedding = self._sentence_transformer_embedding(text)
            if embedding is not None:
                return embedding
        return self._hash_embedding(text)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(n):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            norm_a += av * av
            norm_b += bv * bv
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        score = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
        return max(-1.0, min(1.0, score))

    def _hybrid_similarity(
        self, query: str, text: str, text_embedding: List[float]
    ) -> float:
        query_embedding = self._embed_text(query)
        embedding_score = max(
            0.0, self._cosine_similarity(query_embedding, text_embedding)
        )
        lexical_score = self._semantic_score(query, text)
        return max(0.0, min(1.0, (0.75 * embedding_score) + (0.25 * lexical_score)))

    def add_experience(
        self, text: str, importance: int = 1, meta: Optional[Dict[str, Any]] = None
    ) -> str:
        exp_id = f"exp_{uuid.uuid4().hex[:10]}"
        embedding = self._embed_text(text)
        experience = {
            "id": exp_id,
            "text": text,
            "importance": int(importance or 1),
            "meta": meta or {},
            "embedding": embedding,
            "timestamp": datetime.now(),
        }
        self.experiences.append(experience)
        self._persist()
        return exp_id

    def recall(
        self, query: str, k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        results: List[Tuple[str, str, float]] = []
        query_embedding = self._embed_text(query)
        for exp in self.experiences[-max(k * 4, 10) :]:
            text = str(exp.get("text", ""))
            exp_embedding = exp.get("embedding")
            if not isinstance(exp_embedding, list) or not exp_embedding:
                exp_embedding = self._embed_text(text)
                exp["embedding"] = exp_embedding
            embedding_score = max(
                0.0, self._cosine_similarity(query_embedding, exp_embedding)
            )
            lexical_score = self._semantic_score(query, text)
            score = (0.75 * embedding_score) + (0.25 * lexical_score)
            score = min(1.0, score + (0.04 * int(exp.get("importance", 1))))
            if score >= min_score:
                results.append((str(exp.get("id")), text, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def conversation_search(
        self, query: str, limit: int = 10, roles: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        scored: List[Tuple[float, AgentMessage]] = []
        query_embedding = self._embed_text(q)
        for idx, msg in enumerate(self.messages):
            if roles and msg.role not in roles:
                continue
            content = msg.content or ""
            lexical = self._semantic_score(q, content)
            content_embedding = self._embed_text(content)
            semantic = max(
                0.0, self._cosine_similarity(query_embedding, content_embedding)
            )
            score = (0.75 * semantic) + (0.25 * lexical)
            if score <= 0:
                continue
            recency = (idx + 1) / max(1, len(self.messages))
            scored.append((score + (0.05 * recency), msg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": (
                    m.timestamp.isoformat() if getattr(m, "timestamp", None) else None
                ),
                "score": round(s, 4),
            }
            for s, m in scored[:limit]
        ]

    def archival_memory_insert(
        self, text: str, tags: Optional[List[str]] = None, importance: int = 1
    ) -> str:
        pid = f"arch_{uuid.uuid4().hex[:10]}"
        embedding = self._embed_text(text)
        rec = {
            "id": pid,
            "text": text,
            "tags": tags or [],
            "importance": int(importance or 1),
            "embedding": embedding,
            "timestamp": datetime.now(),
            "owner": {
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "session_id": self.session_id,
            },
        }
        self.archival_passages.append(rec)
        self._persist()
        return pid

    def archival_memory_search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        q = (query or "").strip()
        if not q:
            return []

        results: List[Tuple[str, str, float]] = []
        query_embedding = self._embed_text(q)
        for rec in self.archival_passages:
            if tags:
                rec_tags = {str(t) for t in (rec.get("tags") or [])}
                if not rec_tags.intersection({str(t) for t in tags}):
                    continue
            text = str(rec.get("text", ""))
            rec_embedding = rec.get("embedding")
            if not isinstance(rec_embedding, list) or not rec_embedding:
                rec_embedding = self._embed_text(text)
                rec["embedding"] = rec_embedding
            embedding_score = max(
                0.0, self._cosine_similarity(query_embedding, rec_embedding)
            )
            lexical_score = self._semantic_score(q, text)
            score = (0.8 * embedding_score) + (0.2 * lexical_score)
            score = min(1.0, score + (0.03 * int(rec.get("importance", 1))))
            if score >= min_score:
                results.append((str(rec["id"]), text, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]


__all__ = [
    "AgentMemory",
    "AgentMessage",
    "ToolCallRecord",
    "TaskState",
    "Block",
    "CoreMemory",
    "SerializableMemoryState",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "create_memory_store",
    "create_in_memory_store",
    "create_sqlite_memory_store",
]
