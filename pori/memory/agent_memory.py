"""The `AgentMemory` facade — everything persistent about a session.

Bundles CoreMemory blocks, conversation messages, tool-call records (including
the write-ahead dispatch journal), tasks with resumable checkpoints, archival
passages/experiences, and hybrid (semantic + lexical) memory-record search,
persisting the whole snapshot through a pluggable `MemoryStore`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..memory_contracts import (
    ConflictPolicy,
    MemoryCatalog,
    MemoryHit,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
)
from .models import (
    AgentMessage,
    CoreMemory,
    SerializableMemoryState,
    TaskState,
    ToolCallRecord,
)
from .stores import MemoryStore, create_memory_store

logger = logging.getLogger(__name__)


class AgentMemory:
    _sentence_transformer_models: Dict[str, Any] = {}

    # Weights for blending embedding-based (semantic) and lexical relevance
    # scores into a single hybrid score. Kept here so all search paths stay in
    # sync instead of repeating the literals.
    _SEMANTIC_WEIGHT = 0.75
    _LEXICAL_WEIGHT = 0.25

    @classmethod
    def _blend_scores(cls, semantic: float, lexical: float) -> float:
        """Combine a semantic and a lexical score into one hybrid score."""
        return (cls._SEMANTIC_WEIGHT * semantic) + (cls._LEXICAL_WEIGHT * lexical)

    def __init__(
        self,
        *,
        organization_id: str = "default_org",
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: Optional[str] = None,
        store: Optional[MemoryStore] = None,
    ):
        self.organization_id = organization_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:10]}"
        self.scope = MemoryScope(
            organization_id=self.organization_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )
        self.namespace = self.scope.namespace
        self.legacy_namespace = f"{self.user_id}:{self.agent_id}:{self.session_id}"

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
        self.memory_records: List[MemoryRecord] = []
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
                "organization_id": self.organization_id,
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
            "memory_records": [
                record.model_dump(mode="json") for record in self.memory_records
            ],
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
        self.memory_records = [
            MemoryRecord.model_validate(record)
            for record in (snapshot.get("memory_records") or [])
        ]

        for collection in (self.experiences, self.archival_passages, self.summaries):
            self._hydrate_timestamps(collection)
        if not self.memory_records:
            self._migrate_legacy_records()

    def _load_from_store(self) -> None:
        snapshot = self.store.load(self.namespace)
        if not snapshot:
            snapshot = self.store.load(self.legacy_namespace)
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
            organization_id=self.organization_id,
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
            organization_id=parsed.organization_id,
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

    def _select_window(
        self, max_tokens: int, reserve_tokens: int
    ) -> Tuple[List[AgentMessage], List[AgentMessage]]:
        """Split eligible (user/assistant) messages into ``(kept_tail, dropped)``.

        Keeps the most recent messages that fit the token budget; everything
        older is dropped. Shared by the window builder and the AC-3 compressor so
        the dropped-id set they compute always matches.
        """
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
        dropped = eligible[: -len(chosen)] if chosen else list(eligible)
        return chosen, dropped

    def context_window_dropped(
        self, max_tokens: int, reserve_tokens: int
    ) -> Tuple[List[AgentMessage], List[str]]:
        """Return ``(dropped_messages, dropped_ids)`` for the current budget."""
        _, dropped = self._select_window(max_tokens, reserve_tokens)
        return dropped, [m.id for m in dropped]

    def has_cached_summary(self, dropped_ids: List[str]) -> bool:
        """True if a summary is already cached for exactly this dropped-id set."""
        return any(
            isinstance(s, dict)
            and s.get("source_message_ids") == dropped_ids
            and s.get("summary")
            for s in self.summaries
        )

    def store_context_summary(self, dropped_ids: List[str], summary_text: str) -> None:
        """Cache a compression summary keyed by the dropped-message ids (AC-3)."""
        self._summary_message_id = f"summary_{uuid.uuid4().hex[:10]}"
        self.summaries.append(
            {
                "id": self._summary_message_id,
                "timestamp": datetime.now(),
                "dropped_count": len(dropped_ids),
                "source_message_ids": dropped_ids,
                "summary": summary_text,
            }
        )
        self._persist()

    def get_token_limited_messages(
        self,
        max_tokens: int = 3000,
        reserve_tokens: int = 1200,
        include_summary_message: bool = True,
    ) -> List[Dict[str, Any]]:
        chosen, dropped = self._select_window(max_tokens, reserve_tokens)
        dropped_count = len(dropped)
        structured = [{"role": m.role, "content": m.content} for m in chosen]

        if include_summary_message and dropped_count > 0:
            dropped_ids = [m.id for m in dropped]
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
            # An LLM summary (cached by the AC-3 compressor) is used when present;
            # otherwise fall back to the cheap deterministic role-count summary.
            summary_text = (
                str(cached_summary.get("summary"))
                if isinstance(cached_summary, dict)
                else self._summarize_messages(dropped)
            )
            if summary_text:
                if not cached_summary:
                    self.store_context_summary(dropped_ids, summary_text)
                structured.insert(0, {"role": "system", "content": summary_text})
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

    def record_tool_dispatch(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Write-ahead journal a tool call BEFORE it executes.

        Persisting the dispatch first means a crash mid-tool leaves evidence the
        call was attempted (status stays "dispatched"), so recovery can verify
        the side effect instead of blindly re-running it. Pair with
        :meth:`complete_tool_dispatch` after execution.
        """
        tool_call = ToolCallRecord(
            tool_name=tool_name,
            parameters=parameters,
            result={},
            success=False,
            task_id=self.current_task_id,
            status="dispatched",
        )
        self.tool_call_history.append(tool_call)
        self._persist()
        return tool_call.id

    def complete_tool_dispatch(
        self,
        record_id: str,
        result: Dict[str, Any],
        success: bool,
    ) -> None:
        """Fill in the result for a write-ahead dispatch record.

        Falls back to appending a fresh completed record if the dispatch record
        is missing, so a bookkeeping slip never loses the actual result.
        """
        for record in reversed(self.tool_call_history):
            if record.id == record_id:
                record.result = result
                record.success = success
                record.status = "completed"
                self._persist()
                return
        self.add_tool_call(
            tool_name="unknown", parameters={}, result=result, success=success
        )

    def pending_dispatches(self, task_id: str) -> List[ToolCallRecord]:
        """Tool calls journaled as dispatched but never completed for a task.

        Non-empty after a reload means a previous process died mid-tool; a
        resumed run should verify those side effects before repeating them.
        """
        return [
            record
            for record in self.tool_call_history
            if record.task_id == task_id and record.status == "dispatched"
        ]

    def checkpoint_task_progress(
        self,
        task_id: str,
        *,
        n_steps: int,
        consecutive_failures: int = 0,
        current_activity: str = "",
        plan: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist the run loop's position on the task record (every step).

        This is what makes an interrupted run resumable: a new Agent constructed
        with ``resume_task_id`` restores its step counter and plan from here
        instead of starting over.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return
        task.n_steps = n_steps
        task.consecutive_failures = consecutive_failures
        task.current_activity = current_activity
        if plan is not None:
            task.plan = plan
        task.progress_updated_at = datetime.now()
        self._persist()

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
        except Exception as e:
            logger.debug("sentence-transformers not available, using fallback: %s", e)
            return None
        model = self._sentence_transformer_models.get(self._embedding_model_name)
        if model is None:
            try:
                model = SentenceTransformer(self._embedding_model_name)
            except Exception as e:
                logger.debug(
                    "Failed to load embedding model %s: %s",
                    self._embedding_model_name,
                    e,
                )
                return None
            self._sentence_transformer_models[self._embedding_model_name] = model
        try:
            embedding = model.encode(text or "", normalize_embeddings=True)
            return [float(v) for v in embedding]
        except Exception as e:
            logger.debug("Embedding encode failed, using fallback: %s", e)
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
        return max(0.0, min(1.0, self._blend_scores(embedding_score, lexical_score)))

    def _catalog(self) -> MemoryCatalog:
        return MemoryCatalog(self.memory_records)

    def _replace_catalog(self, catalog: MemoryCatalog) -> None:
        self.memory_records = catalog.records()

    def _migrate_legacy_records(self) -> None:
        for experience in self.experiences:
            text = str(experience.get("text", "")).strip()
            if not text:
                continue
            meta = experience.get("meta") or {}
            self.memory_records.append(
                MemoryRecord(
                    id=str(experience.get("id") or f"exp_{uuid.uuid4().hex[:10]}"),
                    scope=MemoryScope(
                        organization_id=self.organization_id,
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                    ),
                    content=text,
                    importance=int(experience.get("importance", 1)),
                    provenance=MemoryProvenance(
                        source=str(meta.get("source", "legacy_experience")),
                        metadata=meta,
                    ),
                    created_at=self._safe_from_iso(experience.get("timestamp"))
                    or datetime.now(),
                    metadata={
                        "legacy_collection": "experience",
                        "embedding": experience.get("embedding"),
                    },
                )
            )
        for passage in self.archival_passages:
            text = str(passage.get("text", "")).strip()
            if not text:
                continue
            self.memory_records.append(
                MemoryRecord(
                    id=str(passage.get("id") or f"arch_{uuid.uuid4().hex[:10]}"),
                    scope=MemoryScope(
                        organization_id=self.organization_id,
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                    ),
                    content=text,
                    tags=[str(tag) for tag in (passage.get("tags") or [])],
                    importance=int(passage.get("importance", 1)),
                    provenance=MemoryProvenance(source="legacy_archival"),
                    created_at=self._safe_from_iso(passage.get("timestamp"))
                    or datetime.now(),
                    metadata={
                        "legacy_collection": "archival",
                        "embedding": passage.get("embedding"),
                    },
                )
            )

    def add_memory_record(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        tags: Optional[List[str]] = None,
        importance: int = 1,
        confidence: float = 1.0,
        sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL,
        provenance: Optional[MemoryProvenance] = None,
        retention: Optional[MemoryRetention] = None,
        conflict_key: Optional[str] = None,
        conflict_policy: ConflictPolicy = ConflictPolicy.KEEP_BOTH,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scope: Optional[MemoryScope] = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            scope=scope
            or MemoryScope(
                organization_id=self.organization_id,
                user_id=self.user_id,
                agent_id=self.agent_id if agent_id is None else agent_id,
                session_id=session_id,
            ),
            kind=kind,
            content=content,
            tags=tags or [],
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            provenance=provenance or MemoryProvenance(),
            retention=retention or MemoryRetention(),
            conflict_key=conflict_key,
            metadata=metadata or {},
        )
        catalog = self._catalog()
        catalog.add(record, conflict_policy=conflict_policy)
        self._replace_catalog(catalog)
        self._persist()
        return record

    def delete_memory_record(self, record_id: str, *, hard: bool = False) -> bool:
        catalog = self._catalog()
        deleted = catalog.delete(record_id, self.scope, hard=hard)
        if deleted:
            self._replace_catalog(catalog)
            self._persist()
        return deleted

    def export_memory_records(
        self, *, include_deleted: bool = False
    ) -> List[Dict[str, Any]]:
        return [
            record.model_dump(mode="json")
            for record in self._catalog().export(
                self.scope, include_deleted=include_deleted
            )
        ]

    def prune_expired_memory(self) -> List[str]:
        catalog = self._catalog()
        expired = catalog.prune_expired()
        if expired:
            self._replace_catalog(catalog)
            self._persist()
        return expired

    def search_memory_records(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.0,
        kinds: Optional[List[MemoryKind]] = None,
        tags: Optional[List[str]] = None,
        legacy_collection: Optional[str] = None,
    ) -> List[MemoryHit]:
        query_embedding = self._embed_text(query)

        def scorer(record: MemoryRecord) -> MemoryHit:
            embedding = record.metadata.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                embedding = self._embed_text(record.content)
                record.metadata["embedding"] = embedding
            semantic = max(0.0, self._cosine_similarity(query_embedding, embedding))
            lexical = self._semantic_score(query, record.content)
            final = self._blend_scores(semantic, lexical)
            final = min(
                1.0,
                final + (0.03 * record.importance) + (0.02 * record.confidence),
            )
            matched_by = []
            if semantic > 0:
                matched_by.append("semantic")
            if lexical > 0:
                matched_by.append("lexical")
            return MemoryHit(
                record=record,
                semantic_score=semantic,
                lexical_score=lexical,
                final_score=final,
                matched_by=matched_by,
            )

        hits = self._catalog().search(
            self.scope,
            scorer,
            k=max(k * 3, k),
            kinds=kinds,
            tags=tags,
            min_score=min_score,
        )
        if legacy_collection:
            hits = [
                hit
                for hit in hits
                if hit.record.metadata.get("legacy_collection") == legacy_collection
            ]
        return hits[:k]

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
        record = MemoryRecord(
            id=exp_id,
            scope=MemoryScope(
                organization_id=self.organization_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
            ),
            content=text,
            importance=int(importance or 1),
            provenance=MemoryProvenance(
                source=str((meta or {}).get("source", "agent")),
                metadata=meta or {},
            ),
            metadata={
                "legacy_collection": "experience",
                "embedding": embedding,
            },
        )
        catalog = self._catalog()
        catalog.add(record)
        self._replace_catalog(catalog)
        self._persist()
        return exp_id

    def recall(
        self, query: str, k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        hits = self.search_memory_records(
            query,
            k=k,
            min_score=min_score,
            legacy_collection="experience",
        )
        return [(hit.record.id, hit.record.content, hit.final_score) for hit in hits]

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
            score = self._blend_scores(semantic, lexical)
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
        record = MemoryRecord(
            id=pid,
            scope=MemoryScope(
                organization_id=self.organization_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
            ),
            content=text,
            tags=tags or [],
            importance=int(importance or 1),
            provenance=MemoryProvenance(source="agent"),
            metadata={
                "legacy_collection": "archival",
                "embedding": embedding,
            },
        )
        catalog = self._catalog()
        catalog.add(record)
        self._replace_catalog(catalog)
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

        hits = self.search_memory_records(
            q,
            k=k,
            min_score=min_score,
            tags=tags,
            legacy_collection="archival",
        )
        return [(hit.record.id, hit.record.content, hit.final_score) for hit in hits]
