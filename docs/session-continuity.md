# Session Continuity

Pori separates three lifecycles:

- `ContextEngine` decides what enters a model call and emits diagnostics.
- `SessionRepository` owns transcripts, resume, search, export, delete, and
  branch lineage.
- `AgentMemory` owns curated and long-term memory under its independent scope,
  provenance, conflict, retention, and deletion policy.

An agent freezes compiled core memory and retrieved evidence at construction.
Memory tools may write during a run, but those writes are visible to the next
run rather than mutating the current prompt prefix. Trace output records context
diagnostics and fingerprints the frozen core-memory view.

`SQLiteSessionRepository` applies organization and user filters before local
search. `fuse_retrieval` retains source type and source ID when session and
memory evidence are combined.
