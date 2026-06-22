# Hermes Release C Handoff

Release C adds a `ContextEngine` boundary with diagnostics, frozen per-run core
memory and retrieval evidence, provenance-preserving retrieval fusion, portable
session contracts, and a local SQLite session repository with search, export,
delete, resume, and branching.

Core verification passes 196 tests. Full mypy cannot complete in the current
environment because optional FastAPI, Starlette, and sentence-transformers
dependencies are absent.
