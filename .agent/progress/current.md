# Current State

## Active Task

Hermes Release C is implemented on `feat/hermes-release-c` and is uncommitted.

Core now separates context-window policy from durable memory, freezes curated
core memory and recalled evidence per run, records context diagnostics in
traces, provides provenance-preserving retrieval fusion, and defines session
lifecycle contracts with a local SQLite implementation supporting resume,
search, export, delete, and branch lineage.

## Verification

- `uv run pytest -q`: 196 passed.
- Black/isort and `git diff --check` pass on touched files.
- Broad mypy is blocked by optional FastAPI/Starlette/sentence-transformers
  imports absent from this environment, not by a reported Release C type error.

## Remaining Risks

- Local session search is bounded lexical SQL search, not FTS5 yet.
- Default summaries are deterministic extracts; no model summarizer is enabled.

## Next Session Should Start With

Review and commit Release C together with the matching Pori Cloud branch.
