# Hermes Release A Handoff

Branch: `feat/hermes-release-a`

Implemented immutable run identity, deterministic prompt/tool fingerprints,
typed tool receipts, trace/result evidence, scoped memory construction and
validation, and team child lineage. Local orchestrator memory reuse remains
backward compatible.

Verification:

- `uv run pytest -q`: 186 passed
- `uv run mypy pori/ --ignore-missing-imports`: passed, 62 files
- Black and isort: passed

Changes are uncommitted and unpushed.
