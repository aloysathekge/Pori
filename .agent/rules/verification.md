# Verification

Safe verification:

Python tests likely; prefer uv run pytest, pytest, or repo README command when dependencies are available.

Rules:

- Run the smallest useful verification first.
- Report commands that are missing or skipped.
- Do not deploy or trigger production jobs without approval.
- If dependencies are missing, report the dependency issue instead of guessing results.
