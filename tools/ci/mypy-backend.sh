#!/usr/bin/env bash
# Run the backend's mypy with the backend's own venv (exact CI parity),
# working on both Windows (Git Bash) and POSIX venv layouts.
set -euo pipefail
cd "$(dirname "$0")/../../products/aloy/backend"
if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi
exec "$PY" -m mypy aloy_backend/
