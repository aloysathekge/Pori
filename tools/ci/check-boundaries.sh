#!/usr/bin/env bash
# Run the monorepo dependency-boundary check (products -> ext -> pori).
#
# STAGED: activates once the packages/ and products/ packages exist (Phase 4).
# Until then this prints a notice and exits 0 so it is safe to call anywhere.
set -euo pipefail

CONFIG="$(dirname "$0")/importlinter.ini"

if ! command -v lint-imports >/dev/null 2>&1; then
  echo "[boundaries] import-linter not installed. Install with: pip install import-linter"
  echo "[boundaries] (staged — skipping)"
  exit 0
fi

# The kernel is not yet packaged under packages/pori (still repo-root ./pori).
# Once migrated, remove this guard and let lint-imports run for real.
if [ ! -d "packages/pori" ] || [ -z "$(find packages/pori -name '*.py' 2>/dev/null | head -n1)" ]; then
  echo "[boundaries] packages/pori has no code yet (pre-Phase-4). Boundary check staged — skipping."
  exit 0
fi

echo "[boundaries] running import-linter with $CONFIG"
lint-imports --config "$CONFIG"
