#!/usr/bin/env bash
# Run the monorepo dependency-boundary check (products -> extensions -> pori).
#
# ACTIVE: enforced in CI (`boundaries` job). Run locally with:
#   pip install import-linter && bash tools/ci/check-boundaries.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$REPO_ROOT/tools/ci/importlinter.ini"

if ! command -v lint-imports >/dev/null 2>&1; then
  echo "[boundaries] import-linter not installed. Install with: pip install import-linter" >&2
  exit 1
fi

# pori lives at the repo root; aloy_backend under the Aloy backend. Both must be
# importable for grimp to build the graph. On Git Bash / MSYS, Windows Python
# needs Windows-style paths and the ';' separator.
SEP=":"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    SEP=";"
    REPO_ROOT="$(cygpath -w "$REPO_ROOT")"
    ;;
esac
export PYTHONPATH="${REPO_ROOT}${SEP}${REPO_ROOT}/products/aloy/backend${PYTHONPATH:+${SEP}${PYTHONPATH}}"

echo "[boundaries] running import-linter with $CONFIG"
lint-imports --config "$CONFIG"
