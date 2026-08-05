#!/usr/bin/env bash
# Dry-run-first wrapper for the official Meta-World MT10/MT50 REIM pipeline.
# Execute mode resolves the validation-tuned deployment gate just before each
# evaluation and ends with the read-only five-bank separation audit.

set -Eeuo pipefail

REIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REIM_PYTHON_BIN="${REIM_PYTHON:-}"

if [[ -z "$REIM_PYTHON_BIN" ]]; then
  if [[ -x "$REIM_ROOT/.venv/bin/python" ]]; then
    REIM_PYTHON_BIN="$REIM_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    # A system interpreter is sufficient to inspect the default dry-run plan.
    # Real execution still validates dependencies inside each fail-fast stage.
    REIM_PYTHON_BIN="$(command -v python3)"
  else
    printf 'run_multitask.sh: no Python found; run ./setup.sh or set REIM_PYTHON\n' >&2
    exit 2
  fi
fi

exec "$REIM_PYTHON_BIN" "$REIM_ROOT/scripts/run_multitask_pipeline.py" "$@"
