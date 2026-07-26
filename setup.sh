#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REIM_VENV_DIR:-${PROJECT_DIR}/.venv}"
PYTHON_BIN="${REIM_PYTHON:-python3}"
SYSTEM_SITE_ARGS=()

if [[ "${REIM_USE_SYSTEM_SITE_PACKAGES:-0}" == "1" ]]; then
  SYSTEM_SITE_ARGS+=(--system-site-packages)
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"REIM requires Python 3.10--3.13; found {sys.version.split()[0]}"
    )
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${SYSTEM_SITE_ARGS[@]}" "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install --no-deps --editable "${PROJECT_DIR}"

mkdir -p \
  "${PROJECT_DIR}/datasets/demonstrations" \
  "${PROJECT_DIR}/datasets/failures" \
  "${PROJECT_DIR}/checkpoints" \
  "${PROJECT_DIR}/results/tables" \
  "${PROJECT_DIR}/results/figures" \
  "${PROJECT_DIR}/results/logs" \
  "${PROJECT_DIR}/paper_assets"

"${VENV_DIR}/bin/python" - <<'PY'
import gymnasium
import metaworld
import mujoco
import stable_baselines3
import torch

print("REIM environment ready")
print(f"  torch={torch.__version__}, cuda={torch.cuda.is_available()}")
print(f"  gymnasium={gymnasium.__version__}")
print(f"  metaworld={getattr(metaworld, '__version__', '3.1.1')}")
print(f"  mujoco={mujoco.__version__}")
print(f"  stable-baselines3={stable_baselines3.__version__}")
PY

echo "Activate with: source ${VENV_DIR}/bin/activate"
