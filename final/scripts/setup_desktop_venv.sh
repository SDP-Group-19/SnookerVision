#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/5] Creating virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "[2/5] Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "[3/5] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/5] Installing desktop dependencies"
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "[5/5] Verifying environment"
python "$ROOT_DIR/final/scripts/check_env.py"

cat <<EOF

Desktop virtual environment is ready.

Activate it:
  source "$VENV_DIR/bin/activate"

Run SnookerVision:
  python "$ROOT_DIR/final/src/snookervision/app/main.py" --no-interface

EOF
