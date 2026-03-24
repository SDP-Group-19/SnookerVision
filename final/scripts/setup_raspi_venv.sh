#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv-raspi}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[0/6] Installing Raspberry Pi system packages for Camera Module support"
echo "Run this first if you have not already:"
echo "  sudo apt update"
echo "  sudo apt install -y python3-venv python3-picamera2 python3-libcamera"

echo "[1/6] Creating Raspberry Pi virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR" --system-site-packages

echo "[2/6] Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "[3/6] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing Raspberry Pi stream dependencies"
python -m pip install -r "$ROOT_DIR/requirements-raspi.txt"

echo "[5/6] Checking core imports"
python - <<'PY'
import importlib
modules = ["cv2", "flask", "picamera2", "numpy"]
for name in modules:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
    except Exception as exc:
        print(f"[MISSING] {name}: {exc}")
PY

echo "[6/6] Done"
cat <<EOF

Raspberry Pi virtual environment is ready.

Activate it:
  source "$VENV_DIR/bin/activate"

Start the camera stream:
  python "$ROOT_DIR/final/scripts/raspi_camera_mjpeg_server.py" --host 0.0.0.0 --port 8000 --width 1280 --height 720

EOF
