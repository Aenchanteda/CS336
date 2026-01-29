#!/usr/bin/env bash
set -euo pipefail

echo "Starting cs336 assignment environment setup..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.11 or 3.12 and retry." >&2
  exit 1
fi

PY=python3

echo "Creating virtual environment at .venv"
$PY -m venv .venv
source .venv/bin/activate

echo "Upgrading pip, setuptools, wheel"
python -m pip install --upgrade pip setuptools build wheel

OS=$(uname -s)
ARCH=$(uname -m)
echo "Detected OS=$OS ARCH=$ARCH"

install_torch() {
  if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
    echo "Detected Intel macOS (x86_64) — installing torch~=2.2.2"
    python -m pip install "torch~=2.2.2"
  else
    echo "Installing torch~=2.6.0 (CPU / Apple Silicon)."
    echo "If you need a CUDA-enabled build on Linux, follow instructions at https://pytorch.org/get-started/locally/"
    python -m pip install "torch~=2.6.0"
  fi
}

echo "Installing local packages in editable mode"
python -m pip install -e ./cs336-basics || true
python -m pip install -e . || true

echo "Installing/ensuring torch separately (safer for platform-specific wheels)"
install_torch || echo "Warning: automatic torch install failed — please install manually per https://pytorch.org/get-started/locally/"

echo "Installing remaining common dependencies"
python -m pip install numpy matplotlib pandas tqdm wandb einops einx jaxtyping psutil submitit tiktoken regex humanfriendly || true

echo "(Optional) Installing ipykernel and registering kernel 'cs336-venv'"
python -m pip install ipykernel || true
python -m ipykernel install --user --name=cs336-venv --display-name "cs336-venv" || true

echo "Setup complete. Verify by running:"
echo "  . .venv/bin/activate"
echo "  python -c 'import cs336_basics; print(cs336_basics.__name__)'"
echo "Make sure your Jupyter kernel uses 'cs336-venv' if running the notebooks."

exit 0
