#!/bin/bash
set -euo pipefail

# Auto-detect a sensible parallel build level.
detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
    return
  fi
  python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
}

CPU_COUNT="$(detect_cpu_count)"
# Leave one core free for system responsiveness; cap to avoid memory-heavy over-parallel builds.
DEFAULT_BUILD_JOBS=$((CPU_COUNT > 2 ? CPU_COUNT - 1 : 1))
if [ "${DEFAULT_BUILD_JOBS}" -gt 16 ]; then
  DEFAULT_BUILD_JOBS=16
fi
BUILD_JOBS="${FLA_BUILD_JOBS:-$DEFAULT_BUILD_JOBS}"

# Common env vars used by native build backends (ninja/cmake/setuptools/custom setup.py).
export MAX_JOBS="${BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
export MAKEFLAGS="-j${BUILD_JOBS}"

echo "Detected ${CPU_COUNT} CPUs; using ${BUILD_JOBS} parallel build jobs."

# Recreate the venv from scratch to avoid stale packages.
uv venv .venv --clear
source .venv/bin/activate

# Ensure the lock reflects current pyproject constraints, then sync deps.
uv lock
uv sync --extra dev --extra test --extra benchmark

# Install this repo in editable mode for local development.
uv pip install -e .

uv pip install setuptools
uv pip install ipython gpustat

python - <<'PY'
import importlib.metadata
import torch
import triton
print("flash-attn", importlib.metadata.version("flash-attn"))
print("torch", torch.__version__)
print("triton", triton.__version__)
PY
