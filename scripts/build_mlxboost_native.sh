#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT="${1:-local/mlxboost-phase0/dist-native}"
mkdir -p "$OUTPUT"
run_build() {
    test "$(uname -sm)" = "Darwin arm64"
    "$PYTHON" -c 'from importlib.metadata import version; assert version("mlx") == version("mlx-metal") == "0.32.2"'
    xcodebuild -version
    xcrun --find metal
    xcrun --find metallib
    cmake --version
    MLX_ONE_BUILD_NATIVE=1 "$PYTHON" -m build --wheel --no-isolation --outdir "$OUTPUT"
}
run_build 2>&1 | tee -a "$OUTPUT/build.log"
