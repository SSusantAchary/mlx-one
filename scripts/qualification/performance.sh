#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec "${MLX_ONE_PYTHON:-$repo_dir/.venv/bin/python}" "$repo_dir/scripts/qualification/run_inference_optimization.py" performance "$@"
