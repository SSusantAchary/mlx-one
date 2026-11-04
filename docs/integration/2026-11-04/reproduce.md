# Reproduce the 0.1.1 integration checks

Use an Apple Silicon Mac with the locked MLX/Metal 0.32.2 environment. Start from
the prepared integration branch in a fresh checkout. Raw fixture assets are not
redistributed; provide the pinned `local/native_tasks` artifact bundle indexed
by the unchanged source/checkpoint locks and M1–M5 manifests. Honor the original
model and dataset licensing terms. Source-oracle replay requires those sealed
NPZ/JSON files, not an arbitrary floating checkpoint or replacement dataset.

Create a native-only wheel environment outside the checkout:

```sh
python3 -m build --wheel --no-isolation --outdir dist
python3 -m venv /tmp/mlx-one-0.1.1-check
/tmp/mlx-one-0.1.1-check/bin/python -m pip install \
  --constraint docs/integration/2026-11-04/native-wheel-constraints.txt \
  './dist/mlx_one-0.1.1-py3-none-any.whl[server,vision,audio]'
/tmp/mlx-one-0.1.1-check/bin/python -m pip check
```

The build environment needs `build` and the repository's declared setuptools
version. The runtime environment must not contain Torch, Transformers, ONNX
Runtime, MLX-LM or oMLX. The committed verifier asserts this before task execution.
Replace `/path/to/pinned/native_tasks` below with the local artifact bundle.
Choose new output paths on every run:

```sh
MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 python3 \
  docs/integration/2026-11-04/replay.py \
  --checkout . --fixtures /path/to/pinned/native_tasks \
  --python /tmp/mlx-one-0.1.1-check/bin/python \
  --output /tmp/mlx-one-replay-0.1.1

MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 \
  /tmp/mlx-one-0.1.1-check/bin/python -I \
  docs/integration/2026-11-04/live_http.py \
  /path/to/pinned/native_tasks /tmp/mlx-one-live-http-0.1.1
```

The replay exports fresh native traces through the existing harness, applies
unchanged FP32 tolerances to sealed source traces, recomputes task quality and
runs installed Python/CLI/ASGI checks. The second command exercises real CLI
servers over loopback. Loading/quality runtimes are not published as benchmarks.

For the source regression suite, install the existing development dependencies
in a separate environment and make the pinned native fixture bundle available at
ignored `local/native_tasks`. To run retained MLXBoost provenance tests, copy
its locked captured source files inside the checkout at the manifest paths;
external source-capture symlinks are deliberately rejected. The original HTTP
comparison evidence is also a local fixture. It is not regenerated.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH=src python -m pytest --no-cov -q -rs

MLX_ENABLE_TF32=0 MLX_ONE_RUN_MLX_TESTS=1 MLXBOOST_NATIVE_TESTS=1 \
  PYTHONPATH=src python -m pytest --no-cov -q -rs \
  tests/native_tasks tests/mlxboost tests/ocr/test_native.py \
  tests/test_vlm_pipeline_native.py tests/test_audio_pipeline_native.py \
  tests/test_segmentation_task_native.py \
  tests/models/test_numerical_invariants_native.py \
  tests/models/test_reference_first_native.py \
  tests/test_native_whisper_integration.py \
  tests/test_native_minicpm5_integration.py tests/test_native_gpt2_mlx.py
```

Fresh public checkouts without local checkpoints/reference archives will skip
fixture-gated tests. That is not equivalent to this integration's real-checkpoint
evidence. [Verification details](verification.json) identify exact revisions,
artifact hashes, successful counts and retained skips.
