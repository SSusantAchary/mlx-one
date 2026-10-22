> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Reproducing the numerical correction run

Use the existing repository environment on Apple Silicon. Exact source revisions and paths are pinned in this cohort's `asset-resolution.json` and all worker requests; use those paths rather than a later cached main reference. No source-library download is required.

```sh
.venv/bin/python -m pytest -q -o addopts=''
MLX_ONE_RUN_MLX_TESTS=1 .venv/bin/python -m pytest -q -o addopts='' tests/models tests/qualification/test_four_path_oracle.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmarks/model_parity/four_path_oracle.py CHECKPOINT_PATH --output /tmp/oracle-NEW.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/qualification/qualify_ten_models.py --check numerical --check generation --check cli --check http --timeout 360 --output /tmp/gates-NEW
```

For a primary derived track, add `--precision affine_q8` (M02/M03) or `affine_q4` (M04/M05/M06/M07/M08/M10). GPT-2 uses stored FP32; Gemma3 uses stored BF16. In-memory variants deliberately report CLI UNSUPPORTED. Saved artifacts are tested separately.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmarks/model_parity/quality_requalification.py CHECKPOINT_PATH --precision stored --output /tmp/quality-NEW.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmarks/model_parity/equal_operand_projection.py CHECKPOINT_PATH --precision affine_q8 --target model.layers.0.self_attn.q_proj --output /tmp/projection-NEW.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python benchmarks/model_parity/window_boundary_control.py CHECKPOINT_PATH --output /tmp/window-NEW.json
```

`--linear rowwise --attention rowwise` on the four-path tool explicitly substitutes diagnostic operators and sets `qualification_eligible=false`; it must never qualify the canonical profile. The source and primary records retain frozen tolerances. Arrays are materialized and independently copied before later cache updates. Local references are imported only by development tooling under an isolated namespace with bytecode writes disabled.

The exact executed campaign scripts, per-process logs, memory/timeout supervision, environment, source hashes, failed probes, corrected reruns, original reports and new checkpoint records are retained in the new cohort. `run-investigations.py` executes the central gates, independent oracles and task-quality requalification; `run-supplemental.py` preserves the first supplemental probes. Failed metadata/argument probes have separately named corrected results. Do not rerun an archived script into an existing output directory.

The wheel is built without isolation, installed into a fresh environment from existing cached dependency wheels, and tested from /tmp with reference-import rejection. No checkpoint weights or reference source trees are packaged. No hardware-performance extension is authorized by a diagnostic PASS alone.

The exact complete suite commands (rather than only the short native subset above) are saved in `regression-summary.json`. `parameter-integrity-executed.py` compares every shared text parameter, including explicit norm/conv representation mappings. Vision/MTP arrays absent from the text comparator remain outside its claim. `lfm-convolution-replay-executed.py`, `pointwise-replay-executed.py` and the Phi3 contiguous-rowwise controls complete the targeted component investigations. Quality-reviewed records derive only more specific failure categories from immutable generation outputs and retain their raw evidence SHA256.
