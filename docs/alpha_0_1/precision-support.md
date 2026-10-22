# Precision and checkpoint contract

Latest independent qualification: **Qwen3 quality FAIL, bounded coverage PASS**, 15/27 distinct cases passing across five deterministic repetitions. Native execution/lifecycle and stream/generate consistency passed. See [release-readiness.md](release-readiness.md). The results below retain their stated earlier source identity and scope. No alpha support promotion.

Current measured precision cell: pinned Qwen3 affine Q8/group64 executed natively with FP16 activation/logits and FP16 KV. This does not qualify unquantized FP16 weights. Independent quality/coverage and approved-source installation are pending; released precision claims remain NOT_QUALIFIED. See [support matrix](support-matrix.md) and [native evidence](native-qwen3-validation.md).

Five engineering targets: FP32, FP16, BF16, MLX affine Q4, MLX affine Q8. All current-release checkpoint cells remain NOT_QUALIFIED. INT4/INT8 enum values remain planning labels; neither signed INT8, GGUF, mxfp4/nvfp4 nor casting establishes an affine cell.

`mlx_one.core.precision.describe_precision(ModelSpec, activation_dtype=..., kv_cache_dtype=..., recurrent_state_dtype=..., accumulation_dtype=...)` adds a `checkpoint-precision-v1` metadata view to existing schema 1.0. No persisted schema fields or enum values were changed, so no migration is needed; old round trips are tested. Unknown compute/cache dtypes remain null. Per-layer exceptions remain in metadata and are labeled mixed. Positive group sizes describe assets, not qualified support (initial target 64). Quantized metadata requires exact integer bits/group size and affine mode/method.

`validate_finite_safetensors(path)` uses safetensors offset/shape validation followed by bounded 8 MiB CPU reads, including BF16 exponent checks and untouched packed integer payloads. It rejects NaN/Inf before native allocation when called by a loader. NumPy is imported lazily; A9 owns making this already-used runtime dependency explicit. A4–A7 own loader integration; a helper test alone is not proof every model loader is wired. No auto-quantization or casting added.

Checkpoint identity, hash/revision rejection, mixed quantized shape validation and immutable ModelSpec behavior reuse existing contracts. Finite tests include FP16, FP32, BF16, NaN/Inf and uint32 packed payloads. Command: `PYTHONPATH=src:tests:. .venv/bin/python -m pytest --no-cov -q tests/alpha_0_1/test_precision_contract.py tests/test_schemas.py tests/test_checkpoint.py tests/test_checkpoint_identity.py tests/test_text_runtime.py tests/test_execution_arithmetic.py` — **89 passed**, 0.73s; focused Ruff PASS. Ignored evidence: `a2.log`, `a2.junit.xml`.

Owner queues: A4 rejects malformed quantization before allocation and checks file containment/finite payloads; A5/A6/A7 wire applicable loader checks. No precision is ALPHA_SUPPORTED until current-source real checkpoint task and hardware evidence passes. See support-matrix.json for versioned status.

A10 owner-mediated A2 correction: malformed safetensors native parser exceptions are normalized to ValueError, so all wired loaders retain their documented public load-error types. Added a CPU regression; all 77 alpha contracts and the final 1666-test backend-free pass include this change. Stored schema compatibility remains unchanged.
