# Opt-in use and rollback

This experiment is disabled by default. No model/checkpoint is promoted by this
work; cached/fresh numerical qualification remains failed. Existing
`reference_dense`, default inference selection, math policy, native model source,
sampler, detokenizer and public server remain present.

**Caveat:** Under the [2026-11-02 promotion policy](promotion-policy.md), strict
numerical failures remain model diagnostics rather than an automatic veto on an
independently proven mathematically neutral optimization. Scoped optimization
promotion, model qualification and default enablement are separate decisions.
This policy edit changes no runtime selection or feature-flag default.

For the measured Qwen2.5 checkpoint, the existing CLI can select the experiment:

```sh
mlx-one serve /path/to/pinned-qwen2.5-q8 --offline \
  --inference-mode mlxboost_b1 --mlxboost-fast-decode \
  --execution-dtype float16 --cache-type-k f16 --cache-type-v f16 \
  --no-cache-prompt --prefix-cache-mib 0 --parallel 1
```

Use `bfloat16` and `bf16` KV for the measured Qwen3 and SmolLM2 checkpoints.
Python uses the existing `Engine(..., inference_mode="mlxboost_b1",
mlxboost_config=MLXBoostConfig(fast_decode=True), execution_dtype=...)` interface;
`MLXBoostConfig` is imported from `mlx_one.mlxboost`. ServerConfig and
GenerationEngine accept the same immutable configuration. No environment-global
mutable switch or monkey-patch is used in production.

Roll back the single experiment with `--no-mlxboost-fast-decode`, or return to
`--inference-mode reference_dense` and remove the MLXBoost flag. In Python use
`MLXBoostConfig()` or the unchanged reference mode. Recreate the Engine/server
to change its immutable configuration.

The dispatch envelope is dense B1, no prefix reuse, MLX 0.32.2 on Metal:
Qwen2/Q8/group64/FP16, Qwen3/Q8/group64/BF16, or Llama-family unquantized/BF16.
Unsupported combinations fall back to `reference_dense` with the reason in
`capabilities()["selection"]`. `strict_reference=True` rejects them instead.
This envelope is implementation eligibility, not checkpoint qualification.

The independent switches `native_quantized_dispatch`, `dense_cache_capacity`,
`fused_attention`, `stream_bridge`, `compile_experiment`, `experimental_qgemv`
are reserved, false by default and unimplemented. Requesting them triggers visible
fallback or strict failure. No optional extension is required to install/import.

Only the absence of an activation calibration candidate enables the shortcut.
A live `activation_peak` record immediately restores all existing scope checks.
Conservative activation forecasts, transient reservations, host capacity checks,
cancellation and allocator accounting remain active. The experiment intentionally
does not accelerate a registry that contains activation calibration candidates.
