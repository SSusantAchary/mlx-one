# MLXBoost experiments

[Experimental index](README.md) · [Module boundary ADR](../architecture/mlxboost-module-boundary.md)

<!-- mlx-one:mlxboost-summary:start -->
## MLXBoost: optional execution module

MLXBoost is an independent optional module **inside MLX-One**. The current
implementation and defaults are unchanged: `reference_dense` is the default;
`mlxboost_b1` and `fast_decode` require explicit opt-in and compatible execution
facts. Unsupported profiles fall back with a recorded reason, or raise when
strict reference handling is requested. It remains **EXPERIMENTAL**, with no
production rollout approval.

Historical Apple M4 / 32 GiB HTTP measurements (MLX 0.32.2):

| Checkpoint / precision | Flag off (tok/s) | Flag on (tok/s) | Timed requests per flag |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct / Q8 | 22.94 | 109.17 | 50: five processes × ten requests |
| Qwen3-0.6B / Q8 | 21.42 | 96.42 | 50: five processes × ten requests |
| SmolLM2-1.7B-Instruct / BF16 | 15.39 | 24.56 | 6: three processes × two requests |

These are medians of per-request **output tokens / complete HTTP SSE wall time**,
not decode-only or direct-model rates. One warmup per process and model loading
are excluded. Cohorts and run order differ; host load and thermals were not
controlled. The two Qwen cohorts have 100/100 paired exact-output matches across
the models. This is bounded evidence, not a general speedup guarantee.

The measured change skips impossible calibration metadata/hash/inventory work
when no matching activation-peak candidate exists. It adds no custom Metal
kernel and does not alter model equations. Existing cached-vs-fresh model
qualification failures remain failed; a source-neutral execution change can be
evaluated separately under the current policy. See the
[benchmark evidence and limits](../publication/2026-11-04/audit.md#mlxboost-evidence)
and [module boundary ADR](../architecture/mlxboost-module-boundary.md).

A standalone **`mlx-one-boost` distribution is PROPOSED ONLY**. No package split,
dependency, installation command, import migration or default change is shipped.
<!-- mlx-one:mlxboost-summary:end -->

## Design and implemented path

The [source audit](../mlxboost/source-audit.md), [source diff map](../mlxboost/source-diff-map.md)
and [profiling report](../mlxboost/profiling-bottlenecks.md) explain the bounded
host-overhead change. The module keeps equations, weight contracts and cache
semantics in native model/core ownership. Eligibility and recorded fallback
are separate from checkpoint qualification.

## Comparisons, regressions and unimplemented proposals

[Baseline reproduction](../mlxboost/baseline-reproduction.md),
[optimization ablation](../mlxboost/optimization-ablation.md) and
[benchmark comparison](../mlxboost/benchmark-comparison.md) retain different
cohorts, dependency mismatches, request counts, output/EOS differences and
uncontrolled thermal/load limits. The SmolLM2 short-EOS discrepancy across
0.32.2/0.32.4 stacks remains unresolved. These full HTTP rates cannot be described
as kernel or model-forward improvements.

No additional quantized dispatch wrapper, fused attention/capacity path,
stream/async evaluator, custom QGEMV, residual-RMSNorm kernel or compilation path
has been qualified by MLXBoost. Its [qualification matrix](../mlxboost/qualification-matrix.md)
marks those proposals as not implemented or entry gates not met. Existing stock
MLX kernels and native runtime experiments have separate ownership.

## Qualification and rollback

[Numerical parity](../mlxboost/numerical-parity.md) and
[output parity](../mlxboost/output-parity.md) retain failed model gates. The
[promotion policy](../mlxboost/promotion-policy.md) permits independent evaluation
of a proven neutral host optimization without converting those failures to passes.
There is no model/release promotion or production rollout approval.
[API/regression evidence](../mlxboost/api-regression.md) distinguishes isolated
regression passes from a pre-existing monolithic-suite hang.

Future qualification needs exact checkpoint/source/environment identity,
repeatable scope-specific benefits, unchanged affected numerical/state/API gates,
longer-context and cancellation/backpressure evidence, and independent rollout
review. [Deployment and rollback](../mlxboost/deployment-and-rollback.md) retains
the reference path. No flags or defaults change in this refactor.
