# P0.1 native math implementation

This is an implementation and evidence ledger, not a production-speed claim.
The authoritative plan is `mlx_one_p01_native_math_implementation_test_plan.md`.
Existing generation methods, HTTP payloads, SSE framing, defaults, cache owners,
and native model implementations remain in place. No automatic promotion or push.

## Source and baseline

Git base: `53404467afa3cf094bd221c61f5316f2b928b161`.
Preintervention working-source digest:
`554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.
The source includes previously uncommitted P0 repairs; the Git SHA alone does not
identify the measured runtime. The 40-worker frozen baseline completed before runtime installation. Phase A is
now installed; shared Qwen math is undergoing native gates.

Baseline cohort: `qualification/results/p01_math/baseline/1792231798413449000`.
Four pinned checkpoints, five prompt lengths (128/512/2048/4096/8192), two
independent workers each, one warm-up and two measured requests per worker.
These diagnostic repetitions do not satisfy the final ten-repeat gate.
Native host: Apple M4, 32 GiB, macOS 15.7.9, Python 3.12.11, MLX 0.32.2.
The supervised ceiling is 21.33 GiB, below 75% of installed RAM. GPU, child RSS,
and MLX memory preflight are required; the watchdog cannot be disabled.

## Runtime contracts

`NativeMathPolicy` is immutable and travels through request-local execution
context. `reference_dense` retains ALL logits and reference attention/projection
arithmetic. Optimized operators require explicit `native_optimized`, FP16
floating execution and attention KV, B1, no speculation and no optional prefix
reuse. Packed integer weights are preserved. Unsupported custom specializations
and Engine compilation are rejected until implemented and measured.

LAST_ONLY selects normalized hidden rows before the vocabulary projection and
returns `[B,1,V]`. Returned hidden states retain all time positions. NO_LOGITS is
restricted to unused intermediate prefill chunks; state consumers retain logits.
The stock SDPA adapter takes native GQA heads without KV repetition, but both
Qwen families retain reference attention after nonuniform full-state native
failures. Other families' fused dispatch is not validated or publicly available.
Known causal masks use
the lower-right causal contract; FP32 additive masks retain reference precision.
Projection gates inspect actual call rows, affine metadata and packed shapes.
Matrix QMM failed native full-state parity; current public experimental dispatch
retains rowwise reference GEMM and experiments with stock FP16 vector calls only.
Failed GPU execution is not retried against partially mutated cache state.

Explicit chunk sizes are 256/512/1024/2048. Existing growable KV storage remains
the only owner; a second cache implementation is not introduced. Experimental
forecasts retain conservative attention overlap and cannot reuse calibration
from an unmatched math policy. Hybrid state precision is not treated as KV.

## Review order and evidence requirements

| Boundary | Implementation scope | Native gate |
|---|---|---|
| PR-00 | Frozen inventory, watchdog, phase traces, comparisons | Baseline repeatability; separate profiling/timing |
| PR-01 | ALL/LAST_ONLY/NO_LOGITS across five native heads | Same-checkpoint logits/tokens; measured memory |
| PR-02 | Stock SDPA, native GQA, exact mask gates | Qwen3/Qwen2.5 fresh/cached parity and 8K benefit |
| PR-03 | Shape-gated stock packed QMM | Real projections and full-model parity; protected decode |
| PR-04 | Explicit chunk policy and existing bounded KV | All four models, offsets, admission, cancellation |
| PR-05 | Synchronized decode profiles, bounded pure compile experiment | TPOT, long continuation, native graph-memory stability |
| PR-06 | LFM2 component profiling and state validation | Full/chunked convolution and KV isolation |
| PR-07 | Qwen3.5 component profiling and exact GDN oracle | FP32 state, full/chunked parity, measured benefit |
| PR-08 | Matched evidence, lifecycle and competitor reports | Ten repeats, independent ABBA, complete protected matrix |

Phase A validates Qwen3 and Qwen2.5 before hybrid math changes. Shared math is
frozen for the standalone MLX-LM comparison. Hybrid attention and recurrent math
are held back until that gate. No custom Metal kernel has been selected.
MLX 0.32.2 has no exported gated-delta primitive in its pinned Python binding;
the exact sequential GDN remains the oracle. An unavailable specialization is
not an optimized success.

## Tooling

`benchmarks/p01_native_math.py` dispatches baseline, run, micro, external and
comparison commands. Native release workloads reuse the existing correctness,
streaming, state and lifecycle case inventory. Discovery imports no MLX. `qualification/p01_experiments.py`
supervises explicit parity, Engine, controlled-forward and profile workloads.
Every worker validates source, tooling, pinned offline assets and execution
identity. Failed workers and missing telemetry remain visible.

Controlled decode evaluates prompt history before timing fixed continuation
forwards. Its fixture is labeled forced, not generated. Engine timing uses
accepted model-token intervals, excluding prefill and first-token work. Neither
measurement establishes HTTP/SSE service performance.

Profile JSONL contains synchronized inclusive phases and selected dispatch
reasons. Nested phase durations must not be summed. Ordinary throughput does
not enable phase synchronization, logits/state copying or GPU capture. Kernel
counts and unavailable thermal/power telemetry remain null. Xcode capture is
restricted to explicit instrumented profile workers.

Real packed-projection microbenchmarks preserve original modules and expose
reference-output overlap separately. Their peaks are diagnostic and cannot
establish an end-to-end memory win. Final comparisons require matched identities,
independent ABBA chronology, numerical proof, telemetry and protected coverage.

## Commands

Run from the repository with `PYTHONPATH=src:.:tests` and the existing `.venv`.
`BASELINE` refers to a completed P0 cohort containing immutable downloads and
environment manifests, not to a mutable Hugging Face branch.

```sh
BASELINE=qualification/results/p0_engine/runs/1792201272601288000
.venv/bin/python benchmarks/p01_native_math.py run --baseline "$BASELINE" \
  --route optimized --workload parity --model mlx-community/Qwen3-0.6B-8bit \
  --prompt-tokens 1 2 128 512 2048 8192 --repetitions 1 --run
.venv/bin/python benchmarks/p01_native_math.py run --baseline "$BASELINE" \
  --route reference optimized --workload engine \
  --model mlx-community/Qwen3-0.6B-8bit --prompt-tokens 128 512 2048 8192 \
  --repetitions 10 --correctness-proof /absolute/parity/data/experiment.json --run
.venv/bin/python benchmarks/p01_native_math.py external --inspect
PYTHONPATH=src:.:tests .venv/bin/python -m pytest --no-cov -q tests/optimization
```

Add `--projection mlx_qmm` for the separately identified vector candidate.
`--attention mlx_sdpa` records NOT_SUPPORTED for the currently unavailable fused
route; it cannot generate performance qualification. Use `--chunk-tokens` for
chunk experiments, `--workload
profile` for instrumentation, and `--workload controlled --forced-fixture FILE`
for explicitly forced 128/256-forward measurements. Schedule only one GPU
worker at a time. Performance proofs must match each requested shape and policy.

## Sequential shared campaign

`python -m qualification.p01_campaign` discovers the shared Qwen campaign
without initializing MLX. Supply the immutable P0 baseline plus one
`--prompts MODEL=PATH` rendered-token manifest per checkpoint. Add `--run` only
for native execution. Default coverage includes all five prompt widths, two
independent processes per route, reference/optimized ABBA timing, explicit
warm-up and ten measured repetitions. Source, tools and manifests are frozen.
Parity and full release cases precede profiling and timing. Failed model gates
skip dependent work; watchdog, telemetry and source failures stop the campaign.
Worker success does not pass materiality or protected-regression gates.

The separate bounded attention probe is runnable as
`python -m qualification.p01_attention_micro --baseline BASELINE --model MODEL
--run`. It uses actual native head layouts with explicit lower-right causal
offsets, FP16 storage and separately labeled FP16/FP32 stock operands. Paired
scratch is reserved conservatively. Numerical failures stop shape expansion;
diagnostic primitive results cannot enable a failed full-model route.

Hybrid extraction candidates passed CPU tests in a temporary validation copy.
They remain staged outside the runtime while shared Phase A is incomplete.
LFM2 and Qwen3.5 keep their existing hybrid arithmetic in the actual checkout.
The standalone MLX-LM numerical comparison failed, and oMLX remains blocked by
the recorded dependency mismatch. Neither supplies a performance claim.

## Acceptance status

Completion requires the executed evidence in `P01_FOUR_MODEL_RESULTS.md` and
`P01_OMLX_COMPARISON.md`. CPU fixtures and staged candidates are not native
qualification. Missing custom-kernel, compile, lifecycle or competitor evidence
remains pending. Materiality is >=10% target throughput or >=15% matched peak
reduction with no confirmed >3% protected throughput regression. Defaults remain
unchanged even if an experimental comparison passes.
