# E12/HOST-01: impossible calibration lookup

Hypothesis: per-token source hashing, model inventory and host identity queries
dominate the Qwen HTTP decode gap when no activation calibration candidate exists.
The original HTTP device-thread profile directly observed this work; see
`profiling-bottlenecks.md`.

Upstream context: pinned MLX-LM `generate.py::generate_step`, commit
`93e6d4a6447b3080861985947f0820e76fcb957d`, keeps generation/cache state persistent.
The changed lookup itself is MLX-One host metadata logic, independently authored;
no upstream algorithm, kernel or source block was copied for this optimization.

Changed functions: `engine/pressure.py::execution_forecast` and the existing
engine/server/CLI configuration plumbing. New immutable `MLXBoostConfig` and
explicit `mlxboost_b1` mode bind the opt-in choice to the runtime. The reference
branch still constructs and validates scope as before.

Only `fast_decode=True` is implemented. When the live registry has no
`activation_peak` records, scope construction cannot find a match and is skipped.
The conservative forecast, memory reservation/admission and all model execution
remain unchanged. If a candidate appears, full validation resumes immediately.
Flag-off restores the original lookup; unsupported settings visibly fall back.

| Checkpoint | Captured reference/off/on numeric comparisons | Maximum absolute change | Exact native output |
| --- | ---: | ---: | --- |
| Qwen2.5 Q8 FP16 | 544 | 0 | PASS |
| Qwen3 Q8 BF16 | 592 | 0 | PASS |
| SmolLM2 BF16 | 544 | 0 | PASS |

Raw runs: `local/mlxboost/runs/ablation-v2-{qwen2,qwen3,smollm2}-20261101/`.
Each has three timed requests per variant after warmup, one process per model,
plus separate instrumented state captures. Performance and state capture are
separate. Native public Engine medians:

| Model | Reference tok/s | Mode flag-off tok/s | fast_decode tok/s |
| --- | ---: | ---: | ---: |
| Qwen2.5 | 23.88 | 23.02 | 106.21 |
| Qwen3 | 22.23 | 20.72 | 94.62 |
| SmolLM2 | 15.53 | 14.44 | 25.26 |

Current-source original HTTP protocol (3 processes, 6 timed requests each):

| Model | Flag-off E2E tok/s | Flag-on E2E tok/s | Off → on TTFT ms | Off → on peak RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 | 23.02 | 126.67 | 109.44 → 70.77 | 0.694 → 0.699 |
| Qwen3 | 21.56 | 96.21 | 130.97 → 89.62 | 0.797 → 0.799 |
| SmolLM2 | 15.39 | 24.56 | 194.87 → 182.74 | 3.431 → 3.379 |

These are unfiltered medians, not claims that all native output matches upstream.
Expanded confirmation and all cross-stack shared subsets are reported separately.
No combined optimization path exists: the only optimized variant enables this
single flag. No direct forward speedup is claimed; no model operator changed.

Current deployment decision: opt-in **EXPERIMENTAL** for all three checkpoints;
defaults remain reference. Under the [2026-11-02 policy](promotion-policy.md), the
measured gain and exact change parity support independent scoped promotion
assessment of this neutral host optimization. The inherited cached/fresh failures
remain model-qualification failures and are no longer an automatic optimization
veto. This policy edit makes no rollout decision or model/alpha qualification
claim. Kernel, cache, attention, stream and compile proposals were not implemented
or presented as measured optimizations; each requires its own profiling and
affected correctness evidence.
