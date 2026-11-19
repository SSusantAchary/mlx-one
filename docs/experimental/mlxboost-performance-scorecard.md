# MLX-Boost Performance Scorecard

Updated **2026-11-17**, using existing evidence only. **Implementation success is separate from performance qualification.** All entries remain default-off and
unapproved for production. No model inference acceleration is established.

| Kernel / milestone | Implementation achievement | Performance / qualification verdict |
| --- | --- | --- |
| Phase 0 `mlxboost.axpby.v1` | Native C++/Metal bridge, safe dispatch/fallback, synchronized profiling, offline registry; 123 acceptance tests and independent native/standard wheels | Foundation **ACCEPTED**; demo timings **EXPERIMENTAL**, no runtime or model approval |
| G03 `quantized_gemv_q4` | Packed affine Q4 dequantization fused with GEMV; no full dense weight expansion | **UNQUALIFIED**: isolated tests pass, performance and model correctness gates fail |
| G03 `quantized_gemv_q8` | Corresponding affine Q8 family and scoped opt-in adapter | **UNQUALIFIED**: same independent gates; near-neutral cases do not establish a gain |
| Fused MoE combine K4 | Working isolated Metal prototype; 17 focused tests pass | Operator **REJECT**; **MODEL BLOCKED**, checkpoint unavailable |

Documentation reviewed **2026-11-18**; measurements and verdicts are unchanged.

## Hardware and method

Apple **M4 / 32 GiB**, `applegpu_g16g`; MLX/Metal **0.32.2**, macOS **15.8.2**,
Xcode **26.3**. No M2/M3 or physical 8/16/24 GiB qualification is implied.
All latencies below are synchronized wall time, including submission, evaluation
and completed GPU work—**not isolated GPU-command time**. Speedup = stock/native;
below 1 means native is slower. Caches, power and thermals were uncontrolled.
Raw timings exclude repeated integrity checks; different protocols are not pooled.

## Phase 0 — AXPBY

Contiguous B1 vectors `[N]`, `0.75*x - 0.25*y`, output in input dtype. Two runs,
50 randomized/interleaved warm samples/backend/case after 10 warmups: **1,200 timed samples**. Cells show **run 1 / run 2**; latency is µs.

| Dtype | N | Stock µs | Native µs | Speedup | Shared peak MLX B |
| --- | ---: | ---: | ---: | ---: | ---: |
| float16 | 256 | 222.417 / 230.146 | 202.000 / 210.688 | 1.101× / 1.092× | 3,092 |
| float16 | 4096 | 262.000 / 258.812 | 226.312 / 224.103 | 1.158× / 1.155× | 49,172 |
| float16 | 65536 | 272.270 / 221.042 | 240.792 / 210.271 | 1.131× / 1.051× | 786,452 |
| float32 | 256 | 245.604 / 244.480 | 223.209 / 222.270 | 1.100× / 1.100× | 6,168 |
| float32 | 4096 | 261.020 / 256.875 | 225.500 / 209.812 | 1.158× / 1.224× | 98,328 |
| float32 | 65536 | 282.771 / 268.541 | 262.791 / 245.188 | 1.076× / 1.095× | 1,572,888 |

Correctness: **180 FP16/FP32 cases, max absolute error 0**. Demo operator
`atol=rtol=0.001` for FP16 and `0.000001` for FP32; model tolerances are separate.
Both backend records report the same profiling-window peak shown above; it is
not an independently attributable per-call memory saving.

Native execution: the primitive was evaluated and a Metal capture generated.
**Exact dispatch count was not recorded**; GUI capture inspection is unverified.
Offline proposals remain unapproved.
[Acceptance and wheels](mlxboost-native-phase0-evidence.md),
[raw runs](mlxboost-native-phase0-results/native-validated/measurements/),
[numerics](mlxboost-native-phase0-results/native-validated/numerics/),
[capture receipt](mlxboost-native-phase0-results/native-validated/capture-receipt.json).

## G03 — affine Q4/Q8 GEMV

Representative real packed checkpoint weights, identical random activations:
`x[1,K]`, weights `[N,K/(32/bits)]` uint32, scales/biases `[N,K/64]`, FP16 input
and output, group **64**, B1. Fifty randomized warm windows/backend after five
warmups; **16 distinct evaluated calls/window**, latency divided by 16.
These amortized values must not be compared directly with single-call AXPBY/MoE.

| Kernel / role | K×N | Stock µs/call | Native µs/call | Speedup | Max abs error | Shared peak MLX B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4 llama-q4 q_proj | 2048×2048 | 61.845 | 134.120 | 0.4611× | 9.53674316e-07 | 695,513,872 |
| Q4 llama-q4 lm_head | 2048×128256 | 1395.995 | 1636.220 | 0.8532× | 3.05175781e-05 | 704,029,456 |
| Q8 qwen25-q8 q_proj | 896×896 | 22.047 | 29.389 | 0.7502× | 0 | 527,469,840 |
| Q8 qwen25-q8 lm_head | 896×151936 | 1353.178 | 1355.573 | 0.9982× | 0 | 537,683,472 |

Memory columns are **shared process peaks**, equal in stock/native records,
including the model and window outputs—not kernel workspace. Weights stay packed.
Native execution: **912 C++ `eval_gpu` encodings per row** = 57×16 calls including
correctness, first call, warmups and timed work; **800 are timed**. Counters are
read after synchronization. Unused lazy graphs count zero.

Correctness coverage: **105 G03 + 123 Phase 0 = 228 tests**, including independent
Q4/Q8, FP16/BF16, groups 32/64/128, tails, layout/stream rejection, fallback and
restoration. FP16 `(atol,rtol)=(0.01,0.01)` and BF16 `(0.05,0.01)` are unchanged.
These errors describe the named operator scopes, not model parity.
The original **171-candidate** campaign passed isolated parity but **0/171** met
the ≥10% operator target; all eight admitted model scopes failed qualification.
[Complete qualification and original matrix](mlxboost-g03-final-qualification.md),
[amortized follow-up and raw links](mlxboost-g03-root-cause.md).

## Fused MoE combine — corrected prototype

Synthetic LFM2-style `[1,4,2048]` expert outputs and `[1,4]` weights, FP16.
Multiply **and addition** round to activation dtype. Routing, expert order and
weights remain stock. Three interleaved paired trials after three warmups:

| Stock latency | Native latency | Ratio of median latencies | Paired speedups | Extra peak allocation: stock → native |
| ---: | ---: | ---: | --- | ---: |
| 0.536458 ms | 0.540542 ms | 0.9924× | 0.98115×, 0.96578×, 1.08677× | 20,480 → 4,100 B |

Corrected pilot: **bit-identical, max error 0**. **17 focused tests** cover
FP16/BF16, cancellation, tails, layouts, stock routing, fallback and lifecycle.
Native execution: **9 GPU-written markers** (3 timed, 4 first/warmup,
1 correctness, 1 memory probe), read outside timing; marker cost 4 B.
The warmed memory probe starts at 131,152 active B; total peaks are
151,632 / 135,252 B; output 4,096 B. This is operator allocation evidence only.

The initial FP32-accumulation attempt failed two cancellation tests; its timing
is excluded. No tuning followed the corrected negative pilot. No usable MoE
checkpoint was found; this prototype has no product-wheel or model registration.
[Port, focused tests and immutable evidence](mlxboost-moe-port.md).

## Model measurements — separate scopes

Phase 0's profiler ran **reference-only** Qwen2.5-0.5B Q8, 128 input / 16 output
tokens, one warmup and three measured requests: TTFT **49.604 ms**,
prefill **2613.894 tokens/s** (derived from 128/prefill seconds),
decode **126.335 tokens/s**, peak MLX **679,129,024 B**.
`native_model_execution=false`: these are profiler checks, **not AXPBY acceleration**.
[Original model record](mlxboost-native-phase0-results/native-validated/model-reference/).

G03 follow-up: same pinned Qwen2.5-0.5B Q8 and Llama-3.2-1B Q4, FP16 activations,
group 64, 128/4096 input tokens, **128 actual output tokens**, EOS respected, five
paired repeats. Values are **stock → forced G03**, per-column medians; memory is
maximum recorded MLX peak. Reference native-G03 counts are zero.

| Model | Input | TTFT ms | Prefill tokens/s | Decode tokens/s | Peak MLX B | Native calls/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen25-q8 | 128 | 65.830 → 57.649 | 1965.594 → 2243.304 | 131.226 → 108.443 | 679,129,024 → 679,129,024 | 21,463 |
| qwen25-q8 | 4096 | 1719.097 → 1725.769 | 2392.222 → 2380.143 | 114.864 → 105.946 | 2,365,501,224 → 2,365,501,224 | 21,463 |
| llama-q4 | 128 | 116.645 → 115.548 | 1106.017 → 1113.725 | 109.660 → 93.817 | 950,900,936 → 950,900,936 | 14,351 |
| llama-q4 | 4096 | 4273.052 → 4246.649 | 960.565 → 965.733 | 90.745 → 78.754 | 2,320,537,616 → 2,320,537,616 | 14,351 |

Forced-native decode was slower in **every paired repeat**. Greedy token IDs
agree in these diagnostics, while original logits/state gates remain failed.
Filtered G03 selected **zero layers**; filtered timing cannot establish native
acceleration. The independent stock cached/fresh gate remains **OPEN** (maximum
logit error **0.03515625**). No 32k model result was admitted; no tolerance or
baseline was changed. MoE model TTFT/prefill/decode/memory: **NOT RUN—MODEL BLOCKED**.
[Model diagnostics](mlxboost-g03-root-cause.md#longer-generation-diagnostic),
[reference parity](mlxboost-reference-parity-investigation.md).

## Sources, licenses and evidence

MLX-One remains Apache-2.0. Phase 0's bridge/demo patterns were adapted from
[Apple MLX v0.32.2 extensions](https://github.com/ml-explore/mlx/tree/1f8e74e3f12f31365464a6867c6579f0e9b29d85/examples/extensions);
G03's packed-format/arithmetic references are
[quantized.h](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h)
and quantized.metal at that commit. Preserve the
[Apple MIT copyright/permission text](../../src/mlx_one/mlxboost/native/MLX_LICENSE.txt),
[Phase 0 notice](../../src/mlx_one/mlxboost/native/NOTICE.txt) and
[G03 notice](../../src/mlx_one/mlxboost/g03/NOTICE.txt).

Portions of the MoE combine prototype were adapted from **ZMLX**, by Hunter Bown
and contributors, commit `d8cd0d88d4299ca6821d706f9d5b4a3520d688f5`,
[`src/zmlx/kernels/moe.py`](https://github.com/Hmbown/ZMLX/blob/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5/src/zmlx/kernels/moe.py#L1172).
Changes cover K4/B1 eligibility, strides/tails, materialization markers and safe
fallback. The [full MIT notice](../../benchmarks/moe_port/NOTICE.txt) includes
**Copyright (c) 2026 Hunter Bown**, permission and warranty terms; retain it when
redistributing the prototype. Phase 0/G03 studied ZMLX without copying its code.
No new runtime dependency or licensing/package change is made by this scorecard.

[Machine-readable scorecard](mlxboost-performance-scorecard.json) records exact
unrounded values, source selectors and original SHA256 digests. Historical
evidence references and records are unchanged.

### Evidence availability

The publication commit includes this scorecard, its JSON, the linked reports,
Phase 0 evidence records and the Apple MLX and ZMLX notices.

G03 and MoE raw benchmark artifacts under `local/mlxboost-g03/`,
`local/mlxboost-upstream/` and `local/mlxboost-moe/` are ignored and **local-only**.
The published reports retain their paths, and the companion JSON preserves
selected measurements and evidence hashes. The Phase 0 capture receipt is
included as documentation. The capture itself and local build artifacts remain
local-only.

Documentation checks only; **no benchmark, GPU, wheel or test reruns**.
