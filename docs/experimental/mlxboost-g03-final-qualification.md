# MLX-Boost G03 final qualification

**Verdict: UNQUALIFIED — NOT QUALIFIED for correctness and performance.** No model acceleration is established on this M4 / 32 GiB host. All new kernels remain opt-in and default-off; no runtime manifest was approved. Phase 0 is preserved, with its 123 tests passing again. G01/G02 are design-complete and **NOT IMPLEMENTED / NOT ACTIVATED**.

Run: `qualification-20261115-1794761640668291000`. MLX-One HEAD `4cad1634562c5c12933882deb0b8ab3c9461dd97` with an explicitly dirty worktree; the commit alone does not identify the new code. Pinned MLX and mlx-metal 0.32.2, Python 3.12.11, macOS 15.8.2, Apple M4 / applegpu_g16g, Xcode 26.3, SDK 26.2. Both `metal` and `metallib` were located and native compilation succeeded. Compiler/SDK details and initial state are in audit.json (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/audit.json`). GPU approval was restored; no current approval/environment blocker remains.

Evidence: machine-readable verdict (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/summary.json`), SHA256 manifest (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/manifest.sha256`), executed source hashes (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/final/executed-source-hashes.json`), exact campaign commands (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/final/commands.jsonl`), wheel commands (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/distribution-commands.jsonl`). The isolated raw directory contains source/binary snapshots, environment metadata, checkpoint hashes, immutable result JSON, raw timing samples, token IDs and per-layer inventories. The earlier handoff and pilot remain historical evidence; their approval blockage and zero-layer timings are not current qualification.

| Gate | Result |
| --- | --- |
| A0 | FAIL — unresolved reference cached/fresh parity |
| A1 | PASS — 228 current GPU/backend-free tests |
| A2 | NOT QUALIFIED — performance: 0/171 candidates met target |
| A3 | NOT QUALIFIED — correctness and performance in all 8 admitted scopes |
| A4 | PASS — independent wheels and no new regression identities; 27 known failures remain |
| A5 | UNQUALIFIED; default-off; no runtime approval |

## Current correctness and regression results

| Execution | Passed | Failed | Skipped | Evidence |
| --- | ---: | ---: | ---: | --- |
| Current source GPU + backend-free suites | 228 | 0 | 0 | raw log (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/final/correctness.log`) |
| Installed native wheel | 228 | 0 | 0 | raw log (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheel-tests-native.log`) |
| Installed standard wheel | 106 | 0 | 122 | raw log (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheel-tests-standard-final.log`) |
| Full repository | 2496 | 27 | 84 | raw log (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/full-regressions.log`) |
| Preserved baseline | 2268 | 27 | 84 | raw log (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/mlxboost-phase0-baseline-full-fixed.log`) |

The full and baseline runs each also report 9 passing subtests and 2 warnings. The 27 failure identities are identical: 26 existing missing native_tasks checkpoint-asset failures and the existing README summary assertion. No model downloads or unrelated dependency upgrades were used to conceal them. See identity comparison (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/regression-comparison.json`). The full suite is not all green.

Native/source acceptance includes independent Q4/Q8, FP16/BF16, groups 32/64/128, tail widths, streams, unchanged packed inputs, strict/fallback and registry checks, lazy-unused graph counters, output-only memory bounds, and restoration after exceptions. Tests retain the existing FP16 `(atol,rtol)=(0.01,0.01)` and BF16 `(0.05,0.01)` policies. No numerical policy or existing test was weakened. A1 contains 123 Phase 0 and 105 G03 tests.

The first standard-wheel suite invocation erroneously enabled G03 native-only tests: 106 passed, 58 skipped, 64 setup errors because the extension was absent. That failed attempt (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheel-tests-standard.log`) is retained. The corrected invocation explicitly disables only native-only execution, yielding 106 passed/122 skipped; standard fallback is independently executed for all 12 Q4/Q8 dtype/group combinations in the smoke record (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/g03-smoke-standard.json`). The native wheel ran all 228 tests without skips.

## A0: reference cached versus fresh investigation

Status: **FAIL — unresolved reference parity; model promotion blocked.** The stock Qwen2.5-0.5B Q8 checkpoint reproduces the pilot failure with G03 disabled, the same 128-token synthetic prompt, FP16 storage, greedy forced token, fresh versus incremental cache, and unchanged tolerance. Max final-logit absolute error is **0.03515625** (mean 0.0053917840123176575); `mx.allclose(atol=.01,rtol=.01)` fails. The oracle rerun reproduces the same result.

The first bitwise divergence is `model.layers.0.self_attn.q_proj`, before RoPE or cache update: last-token normalized inputs are exactly equal, but stock matrix versus vector projections differ by 0.0078125. Replaying the same normalized vector through the same stock Q/K/V layers exactly reproduces their cached outputs. First tolerance failure in the leaf trace occurs at layer 3 `post_attention_layernorm` (max 0.02294921875). The instrumented fresh result exactly equals the ordinary uninstrumented fresh result. Cache offsets are 128→129 and fresh 129; recorded shapes, causal versus single-token masks and RoPE positions are consistent.

The offline FP32 dequantized small-projection oracle distinguishes rounding: Q-vector versus FP32 oracle max 6.103515625e-05, Q-matrix versus oracle 0.0078125; K-vector versus oracle 0, K-matrix versus oracle 0.0078125. Pinned MLX `quantized.h` has different vector-dot versus matrix tile dequantization/reduction contracts (`qdot`192–289, `dequantize`486ff, `QuantizedBlockLoader`600ff, `qmm_t_impl`1200ff). This establishes a stock shape-dependent projection arithmetic contributor. It does not prove the only cause of accumulated model drift or identify a conclusive MLX-One indexing bug. No safe local cache fix was established.

Diagnostic dequantization is confined to an offline small-projection oracle; it is absent from inference and G03. No reference precision, cache API, reduction policy or baseline route was changed to make the assertion pass. Every final model/context reference control also fails the cached/fresh gate. Evidence: before trace (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/parity-before/55be50e6da46a36cc87bf50d1be3a2a6673f9f721be12fd0b7d0c61e737384ac.json`), oracle replay (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/parity-oracles/64fc8b0355bbbf8e3bd9cbdc933b100e0cee4e5ac122116d57d798c9de88b0e2.json`). Reproduce with `MLX_ONE_G03=0 MLX_ONE_NATIVE=0 .venv/bin/python benchmarks/g03/reference_parity.py --checkpoint-record local/mlxboost-g03/qwen25-q8-path.json --output <new-directory>`.

Next correctness step: reduce the identical-input MLX 0.32.2 matrix/vector rounding divergence to a minimal upstream reproducer and determine a reference-preserving remedy before rerunning the unchanged cached/fresh policy. A precision or policy change needs separate justification; none is adopted here.

## Measurement method and native dispatch proof

Four-output-row SIMD variant, 32 lanes/SIMD, 2/4/8 SIMDs per 64/128/256-thread group, affine MLX uint32 packing, FP32 accumulation, activation-dtype Q4 bias-correction rounding, no dense dequantized weight. Metal uses `-fno-fast-math`; C++ additionally uses `-ffp-contract=off`. Actual compiler rules (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/compiler-rules.json`) are retained. The benchmark and installed native wheel have implementation hash `3a59968a495a9e47a41b29693e2a5d55f8cd96395303910ec7b0462e19b7f19a` and combined executable-asset hash `3e97452fc583c5441531daa1c233259efcb95dc0be06acfe3aa3d838337ab212`.

Latency is **synchronized operator wall time**, including Python/C++ dispatch, evaluation, counter bookkeeping and completed GPU work; it is not an isolated GPU-command duration. Inputs/packed weights/scale/bias are identical to `mx.quantized_matmul`. Each geometry uses seeded randomized/interleaved backends and 50 warm samples per configuration after warmup. Raw samples, stdev/IQR/p90 and shared profiling-window MLX peak allocation are in JSON. CSV adds a descriptive paired-round median-ratio percentile bootstrap 95% interval (1000 resamples, seed 7); this does not replace or relax the existing target rule. Cold first profiled calls are distinguished from true fresh-process model startup. OS caches, clocks, power and thermals are uncontrolled.

A C++ request/layer-owned atomic counter increments only after Metal dispatch is encoded in `eval_gpu`. Reports are read after `mx.eval` and synchronization. An unused lazy graph counts 0; materializing once counts 1; evaluating the already materialized output again does not increment. Synthetic candidates each record 62 actual encodings (correctness+cold+10 warmup + 50 samples); real-weight candidates record 57 (correctness+cold+5 warmup + 50 samples). Layer inventories are separate untimed instrumented generations and cannot contaminate reported timed samples. Stock counts are executed Python operation counts, not asserted Metal launch counts. Reference models remain unpatched; selected scoped adapter objects and arrays are restored.

## Kernel measurements

There are 36 synthetic geometries and 21 model/geometry scopes, each at 3 threadgroup configurations: **171 measured candidates, all pass isolated numerical parity; zero meet the >=10% target**. Synthetic observed speedup range 0.845477–1.002674×; real-weight range 0.831803–0.991148×. All geometries remain unselected. Tables show the fastest measured configuration per geometry; CSV with every configuration, CI, error and evidence path (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/kernel-measurements.csv`) retains all 171 rows. Times below are microseconds of synchronized wall time.

### Synthetic Q4/Q8 matrix

| Model | Bits / dtype / group | K × N | Threads | MLX µs | Native µs | Speedup | >=1.10× |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic | 4/bfloat16/128 | 128 × 3 | 128 | 204.791 | 205.667 | 0.9957× | FAIL |
| synthetic | 4/bfloat16/128 | 4096 × 129 | 128 | 223.146 | 254.667 | 0.8762× | FAIL |
| synthetic | 4/bfloat16/128 | 896 × 128 | 64 | 217.833 | 227.395 | 0.9579× | FAIL |
| synthetic | 4/bfloat16/32 | 128 × 3 | 128 | 205.834 | 210.583 | 0.9774× | FAIL |
| synthetic | 4/bfloat16/32 | 4096 × 129 | 128 | 219.062 | 251.979 | 0.8694× | FAIL |
| synthetic | 4/bfloat16/32 | 896 × 128 | 64 | 216.854 | 220.875 | 0.9818× | FAIL |
| synthetic | 4/bfloat16/64 | 128 × 3 | 256 | 195.958 | 199.103 | 0.9842× | FAIL |
| synthetic | 4/bfloat16/64 | 4096 × 129 | 128 | 222.625 | 256.167 | 0.8691× | FAIL |
| synthetic | 4/bfloat16/64 | 896 × 128 | 128 | 215.520 | 221.876 | 0.9714× | FAIL |
| synthetic | 4/float16/128 | 128 × 3 | 256 | 203.292 | 205.708 | 0.9883× | FAIL |
| synthetic | 4/float16/128 | 4096 × 129 | 256 | 218.458 | 253.625 | 0.8613× | FAIL |
| synthetic | 4/float16/128 | 896 × 128 | 128 | 207.750 | 211.834 | 0.9807× | FAIL |
| synthetic | 4/float16/32 | 128 × 3 | 64 | 178.271 | 180.355 | 0.9884× | FAIL |
| synthetic | 4/float16/32 | 4096 × 129 | 128 | 208.791 | 246.000 | 0.8487× | FAIL |
| synthetic | 4/float16/32 | 896 × 128 | 64 | 193.209 | 200.646 | 0.9629× | FAIL |
| synthetic | 4/float16/64 | 128 × 3 | 256 | 201.125 | 201.375 | 0.9988× | FAIL |
| synthetic | 4/float16/64 | 4096 × 129 | 256 | 219.396 | 247.062 | 0.8880× | FAIL |
| synthetic | 4/float16/64 | 896 × 128 | 256 | 200.750 | 206.375 | 0.9727× | FAIL |
| synthetic | 8/bfloat16/128 | 128 × 3 | 256 | 210.958 | 210.396 | 1.0027× | FAIL |
| synthetic | 8/bfloat16/128 | 4096 × 129 | 64 | 231.062 | 266.771 | 0.8661× | FAIL |
| synthetic | 8/bfloat16/128 | 896 × 128 | 64 | 218.313 | 222.376 | 0.9817× | FAIL |
| synthetic | 8/bfloat16/32 | 128 × 3 | 64 | 206.417 | 208.375 | 0.9906× | FAIL |
| synthetic | 8/bfloat16/32 | 4096 × 129 | 256 | 205.292 | 240.271 | 0.8544× | FAIL |
| synthetic | 8/bfloat16/32 | 896 × 128 | 128 | 213.229 | 219.562 | 0.9712× | FAIL |
| synthetic | 8/bfloat16/64 | 128 × 3 | 128 | 181.834 | 183.979 | 0.9883× | FAIL |
| synthetic | 8/bfloat16/64 | 4096 × 129 | 256 | 234.459 | 270.605 | 0.8664× | FAIL |
| synthetic | 8/bfloat16/64 | 896 × 128 | 128 | 215.020 | 220.125 | 0.9768× | FAIL |
| synthetic | 8/float16/128 | 128 × 3 | 64 | 209.958 | 210.000 | 0.9998× | FAIL |
| synthetic | 8/float16/128 | 4096 × 129 | 256 | 228.833 | 266.667 | 0.8581× | FAIL |
| synthetic | 8/float16/128 | 896 × 128 | 64 | 215.521 | 220.938 | 0.9755× | FAIL |
| synthetic | 8/float16/32 | 128 × 3 | 64 | 207.625 | 207.792 | 0.9992× | FAIL |
| synthetic | 8/float16/32 | 4096 × 129 | 64 | 232.083 | 268.021 | 0.8659× | FAIL |
| synthetic | 8/float16/32 | 896 × 128 | 128 | 219.292 | 224.604 | 0.9763× | FAIL |
| synthetic | 8/float16/64 | 128 × 3 | 128 | 211.520 | 213.605 | 0.9902× | FAIL |
| synthetic | 8/float16/64 | 4096 × 129 | 64 | 230.896 | 268.646 | 0.8595× | FAIL |
| synthetic | 8/float16/64 | 896 × 128 | 64 | 221.562 | 224.584 | 0.9865× | FAIL |
### Real packed checkpoint weights

| Model | Bits / dtype / group | K × N | Threads | MLX µs | Native µs | Speedup | >=1.10× |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3-q8 | 8/bfloat16/64 | 1024 × 1024 | 64 | 203.250 | 228.542 | 0.8893× | FAIL |
| qwen3-q8 | 8/bfloat16/64 | 1024 × 151936 | 128 | 1838.271 | 1854.687 | 0.9911× | FAIL |
| qwen3-q8 | 8/bfloat16/64 | 1024 × 2048 | 64 | 252.375 | 261.750 | 0.9642× | FAIL |
| qwen3-q8 | 8/bfloat16/64 | 1024 × 3072 | 128 | 245.542 | 266.479 | 0.9214× | FAIL |
| qwen3-q8 | 8/bfloat16/64 | 2048 × 1024 | 64 | 231.333 | 241.375 | 0.9584× | FAIL |
| qwen3-q8 | 8/bfloat16/64 | 3072 × 1024 | 64 | 239.229 | 253.688 | 0.9430× | FAIL |
| llama-q4 | 4/float16/64 | 2048 × 128256 | 128 | 1702.458 | 1932.729 | 0.8809× | FAIL |
| llama-q4 | 4/float16/64 | 2048 × 2048 | 64 | 245.812 | 261.020 | 0.9417× | FAIL |
| llama-q4 | 4/float16/64 | 2048 × 512 | 64 | 208.229 | 227.875 | 0.9138× | FAIL |
| llama-q4 | 4/float16/64 | 2048 × 8192 | 128 | 362.146 | 402.104 | 0.9006× | FAIL |
| llama-q4 | 4/float16/64 | 8192 × 2048 | 64 | 354.333 | 388.000 | 0.9132× | FAIL |
| qwen25-q8 | 8/float16/64 | 4864 × 896 | 64 | 283.292 | 301.729 | 0.9389× | FAIL |
| qwen25-q8 | 8/float16/64 | 896 × 128 | 128 | 191.626 | 199.145 | 0.9622× | FAIL |
| qwen25-q8 | 8/float16/64 | 896 × 151936 | 128 | 1649.917 | 1697.500 | 0.9720× | FAIL |
| qwen25-q8 | 8/float16/64 | 896 × 4864 | 64 | 289.354 | 300.500 | 0.9629× | FAIL |
| qwen25-q8 | 8/float16/64 | 896 × 896 | 64 | 208.187 | 219.854 | 0.9469× | FAIL |
| qwen25-15-q4 | 4/float16/64 | 1536 × 151936 | 64 | 1596.438 | 1912.000 | 0.8350× | FAIL |
| qwen25-15-q4 | 4/float16/64 | 1536 × 1536 | 64 | 251.041 | 260.250 | 0.9646× | FAIL |
| qwen25-15-q4 | 4/float16/64 | 1536 × 256 | 64 | 192.479 | 199.792 | 0.9634× | FAIL |
| qwen25-15-q4 | 4/float16/64 | 1536 × 8960 | 256 | 327.979 | 357.459 | 0.9175× | FAIL |
| qwen25-15-q4 | 4/float16/64 | 8960 × 1536 | 128 | 334.105 | 371.250 | 0.8999× | FAIL |

The initial one-output-row Q8 prototype is compared separately in the historical variant table (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/historical-variant-comparison.csv`). Its five original model geometry results and source/binary fingerprints are retained. The original compiled binary/source variant was not preserved for a contemporary rerun. This is an explicitly **historical, non-paired comparison**, not proof that SIMD reuse caused a timing change; no original-variant rerun is claimed.

## End-to-end results

R=unpatched stock reference; S=performance-filtered scoped experiment; U=unfiltered forced-native diagnostic. All S plans select **0 geometries / 0 layers / 0 native dispatches**, so S timing differences are reference-path variation. U actually executes custom kernels but fails model logits/state parity. Every recorded 16-token greedy output agrees with R; token agreement does not waive failed numerical/state controls. Five seeded randomized repetitions per backend/context, 1 warmup, exact 128/4096-token prompt roundtrip, EOS respected. Per-backend first generation after offline search is labeled as such, not process-cold. Per-repeat ratios, exact output IDs/hashes and layer eligibility/dispatch/stock fallback counts remain in the JSON and all 120 timed rows (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/model-measurements.csv`).

| Model | Input | Backend | Decode t/s median | Prefill t/s median | TTFT ms median | Peak MLX GiB max | Native dispatches / run |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| qwen25-q8 | 128 | R | 133.194 | 2604.758 | 49.698 | 0.632 | 0 |
| qwen25-q8 | 128 | S | 136.712 | 2599.684 | 49.909 | 0.632 | 0 |
| qwen25-q8 | 128 | U | 124.655 | 2599.189 | 49.838 | 0.632 | 2535 |
| qwen25-q8 | 4096 | R | 113.122 | 2071.954 | 1986.476 | 2.203 | 0 |
| qwen25-q8 | 4096 | S | 113.874 | 2076.614 | 1977.744 | 2.203 | 0 |
| qwen25-q8 | 4096 | U | 101.986 | 2090.639 | 1964.122 | 2.203 | 2535 |
| qwen3-q8 | 128 | R | 105.495 | 2068.614 | 62.469 | 0.797 | 0 |
| qwen3-q8 | 128 | S | 109.619 | 2086.568 | 62.042 | 0.797 | 0 |
| qwen3-q8 | 128 | U | 98.980 | 2069.497 | 62.454 | 0.797 | 2955 |
| qwen3-q8 | 4096 | R | 37.931 | 1277.635 | 3214.719 | 2.757 | 0 |
| qwen3-q8 | 4096 | S | 36.659 | 1266.258 | 3239.778 | 2.757 | 0 |
| qwen3-q8 | 4096 | U | 35.800 | 1282.609 | 3198.691 | 2.757 | 2955 |
| llama-q4 | 128 | R | 102.207 | 885.654 | 145.204 | 0.886 | 0 |
| llama-q4 | 128 | S | 105.899 | 775.743 | 165.619 | 0.886 | 0 |
| llama-q4 | 128 | U | 78.794 | 914.120 | 140.731 | 0.886 | 1695 |
| llama-q4 | 4096 | R | 78.367 | 682.782 | 6014.350 | 2.161 | 0 |
| llama-q4 | 4096 | S | 76.448 | 592.279 | 6921.421 | 2.161 | 0 |
| llama-q4 | 4096 | U | 53.533 | 634.512 | 6461.468 | 2.161 | 1695 |
| qwen25-15-q4 | 128 | R | 65.558 | 544.403 | 236.477 | 1.034 | 0 |
| qwen25-15-q4 | 128 | S | 70.248 | 556.713 | 230.882 | 1.034 | 0 |
| qwen25-15-q4 | 128 | U | 48.240 | 636.558 | 201.863 | 1.034 | 2955 |
| qwen25-15-q4 | 4096 | R | 63.521 | 541.443 | 7573.574 | 2.545 | 0 |
| qwen25-15-q4 | 4096 | S | 63.434 | 530.538 | 7727.531 | 2.545 | 0 |
| qwen25-15-q4 | 4096 | U | 46.909 | 534.216 | 7673.384 | 2.545 | 2955 |

| Scope | R cached/fresh | S logits | U logits / state | Greedy IDs | Each-repeat >=5% + actual S dispatch | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| qwen25-q8 / 128 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| qwen25-q8 / 4096 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| qwen3-q8 / 128 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| qwen3-q8 / 4096 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| llama-q4 / 128 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| llama-q4 / 4096 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| qwen25-15-q4 / 128 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |
| qwen25-15-q4 / 4096 | FAIL | PASS | FAIL / FAIL | PASS | FAIL | NOT QUALIFIED |

Every admitted scope fails the per-repeat performance rule. Even if one S ratio exceeds 1.05, its native count is 0 and cannot support acceleration. U generally regresses, and no scope has a reproducible >=5% native decode improvement. The JSON retains S/U per-repeat ratios and memory checks; all S memory-growth checks pass. Peak MLX allocation is process-global per request and reported separately from process high-water RSS. OS-wide peak unified memory/physical footprint was not measured; these counters must not be described as total system memory.

### Context admission

The same shared admission function is used for warm/cold runs. It reserves full FP32-logit lower bound `context*vocab*4 + resident_MLX_bytes` under 8 GiB and requires `context+16 <= model_context_length`. Actual measured requests are also checked against the budget. No prompt is shortened or model swapped. All 128/4096 cases were admitted. Rejected cases:

| Model | Input | Reason | Estimated lower bound bytes |
| --- | ---: | --- | ---: |
| qwen25-q8 | 32768 | model_context_limit | 20441902864 |
| qwen3-q8 | 32768 | 8GiB_working_budget | 20547928080 |
| llama-q4 | 32768 | 8GiB_working_budget | 17506015248 |
| qwen25-15-q4 | 32768 | model_context_limit | 20786113552 |

SmolLM2 Q4 is skipped: the requested Q4 repository was unavailable; the unsuffixed repository was unquantized and is not substituted. Only the four cached immutable snapshots were used. Their revisions and file hashes are in each model record and the initial audit. No performance is inferred for physical 8/16/24 GiB devices.

### Fresh-process startup

Each row is a separate model process with no offline kernel search or parity work before its first request. Context 128 is admitted for every model. Driver/OS cache state is uncontrolled; this is process-cold, not a guaranteed cold filesystem/driver. Warm medians are in the preceding table. Cold raw table (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/cold-measurements.csv`).

| Model | Backend | Load ms | First TTFT ms | Decode t/s | Peak MLX GiB | Native count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen25-q8 | g03_unfiltered | 211.228 | 74.181 | 121.521 | 0.632 | 2535 |
| qwen25-q8 | mlx_reference | 386.201 | 64.193 | 135.085 | 0.632 | 0 |
| qwen25-q8 | g03_custom | 216.112 | 67.482 | 135.396 | 0.632 | 0 |
| qwen3-q8 | g03_unfiltered | 253.038 | 81.361 | 90.130 | 0.797 | 2955 |
| qwen3-q8 | mlx_reference | 598.949 | 92.245 | 104.905 | 0.797 | 0 |
| qwen3-q8 | g03_custom | 259.066 | 81.407 | 105.986 | 0.797 | 0 |
| llama-q4 | g03_unfiltered | 358.039 | 142.710 | 84.130 | 0.816 | 1695 |
| llama-q4 | mlx_reference | 704.675 | 138.324 | 113.485 | 0.816 | 0 |
| llama-q4 | g03_custom | 352.890 | 138.518 | 119.139 | 0.816 | 0 |
| qwen25-15-q4 | g03_unfiltered | 283.422 | 175.856 | 59.574 | 1.034 | 2955 |
| qwen25-15-q4 | mlx_reference | 586.483 | 170.141 | 80.072 | 1.034 | 0 |
| qwen25-15-q4 | g03_custom | 288.284 | 171.579 | 82.246 | 1.034 | 0 |

## Independent distribution acceptance

Both wheels were built from this tree, installed into distinct newly created venvs with exact existing dependency constraints and MLX/Metal 0.32.2, then executed from `/private/tmp`. No editable source path was used; smoke asserts imported package resides under its fresh prefix. `pip check` passes. Both environments lack MLX-LM/Transformers. Python Engine generation, real CLI generation, in-process HTTP completions and SSE agreement/cleanup pass on the pinned Qwen2.5 Q8 model with the unchanged reference_dense path.

- **native**: mlx_one-0.1.2-cp312-cp312-macosx_15_0_arm64.whl (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheels/native/mlx_one-0.1.2-cp312-cp312-macosx_15_0_arm64.whl`); SHA256 `fa75ee83fd42a8921954b5b7b32580713ec6801a737d4852dc256f94702ad937`. G03/API smoke (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/g03-smoke-native.json`); Phase 0 smoke (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/phase0-smoke-native.json`).
- **standard**: mlx_one-0.1.2-py3-none-any.whl (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheels/standard/mlx_one-0.1.2-py3-none-any.whl`); SHA256 `743ad0dc2e86d89ea0cf3812cfd58261f7aed4f18183b019e13b2105cf8e9ae9`. G03/API smoke (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/g03-smoke-standard.json`); Phase 0 smoke (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/phase0-smoke-standard.json`).

Native wheel: arm64 extension + operation dylib + Metal library + build identity; 12 installed Q4/Q8 dtype/group executions each count 1 materialized native dispatch and pass numerical checks. Runtime works with compiler PATH absent. Removing its metallib in the isolated venv and using a fresh subprocess produces correct MLX fallback plus typed strict failure; the asset is restored in `finally`. Standard wheel: no `.so`, `.dylib`, `.metallib` or required compiled backend; default-off, missing-extension fallback and strict failure pass. Archive audit (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/wheel-archive-audit.json`) and otool/lipo relocation audit (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/relocation-audit.json`) show relative loader rpaths, system/MLX dependencies and no build/worktree library dependency. The inherited MLX imported-target warning about missing Xcode 26.6 Framework search path did not become an installed runtime dependency.

## Source changes and preservation

This qualification adds request-owned C++ dispatch traces (`native/_build/gemv.h/.cpp`, `bindings.cpp`), per-layer adapter reports, a registry gate requiring positive materialized native evidence, autotuning trace evidence, meaningful lazy/cancellation/memory/admission tests, stock-reference replay diagnostics, shared context admission, serial campaign/distribution runners and CSV/report generation. Kernel math, model-family source, inference APIs, original dispatch/reference paths, numerical policies and default flags remain unchanged during this qualification. Existing README/logo bytes are preserved. All work remains uncommitted; no publish/merge occurred. Packaging and the Q4/Q8 implementation from the prior milestone remain intact. Exact source changes (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/source-changes.json`) and frozen sources/binaries (local-only: `local/mlxboost-g03/qualification-20261115-1794761640668291000/final/source-snapshot`) identify actual worktree bytes; generated local model/evidence caches remain ignored.

Reproduce with new output directories (existing artifacts are retained):

```bash
MLXBOOST_G03_GPU_TESTS=1 MLXBOOST_PHASE0_GPU_TESTS=1 .venv/bin/python -m pytest tests/g03 tests/native -q --no-cov
.venv/bin/python benchmarks/g03/run_campaigns.py --output <new-run>/final
.venv/bin/python benchmarks/g03/report.py --run <new-run>
# Supply exact existing dependency constraints before distribution.py:
.venv/bin/python benchmarks/g03/distribution.py --run <new-run>
# Optional native build (normal build defaults to standard/Python-only):
MLX_ONE_BUILD_NATIVE=1 .venv/bin/python -m build --wheel --no-isolation --outdir <new-native-dist>
MLX_ONE_BUILD_NATIVE=0 .venv/bin/python -m build --wheel --no-isolation --outdir <new-standard-dist>
```

GPU commands require the normal execution approval. The experimental typed config/plan is diagnostic only and cannot approve itself. Runtime `MLX_ONE_G03=1` still requires an exact reviewed registry hit; absent evidence uses `mx.quantized_matmul` (strict mode raises the typed reason). Do not deploy these forced plans.

## G01/G02 design-only result

The [combined study](mlxboost-g01-g02-design-study.md), [source appendix](mlxboost-g01-g02-source-appendix.md), [G01 detailed design](mlxboost-g01-design.md) and [G02 detailed design](mlxboost-g02-design.md) are complete. Each companion map has 42 verified source hashes/spans. They pin MLX v0.32.2 and ZMLX `d8cd0d88d4299ca6821d706f9d5b4a3520d688f5`, inspect actual local family equations, record MIT reuse constraints and give per-family risks and future tensor/model/performance matrices.

G01 requires two outputs and preservation of residual-sum rounding before RMS; inspected ZMLX fused alternatives have incompatible cast points. G02 must compare against the existing compiled standard SwiGLU graph; FP32-precise mode, Gemma GELU and GPT-OSS clipped/alpha/+1 activation are separate contracts. Qwen fidelity exclusions remain a HOLD. Conditional later research starts with a narrow Llama contract after reference controls; **no G01/G02 kernel, registration, model patch, feature flag or build change exists**.

The next required action is the A0 upstream quantized-shape arithmetic reproducer and a reference-preserving correctness remedy. Current negative performance evidence does not justify activating G03 or starting G01/G02 production implementation.
