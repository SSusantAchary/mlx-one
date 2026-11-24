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

## Native Inkling-2-Mini text runtime

The performance commands below refer to the local benchmark fixtures and
receipts used for these measurements. Those files are excluded from Git.

The installed Inkling-2-Mini Bare Speed pack loads directly on Metal, including its
NVFP4 experts and DSpark sidecar. The 48 loop visits use eight full BF16 KV banks and
40 bounded local windows. Masked multirow target verification changed the greedy
continuation. The explicit DSpark path now verifies chronologically with the same
one-row target arithmetic as reference decode.

Measured on Apple M4, 32 GiB, macOS 15.8.3, MLX and MLX Metal 0.32.2. Both
requests used the same 1,424 prompt tokens, greedy settings, and 96 generated tokens
under a configured 131,072-token context limit. These are single paired observations.

| Native B1 path | TTFT | Prefill t/s | Decode t/s | Peak MLX | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Reference | 19.62 s | 73.74 | 16.48 | 15.28 GiB | Completed |
| DSpark, depth 3 | 18.85 s | 76.92 | 13.39 | 15.33 GiB | Completed, all 96 token IDs match reference |

Both requests released their cache allocations. The separate 131,072-token admission
probe with DSpark reserved 4.71 GiB for request state and forecast 20.96 GiB against
the 21.33 GiB live ceiling. Full-window prefill and decode throughput remain
unverified. DSpark accepted 51 of 65 drafts in the measured run and remains opt-in.
**Native text execution verified. Model acceleration unqualified.**

Run the verified reference path from the repository virtual environment.

```sh
.venv/bin/mlx-one serve "$HOME/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed" \
  --offline --alias inkling-2-mini --context-length 131072 --parallel 1 \
  --cache-type-k bf16 --cache-type-v bf16 --no-cache-prompt
```

For experimental DSpark, add `--spec-type draft-mtp --spec-draft-n-max 3` and
send `"temperature": 0` in the completion request.

Powered by MTPLX. [MTPLX by Youssof Altoukhi](https://github.com/youssofal/mtplx)

### Inkling P0 one-row convolution

On M4 Air, the local Bare Speed checkpoint with a real 1,024-token prefill and
128 timed greedy decode steps measured **17.00 → 18.06 tok/s (+6.25%)**.
Peak MLX allocation was **14.908 → 14.907 GiB**. The timed legs ran once each
in separate processes. Logits, generated IDs, and retained KV and convolution
state matched bit for bit. The focused Inkling and native MTP tests passed
18 checks. The route is opt-in with `MLX_ONE_INKLING_FUSED_CONV=1` and falls
back to stock operations on unsupported geometry or a failed first-use check.

The kernel adapts `_SOURCE` and residual handling in
`mtplx/models/inkling/short_conv.py` from MTPLX v2.13.0 at commit
`bebe21b5ba36af3688335ab66d365509fd042fad`. The adaptation handles a single
BF16 row and returns a separate FP32 tail. The Apache-2.0 license, NOTICE,
and in-product attribution are retained. [MTPLX by Youssof Altoukhi](https://github.com/youssofal/MTPLX/releases/tag/v2.13.0)

The gain fell below the 10% end-to-end promotion threshold. No long-context
speedup was qualified. In the separate initialized-KV diagnostic, synchronized
global attention per decode step grew from **18.19 ms at 32k** to **67.39 ms
at 128k**. Those timings include synchronization and use populated synthetic
history with real model weights.

A single follow-up with a real **8,192-token prompt / 2,048-token output**
measured **10.35 → 11.24 tok/s (+8.63%)**. Both paths started
from copies of the same prompt state. Its shared TTFT was **111.37 s**.
Decode peak MLX was **15.43 → 15.47 GiB**.
All generated IDs, final KV and convolution state, and 16 sampled logit rows
matched bit for bit. The run order was reference first and fused second.

### Inkling DSpark verification and draft KV

The first difference observed at two target rows was in the layer 0 packed
query projection, with **0.015625** maximum absolute difference after an
identical embedding on a shared 128-token prompt state. Chronological
verification remains in use. The updated greedy path evaluates the draft
and anchor prediction together and reuses reserved storage for full draft
attention when it has room, without advancing the committed draft offset.

One native pair on M4 Air with **1,039 input / 96 output tokens** measured
**16.84 tok/s** for reference versus **15.62 tok/s** for depth-3 DSpark.
Peak MLX was **15.21 / 15.32 GiB**. DSpark accepted **48 of 68** drafts.
All 96 output token IDs matched and both request caches were released.
**21 focused tests passed**, including rejected proposals, cancellation,
committed-history isolation, and the append fallback. DSpark remains explicit
and greedy-only. Batched verification and end-to-end speedup remain unqualified.

### Inkling 64K native attention on the M4 Air

One greedy B1 request per path used the same 65,536 prompt IDs from the local
Bare Speed pack. TTFT excludes model load. The new native run enabled
`MLX_ONE_INKLING_LONG_ATTN=1` and the existing `MLX_ONE_INKLING_FUSED_CONV=1`.

| Path | Output tokens | TTFT | Decode at 3,072 outputs | Decode over run | Peak MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX-One reference, interrupted by user | 3,110 | 2,156.59 s | 7.71 tok/s | 7.71 tok/s | 17.47 GiB |
| MLX-One native optimized, complete | 4,096 | 1,528.26 s | 7.79 tok/s | 7.78 tok/s | 17.36 GiB |
| MTPLX 2.13.0 DSpark depth 3, complete | 4,096 | 637.56 s | 16.89 tok/s | 17.08 tok/s | 18.08 GiB |

Native TTFT fell by **628.33 s (29.1%)**. The native output prefix matched all
3,110 saved reference tokens by SHA-256. Both first-use attention checks matched
the native BF16 output bit for bit. Cache cleanup passed. The 4,096-token
completion hashes differ between engines. A single native pair at the common
3,072-token checkpoint establishes no sustained decode speedup.

Long global visits now read the BF16 banks directly, with a bounded group of query
rows per launch. Prompt tiles and one-row decode keep their respective MLX reduction
orders and sink positions. Unsupported shapes, devices, block overrides and failed
bit checks use the existing attention. The native verifier still processes one
target row at a time. The earlier two-row packed projection mismatch remains open.
The exact Metal reduction adapts `fast/attend.py` and `fast/prefill_attend.py` in
MTPLX 2.13.0. [MTPLX by Youssof Altoukhi](https://github.com/youssofal/mtplx)

### Inkling native TTFT and decode candidates

`MLX_ONE_INKLING_FLASH_PROMPT=1` enables M4 simdgroup prompt attention with
up to 512 rows per prefill chunk on long prompts. It computes Inkling's causal
relative bias and query-dependent sink without materializing the full mask.
The BF16 reduction order differs from the reference. Its finite-output
check confirms dispatch only. Model parity requires the manual comparison.

`MLX_ONE_INKLING_VERIFY_PANEL=1` with greedy `draft-mtp` verifies the
anchor and up to three proposals in one native target forward. It reuses the
native model, KV banks, and DSpark proposal lifecycle. First use per width and
history tier checks logits, draft features, appended KV, and convolution
tails bit for bit against chronological target steps. A mismatch selects
chronological verification. The proposal commits only the consumed prefix.

Both switches default off. MLX-One owns execution and cache state and uses
the installed MLX Metal headers. The SIMD prompt kernel and affine product
adapt MTPLX v2.13.0 `fast/prompt_flash.py`, `fast/affine_lane.py`, and
`fast/qproduct.py`. The bundled Apache-2.0 license and notice apply.

The first combined 64K/4K run completed at **8.84 tok/s** and **1,018.76 s TTFT**
with **18.75 GiB** peak MLX memory. All 978 target panels used the chronological
fallback. Generated IDs differed from the saved native reference.

The revised native verifier uses one-row depth-read and cross-pass reductions.
It qualifies packed projections, expert products, and convolution separately
against the native one-row path. Passed stages share work across draft rows.
Qualified panels consume draft IDs on the GPU and return one packet with the
accepted prefix and predictions. Whole-model verification remains bitwise.

The local script's `panel --smoke` mode uses 1,024 input and 128 output tokens.
It records the first visible mismatch with the token row and visit or output.
Use `panel` for 64K/4K with the exact prefill path. `target` also enables
experimental flash prefill.

The manual M4 Air smoke run completed **1,024 input / 128 output tokens** at
**14.41 tok/s** with **13.94 s TTFT** and **15.49 GiB** peak MLX allocation.
Both tested target panel widths (two and four rows) matched chronological
verification bit for bit. The request used 36 verified panels, accepted 51
of 112 drafts, and released its cache. The subsequent manual 64K/4K run
used 960 verified panels, matched all 4,096 reference IDs, and measured
**9.64 tok/s** with **1,548.92 s TTFT** and **18.09 GiB** peak MLX allocation.

### Inkling fused prefill convolution

`MLX_ONE_INKLING_FUSED_PREFILL_CONV=1` reuses the native Metal convolution
and residual-add kernel for B1 BF16 prefill chunks of 2 to 128 tokens.
The first call per shape and operand form compares outputs and FP32 tails
bit for bit against the existing multirow prefill computation. A failed
check keeps the reference operation. The switch defaults off. The candidate
uses the existing MLX-One kernel and tap order. The prefill fusion approach
was also applied upstream in [oMLX #5538](https://github.com/jundot/omlx/pull/5538).

On the M4, **9 focused convolution tests passed**. Eight separate synthetic
128-row operand forms passed bitwise checks, covering static and dynamic
taps, cold and retained tails, and BF16 and FP32 residual outputs.

One manual cold pair on the M4 Air used identical 4,096-token prompts and
128-token chunks. The stock leg ran first.

| Path | TTFT | Prefill | Peak MLX |
| --- | ---: | ---: | ---: |
| Stock | 53.87 s | 76.03 tok/s | 15.192 GiB |
| Fused | 53.04 s | 77.23 tok/s | 15.068 GiB |

All ten operand forms activated. Final logits, retained KV, convolution
tails, and the first token matched bit for bit. Both requests released
their cache allocations. Fusion saved **0.83 s (1.55%)** in this pair and
**127.6 MiB** of peak MLX allocation. The single fixed-order pair qualifies
parity at 4K. A sustained speedup and long-context performance remain open.

```sh
.venv/bin/python -u local/inkling-p0/measure_prefill_conv.py stock
.venv/bin/python -u local/inkling-p0/measure_prefill_conv.py fused
.venv/bin/python local/inkling-p0/measure_prefill_conv.py compare
```

### Inkling expert output scatter

`MLX_ONE_INKLING_EXPERT_SCATTER=1` uses one vector Metal scatter to put BF16
expert outputs back in token order for native grouped prefill of 33–128
tokens. The existing route selection, NVFP4 grouped matmul, and weighted
sum retain their arithmetic and slot order. A first-use bitwise check
against the inverse sort and gather keeps the reference path available.
The vector scatter is adapted from OpenAI's GPT-OSS Metal operation as
published in the Hugging Face kernel catalog. MLX-One's implementation
writes each routed BF16 row back to its original slot. [Catalog source](https://huggingface.co/kernels-community/gpt-oss-metal-kernels/blob/9e86c1c0b9935387e73b452b6d97f8c3e9251ee0/gptoss_kernels/source/scatter.metal)

On M4, a synthetic **1,024 × 2,304 BF16** operand matched bit for bit.
The median synchronized wall time over three alternating 16-call batches
per path fell from **0.679 ms to 0.538 ms (20.8%)**. Ten focused scatter and
convolution tests passed, including strided input parity.

The manual cold 4K run with 128-token chunks measured **53.808 s TTFT** and
**76.12 tok/s**, compared with **53.873 s** and **76.03 tok/s** in the saved
stock run. Peak MLX was **15.193 GiB** against **15.192 GiB**. The scatter
activated for the measured `(1024, 2304)` form. Final logits, retained KV,
convolution tails, and the first token matched bit for bit. Both runs
released their caches. The measured TTFT difference was 65 ms (0.12%).
End-to-end acceleration remains unqualified.

```sh
.venv/bin/python -u local/inkling-p0/measure_prefill_conv.py scatter
.venv/bin/python local/inkling-p0/measure_prefill_conv.py compare --candidate scatter
```

### Inkling grouped NVFP4 verification

`MLX_ONE_INKLING_GROUPED_EXPERTS=1` selects a native Metal projection for
2–4-row DSpark panels on base M4 with MLX 0.32.2. Sorted routes for one expert
reuse decoded weight and scale values. The kernel returns FP32 products in
original router order. The existing activation quantizer, interleaved
gate/up layout, outer scales, SwiGLU, and weighted sum supply the rest of
the calculation.

Each projection shape must match the existing FP32 `gather_qmm` result
bit for bit on first use. A failed proof uses the existing gather. The
full panel must still pass the native logits, capture, KV, and convolution
state checks before it can drive generation. The kernel is adapted from
Neywa Labs' MIT-licensed source. [Pinned grouped expert kernel](https://github.com/Neywa-Labs/mlx-vlm/blob/8e80d434fe59f4d4e00bc91a418e0a16ad951326/mlx_vlm/models/inkling2/grouped_experts.py)

One manual 2,048-input / 1,024-output pair on the M4 Air used the same
greedy prompt and DSpark depth 3. The stock leg ran first. TTFT excludes
model load.

| Path | TTFT | Decode | Peak MLX |
| --- | ---: | ---: | ---: |
| Stock panel | 27.23 s | 15.89 tok/s | 15.55 GiB |
| Grouped NVFP4 panel | 27.32 s | 15.58 tok/s | 15.55 GiB |

Both grouped projections passed their bitwise check on the four-row
panel. The two runs matched all 1,024 output IDs, verified 290 panels each,
accepted 440 of 873 draft tokens each, and released their cache allocations.
Grouped decode was 1.93% slower in this pair. Peak MLX differed by 0.22 MiB.
Four-row parity passed at this workload. Acceleration remains unqualified,
and the grouped flag stays default-off. The long-context path is unmeasured
with this kernel.

```sh
.venv/bin/python -u local/inkling-64k-4k/run.py panel --short
.venv/bin/python -u local/inkling-64k-4k/run.py panel --short --grouped-experts
```

### Inkling DSpark accepted-prefix commit

The saved 64K/4K MTPLX 2.13.0 run forced 512-token sustained prefill chunks
and used fixed depth-3 greedy speculation. The live round programs reported
at startup record the draft, target panel, and decision as one Metal command
buffer per geometry. The upstream default applies the kept prefix after the
decision in `models/inkling/fast/decoder.py`. The optional in-program commit
defaults off upstream. The saved result has no timing breakdown by stage.

`MLX_ONE_INKLING_PANEL_PREFIX_COMMIT=1` adapts the default prefix-selection
approach to the existing native panel verifier. A qualified panel keeps the
old boundary and the panel tape. After stop and length checks, it restores
only the consumed prefix and passes the corresponding capture slice to the
existing drafter. The pre-round boundary handles cancellation. The reference
path, model arithmetic, and reservation remain unchanged. The flag defaults
off. Seven focused cache tests passed, including truncated, full, and
cancelled commits. The manual 2K/1K run matched all output IDs and verified
290 panels. It measured 15.06 tok/s versus stock's 15.89 tok/s (5.18% lower),
27.39 s TTFT versus 27.23 s, and 15.55 GiB peak MLX for both paths. Long-context
throughput remains unmeasured for this flag.

The 2K/1K comparison uses the saved stock panel receipt and writes a separate
candidate result with TTFT, throughput, peak MLX memory, and output ID parity.

```sh
.venv/bin/python -u local/inkling-64k-4k/run.py panel --short --prefix-commit
```
