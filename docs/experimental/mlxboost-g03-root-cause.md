# G03 slowdown investigation

**Decision: retain stock MLX for B1; do not pursue another broad G03 tuning sweep. G03 stays UNQUALIFIED.** Correct isolated operations and successful compilation do not waive prior logits/state failures. No G03 source/binary/registry was altered.

## Source-grounded explanation

[mlx `mlx/backend/metal/kernels/quantized.h:758`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h#L758) already processes four output rows/SIMD, uses two packed words/thread for Q4/Q8, vectorized activation loads, scale/bias reuse and specialized aligned paths. G03's four-row reuse duplicates an existing stock strategy. Its one-pack loop performs more iterations, scalar extraction/conversion and metadata work. Stock also specializes tiny K and tails. G03 groups64/128/256 threads do not automatically improve stock's64-thread/8-output tile. SIMD occupancy/spill counts were **not measured**; they remain hypotheses, not proven causes.

The G03 C++ `eval_gpu` also resolves its library/pipeline via dladdr/filesystem/name construction. That is a possible evaluation-side host cost, but graph construction alone is sub-microsecond in these direct probes and does not explain the whole gap. Profiling uses the raw loaded extension; it does **not** repeatedly run integrity hashing/registry lookup per timed call. Session verification/setup is outside the sustained decode interval. Blaming the historical16ms Phase 0 integrity-guard demonstration would be incorrect here.

Single-call wall latency includes a large synchronization/submission floor. We also time16 distinct evaluated input views per window, and assert57×16 native encodings for correctness+cold+warmup+50 samples; identical inputs are not collapsed into one custom primitive. Amortization exposes a larger small-projection gap. Q8 q_proj single stock226.900µs/G03241.810µs becomes22.050µs/29.390µs per call; Q4 q_proj260.100µs/299.830µs becomes61.850µs/134.120µs. Graph-only construction is separately timed without eval and never increments native counters. Combined evaluation/encoding/GPU work remains the measured quantity; no isolated GPU-command duration is claimed.

## Real packed checkpoint arrays, M=1,16-call windows

| Checkpoint | Projection | K×N | MLX µs/call | G03 µs/call | Speedup | Operation parity |
| --- | --- | --- | --- | --- | --- | --- |
| qwen25-q8 | q_proj | 896×896 | 22.047 | 29.389 | 0.7502× | True |
| qwen25-q8 | gate_proj | 896×4864 | 56.111 | 68.352 | 0.8209× | True |
| qwen25-q8 | down_proj | 4864×896 | 57.824 | 67.134 | 0.8613× | True |
| qwen25-q8 | lm_head | 896×151936 | 1353.178 | 1355.573 | 0.9982× | True |
| llama-q4 | q_proj | 2048×2048 | 61.845 | 134.120 | 0.4611× | True |
| llama-q4 | gate_proj | 2048×8192 | 104.987 | 129.449 | 0.8110× | True |
| llama-q4 | down_proj | 8192×2048 | 107.734 | 128.109 | 0.8410× | True |
| llama-q4 | lm_head | 2048×128256 | 1395.995 | 1636.220 | 0.8532× | True |

Complete50-sample single/windowed data, p10/p90/stdev, first/last output checks, host-only samples and counters: measurements/quantized-summary.csv (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/measurements/quantized-summary.csv`), measurements/quantized-raw-samples.csv (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/measurements/quantized-raw-samples.csv`). These use original packed checkpoint weights with identical random activation inputs across candidates. Real model activations are used separately for the promising verifier validation, not substituted here.

For large LM heads the one-pass unique packed weight+metadata payload divided by amortized stock wall time is:

| Checkpoint | Unique encoded bytes | Effective GB/s | Fraction of Apple 120GB/s specification |
| --- | --- | --- | --- |
| qwen25-q8 | 144643072 | 106.891 | 89.1% |
| llama-q4 | 147750912 | 105.839 | 88.2% |

Apple lists 120GB/s for [M4 MacBook Air](https://support.apple.com/en-ie/122209). These ratios are a **payload-based inference**, not hardware DRAM counters: cache hits, repeated metadata reads, output traffic and clocks are not resolved. They suggest little B1 headroom at the large Q8 head, while the small Q4 projection has extra work/latency. They do not prove bandwidth saturation or promise an idealized remaining gain. GPU-only timing, occupancy and register spills were unavailable.

## Other upstreams and misses

Unmodified oMLX classic affine tiles0/4/8 compiled on this pinned stack; groups64/128 and aligned dimensions only. All240 small-matrix tile cases pass numerical policy but none meets10%; padding small M into BM16/32/64 and its tile work lose here. Group32 and unsupported N tails are rejected, not silently repacked. NAX branches require newer hardware/OS and were not compiled/executed. See [omlx `omlx/custom_kernels/quant_matmul/csrc/affine_qmm.cpp:70`](https://github.com/jundot/omlx/blob/166ecf5d84e29abb0622cc2979fd0bad1dc277fd/omlx/custom_kernels/quant_matmul/csrc/affine_qmm.cpp#L70) and [omlx `omlx/custom_kernels/quant_matmul/csrc/affine_qmm.metal:1`](https://github.com/jundot/omlx/blob/166ecf5d84e29abb0622cc2979fd0bad1dc277fd/omlx/custom_kernels/quant_matmul/csrc/affine_qmm.metal#L1).

MTPLX wins only narrower large-N multi-row scopes, analyzed in [the verifier report](mlxboost-quant-verify-opportunity.md). The generic180-workload matrix has no 10% MTPLX win on its small aligned matrices. llama.cpp's [llama.cpp `ggml/src/ggml-metal/ggml-metal-device.cpp:1027`](https://github.com/ggml-org/llama.cpp/blob/a946eb541a6d3051496bdcc451a0fa85772b8dfd/ggml/src/ggml-metal/ggml-metal-device.cpp#L1027) and [llama.cpp `ggml/src/ggml-metal/kernels/mul_mv.metal:100`](https://github.com/ggml-org/llama.cpp/blob/a946eb541a6d3051496bdcc451a0fa85772b8dfd/ggml/src/ggml-metal/kernels/mul_mv.metal#L100) are useful SIMD/tile/packing references, but GGML Q4_0/Q4_1/Q8_0 block layouts and output conventions differ from MLX affine. No non-equivalent binary timing was presented and no GGML dependency or weight conversion added.

The initial one-row G03 prototype binary/source were not retained; its earlier results are historical, non-paired only. No reconstruction is called a rerun. Current code and compiled assets are exact and hashed.

## Longer generation diagnostic

Five randomized paired repeats,128/4096 exact input tokens,128 requested output tokens, EOS respected. **Every sample reaches128 tokens and all paired token sequences agree**. Forced G03 remains slower in every paired repeat. This is a separate research generation scope, not a replacement for the original failed logits/state/fresh controls.

| Checkpoint | Input | Backend | Decode t/s median | Prefill t/s median | TTFT ms median | Peak MLX bytes max | Custom dispatches/request |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen25-q8 | 128 | forced_g03 | 108.443 | 2243.304 | 57.649 | 679129024 | 21463 |
| qwen25-q8 | 128 | mlx_reference | 131.226 | 1965.594 | 65.830 | 679129024 | 0 |
| qwen25-q8 | 4096 | forced_g03 | 105.946 | 2380.143 | 1725.769 | 2365501224 | 21463 |
| qwen25-q8 | 4096 | mlx_reference | 114.864 | 2392.222 | 1719.097 | 2365501224 | 0 |
| llama-q4 | 128 | forced_g03 | 93.817 | 1113.725 | 115.548 | 950900936 | 14351 |
| llama-q4 | 128 | mlx_reference | 109.660 | 1106.017 | 116.645 | 950900936 | 0 |
| llama-q4 | 4096 | forced_g03 | 78.754 | 965.733 | 4246.649 | 2320537616 | 14351 |
| llama-q4 | 4096 | mlx_reference | 90.745 | 960.565 | 4273.052 | 2320537616 | 0 |

measurements/sustained-model-samples.csv (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/measurements/sustained-model-samples.csv`); immutable full records contain token IDs, five ratios, order, RSS high-water and source/environment metadata. No 32k model run was added; previous admission skips stand. There is no new model speedup claim. Do not infer physical 8/16/24GiB performance from this host.
