# Measured performance and limits

Full GPT-OSS-20B TQ3 checkpoint performance: **PENDING_USER_DOWNLOAD**.
The >=90% warm-decode target and full-model prefill ratio remain unmeasured.
No 20B load, generated answer or model TPS claim is made.

Measured host: Apple M4, 32 GB unified memory, macOS 15.7.9, Python 3.12.11,
MLX/MLX-Metal 0.32.2. Reference TurboQuant commit
`54443c8b83424adb5dac7e6ffc7ac2d91fe88af1`, isolated MLX-LM 0.32.2.
Reports include exact host/device metadata. Twelve timed samples/case follow three warmups.
Medians below include normal Python operator dispatch; see raw p95, compile and memory fields.

| Synthetic operation | Shape / tokens | Native ms | Reference ms | Reference/native speed ratio |
|---|---|---:|---:|---:|
| Dense q projection | B1, O4096 K2880 | 1.206 | 1.222 | 1.01 |
| Dense k/v projection | B1, O512 K2880 | .275 | .273 | .99 |
| Dense o projection | B1, O2880 K4096 | 1.002 | .992 | .99 |
| Dense expert-sized projection | B1, O2880 K2880 | .389 | .393 | 1.01 |
| Dense packed GEMM | 128, O4096 K2880 | 3.246 | 3.247 | 1.00 |
| Dense packed GEMM | 256, O2880 K4096 | 5.967 | 6.567 | 1.10 |
| Expert shared input, top four | B1, E32 O/K2880 | 2.132 | 1.242 | .58 |
| Expert per-route input, top four | B1, E32 O/K2880 | .855 | 2.422 | 2.83 |
| Unsorted expert shared prefill | 128 tokens, top four | 11.805 | 71.743 | 6.08 |
| Unsorted expert multi prefill | 256 tokens, top four | 20.932 | 134.858 | 6.44 |
| Complete MoE | B1 | 1.725 | 2.438 | 1.41 |
| Complete decoder block | B1 | 2.079 | 2.638 | 1.27 |
| Complete MoE | 128 tokens | 38.868 | 37.004 | .95 |
| Complete decoder block | 128 tokens | 46.355 | 43.153 | .93 |
| Complete decoder block | 256 tokens | 77.956 | 76.653 | .98 |

The unsorted standalone reference expert layer expands the entire bank at large routing counts.
Native tiled sorted execution avoids that branch: at 128 shared-input tokens peak active increment
is 12.55 MB versus 534.53 MB (decimal MB); at 256 per-route tokens 23.63 versus 542.64 MB.
The reference's complete MoE sorts earlier, so that standalone 6x advantage does not carry to the
whole block. At 128 tokens the native decoder increment is 55.07 MB versus 38.80 MB; at 256,
110.24 versus 77.58 MB. Native repeated gate/up sorting and separate activation intermediates are candidates
for the larger peak; allocation-by-allocation attribution remains unmeasured. Shared-input B1 projection and router-plus-projection are slower
than reference in this run and remain bottlenecks. Hardware clock/thermal variation on a fanless
Air and the small sample count limit close ratio comparisons.

## Optimization evidence

The initial functional baseline and independent reconstruction passed before tuning.
A direct dispatch sweep compared reference GEMV, coalesced GEMV and tiled GEMM on identical inputs.
Coalesced B1 GEMV improved measured K4096/O2880 from .990 to .865 ms and K2880/O2880 from
.395 to .374 ms; O4096/K2880 regressed from .633 to .697 ms. It is not a uniform win.
Two-row GEMV and the 128-routing sparse crossover were retained with boundary correctness tests.
At 256 routings the sweep measured 20.302 ms coalesced versus 8.539 ms sorted tiled including sorting.
Final full regressions passed after retaining those changes.

A later experiment embedded the routing stride as a constant and staged expert codebooks in
registers. Shared B1 improved 2.132 -> 1.877 ms, but multi-input regressed .855 -> 1.179 ms and
whole decoder 2.079 -> 2.556 ms. It was **rejected and reverted**. Its raw results are retained,
not presented as final performance. No kernel/source optimization was retained solely to force a
speed PASS. Current opportunities include shared-input expert bandwidth, gate/up rotation and
sorting reuse, activation fusion, launch/graph overhead and larger-prefill tiling.

The full-model driver accepts local weights, saved exact IDs, input matrix, output budget,
sampling, FP16 KV, repetitions and backend. It records fresh-process load, cold/warm TTFT,
prefill/decode TPS, real EOS/counts, active/RSS/state memory, warmups, dispatch and optional layer/MoE
samples. Both direct backends apply last-only prompt logits; the reference gets a benchmark-only
head wrapper. Comparisons reject host/MLX/checkpoint hash/prompt/settings differences and mark
unequal EOS output counts incomparable. Tiny synthetic driver checks are clearly labeled as such.
Python Engine, fresh CLI process and resident HTTP results have separate reports and scores.
See [manual-validation](manual-validation.md) for real checkpoint commands.

The pinned MLX-LM 0.32.2 YaRN helper always floors/ceils correction bounds; native GPT honors
the checkpoint's `truncate=false`. This source difference is retained and recorded in the full-model
runner's comparison notes. It can affect generated IDs/EOS; it is distinct from TQ3 packed arithmetic.
The helper's file hash is recorded in source provenance. Large-vocabulary lm_head latency is not
in the operator grid; the owner's full-model profile must include its contribution.
