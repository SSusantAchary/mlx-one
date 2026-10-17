# Historical model performance

These are older measured native Python Engine scores from the completed P0
reference cohort on 17 October 2026. They are not results from the new matched
8K reference-versus-optimized comparison and do not qualify the current P0.1
implementation. The archived cohort recorded `GO` for its tested reference
scope; public capabilities were not promoted.

## Measured scores

| Model | Parameters | Weight precision | Prompt tokens | Decode TPS | Peak MLX memory (GiB) | Historical result |
|---|---:|---|---:|---:|---:|---|
| Qwen3-0.6B | ~0.60B | INT8 | 8,192 | 5.82 | 8.614 | P0 reference pass |
| Qwen2.5-0.5B-Instruct | ~0.49B | INT8 | 8,192 | 13.71 | 7.041 | P0 reference pass |
| LFM2-350M | ~0.35B | 4-bit | 8,192 | 38.16 | 5.299 | P0 reference pass |
| GPT-2 | ~0.124B | Floating weights; FP16 execution | 512 | 103.95 | 0.722 | P0 reference pass within 1,024 positions |
| Qwen3.5-0.8B | ~0.75B active text | 4-bit | 8,192 | 12.30 | 11.034 | P0 text-only reference pass |

GPT-2 cannot execute an 8K prompt: its total context limit is 1,024 positions.
Its 512-token control is a different workload and must not be ranked as an
equivalent 8K result. Qwen3.5 has approximately 0.853B total loaded logical
parameters, including components outside its approximately 0.752B active text
path; vision and MTP execution were not tested here.

| Model | Prompt tokens | Prefill TPS | Median TTFT (seconds) | Measured repetitions |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 8,192 | 55.35 | 148.393 | 5 |
| Qwen2.5-0.5B-Instruct | 8,192 | 89.07 | 92.321 | 5 |
| LFM2-350M | 8,192 | 234.13 | 35.260 | 5 |
| GPT-2 | 512 | 5,931.76 | 0.110 | 5 |
| Qwen3.5-0.8B | 8,192 | 74.09 | 110.962 | 5 |

## Measurement scope

- Apple M4, 32 GiB RAM; macOS 15.7.9; Python 3.12.11; MLX 0.32.2.
- Native `reference_dense`, actual GPU B1, greedy decoding, no optional prefix
  reuse, speculation, paged execution, SSD persistence, or optimized batching.
- Original packed INT8/4-bit weights preserved; floating execution and attention
  KV were FP16. Qwen3.5 recurrent state retained its required FP32 precision.
- One unmeasured warm-up followed by five measured fresh-state requests per
  workload. Every selected repetition generated 128 tokens.
- TPS values are medians across repetitions. Decode TPS measures accepted-token
  intervals: `(N - 1) / (last_token_time - first_token_time)`, excluding prefill
  and first-token work. TTFT includes request/tokenization overhead.
- Memory is the maximum observed MLX allocator peak across the five measured
  requests; `1 GiB = 2^30 bytes`. It is not RSS, cache-state size, or an estimate
  of total device use. Overlapping RSS and MLX counters are not summed.
- These are Python Engine measurements, not HTTP/SSE service benchmarks,
  continuous-batching claims, or competitor comparisons.

## Checkpoint pins and evidence

| Checkpoint | Revision |
|---|---|
| `mlx-community/Qwen3-0.6B-8bit` | `11de96878523501bcaa86104e3c186de07ff9068` |
| `mlx-community/Qwen2.5-0.5B-Instruct-8bit` | `6474d4c1a05bd7bd634a9c993560e4e7272eda3d` |
| `mlx-community/LFM2-350M-4bit` | `18dc72abf3b2337f9123cfd6eeeb58dfa7947066` |
| `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| `mlx-community/Qwen3.5-0.8B-4bit` | `da28692b5f139cb0ec58a356b437486b7dac7462` |

Measured Git base: `53404467afa3cf094bd221c61f5316f2b928b161`.
Measured working-source SHA256:
`554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.
The measured tree included uncommitted P0 repairs; the Git base alone does not
identify that runtime.

Local evidence directory: `qualification/results/p0_engine/runs/1792201272601288000/`.
The report was checked against `inference_benchmarks.json`, `summary.json`,
`download_manifest.json`, and their `result-integrity.json` checksums. Raw
timings, token IDs, manifests, weights, and large artifacts remain local and
are intentionally excluded from the code/report commit.
