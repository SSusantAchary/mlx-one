# P0 Engine native benchmarks

Measured source: `53404467afa3cf094bd221c61f5316f2b928b161`; working-source SHA256 `554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.

Local evidence: `qualification/results/p0_engine/runs/1792201272601288000`. Checkpoints, raw traces and results are uncommitted.

Every model correctness/lifecycle suite completes before this performance phase. Source identity is frozen across the cohort.

One unmeasured warm-up and five measured fresh-state repetitions per legal workload. Tokenization is included in request wall/TTFT timing and reported separately. Synchronized forward durations include native evaluation and exclude admission/reclamation; request wall/TTFT timing includes that work. Decode interval rate is `(accepted_tokens - 1) / (last_model_token_time - first_model_token_time)`; zero/one-token intervals are null. End-to-end rate uses actual generated tokens and full request wall time. HTTP/SSE performance is not measured here.

| Model | Prompt tokens | Runs | Median TTFT ms | Median prefill tok/s | Median decode tok/s | Median end-to-end tok/s | Max MLX peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| openai-community/gpt2 | 128 | 5 | 62.89 | 3102.86 | 95.36 | 91.78 | 0.337 |
| openai-community/gpt2 | 512 | 5 | 110.36 | 5931.76 | 103.95 | 95.97 | 0.722 |
| mlx-community/Qwen3-0.6B-8bit | 128 | 5 | 573.42 | 232.01 | 21.01 | 19.35 | 0.726 |
| mlx-community/Qwen3-0.6B-8bit | 512 | 5 | 2610.57 | 198.71 | 20.34 | 14.45 | 1.128 |
| mlx-community/Qwen3-0.6B-8bit | 2048 | 5 | 15753.67 | 130.50 | 17.03 | 5.50 | 2.277 |
| mlx-community/Qwen3-0.6B-8bit | 8192 | 5 | 148392.61 | 55.35 | 5.82 | 0.75 | 8.614 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 128 | 5 | 400.66 | 342.56 | 26.09 | 24.25 | 0.570 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 512 | 5 | 1901.46 | 273.29 | 25.18 | 18.45 | 0.869 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 2048 | 5 | 10967.35 | 187.49 | 23.76 | 7.84 | 1.799 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 8192 | 5 | 92321.01 | 89.07 | 13.71 | 1.26 | 7.041 |
| mlx-community/LFM2-350M-4bit | 128 | 5 | 210.58 | 695.81 | 61.98 | 56.26 | 0.264 |
| mlx-community/LFM2-350M-4bit | 512 | 5 | 781.85 | 680.23 | 57.07 | 42.52 | 0.535 |
| mlx-community/LFM2-350M-4bit | 2048 | 5 | 4850.52 | 425.54 | 56.97 | 18.09 | 1.280 |
| mlx-community/LFM2-350M-4bit | 8192 | 5 | 35259.55 | 234.13 | 38.16 | 3.31 | 5.299 |
| mlx-community/Qwen3.5-0.8B-4bit | 128 | 5 | 981.60 | 134.40 | 17.58 | 15.62 | 0.834 |
| mlx-community/Qwen3.5-0.8B-4bit | 512 | 5 | 4352.03 | 118.41 | 17.08 | 10.83 | 1.313 |
| mlx-community/Qwen3.5-0.8B-4bit | 2048 | 5 | 19902.84 | 103.16 | 16.88 | 4.67 | 3.160 |
| mlx-community/Qwen3.5-0.8B-4bit | 8192 | 5 | 110961.87 | 74.09 | 12.30 | 1.06 | 11.034 |

## Timing and memory details

| Model | Prompt | Output range | Median tokenization ms | Median per-run p95 gap ms | Max RSS GiB | Peak request state MiB |
|---|---:|---:|---:|---:|---:|---:|
| openai-community/gpt2 | 128 | 128-128 | 0.16 | 13.07 | 0.560 | 8.965 |
| openai-community/gpt2 | 512 | 128-128 | 0.55 | 13.85 | 0.561 | 22.465 |
| mlx-community/Qwen3-0.6B-8bit | 128 | 128-128 | 0.37 | 53.81 | 0.749 | 27.891 |
| mlx-community/Qwen3-0.6B-8bit | 512 | 128-128 | 1.39 | 60.24 | 0.751 | 69.891 |
| mlx-community/Qwen3-0.6B-8bit | 2048 | 128-128 | 5.59 | 92.31 | 0.755 | 237.891 |
| mlx-community/Qwen3-0.6B-8bit | 8192 | 128-128 | 20.78 | 210.14 | 0.765 | 909.891 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 128 | 128-128 | 0.38 | 43.48 | 0.670 | 2.988 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 512 | 128-128 | 1.06 | 46.60 | 0.672 | 7.488 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 2048 | 128-128 | 4.49 | 59.15 | 0.674 | 25.488 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 8192 | 128-128 | 22.96 | 107.57 | 0.654 | 97.488 |
| mlx-community/LFM2-350M-4bit | 128 | 128-128 | 0.44 | 18.43 | 0.317 | 3.027 |
| mlx-community/LFM2-350M-4bit | 512 | 128-128 | 1.27 | 19.97 | 0.319 | 7.527 |
| mlx-community/LFM2-350M-4bit | 2048 | 128-128 | 4.37 | 22.07 | 0.320 | 25.527 |
| mlx-community/LFM2-350M-4bit | 8192 | 128-128 | 19.27 | 33.43 | 0.337 | 97.527 |
| mlx-community/Qwen3.5-0.8B-4bit | 128 | 128-128 | 0.35 | 75.01 | 0.879 | 21.621 |
| mlx-community/Qwen3.5-0.8B-4bit | 512 | 128-128 | 0.95 | 77.54 | 1.048 | 26.121 |
| mlx-community/Qwen3.5-0.8B-4bit | 2048 | 128-128 | 4.18 | 90.10 | 1.730 | 44.121 |
| mlx-community/Qwen3.5-0.8B-4bit | 8192 | 128-128 | 15.79 | 128.16 | 3.746 | 116.121 |

Per-request evaluated cache/state bytes are captured without logits recording. The JSON also reports tokenization time, load time, actual token IDs/counts, synchronized prefill/decode duration, p50/p95 model-token gaps and end-to-end wall time.

Peak RSS is the maximum 100-ms supervised sample within each request, with null when unavailable; allocator active/cached/peak values, token counts, per-forward timing and token IDs are retained in JSON/CSV. RSS and MLX counters overlap unified memory and are never summed. Benchmark failures and inapplicable context lengths remain in the test matrix. No TPS threshold is used to manufacture correctness qualification.
