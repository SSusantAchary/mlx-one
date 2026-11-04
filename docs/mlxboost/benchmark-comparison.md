# Performance comparison and measured scope

Current deployment status: opt-in **EXPERIMENTAL**, defaults unchanged. The single host-lookup flag has a repeatable HTTP gain. Under the [2026-11-02 promotion policy](promotion-policy.md), pre-existing numerical self-consistency failures remain model-qualification caveats and do not automatically block independent promotion of a proven neutral optimization. This documentation edit issues no rollout decision. No direct-forward or kernel speedup is claimed.

## Original whole-stack protocol: frozen and reproduced

All columns below use the shared full-length subset across all three runtimes within each run. Qwen models have six samples, SmolLM2 five; excluded EOS cases remain in unfiltered records. MLX-One is flag-on only in the last column. The original dependency mismatch (MLX-One 0.32.2 versus competitors 0.32.4) remains.

| Model | Runtime | Frozen historical tok/s | Reproduction tok/s | New full comparison tok/s |
| --- | --- | ---: | ---: | ---: |
| qwen2 | mlx-one | 21.23 | 20.81 | 126.67 |
| qwen2 | mlx-lm | 134.82 | 133.86 | 138.88 |
| qwen2 | omlx | 156.72 | 156.84 | 157.60 |
| qwen3 | mlx-one | 20.13 | 19.99 | 96.21 |
| qwen3 | mlx-lm | 113.21 | 111.98 | 112.29 |
| qwen3 | omlx | 128.74 | 129.47 | 130.33 |
| smollm2 | mlx-one | 14.75 | 14.64 | 24.57 |
| smollm2 | mlx-lm | 26.60 | 25.39 | 25.08 |
| smollm2 | omlx | 27.36 | 27.05 | 27.12 |

## Same-source original protocol, flag off/on

Three independent processes, one excluded warmup and two timed requests/process; actual output counts and full SSE wall time. All native samples are retained, with no EOS filtering in this table.

| Model | Off tok/s | On tok/s | Off → on TTFT ms | Off → on latency s |
| --- | ---: | ---: | ---: | ---: |
| qwen2 | 23.02 | 126.67 | 109.44 → 70.77 | 5.561 → 1.010 |
| qwen3 | 21.56 | 96.21 | 130.97 → 89.62 | 5.936 → 1.330 |
| smollm2 | 15.39 | 24.56 | 194.87 → 182.74 | 8.318 → 5.212 |

## Expanded Qwen confirmation

Five processes per flag/model, one excluded warmup and ten timed requests/process: 50 paired requests per flag/model. Expanded prompts are separately prepared from the same cached snapshots, with exact IDs and hashes. Qwen2 runs on then off; Qwen3 reverses that order. External host load, power and thermals were not controlled. All 100 paired cases across the two models match exact text, counts and finish reason.

| Model | Off tok/s | On tok/s | Median process gain | Process bootstrap 95% interval | Off → on TTFT ms |
| --- | ---: | ---: | ---: | --- | ---: |
| qwen2 | 22.94 | 109.17 | 4.77× | 4.49–4.83× | 111.22 → 86.56 |
| qwen3 | 21.42 | 96.42 | 4.50× | 4.45–4.55× | 126.62 → 92.62 |

The interval resamples five paired process medians, not 50 independent requests. Its small process count and unmeasured thermals limit generalization; it is not a latency SLA. Peak RSS stayed about 0.70 GiB (Qwen2.5) and 0.80 GiB (Qwen3); MLX allocation is reported separately in native/API evidence.

## Direct and native generation diagnostics

One process/model, warmup plus three timed requests. These separate before/after diagnostics measure an unchanged model path at different times, **not a causal forward gain**. The causal native flag experiment is separately recorded in optimization-ablation.md.

| Model | Direct forward before → after tok/s | Bare native generation before → after tok/s | Post direct decode tok/s |
| --- | ---: | ---: | ---: |
| qwen2 | 131.94 → 128.67 | 125.65 → 108.62 | 134.48 |
| qwen3 | 107.85 → 100.36 | 106.24 → 88.54 | 104.89 |
| smollm2 | 24.81 → 23.54 | 24.63 → 21.70 | 24.42 |

## Matched dependency native ablation

Pinned MLX-LM source and MLX-One both use MLX/Metal 0.32.2. The source probe has one process/model, two timed repeats each on prompt indices 1 and 2 after excluded warmup; figures below are native request rates, not HTTP. Its full/split/async variants have identical IDs for these prompts, including 128-token SmolLM2 outputs where the original whole-stack 0.32.4 server had a short EOS. This does not uniquely attribute the historical EOS difference.

| Model | Pinned source full prefill | Pinned source 127+1 prefill | Pinned source async generate_step |
| --- | ---: | ---: | ---: |
| qwen2 | 104.85 | 108.67 | 113.67 |
| qwen3 | 101.44 | 100.42 | 100.22 |
| smollm2 | 23.25 | 23.67 | 26.14 |

The matched dependency experiment is native-only. No matched-dependency HTTP or oMLX 0.32.2 comparison was performed. No 0.32.2→0.32.4 primitive speedup is claimed.

## Secondary checkpoints and evidence limits

Cached LFM2 Q4 revision `18dc72abf3b2337f9123cfd6eeeb58dfa7947066` has a new native FP16 baseline of **254.73 tok/s**, one process/two timed requests after warmup, 128 input/128 output. Its 128-step source comparison passed all 194 logits/KV/convolution-carry checks. This is a diagnostic baseline, with no HTTP comparison, optimization or broad model qualification. Llama-3.2 Q4 is **NOT MEASURED**: no matching Q4 artifact was resolved in the inspected cache/local provenance paths; a floating upstream snapshot is not substituted.

No 4k–32k checkpoint performance, 128k result, kernel GPU capture, full activation-boundary trace, quality-corpus requalification or cross-device qualification is claimed. Those scopes lack supporting evidence; the model's strict numerical qualification also remains failed. Neither limitation invalidates the measured neutral-change comparison within its tested scope.

## Reproduce and raw artifacts

`benchmarks/mlxboost/final_campaign.py --expanded` serializes the original protocol and expanded experiments. `http_experiment.py` extends the existing benchmark with an explicit source launcher; it preserves request, sampling, natural EOS, cache, readiness and accounting logic. It never edits the owner runner.

Raw directories: `local/mlxboost/runs/baseline-original-20261101/`, `confirmation-20261101/{original-protocol-fast,original-protocol-off,expanded-qwen2-on,expanded-qwen2-off,expanded-qwen3-off,expanded-qwen3-on}/`, and `source-requalification-20261101/`. Each HTTP directory has original raw results/events/logs plus a sealed manifest and per_request.csv. `performance-summary.json` is generated from these unfiltered records.

Initial exploratory failures (source metadata import, local launcher selection and observer adaptation) are retained in audit logs. The first Qwen2 native diagnostic preceded per-run manifest instrumentation; its runtime source is bound to the pre-change source audit, and its script snapshot is incomplete. HTTP manifests explicitly disclose that the original runner captures production hashes rather than a full dirty-tree/wrapper snapshot at launch. No missing launch hash was fabricated.
