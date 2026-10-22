# P0 H8192 admission fix

Native source: `53404467afa3cf094bd221c61f5316f2b928b161`; working-source SHA256 `554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.

Final cohort: `qualification/results/p0_engine/runs/1792201272601288000`. Prior failed cohort: `qualification/results/p0_engine/runs/1792175822328500000`. The prior source digest was `c6b31d17992f5af41a820c227ff199b4d6bd806381ab99a4b685823df7331943`; it matched the checkout before this repair. Historical evidence remains local and unchanged.

Release recommendation: **GO**, limited to the recorded five checkpoints, explicit floating FP16 conversion, FP16 attention KV, reference dense GPU B1, and measured hardware. No capabilities, defaults, commits, pushes, tags or releases are promoted.

## Observed cause

All four cold 8K requests succeeded before repair. Shorter warmed workloads left large unused MLX allocator buffers; the original host check could reject before pressure shedding ran. Qwen2.5, LFM2 and Qwen3.5 also reproduced rejection on the next repeated 8K request. Active handles, outstanding reservations and owned request state returned to zero after each request. These were typed admission rejections, not observed GPU OOMs or numerical failures.

The formula remains `max(RSS, MLX active + cached, model + physical owner) + outstanding reservations + requested growth`. Overlapping RSS, allocator and owner counters are not summed. Cached bytes are never subtracted from RSS. The ceiling remains the lower configured, recommended or 75%-RAM limit: 22,906,503,168 bytes (21.33 GiB) on this M4/32 GiB host. The cache/transient budget remains 16 GiB.

Baseline Qwen3 warm rejection: working set 20,857,283,372 bytes, outstanding 968,884,224 bytes, requested transient 9,474,932,736 bytes. Their sum exceeds 22,906,503,168 bytes. MLX active was 633,372,680 bytes and cached was 20,223,910,692 bytes. Owner-thread cleanup with zero request handles/reservations cleared unused buffers and admitted the unchanged full-width 8K request.

## Safe recovery and accounting

A failed host check now permits one device-owner recovery when unused allocator/prefix/free-pool storage exists and no transient reservation is active. It synchronizes completed work, releases eligible index/free references, clears unused MLX allocator buffers, invalidates telemetry, resamples, and recalculates once. Active request KV/carries and reservations remain owned. Unsafe-thread, permanent oversize, critical-pressure and still-over-ceiling paths retain typed failures; reclamation errors propagate. Admission recovery and the cold/warm ownership probes use no forced garbage collection. The pre-existing reload-memory calibration still collects released lifecycle objects; each admission/lifecycle 8K probe asserts zero request ownership before any explicit allocator cleanup.

Persistent admission refreshes host telemetry. Forward boundaries read current allocator counters; full host samples remain bounded to one per second until reclamation forces a fresh sample. Pressure-controller snapshots are refreshed after shedding. Diagnostics distinguish owner storage, retention, allocator observations, outstanding commitments, and predicted transient allocation.

Dense hybrid accounting now includes convolution/recurrent carries independently of restoration capability. Actual declared compound FP16/FP32 layouts remain intact. Existing evaluated KV/carry storage is excluded from new growth while remaining in owner accounting. Dense hybrid prefix restoration remains unsupported.

## Prefill policy

No prefill chunking or execution default changed. Before repair, cold evaluated MLX peaks were 8.614 GiB (Qwen3), 7.041 GiB (Qwen2.5), 5.299 GiB (LFM2) and 10.955 GiB (Qwen3.5), below the unchanged ceiling. The qualified profile records `prefill_chunk_tokens=null`, 8192 exact input tokens, output budget 128, B1, fixed pinned packed weights, native tolerances and disabled optional prefix reuse. Forward timing excludes admission/reclamation overhead; TTFT and end-to-end timing retain it.

## Initial and final reclamation calculations

| Model | Working before GiB | Allocator cached before GiB | Working after GiB | Cached after GiB | Reserved GiB | Requested GiB | Retry |
|---|---:|---:|---:|---:|---:|---:|---|
| mlx-community/Qwen3-0.6B-8bit | 20.239 | 19.649 | 0.590 | 0.000 | 0.246 | 1.456 | admit |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 17.535 | 17.044 | 0.601 | 0.000 | 0.097 | 8.367 | admit |
| mlx-community/LFM2-350M-4bit | 19.257 | 19.071 | 0.312 | 0.000 | 0.097 | 6.297 | admit |
| mlx-community/Qwen3.5-0.8B-4bit | 20.259 | 19.677 | 0.644 | 0.000 | 0.045 | 2.070 | admit |

Each row above is an observed matched reclaim/retry pair; full per-request records retain exact bytes, stage, request tokens, precision, component dtypes, OS pressure/swap, source and case identity in `admission_diagnostics.json`.

## Growth estimates and observed state

| Model | Planned KV growth MiB | Planned carry growth MiB | Observed allocated state MiB | Observed logical state MiB |
|---|---:|---:|---:|---:|
| mlx-community/Qwen3-0.6B-8bit | 924.000 | 0.000 | 924.000 | 909.891 |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 99.000 | 0.000 | 99.000 | 97.488 |
| mlx-community/LFM2-350M-4bit | 99.000 | 0.039 | 99.039 | 97.527 |
| mlx-community/Qwen3.5-0.8B-4bit | 99.000 | 18.633 | 117.633 | 116.121 |

Allocated state is observed at scheduling boundaries and includes capacity rounding; logical state contains only committed real tokens plus hybrid carries. Transient forward buffers and unused allocator cache are reported separately. Carry dtypes/shapes are validated in the numerical/state artifacts.

## Cold, warmed and lifecycle 8K checks

| Model | Original cold | Original warm/repeat | Original cleanup | Fixed cold | Fixed warm | Post-cancel | Post-cleanup | Repeated | Fresh independent worker |
|---|---|---|---|---|---|---|---|---|---|
| mlx-community/Qwen3-0.6B-8bit | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | PASS | PASS,FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mlx-community/LFM2-350M-4bit | PASS | PASS,FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mlx-community/Qwen3.5-0.8B-4bit | PASS | PASS,FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

I01–I06 compare greedy tokens across fresh state, warmed state, decode cancellation, explicit owner cleanup, consecutive requests and a separately loaded process. Ownership is asserted after every warm-up and completed request, before explicit cleanup, without forced GC. Component state and unchanged numerical tolerances are covered by D/E and the pinned checkpoint regressions.

## Five-repeat 8K performance

| Model | 8K admission | 8K inference | TTFT s | Prefill TPS | Decode TPS | Peak memory | Final |
|---|---|---|---:|---:|---:|---|---|
| mlx-community/Qwen3-0.6B-8bit | PASS | PASS | 148.39 | 55.35 | 5.82 | 8.614 GiB MLX peak | QUALIFIED |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | PASS | PASS | 92.32 | 89.07 | 13.71 | 7.041 GiB MLX peak | QUALIFIED |
| mlx-community/LFM2-350M-4bit | PASS | PASS | 35.26 | 234.13 | 38.16 | 5.299 GiB MLX peak | QUALIFIED |
| mlx-community/Qwen3.5-0.8B-4bit | PASS | PASS | 110.96 | 74.09 | 12.30 | 11.034 GiB MLX peak | QUALIFIED |

TPS/TTFT entries are medians of five measured fresh-state requests after one warm-up. Peak memory is the largest observed MLX active allocation peak, not summed unified-memory counters. Request-level sampled peak RSS, allocator cached memory, synchronized forward durations, accepted token counts/IDs, state bytes, tokenization, model-token gaps and end-to-end timing are retained in JSON/CSV. Decode TPS excludes prefill and the first token. No HTTP/SSE claim is inferred. GPT-2 retains its legal 128/512-token baseline in the overall five-model reports; 8K is outside its 1024-position context.

## Changed source and regressions

- `engine/admission.py`: bounded safe recovery and structured telemetry decisions.
- `engine/cache_manager.py`: delegate host decisions, account for hybrid carries/new growth, preserve reservation ownership and typed rejection.
- `engine/cache_planner.py`: explicit floating convolution-carry layout where no state dtype was declared.
- `engine/pressure.py` and `engine/actor.py`: owner-only shedding, existing actor call routed through the owner, and refreshed telemetry.
- `text/generation.py`: synchronized forward timing starts after admission/staging.
- `p0_engine_qualification.py` and `p0_engine_worker.py`: workload selectors, non-qualifying diagnostics, I01–I06, request RSS attribution and supervision-safe results.
- `test_h8192_admission.py`, `test_h8192_checkpoints.py`, qualification fixtures, actor pressure-owner and compound carry-growth regressions.

Backend-free coverage includes ceiling boundaries, one retry, failed/no/off-owner reclaim, live state preservation, critical pressure, fresh telemetry, reservation isolation/cleanup, prior workloads, declared carry layouts, supervision failures and request RSS sampling. Native checkpoints exercise the actual full-width cases under supervision. CPU success alone never replaces the native gates.

## Remaining risks and boundaries

Full-width 8K prefill has high latency and allocator churn; baseline qualification imposes no inherited P1 throughput target. Safe one-retry rejection remains necessary under larger live workloads or host pressure. RSS sampling is 100 ms and cannot establish continuous OS physical footprint or swap-stress behavior. Optimized batching, paging, auto selection, quantized KV, SSD and speculation remain outside this reference-only result. Qualification is specific to the recorded source, pinned assets, hardware and execution policy.

Final case counts: `{'PASS': 404, 'NOT_SUPPORTED': 11}`. See `P0_ENGINE_TEST_MATRIX.md` for all mandatory and inapplicable outcomes; no missing or unsupported test is promoted to PASS.
