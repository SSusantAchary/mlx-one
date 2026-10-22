# P0 native Python Engine qualification

Measured source: `53404467afa3cf094bd221c61f5316f2b928b161`; working-source SHA256 `554ce34479cf89daf094caf9843c84a2b77eb1d45968e227fafd7d91f1ea68f6`.

Local evidence: `qualification/results/p0_engine/runs/1792201272601288000`. Checkpoints, raw traces and results are uncommitted.

Release recommendation: **GO**. This is qualification of the recorded source/checkpoint/hardware configuration, not a capability promotion or release publication.

Apple M4, 32 GiB unified memory; macOS 15.7.9; Python 3.12.11; MLX 0.32.2.

Policy: native `reference_dense`, GPU B1, explicit FP16 floating conversion preserving native packed weights, FP16 attention KV, architecture-specific carries/recurrent precision. Prefix reuse, SSD, speculation, paging, batching and automatic selection are disabled.

## Model decisions

| Model | Precision | Load | Generation | Streaming | KV/state | Async | Cancellation | Lifecycle | 8K admission | Benchmark | Final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| openai-community/gpt2 | float | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_SUPPORTED | PASS | QUALIFIED |
| mlx-community/Qwen3-0.6B-8bit | 8 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | QUALIFIED |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 8 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | QUALIFIED |
| mlx-community/LFM2-350M-4bit | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | QUALIFIED |
| mlx-community/Qwen3.5-0.8B-4bit | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | QUALIFIED |

## Reproduction

```sh
source .venv/bin/activate
export PYTHONPATH=src:.:tests
python benchmarks/p0_engine_qualification.py --download
python benchmarks/p0_engine_qualification.py --all --offline
```

Use `--model` and `--suite correctness|streaming|state|lifecycle|admission|benchmark` to select work. Filtered mandatory cases stay NOT_RUN and cannot fully qualify a model. Benchmarks require completed prerequisite gates; isolated performance invocation cannot bypass them.

One native worker runs at a time. The watchdog uses the lower recommended/configured/75%-RAM ceiling. Every case and worker has a bounded timeout. Actual native failures are retained; no skipped case is counted as passing.

## Repairs and boundaries

- Request ContextVars now survive lazy device execution and submitted reference requests, enabling isolated observation.
- Explicit floating conversion validates both native affine 4-bit and 8-bit payloads; optimized INT8 eligibility remains unchanged.
- Active requests finish result construction from their admitted bundle identity during Engine shutdown.
- Incremental decoding holds incomplete UTF-8 replacement tails and flushes final text. HTTP forwards that text without adding a model token or changing SSE framing.
- Async cancellation now signals the admitted request handle; already-open streams stop at the shutdown boundary.
- Prompt context validation precedes cache reservation and allocation.
- Explicit FP16 affine 4-bit hybrid execution uses shape-stable native projections and FP32 attention reductions; full and cached execution pass the original tolerances.
- Qwen3.5 publisher weight names, convolution axes and vision patch axes are validated and normalized for strict loading; only text execution is in scope.
- Removed an obsolete package license classifier that prevented wheel builds with the installed setuptools; the Apache-2.0 license is unchanged.
- Harness fixes: exact tokenizer-roundtrip prompt lengths, clone/inspection on the device owner, truncated payload rejection, evaluated metadata capture, workload-scoped diagnostic exits, and sampled per-request peak RSS.
- H8192 admission repair: one owner-thread synchronize/reclaim/resample retry under the unchanged ceiling; fresh allocator observations and separate dense hybrid carry-growth accounting. Full-width prefill remains unchanged. See `P0_H8192_ADMISSION_FIX.md`.

Earlier failure/interruption cohorts remain under `qualification/results/p0_engine/runs/`. The known experimental paged INT8 B2 issue is outside this reference-only qualification.

## Validation

Backend-free suite: **1061 passed, 35 skipped**, two inherited dependency warnings. Opt-in pinned-checkpoint state regressions: **5 passed**; H8192 regressions: **4 passed**, verifying the full cohort’s 24 executed cold/warm/cancellation/cleanup/repeated/independent-process cases. Scoped Ruff: **65 inherited findings, zero added**; a whole-checkout scan additionally finds 19 issues under unrelated `proofcart_starter/`, which is left untouched, plus 30 findings in archived uncommitted baseline diagnostic scripts under result artifacts. None are added to code, tests or benchmark tooling. Wheel build and working/staged diff checks pass. These checks supplement native results; skipped tests are not passes.

Native lifecycle workload includes 100 generation cycles on one loaded model and 20 load/close cycles with five additional requests per cycle. Memory compares the median first three and last three warmed reload samples against 128 MiB RSS and 32 MiB MLX-active limits. Raw retained-storage and queue counters are separate.

No commits, pushes, tags, dependency upgrades, public promotion or default backend changes are performed. Existing UI/logo/port changes are preserved.

## Regression evidence

The original four H8192 rejections were reproduced cold/warm/cleanup before repair. The current cohort reruns every mandatory gate, including five-repeat 8K benchmarks and I01–I06. An earlier diagnostic cohort was interrupted before its first measured request completed; it remains historical and supplies no qualification.

| Defect | Correction and regression | Native evidence / final gate |
|---|---|---|
| Private observer lost at thread boundary | `engine/device_executor.py`, `engine/public.py`; `test_lazy_device_iterator_preserves_request_context` | Current A08 / D01 |
| FP16 rejected valid affine 4-bit weights | `engine/modes.py`; `test_fp16_conversion_preserves_native_affine_payload` | Red cohort `1792167370640237000`; current LFM2/Qwen3.5 A03/A05 |
| Closing an admitted request read a closed public accessor | `engine/public.py`; `test_close_with_running_submission_does_not_read_closed_public_model` | Red `1792167423200872000`; current F09 |
| Incremental byte decoding duplicated/replaced Unicode | `text/generation.py`, `server/generation_engine.py`; UTF-8/HTTP terminal-flush tests in `test_engine_observation.py` | Red `1792168431681037000`; current C01-C10 |
| Async cancellation did not signal native work | `engine/public.py`; `test_async_task_cancellation_signals_running_reference_request` | `async-cancel-before/` and `async-cancel-after/`; current F11 |
| Open direct streams continued after Engine.close | `engine/public.py`; `test_active_direct_stream_observes_engine_shutdown` | `direct-stream-before/` and `direct-stream-after/`; current F12 |
| Invalid context reserved memory before validation | `text/generation.py`; `test_invalid_context_is_rejected_before_cache_handle_allocation` | Red `1792170967726794000`; current B05 |
| Hybrid full/cached arithmetic changed with row/mask shape | Shared native projection/attention helpers and LFM2/Qwen3.5 models; projection and mask regressions | LFM2 red `1792172610054434000`, Qwen3.5 red `1792173848660884000`; current D/E and opt-in checkpoint tests |
| Qwen3.5 publisher names and tensor axes differed | `models/vision_language/qwen3_5/weights.py`; publisher-layout tests in `test_native_qwen_multimodal_architectures.py` | Red `1792172610054434000`; current Qwen3.5 A03 |
| Installed setuptools rejected obsolete license classifier | `pyproject.toml`; wheel build with `--no-deps --no-build-isolation` | Local validation build log; Apache-2.0 license unchanged |

Paths to red cohorts are relative to `qualification/results/p0_engine/runs/`; named before/after probes are beneath `qualification/results/p0_engine/`. The final immutable cohort above contains current native reruns. Harness-only failures and interruptions are preserved separately from model failures.
