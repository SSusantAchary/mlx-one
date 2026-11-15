# Serving overhead correction

The activation-peak lookup used to build hardware/source identities and a full
native runtime report on every forward whenever a qualification registry existed.
GPT-OSS MLXBoost serving did not enable the older `fast_decode` shortcut, so an
empty activation-calibration registry still incurred this work.

The lookup now inspects live registry records first, in all execution modes.
With no activation-peak records, it immediately uses the same conservative
analytical forecast. When a record appears, full scope, source, dimensions,
freshness and evidence validation still runs. No negative lookup is cached.
Memory admission, host telemetry, reservations and cache accounting are unchanged.

Validation on 14 November 2026:

- 141 dispatch, calibration, paged/hybrid and GPT-OSS tool tests passed.
- 99 server, scheduling, memory, admission and GPT-OSS configuration tests passed.
- Ruff and whitespace checks passed for the changed code.
- A 20-call isolated CPU benchmark with a stubbed native report and no loaded
  model reduced empty-calibration lookup time from 18.0678 ms/call to 0.00244
  ms/call. Both returned the identical 977664-byte analytical forecast. These
  measurements exclude full model reporting and are not decode benchmarks.

Hardware before/after testing completed sequentially with only one model loaded.
The installed original and local patched package differ in only `engine/pressure.py`.
Apple M4 / 32 GiB, Python 3.13.4, MLX 0.32.2, mixed MXFP4/Q4 weights, BF16 KV,
16,384 configured context, parallel 1, prefix reuse off, `gptoss_mlx_lm` on.
Each server received three identical non-streaming requests: 88 input tokens,
256 generated tokens, greedy sampling with seed 0 and `tools: []`. Startup model
warmup was enabled. All six response messages have the same SHA-256.

| Median of three requests | Original | Patched |
|---|---:|---:|
| Decode throughput, first-to-last model token | 19.39 tok/s | 38.14 tok/s |
| Whole engine request throughput | 18.60 tok/s | 35.71 tok/s |
| Engine request duration | 13.762 s | 7.168 s |
| Recorded decode-forward time | 6.491 s | 6.414 s |
| First model token | 561 ms | 483 ms |
| First visible text | 7.218 s | 3.830 s |

The GPU-forward duration is nearly unchanged; reduced host work explains the
measured serving improvement. First-visible-text timing remains separate from
first-model-token timing; output handling and reasoning affect visibility.

The 30+ tok/s target is demonstrated for this bounded workload. This is not a
matched llama.cpp comparison, full-context qualification, or Kilo tool-execution
benchmark. The prompt reaches the 256-token output cap. No cache quantization,
model arithmetic, or memory guards were changed. Raw local response records
are in `local/serving-performance/`; the shareable request, per-run summaries,
source digest and limitations are in [the result JSON](serving-overhead-results.json).

To test the local patch after stopping other model servers, use the active `mlx`
Python environment from this checkout:

```bash
PYTHONPATH="$PWD/src" python -m mlx_one.cli serve \
  "$HOME/mlx-one-models/gpt-oss-20b-MXFP4-Q4" \
  --offline --host 127.0.0.1 --port 8888 \
  --alias gpt-oss-20b --api-key kilo-local --context-length 16384 \
  --inference-mode mlxboost_b1 --mlxboost-gptoss-mlx-lm \
  --parallel 1 --prefill-chunk-size 2048 --prefix-cache-mib 0 \
  --no-cache-prompt --cache-type-k bf16 --cache-type-v bf16 --timeout 0
```

This source override is for testing unpushed code. A normal PyPI installation
continues using the published implementation until a subsequent release.

## Additional API checks

The running patched source was exercised through HTTP on 14 November 2026:

- Streaming text: exact `OK` response, normal stop, valid SSE events and `[DONE]`.
  Decode throughput was 36.76 tok/s; first visible text arrived in 1.217 seconds.
- Automatic function selection: called `add_numbers` with integer arguments
  `a=17`, `b=25` and returned `finish_reason=tool_calls`.
- Tool-result continuation: supplied a simulated result of 42; the model returned
  “The sum of 17 and 25 is **42**.” with a normal stop and no further tool calls.
- Forced/named function selection: explicitly rejected with HTTP 400 capability
  error. This existing interface restriction remains; it is not an auto-tool test.

These are bounded API checks, not a full Kilo agent session. An additional code
answer became repetitive and reached the 512-token output limit; its comparison
against the installed unpatched package produced exactly identical visible text
(SHA-256 `a2d1671d48f3a950b2fa49b57dc9b652eb11e20cd17f831d7242b0d7dbbbf935`).
Decode speed was 18.10 tok/s before versus 36.48 tok/s after. This confirms an
existing quality failure, rather than a regression from the overhead patch;
this patch does not fix that failure. See [functional results](serving-functional-results.json).

## Nonzero-temperature sampling fix

The first overhead correction left the nonzero-temperature sampler doing a
Python object sort across the entire vocabulary for each generated token.
The UI defaults to temperature 0.7 and top-p 0.9, so greedy benchmarks did not
represent that path.

Ranking and cumulative probabilities now use NumPy, retaining Python's seeded
RNG and normalization sums. Exactly representable nonnegative BF16 probabilities
use stable uint16 key ranking without rounding or changing weights. Other values
use stable numeric ranking. Signed-zero ties retain original token order.
Float64 cumulative sums retain the old sequential Python-float CDF boundaries.
The existing temperature-zero argmax path and memory guards are unchanged.

Matched HTTP test: M4 / 32 GiB, Python 3.13.4, MLX 0.32.2, 85 input tokens,
128 generated tokens, temperature 0.7, top-p 0.9, seed 0, no top-k, same server
configuration as above. Both sides already include the calibration-lookup fix.
Three measured runs per side produced identical response-message hashes.
The first baseline run overlapped small-array MLX checks and was excluded;
baseline runs 2–4 are compared with final sampler runs 1–3.

| Median | Original sampler | Optimized sampler |
|---|---:|---:|
| Decode throughput | 12.90 tok/s | 32.19 tok/s |
| Whole-request throughput | 12.35 tok/s | 28.95 tok/s |
| Request duration | 10.365 s | 4.421 s |

A separate 512-output-token request at temperature 0.7 measured **32.90 decode
tok/s**, **31.91 overall tok/s**, and **517 ms first-model-token latency**.
This is one longer run, not a median or a matched longer baseline.

Validation: 192 regression tests passed, including 33 sampler compatibility tests;
30 additional actual MLX float32/BF16 seed comparisons matched the old sampler.
Tests cover full GPT-OSS vocabulary size, ties, signed zero, top-k/top-p, CDF
boundaries, invalid mass and RNG state. These are bounded compatibility checks,
not exhaustive proof across every input or supported Python/NumPy version.
See [sampling results](sampling-performance-results.json) for the request,
per-run values, source hashes and limitations. Existing repetitive output on
one earlier prompt remains a separate issue; sampling optimization does not
claim to fix model quality.

## Output allowance and sustained decode follow-up

The GPT-OSS UI recommendation is now 8192 output tokens, bounded by configured
context with input space reserved; manually edited limits remain untouched.
Streamed text updates are coalesced over 80 ms before Markdown/DOM rendering,
with a final flush on completion, cancellation and errors. Pending chunks are not
dropped. Progress-only updates do not needlessly invalidate Markdown content.

For the existing `gptoss_mlx_lm` recipe only, B1/BF16 full-history single-query
attention can omit a redundant all-zero mask while using stock MLX SDPA and the
same sinks, scale and cache values. Prefill, padding, other layouts and other
recipes retain their reference path. A sliding-window cropping experiment changed
the seeded response and was rejected; sliding attention remains unchanged.
Fifteen Metal tests cover exact BF16 equality of eligible full-history outputs
and fallback eligibility. No memory protections or precision settings changed.

A fixed 83-input/1024-output request at temperature 0.7, top-p 0.9 and low effort
measured **19.38 → 28.91 decode tok/s**, with identical visible response hashes.
This is a single sequential before/after observation, not a thermal-controlled
benchmark or proof of general bitwise identity. A subsequent live UI response
using the new 8192-token budget stopped normally at 2016 generated tokens,
but measured only **15.39 decode tok/s** and **15.33 overall tok/s**. Its visible
text started after 2.345 seconds as observed by the browser. Sustained
long-response throughput therefore remains unresolved. Do not generalize the
1024-token improvement to 4096/8192-token outputs or full-context operation.
A subsequent browser-free request also stopped normally at 2016 tokens, averaging
**21.11 decode tok/s**; its rolling speed fell from about 32.6 to about 12.5 tok/s.
This shows the slowdown also occurs without rendering, but sequential runs do not
isolate browser cost or establish the underlying bottleneck.
See [all measurements and limitations](long-response-results.json).

Validation: 135 backend regression tests, 15 Metal attention tests, 14 UI tests,
9 browser scenarios and Svelte checking without errors or warnings passed.
