# Bounded Qwen3 native M4 validation

## Qwen3 upstream native adoption — candidate, 20 October 2026

**Parallel port implemented; canonical model retained.** The candidate adopts pinned Apple mlx-lm v0.31.3 projections, QKNorm, RoPE, fused SDPA, SwiGLU, residuals and tied head as first-party native MLX code. Existing registry/default model, Engine/cache ownership and API paths are preserved. No production mlx-lm dependency was added. Candidate scope is unpadded contiguous reference B1 with dense KV; unsupported modes fail explicitly. See [qwen3-upstream-native-adoption.md](../qualification/archived-evidence.md).

Final current-source backend-free regression: **1703 passed / 48 skipped / 3 inherited warnings**. One newly introduced metadata-probe guard assumption failed the first full run and was corrected; its original failure receipt remains. Candidate Metal comparison, five-repeat frozen v1 quality, native APIs/lifecycle and matched performance are **NOT_RUN, awaiting the explicit authorization required by Phase 5**. A concrete bounded runner and seven existing-framework quality profiles are prepared. Earlier native/quality/parity receipts below retain their original source and do not qualify this candidate. No promotion, packaging or source-inclusion approval is inferred. Alpha **NO-GO**, version **0.1.0a1**; owner inclusion remains pending.

Native source SHA `280317b00742ccd32015a9c823cd303e2d8bde0598111a283c83424905407682`; broader validation SHA `3d453f4acfb73d08da393defc6519092f093419ff1decfab1e03d8903eb9a78c`. Evidence: `.venv/alpha_0_1/qwen3-adoption-20261020-1/`. Resume at the pending candidate Metal authorization; do not restart A0–A10.

## Independent mlx-lm comparison — 20 October 2026

**Decision D: INCONCLUSIVE_OR_EXECUTION_PENDING; all 14 authorized runs completed.** Current-source Qwen3 Q8 matches pinned mlx-lm 0.31.3 in every visible token, text and finish reason, including all 12 frozen v1 failures. Numerical parity **FAILS** the unchanged FP16 gate: 2/140 full-logit boundaries and 498/2352 KV components pass, despite 140/140 greedy tokens and all cache positions agreeing. Cached/fresh and reset logits reproduce exactly within each implementation. No native production defect is confirmed; production source and tolerances were preserved.

See [qwen3-mlx-lm-parity.md](../qualification/archived-evidence.md) for the 14-operation audit, exact upstream/source hashes, layer diagnostics, harness-only EOS bookkeeping repair and measured limitations. New regressions: 43 passed; diagnostic adapter 8 passed (7 pass/1 fail before its copy-return fix). Original Metal reports are preserved and the corrected visible-ID readout is derived on CPU from saved raw IDs. All workers exited within bounds. Five-repeat v1 quality remains 15/27, FAIL. Release **NO-GO**, version **0.1.0a1**; source inclusion remains explicitly pending and distributions NOT_RUN. Following entries retain their original scope and source.

Latest independent qualification: **Qwen3 quality FAIL, bounded coverage PASS**, 15/27 distinct cases passing across five deterministic repetitions. Native execution/lifecycle and stream/generate consistency passed. See [release-readiness.md](release-readiness.md). The results below retain their stated earlier source identity and scope. No alpha support promotion.

**SCOPED_NATIVE_WORKFLOW_PASS; alpha NO-GO.** Authorized checkpoint: `mlx-community/Qwen3-0.6B-8bit` at `11de96878523501bcaa86104e3c186de07ff9068`. Affine Q8/group64; safetensors SHA256 `3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397`. No other checkpoint executed.

Development SHA `8c57a6aff778f210ea0d212d0648df75dd3b920d4f96d93355e7edab268fb7c0`; broader validation SHA `a9a3f23771a2a116bc3bfaf9953341163aeb2faf573f2346feef230fe11e557c`; native source-provenance SHA `554a1a98fd16e7cb6530b9fa8e81cfe3401034752f07e5d5808ad172f39bcdf4`. These are three different hashing scopes. HEAD remains `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`; version remains `0.1.0a1`. Approved release-source SHA is unavailable.

Observed host: Apple M4, 34359738368 RAM bytes, macOS 15.7.9, Python 3.12.11, MLX 0.32.2, MLX-One 0.1.0a1. tokenizers 0.23.2, huggingface-hub 1.30.0, safetensors 0.8.0, uvicorn 0.52.4. Real GPU preflight result 1240; Metal available. Configured ceiling 24 GiB; actual enforced process ceiling 22906503168 bytes (lower MLX recommended working set). Native generation workers were supervised at 120 seconds; each request <=512 input / <=64 output tokens. Actual outputs used 32 tokens or fewer. No download, conversion, optimization, stress or concurrency sweep.

## Native results and scope

| Check | Result | Observation |
|---|---|---|
| Native checkpoint loading and forwards | PASS | MLX-One Qwen3 mathematics on MLX Metal; no external inference engine imported in inspected Python/ASGI workers |
| Precision | PASS in observed scope | Actual FP16 + U32 parameters; FP16 logits and all 28 KV layers in Python/HTTP; Q8 pack metadata/hash/layout verified |
| Backend | PASS | Explicit reference_dense, B1, native_math.reference, no fallback in Python/HTTP; CLI uses preserved native generation |
| Greedy repeatability | PASS | Two prompts; two measured repeats per prompt with exact token equality |
| Tokenizer/chat template | PASS | Immutable asset hashes; CPU token IDs; native chat rendered with reasoning off and answered Paris |
| Existing numerical parity | PASS | Eight forced steps; existing atol=rtol=0.01; max/mean error 0; greedy agreement 1; all 56 state-carry checks true; RNG/cancel isolation true |
| Factual smoke quality | PASS, limited | France continuation begins Paris; arithmetic begins 4; full continuation below shows repetition/irrelevant elaboration |
| General task quality / coverage | NOT_RUN | No independent versioned coverage protocol/golden dataset established; not ALPHA_SUPPORTED |
| Python Engine/stream | PASS | Real model, token IDs, repeated generation and recovery |
| CLI generation / streaming | PASS | 32-token JSON generate, 16-token streamed generate; default native floating dtype not separately certified as FP16 |
| ASGI HTTP / live TCP HTTP | PASS | Existing FastAPI app and CLI uvicorn server; actual model; 401 missing auth, 404 wrong alias (ASGI), correct chat response |
| SSE | PASS | Content Paris, finish event and [DONE], matches JSON; real TCP client event times retained |
| Cancellation | PASS | Python cancel after first token; live TCP client closes after content; handles/reservations return to zero; next response correct |
| Unload/cleanup | PASS | Engine close twice; model weak-reference released; 0 leases, 0 handles/reservations; 8 active / 0 cached MLX bytes |
| Controlled server shutdown | PASS | Application shutdown complete; supervisor sent SIGTERM, process exit -15 retained |

The first sandboxed preflight was BLOCKED by sysctl denial before model execution (`first-native-1792433022457397000`). Authorized execution outside that restriction passed. Earlier successful receipts at the preceding source remain historical; the final-source repeats below are authoritative. The incomplete CPU cache entry remains a discovery failure, not a native correctness failure.

## Actual outputs

Prompt 1: `The capital of France is`. Prompt IDs: `[785, 6722, 315, 9625, 374]`. Greedy output IDs: `[12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 8585, 5429, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 8585, 3033, 13, 576, 6722, 315, 9625]`.

```text
 Paris. The capital of France is also the capital of the French Republic. The capital of France is also the capital of the French government. The capital of France
```

Prompt 2: `2 + 2 =`. Prompt IDs: `[17, 488, 220, 17, 284]`. Greedy output IDs: `[220, 19, 11, 773, 279, 2629, 374, 220, 19, 382, 3983, 3783, 11, 279, 3491, 2727, 330, 1782, 2629, 315, 279, 5109, 1, 892, 525, 279, 5109, 304, 279, 1156, 2802, 13]`.

```text
 4, so the sum is 4.

But wait, the problem says "the sum of the numbers" which are the numbers in the first row.
```

Live HTTP user prompt: `What is the capital of France? Answer with only the city name.` Rendered chat: 26 input tokens; response text **Paris**, one output token. Full wire responses and chat-token IDs are retained in the ignored reports. CLI output for the first prompt matched the recorded continuation, finish_reason=length, 32 generation tokens; generation wall 1.043s. Deterministic behavior does not imply high answer quality on unrelated prompts.

## Performance disclosure

| Metric | Observed value | Scope |
|---|---|---|
| Fresh-process model load | 0.881463s | Python load + Engine/evaluation; filesystem cache warm |
| Warm TTFT | 77.0125–86.9052ms | Four measured requests, two prompts |
| Prefill | 96.2134–108.5928 tokens/s | 5 prompt tokens each; existing synchronized forward observer |
| Decode | 25.4123–26.6391 tokens/s | 32 output tokens each; existing RequestTiming |
| Python MLX peak | 729333940 bytes | Worker process |
| Python sampled process RSS peak | 871219200 bytes | Existing supervisor, not a system-wide memory maximum |
| ASGI HTTP sampled RSS peak | 871448576 bytes | Separate process |
| After unload | 8 active / 0 cached MLX bytes | Model released; tiny runtime allocation remains |
| HTTP decode tokens/s | null | One output token; no meaningful inter-token decode interval |

No minimum speed, competitive comparison, disk-cold load, thermal steady-state or large-context qualification is claimed. Earlier attempts' measurements are not pooled into these ranges.

## A10 receipt implementation

`src/mlx_one/task_qualification_worker.py` now probes cancellation/recovery for native B1 text, records idempotent close and cache/lease cleanup, drops the bundle/engine, observes the model weak reference and records post-unload MLX memory. `task-lifecycle.json` is mirrored in the checksummed measurement; gates are recomputed after cleanup. Missing cancellation in other modes stays pending. Ground-truth presence and observed case counts are recorded without setting quality coverage true. Five new focused tests reject ignored cancellation, leaked handles, changed recovery and manufactured quality coverage; focused 90 pass, whole regression 1671 pass / 47 skip / 3 warnings, changed-file Ruff pass.

The existing task runner was exercised with B1, one warm-up and one measured repetition for two short prompts, then cancellation/recovery. It recorded task_lifecycle=true, parity/native_execution/checkpoint_scope/telemetry/cleanup=true. Aggregate task gates remain **failed**, five_repetitions=false; quality/coverage/service/performance/fairness fields remain pending. This bounded receipt verifies lifecycle instrumentation; it is not a completed benchmark/quality qualification. No existing gate or tolerance changed, no registry promotion occurred. Independent quality/coverage fixtures remain the earliest A4/A10 work.

## Reproduction and evidence

The following guarded command reruns only this already authorized bounded checkpoint workflow, with new directories and existing supervision:

```sh
.venv/bin/python .venv/alpha_0_1/resume-release-blockers-20261019-1/run_final_native.py --allow-metal
```

Actual implemented standalone CLI options (cached model ID is equivalent to the private local snapshot path used in the worker):

```sh
.venv/bin/python -m mlx_one.cli generate mlx-community/Qwen3-0.6B-8bit \
  'The capital of France is' --revision 11de96878523501bcaa86104e3c186de07ff9068 \
  --offline --max-tokens 32 --temperature 0 --seed 0 --json-output

.venv/bin/python -m mlx_one.cli serve mlx-community/Qwen3-0.6B-8bit \
  --revision 11de96878523501bcaa86104e3c186de07ff9068 --offline \
  --host 127.0.0.1 --port 8765 --alias alpha-qwen3 --api-key local-qualification-only \
  --context-length 512 --parallel 1 --timeout 120 --no-warmup --no-cache-prompt \
  --prefix-cache-mib 0 --inference-mode reference_dense --execution-dtype float16 \
  --prefill-chunk-size 512 --max-prefill-tokens 512 --max-batch-tokens 512 \
  --queue-size 2 --reasoning off
```

Standalone CLI commands do not install the watchdog; use the guarded workflow for the authorized bounded checks. It starts the server on an ephemeral port, uses a local qualification key, exercises real requests and shuts it down. Do not use the older P1 correctness --backends reference_dense command as a pure B1 smoke: its selected tests include B1/B2/B4 and experimental batching. Existing broader numerical suites remain intact and unexecuted in this scope.

Final authoritative ignored evidence:

- `run_first_native_cli.py`: `.venv/alpha_0_1/resume-release-blockers-20261019-1/first-native-1792433771457120000`.
- `run_native_workflow.py`: `.venv/alpha_0_1/resume-release-blockers-20261019-1/native-workflow-1792433774732209000`.
- `run_live_http.py`: `.venv/alpha_0_1/resume-release-blockers-20261019-1/live-http-1792433791492537000`.
- `run_task_lifecycle.py`: `.venv/alpha_0_1/resume-release-blockers-20261019-1/task-lifecycle-1792433799685333000/native`.
- Backend-free regression: `.venv/alpha_0_1/resume-release-blockers-20261019-1/final-validation/backend-free-1792433779259685000-8208d610` (JSON/CSV/JUnit/logs/checksum manifest).
- Native aggregate: `.venv/alpha_0_1/resume-release-blockers-20261019-1/final-native-1792433770647560000/result.json`.
- Exact hashes/preservation: final-development-source.private.json, final-validation-source.private.json, preservation-native-final.private.json and native-public-summary.json in the same ignored root.
- Inclusion review: proposed-release-manifest.final.private.json, 324 candidates / 11 local-only, approved=false. Original proposals preserved.

Final disposition: local native text usability demonstrated, full quality/coverage NOT_RUN, real floating-weight native evidence HARDWARE_PENDING, wheel/sdist/install BLOCKED_ON_RELEASE_INCLUSION, release **NO-GO**. No process remains running. Next owner action is approval/rejection of the final private inclusion proposal; separate floating-checkpoint execution requires its own authorization. Preserve all current/older receipts and resume from the earliest unfinished quality gate without restarting completed stages.
