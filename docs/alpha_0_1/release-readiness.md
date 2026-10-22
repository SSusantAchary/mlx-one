# Remaining release qualification — measured result

## Qwen3 upstream native adoption — candidate, 20 October 2026

**Parallel port implemented; canonical model retained.** The candidate adopts pinned Apple mlx-lm v0.31.3 projections, QKNorm, RoPE, fused SDPA, SwiGLU, residuals and tied head as first-party native MLX code. Existing registry/default model, Engine/cache ownership and API paths are preserved. No production mlx-lm dependency was added. Candidate scope is unpadded contiguous reference B1 with dense KV; unsupported modes fail explicitly. See [qwen3-upstream-native-adoption.md](../qualification/archived-evidence.md).

Final current-source backend-free regression: **1703 passed / 48 skipped / 3 inherited warnings**. One newly introduced metadata-probe guard assumption failed the first full run and was corrected; its original failure receipt remains. Candidate Metal comparison, five-repeat frozen v1 quality, native APIs/lifecycle and matched performance are **NOT_RUN, awaiting the explicit authorization required by Phase 5**. A concrete bounded runner and seven existing-framework quality profiles are prepared. Earlier native/quality/parity receipts below retain their original source and do not qualify this candidate. No promotion, packaging or source-inclusion approval is inferred. Alpha **NO-GO**, version **0.1.0a1**; owner inclusion remains pending.

Native source SHA `280317b00742ccd32015a9c823cd303e2d8bde0598111a283c83424905407682`; broader validation SHA `3d453f4acfb73d08da393defc6519092f093419ff1decfab1e03d8903eb9a78c`. Evidence: `.venv/alpha_0_1/qwen3-adoption-20261020-1/`. Resume at the pending candidate Metal authorization; do not restart A0–A10.

## Independent mlx-lm comparison — 20 October 2026

**Decision D: INCONCLUSIVE_OR_EXECUTION_PENDING; all 14 authorized runs completed.** Current-source Qwen3 Q8 matches pinned mlx-lm 0.31.3 in every visible token, text and finish reason, including all 12 frozen v1 failures. Numerical parity **FAILS** the unchanged FP16 gate: 2/140 full-logit boundaries and 498/2352 KV components pass, despite 140/140 greedy tokens and all cache positions agreeing. Cached/fresh and reset logits reproduce exactly within each implementation. No native production defect is confirmed; production source and tolerances were preserved.

See [qwen3-mlx-lm-parity.md](../qualification/archived-evidence.md) for the 14-operation audit, exact upstream/source hashes, layer diagnostics, harness-only EOS bookkeeping repair and measured limitations. New regressions: 43 passed; diagnostic adapter 8 passed (7 pass/1 fail before its copy-return fix). Original Metal reports are preserved and the corrected visible-ID readout is derived on CPU from saved raw IDs. All workers exited within bounds. Five-repeat v1 quality remains 15/27, FAIL. Release **NO-GO**, version **0.1.0a1**; source inclusion remains explicitly pending and distributions NOT_RUN. Following entries retain their original scope and source.

**NO-GO: independent bounded Qwen3 quality FAIL; complete dataset coverage PASS.** The 27-case protocol was frozen before execution at SHA256 `04570dd4ff607017e0d9d890fe63f62cc3a21af13469b33167bb52e8eac4b642`. Criteria and prompts were not changed after seeing outputs. Five measured repetitions of every case completed: **75/135 case-repetitions passed; 15/27 distinct cases passed all criteria; 12 distinct cases failed**. All repetitions were deterministic and stream/generate agreed, including declared invalid-input errors.

Development SHA `08826936f2d0bc8e3cc68cd48868976cece9bb5fb29ae9fc21e7ff8639f54c52`; broader validation SHA `a0e8934260dc9efa3b31bbc2c149b5792783f404f325d96687372b169f72dcb5`; native source-provenance SHA `65f462cfe1bfb2ec3c57c549dbf56d4a12ea5df684a2bb3116493b40e8264af7`. HEAD `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`, version `0.1.0a1`. Approved release-source SHA remains unavailable.

Source changes are limited to the existing quality_dataset/task qualification modules and worker, their existing dataset tests, and one fixture JSON under tests/alpha_0_1. No inference mathematics, runtime, model registry, attention or historical receipt changed. Existing long-context held-out requirements remain untouched. The task correctness validator now enforces the already-existing five-repetition gate as well as the other correctness checks; its earlier omission was confirmed during inspection.

## Independent readiness axes

| Axis | Result | Scope / remaining evidence |
|---|---|---|
| Native inference correctness | PASS at measured scope | Seven native Metal workers exited 0; fresh reference execution matched every measured token/text/error result across five repeats; B1 forwards observed. Prior forced-logit parity remains bound to its earlier source; no new external numerical oracle claim. |
| Task quality | FAIL | Independent exact factual/arithmetic/instruction/structured/termination anchors, 15/27 cases passed; no threshold relaxed. |
| Quality coverage | PASS, bounded protocol | All 27 required cases, ten categories, five measured repetitions, complete outputs and stream evidence. Not a broad held-out or long-context benchmark. |
| Precision / checkpoint | Q8 native execution observed; NOT_QUALIFIED | Qwen3 immutable revision `11de96878523501bcaa86104e3c186de07ff9068`, packed U32/FP16 parameters and FP16 cache trace components; floating/Q4/other Q8 native runs remain NOT_RUN. |
| Python / CLI / HTTP | Python stream+generate PASS now; CLI/HTTP prior-source PASS | Native CLI/live HTTP/SSE were not unnecessarily repeated; their original source identities remain explicit in native-qwen3-validation.md. Production paths unchanged; current backend-free API tests pass. |
| Performance / lifecycle | Measured / lifecycle PASS | Seven unload/cancel/recovery receipts passed. Service completion deadline met; no competitive comparison, HTTP-load or fairness qualification. |
| Release-source inclusion | PENDING | Owner explicitly replied “Keep inclusion pending while I review”; no contents approved. |
| Wheel / sdist / isolated install | NOT_RUN | Blocked on exact source/member approval; editable-install evidence is not wheel evidence. |
| Final alpha | NO-GO | Required quality failed, floating cell unexecuted, release inclusion/install incomplete. |

## Coverage and failure trace

| Category | Passed case-repetitions | Total | Result |
|---|---:|---:|---|
| arithmetic | 0 | 15 | FAIL |
| chat | 10 | 10 | PASS |
| eos | 5 | 10 | FAIL |
| factual | 5 | 15 | FAIL |
| instruction | 5 | 15 | FAIL |
| invalid | 15 | 15 | PASS |
| reasoning | 5 | 15 | FAIL |
| stop | 5 | 10 | FAIL |
| structured | 10 | 15 | FAIL |
| unicode | 15 | 15 | PASS |

Exact first measured output for every failed case (the same failure repeated five times):

| Case | Independent expected answer | Actual text | Finish |
|---|---|---|---|
| fact-planet | `{"text": "Jupiter"}` | `"Mercury."` | stop |
| fact-water | `{"text": "H2O"}` | `"3H2O"` | stop |
| add | `{"text": "12"}` | `"7 + 5 = 12\n\nAnswer: **12**"` | stop |
| multiply | `{"text": "54"}` | `"9 × 6 = 54\n\nAnswer: **54**"` | stop |
| subtract | `{"text": "13"}` | `"20 - 7 = 13\n\nAnswer: **13**"` | length |
| count | `{"text": "5"}` | `"The box contains **3 red balls** and **2 blue balls**, so the"` | length |
| order | `{"text": "8"}` | `"The integer 8 is larger than 3.  \n\nAnswer: **8**"` | length |
| lowercase | `{"text": "mixed"}` | `"midxed"` | stop |
| select | `{"text": "blue"}` | `"The second word from the list [red, blue, green] is **blue"` | length |
| json-types | `{"json": {"ok": true, "count": 2}}` | `"```json\n{\n  \"ok\": true,\n  \"count\": 2"` | length |
| eos-ok | `{"text": "OK"}` | `"OK."` | stop |
| stop-end | `{"text": "one"}` | `"END two"` | stop |

Factual and literal-copy failures are not explained away as numerical PASS. Some responses also exhausted the predeclared 16-token budget; markdown/explanation already violated exact-answer instructions. A larger budget or changed prompt belongs to a new versioned experiment, not a rewrite of this FAIL. These observations alone do not identify a confirmed model-math defect; no speculative runtime fix or optimization was made.

## Why aggregate gates fail or remain pending

The previous one-repetition receipt remains unchanged, with five_repetitions=false. All seven new shard receipts have five_repetitions=true, parity=true, native_execution=true, checkpoint_scope=true, telemetry=true, cleanup=true, task_lifecycle=true and service_limits=true. They remain aggregate failed because quality=false. Shard task_quality_coverage stays pending because each shard is only part of the frozen dataset. The aggregate scorer, on the same source/checkpoint/protocol, observes all 27 cases and sets bounded coverage true; this is a derived readout, not registry promotion or rewritten worker evidence.

matched_reference and measured_advantage are pending because no paired competing workload/reference run or minimum ratio was declared. fairness is pending because no multi-request scheduling/fairness workload was run; a serial B1 quality workload cannot establish it. dynamic_membership and actual_batched_forward keep the existing B1 gate behavior. Missing performance/service-comparison evidence is retained; no gate is removed or replaced by N/A. The measured p95 deadline is native task completion within the existing 120s bound, not an HTTP service load claim. There is no competitive-speed alpha gate.

## Performance and implementation verification

```json
{
  "cold_load_seconds_range": [
    0.39224679197650403,
    0.4200852500507608
  ],
  "warm_ttft_ms_range": [
    161.7041249992326,
    296.03658406995237
  ],
  "decode_tokens_per_second_range": [
    15.20375309814537,
    22.68485606771676
  ],
  "max_task_response_p95_ms": 2114.001445611939,
  "peak_mlx_bytes": 2404673168,
  "peak_sampled_process_rss_bytes": 866746368,
  "max_after_unload_active_bytes": 8,
  "max_after_unload_cached_bytes": 0,
  "scope": "seven processes; existing Engine default prefix reuse; synchronized trace observer; paired generate excluded from token timing; no speed comparison"
}
```

These are quality-harness measurements, with the existing Engine default dense prefix reuse and synchronized forward tracing, and must not be pooled with earlier prefix-disabled usability measurements. One-token outputs have no decode interval and retain null throughput. Actual prompt/forward/token timing records are retained per shard. Fresh-process load excludes Engine construction; filesystem cache state is uncontrolled.

Final backend-free regression: **1680 passed / 47 skipped / 3 warnings**, 56.70s. Focused scorer/task/receipt tests: **30 passed**. Changed-file Ruff PASS; git diff --check PASS. All entry source remains present; four existing source/test files changed and one fixture added. All 203 prior native evidence files and all 6494 historical files are hash-identical; staged contents unchanged.

## Reproduction and additional checkpoint plans

Frozen protocol: [text-quality-protocol.md](text-quality-protocol.md). Dataset: `tests/alpha_0_1/fixtures/text-quality-v1.json`. Current native receipts: `.venv/alpha_0_1/quality-gates-20261019-1/quality-run-1792434914820604000`. Each shard uses existing native_tasks profiles and worker; aggregate scoring reuses quality_dataset. The existing Qwen3 authorization persists; no new checkpoint is authorized.

```sh
.venv/bin/python .venv/alpha_0_1/quality-gates-20261019-1/run_quality.py --allow-metal
```

That command creates unique bounded receipts and exits 1 on semantic quality FAIL. Native workers themselves exited 0. Do not rerun to search for a passing sample or edit the frozen fixture. Exact token IDs, prompts, reference outputs and all five repeats are in outputs-aggregate.json and the per-shard outputs/oracle/forward-traces files.

Additional local checkpoints were inspected again by safe headers and complete config/tokenizer/weight hashes, with no tensor load into MLX:

| Candidate | Revision | Stored weights | Native status |
|---|---|---|---|
| openai-community/gpt2 | 607a30d783dfa663caf39e06633721c8d4cfcd7e | F32, unquantized | NOT_RUN; separate authorization required |
| mlx-community/LFM2-350M-4bit | 18dc72abf3b2337f9123cfd6eeeb58dfa7947066 | Affine Q4/group64, BF16 exceptions/U32 | NOT_RUN; separate authorization required |
| mlx-community/Qwen2.5-0.5B-Instruct-8bit | 6474d4c1a05bd7bd634a9c993560e4e7272eda3d | Affine Q8/group64, F16 exceptions/U32 | NOT_RUN; separate authorization required |

Bounded commands and private profiles are in `.venv/alpha_0_1/quality-gates-20261019-1/additional-checkpoint-plans.md`. All three existing native_tasks plan commands passed. Prepared native commands use reference B1, five repetitions, 32 output tokens and a 120s worker timeout, preserving native floating dtype. Their initial native smoke profiles intentionally leave independent model-specific quality pending; executing them would not automatically qualify a model.

## Owner review and exact resume

The requested 11-row table, with full relative paths/SHA256/dependencies/intended wheel+sdist inclusion/risks/recommendations, is private at `.venv/alpha_0_1/quality-gates-20261019-1/owner-review.md`. All 11 contents match the previous final proposal. The new complete 324-file proposal refreshes the changed tracked qualification module hashes; no file is auto-approved. Three local-only inputs are required, one is optional experimental source and seven form a coherent optional UI set. UI approval must also cover its changed tracked index/version assets. Sdist source/tests/fixture/docs inclusion requires its own member review. No wheel, sdist, isolated install, staging, commit, tag, push or publication occurred.

Resume from the reproducible Qwen3 quality failures; inspect exact rendered prompts/tokens and independent expected answers without changing v1 criteria. A confirmed defect can receive a narrow owned fix and fresh-source qualification; otherwise keep this checkpoint unqualified and use a separately authorized candidate qualification. Owner inclusion remains pending at the owner's explicit request. Do not ask again until that review is ready; elapsed time is not approval. All authorized workers are finished. No inference optimization was attempted.
