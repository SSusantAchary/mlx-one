<!-- TILED_FEASIBILITY_CURRENT -->
# Reference-compatible attention feasibility — 2026-10-19

**Decision A for the scoped native primitive route; global Phase 1 acceptance remains NOT_PASSED.** The opt-in `native_gather_tiled` implementation preserves the reference arithmetic and FP16 boundary. This multipass route executes zero stock SDPA calls. Stock full/vector SDPA failures remain unresolved. Defaults remain reference; qualification promotion is false; Phase 2 is NOT_RUN.

Read the [feasibility report and exact M4 commands](../qualification/archived-evidence.md), [current numerical/performance tables](../qualification/archived-evidence.md), and [current source identity](../qualification/archived-evidence.md). Earlier diagnostic tables retain their own source/dispatch identities and are not substituted for integrated measurements.

| Final-source evidence | Outcome |
|---|---|
| Backend-free regression | 1,589 passed / 47 skipped / 2 warnings; 38.07 s; skips excluded from passes |
| Integrated Qwen3 full/chunk 7/forced 32 | 3,818 comparisons and 2,100 strict boundaries PASS; FP32/logit/KV error, FP16 unequal bits and ULP all zero |
| Uninstrumented model confirmation | 3,818 comparisons PASS, zero model error; zero per-core/mask evaluations; 2,100 actual native-gather dispatches, zero fallback |
| Native attention Q=1/7/32/64/512/2048/4096 | 7/7 exact; diagnostic primitive latency 2.90%–8.65% speedup; longer shapes remain outside production capability |
| Reference B1/B2/B4 | 6/6 PASS, zero numerical error; current-source JUnit 211.626 s |
| Integrated B1 64/32, 21 matched pairs | TTFT 289.455375→288.021291 ms; prefill 221.330401→222.426306 TPS; decode 22.194702→22.353946 TPS |
| Peak MLX / observed GQA repeat-output bytes per request | 720,275,688→717,574,536 B peak; 605,552,640→0 B cumulative repeat output |

The integrated model medians improve only 0.50% prefill / 0.72% decode, with candidate slower in 12/21 prefill and 7/21 decode pairs. This does not establish a robust whole-model benefit. Keep the route experimental and opt-in. B2/B4 candidate throughput, other checkpoints, longer model contexts, BF16, physical cache-copy bytes and physical kernel counts remain unqualified/unmeasured. The original stock SDPA FAIL, 18 unsupported configurations, initial prototype FAIL and earlier invocation failures are preserved.

Current implementation SHA `3e32d7bb5e7cec8c4d800e0d7acfc5c06defd4dda3b9a0ddc26564d955ca5e98`; tests/scripts `e2340e4eb39ce31e88e5c36403f31e82217ba2a0d968cfcbfe2f9312f38421c1`. Starting commit, original artifacts and protected user changes remain unchanged. Resume from this report and the executable `tiled_feasibility/rerun-m4.sh`; do not infer global Phase 1 acceptance or begin Phase 2.
<!-- /TILED_FEASIBILITY_CURRENT -->

# Current Qwen3 causal investigation — 2026-10-18

**Actual fused parity remains FAIL.** The first causal change is layer 0 attention output, amplified at its FP16 cast and propagated through the unchanged INT8 projection/MLP. The first frozen trace violation is layer 1 `q_rope`; layer 2 key-state failure is downstream. A stable INT8 dispatch guard is implemented and validated; it is a reference safety fallback, not a fused numerical fix.

Current implementation `018549f002ca495e92ac2a8ee9a9dd7092b2ccf4f820cc4e7c48a84826f746bb`; tests/scripts `495300380d4fb5b4b464b7eaa14fad24465e1921b8de5294781894f73ab9b972`. Starting commit and pinned checkpoint are unchanged. Read [current causal results, limitations and exact M4 reruns](../qualification/archived-evidence.md), [measured tensor/replay/layer-control tables](../qualification/archived-evidence.md), and [current source identity](../qualification/archived-evidence.md).

| Current-source validation | Outcome / actual dispatch |
|---|---|
| Backend-free regression | 1,453 passed / 47 skipped / 2 warnings; 35.29s |
| Forced fused operators | 205 PASS / 18 UNSUPPORTED / 0 FAIL / 0 NOT_RUN |
| Direct Qwen3 full/chunk 7/forced 32 | FAIL: 3,164/3,818 comparisons; 2,100 forced FP32 SDPA invocations |
| Forced Qlen=1 vector alternative | FAIL: 3,111/3,818 comparisons; 5,376 forced FP32 SDPA invocations |
| Guarded candidate full/chunk 7/forced 32 | PASS: zero error across 3,818 comparisons; 2,100 reference calls, zero fused |
| First-three-layer trace | 72 comparisons, nine violations; 33 exact metadata checks PASS |
| One fused layer at a time | 28 controls executed; layer 0 alone reproduces first layer 2 key failure |
| Reference B1/B2/B4 correctness | PASS: 6/6, zero logit/component error; 197.92s |
| Whole-model current-source performance | HARDWARE_PENDING / BLOCKED by direct fused numerical FAIL |

Phase 1 acceptance **NOT PASSED**. Phase 2 **NOT_RUN**. No backend promotion or tolerance/reference-arithmetic change. The stock fused reduction/softmax ordering still does not preserve the reference FP16 rounding boundary; the exact private Metal suboperation is not uniquely isolated. An oracle boundary substitution proves causality only and is never used for runtime output, qualification or timing.

All new raw outputs are in `qualification/results/inference_optimization_qwen_causal/`. Existing 161-file original and 335-file repair artifact trees remain unchanged. The 485-file starting source archive and both original aggregate digests are preserved. Current reports distinguish failed forced execution from the passing reference fallback, including actual dispatch and command/source identities.

Resume from the causal report: develop a numerically compatible fused implementation, repeat backend-free and frozen full/chunk/KV/continuation gates, then run matched current-source model performance. Do not use operator passes, greedy agreement, last-layer-only controls or reference-fallback passes as fused acceptance. Do not begin Phase 2.

The earlier reports below retain their original source identities and historical measurements.

---

# Phase 1 repair execution report — 2026-10-18

Phase 1 acceptance has **not passed**; Phase 2 remains **NOT_RUN**. No runtime
backend or checkpoint was promoted. Full measured tables, raw source identities,
negative results and copyable M4 rerun commands are in
[phase1_repair/results.md](../qualification/archived-evidence.md).

The immutable comparison commit is `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`. Final implementation
SHA-256 is `3b51a5c11c69819715277ce4c694b4e04754140e9afa463d736f73fcdd0414c0`; test/script SHA-256 is
`0d3e64354026fce726b2848850be19bd35a3696d5a29f082f2a675bf4fff21ab`. Original 161-file hardware tree SHA-256 remains
`eaadabf890349eba25fd6377b95cfc8ef5dacf9980d150c14d1c9dd323a185de`.

Implemented loader/manager separation of canonical ID, resolved checkpoint path,
revision and content hashes; strict benchmark path/hash checks and before/after
asset checks; real HTTP positive and negative identity qualification. HTTP model
validation was retained unchanged. Converted checkpoint source/artifact identity
and remote symbolic-reference resolution have explicit regression coverage.

Implemented MLX 0.32.2 vector/full capability checks before runtime cache mutation,
without imposing the vector product limit on full attention. Diagnostics retain
forced fused execution, expected unsupported-shape/reference-fallback receipts,
FAIL on unexpected exceptions, complete operator collection and independent model
qualification. The diagnostic schema is now v2. Known Qwen production fallback
remains active.

Added actual attention-dispatch and materialized GQA-repeat allocation receipts.
Fixed benchmark early EOS through an explicit qualification-only output policy;
the exact output-length assertion remains. Ordinary HTTP still respects EOS.
Paired profiles must match EOS policy. Added `reference-smoke.sh` with five warm
samples and explicit omitted-case statuses; full twelve-request groups remain
available with 3600-second workers. Bounded smoke workers allow 1800 seconds.

| Executed final-source validation | Result |
|---|---|
| Backend-free tests | 1,418 passed / 47 skipped / 2 warnings; 38.11s |
| Static checks | Changed files Ruff PASS; 65 inherited full-tree findings, zero new; shell syntax/diff PASS |
| Real native HTTP identity | canonical 200 + completed inference; wrong ID/local path 404 |
| Native reference-attention dense parity | 6/6 PASS, zero numerical/component error, 100% greedy agreement; 288.08s |
| Forced-fused operator numerical cases | 205 PASS / 18 UNSUPPORTED / 0 FAIL / 0 NOT_RUN |
| Qwen3 INT8 whole-model fused | FAIL: 3,164/3,818 comparisons; 2,100 evaluated selected fused calls |
| Bounded reference benchmark | 7 completed exact 512/128 requests, 5 warm groups, zero failed requests, cleanup PASS |
| Reference TTFT/prefill/decode | 3782.7811 ms p95 / 148.6876 TPS / 17.1571 TPS |
| Reference controlled decode | Median 21.8939 TPS, five samples, four serial prepared histories |
| Reference actual dispatch/memory | 25,088 reference attention calls; actual B1; peak MLX 1211149036 bytes |
| GQA-repeat output allocation bytes | 118277275648; partial allocation count, excludes cache concat/FP32 casts |
| Operator performance | 204 actual pairs: 191 faster, 13 slower; B1/B2/B4 operator throughput measured |
| Whole-model optimized performance | BLOCKED; no model speedup or optimized peak-memory result |

Unresolved numerical failures retain atol/rtol 0.01/0.01: full-prefill logits
max 0.01953125, chunked-prefill logits 0.025390625, continuation logits 0.125,
KV state 1.765625. All 66 greedy logit comparisons agree; state/logit qualification
still fails. First reported KV failure is full prefill layer 2 key state. The
precise arithmetic divergence remains unresolved and needs operand-level tracing.

The measured reference p95 TTFT exceeds its 2000ms target. `gate-p1.json` is
pending for missing matched optimized runs. The full five-case reference matrix,
model B2/B4 performance, other pinned/four-model current-source coverage, complete
serial cache-copy accounting and physical GPU-kernel counts remain unmeasured.
Operator timings cannot substitute for model timing or qualify a failing model.

All failed attempts were retained: initial pytest import-path collection error,
a corrected manifest-fixture assertion, the 74/128 early-EOS worker failure,
an explicitly interrupted full replay with an insufficient 600-second budget,
and an interrupted stale test-scope native run. They remain failures/incomplete
runs; partial requests were not used as accepted performance evidence. The
original M4 artifacts were never edited.

Resume from [progress.md](progress.md). Repair Qwen numerical/state divergence,
then regenerate source-bound proofs and measure identical optimized/reference
model workloads with verified dispatch. Hold Phase 2 until acceptance is supported.

The earlier report below is historical; its source IDs and pending labels describe the pre-repair implementation.

---

# Per-phase execution report

Starting commit: `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`, with existing user
changes. Final commit is unchanged; exact implementation/source digests and file
checksums are in `baseline/final-source.json`. Python 3.12.11, MLX 0.32.2, optional
reference mlx-lm 0.31.3, Darwin arm64. Declared hardware is Apple M4 / 32 GiB.
Sandbox host telemetry was unavailable. Expensive native runs were left to the
user as instructed. All new native results are HARDWARE_PENDING.

| Workstream | Files/mechanisms | Implementation | Tests | Performance |
|---|---|---|---|---|
| Phase 0 / OPT-07 | native_correctness, native_qualification, inference_baseline/worker/qualification, p1_decode, batching, qualification scripts | Available implementation complete | PASS | HARDWARE_PENDING |
| Phase 1 / OPT-01 | ops/attention, CPU/native SDPA tests, fused scripts | Dispatch validation and forced-fused diagnostics implemented; full phase in progress | Backend-free PASS; Metal HARDWARE_PENDING | HARDWARE_PENDING |
| OPT-02/04/05/06 | Existing modules audited, production changes await sequential phases | NOT_RUN | NOT_RUN | NOT_RUN |
| OPT-03 | Existing direct-paged diagnostics/failure guards retained | DEFERRED | No new numerical claim | DEFERRED |

The actual model path remains projection, QK normalization, RoPE, optional
cache-owned attend, validated backend selection, cache append, stable reference
or existing experimental fused attention, and output projection. Cache-owned
attention keeps precedence. Existing failed Qwen/single-query routes select
reference. Dense runner telemetry observes pack/forward/evaluated staging/commit
without adding GPU synchronization or changing transaction ownership.

Qualification selects dense or paged correctness from the requested execution
backend. A failed/deferred paged row cannot block a separately passing selected
dense cohort. Omitted selection conservatively requires both for compatibility.
Schema v2 binds all source/software/test/script identities, checkpoint files,
actual commands/JUnit cases, numerical/state/isolation receipts and checksums.
BLOCKED is retained. DEFERRED, HARDWARE_PENDING and other non-PASS statuses cannot
admit their selected backend. Existing performance gates remain authoritative.
No production promotion or default change occurred.

Actual changes, mathematical/storage behavior, reference repositories and inspected
symbols are detailed in implementation.md and architecture.md. All 19 pre-existing
tracked modifications match their original diff checksum. No user UI, CLI,
configuration or prior qualification artifacts were overwritten.

Validation actually executed:

| Check | Status | Result |
|---|---|---|
| Startup backend-free suite | PASS | 1,301 passed, 46 skipped, 2 warnings; 22.44s |
| Phase 0 boundary suite | PASS | 1,331 passed, 46 skipped, 2 warnings; 23.45s |
| Final backend-free suite | PASS | 1,395 passed, 47 skipped, 2 warnings; 28.64s |
| Ruff src/tests/benchmarks | Inherited findings | 65 before and after; zero new |
| Qualification scripts Ruff/shell syntax | PASS | All checks pass |
| Diff whitespace | PASS | git diff --check |
| Hardware command help and exact pytest parameter identity collection | PASS | No MLX initialization |
| M4 numerical/performance runs | HARDWARE_PENDING | Not executed |

One intermediate run exposed an inherited cancellation-test timing race. The test
now verifies cancellation in generator cleanup and adds a deterministic suspended
yield case; scheduler production code is unchanged. Its failed log is retained in
`baseline/final-pytest.log`. Another intermediate run rejected two new test fixtures
with invalid synthetic revision strings; the fixtures were corrected to immutable
revision format. Final validation is `baseline/validated-regression.log`.

Numerical thresholds are unchanged: FP32 atol/rtol 0.0001/0.0001; FP16 0.01/0.01;
BF16 0.05/0.01. New native maximum/mean errors, TTFT, prefill TPS, decode TPS,
aggregate throughput, RSS and peak memory have no measured values yet. Their
machine-readable entries are null/HARDWARE_PENDING in hardware_pending.json.
No speedup against the original commit can be calculated without comparable
native runs. Historical P0/P0.1 results are preserved and are not borrowed for
this source. Logical copy instrumentation does not claim physical Metal traffic.

| Execution path | Implemented | Backend-free | Native correctness | Native performance | Newly production selectable | Default enabled |
|---|---|---|---|---|---|---|
| Existing reference dense | Yes | PASS | HARDWARE_PENDING for new source | HARDWARE_PENDING | Existing selection retained | Yes |
| Existing dense/hybrid batching | Yes | PASS | HARDWARE_PENDING for new source | HARDWARE_PENDING | No new promotion | No |
| Fused diagnostics | Yes | PASS | HARDWARE_PENDING | HARDWARE_PENDING | No | No |
| Direct paged INT8 | Existing experimental path | PASS for affected CPU gates | DEFERRED / historical failure | HARDWARE_PENDING | No | No |
| Persistent batch KV / new arena and prefix optimizations | Not implemented in this milestone | NOT_RUN | NOT_RUN | NOT_RUN | No | No |

Reproduction, from the repository:

```sh
PYTHONPATH=src:.:tests .venv/bin/python -m pytest --no-cov -q
.venv/bin/python -m ruff check src tests benchmarks
.venv/bin/python -m ruff check scripts/qualification
git diff --check
scripts/qualification/smoke.sh
scripts/qualification/full.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
scripts/qualification/fused-full.sh --kernel-only
scripts/qualification/fused-correctness.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
```

The saved checkpoint is the startup-local Qwen3-0.6B-8bit snapshot at
`11de96878523501bcaa86104e3c186de07ff9068`. Additional P1 assets/conversions require
explicit paths. Separate correctness/performance commands and output locations
are in benchmarks.md. Preserve whole result directories, not only stdout.

Outstanding gates: native Phase 0 correctness/reference timings; Phase 1 fused
checkpoint/state parity, compatible hybrid dispatch and additional four-model
coverage, followed by whole-model comparative timing/memory qualification.
The 223-case kernel diagnostic cannot establish model or serving qualification.
Architectures outside the shared dispatcher are UNSUPPORTED in that diagnostic.
Phases 2–7 remain unimplemented in this milestone. Continue from progress.md;
keep reference defaults and optimized modes experimental until their gates pass.
