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

# Phase 1 resumption record — 2026-10-18

Phase 1 acceptance: **NOT PASSED**. Phase 2: **NOT_RUN**. No backend was promoted.
Current implementation source: `3b51a5c11c69819715277ce4c694b4e04754140e9afa463d736f73fcdd0414c0`; tests/scripts:
`0d3e64354026fce726b2848850be19bd35a3696d5a29f082f2a675bf4fff21ab`. Immutable starting commit remains
`b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`. All 161 original artifacts and the 19 protected user
diffs are unchanged. New runs live in `qualification/results/inference_optimization_phase1_repair/`.

Read [current measured results and exact M4 commands](../qualification/archived-evidence.md),
[source identity](../qualification/archived-evidence.md), and [execution report](execution_report.md).

| Current scope | Outcome |
|---|---|
| Backend-free regression | 1,418 passed; 47 skipped; 2 warnings; 38.11s |
| Strict real HTTP identity | PASS; canonical request 200, wrong ID/local-path requests 404 |
| Reference-attention dense parity | 6/6 PASS; zero logit/component error; B1/B2/B4 |
| Forced-fused operators | 205 PASS, 18 UNSUPPORTED, zero FAIL/NOT_RUN |
| Qwen3 INT8 fused parity | FAIL; 3,164/3,818 comparisons outside frozen tolerance |
| Bounded reference 512/B1 measurement | 7 completed requests, 5 warm samples, zero failures |
| Reference TTFT / prefill / decode | 3782.7811 ms p95 / 148.6876 TPS / 17.1571 TPS |
| Reference peak MLX | 1211149036 bytes; weights included |
| Paired operator performance | 204 timed; 191 faster, 13 slower |
| Whole-model fused performance | BLOCKED by numerical failure |
| Full reference matrix / other assets | HARDWARE_PENDING |

Qwen continuation logit error reaches 0.125, KV error 1.765625 at atol/rtol
0.01/0.01. All 66 greedy logit comparisons agree; that is insufficient.
Production Qwen fallback remains reference. Reference TTFT exceeds its 2000ms
budget; `gate-p1.json` is pending, not passed. Serial cache append/concat bytes
and physical GPU kernel counts remain unmeasured; the measured GQA head-repeat
output allocation count is 118277275648 bytes across HTTP forwards.

Resume by localizing projected/normalized/rotated operands and scale/reduction/
cast ordering at the first failing full-prefill KV layer (layer 2), without
loosening tolerances. Run backend-free validation before freezing each new
source scope. The current report provides exact native proof paths and separate
smoke/correctness/performance/full commands. Do not reuse a stale proof after
source, test, Git/package or hardware changes. Full reference workers allow one
hour; bounded smoke workers allow 30 minutes. The bounded seven-request result
is not the original twelve-request-per-group full baseline.

Executed failed/aborted attempts remain FAIL in their raw directories: 74/128
early EOS, the incomplete 600-second full replay, and the stale test-scope run.
The corrected backend-free fixture failure is retained in its log. No partial
worker or skipped/unsupported test was converted into a qualification pass.

The earlier handoff below is historical.

---

# Resumption record

Date: 2026-10-18 (Asia/Kolkata).
Immutable starting commit: `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`.
Current source/diff identity: `baseline/final-source.json`.
Machine-readable state: `progress.json`; hardware rows: `hardware_pending.json`.

Phase 0: available implementation and backend-free acceptance complete.
Native correctness and the required five reference workloads: HARDWARE_PENDING.
Phase 1: dispatcher hardening and executable diagnostics implemented; full
numerical/performance acceptance remains in progress. Phases 2–7: NOT_RUN.
OPT-03 direct paged INT8 parity: DEFERRED, never PASS.

Baseline: 1,301 passed / 46 skipped / 2 inherited warnings, 22.44s.
Phase 0 boundary: 1,331 passed / 46 skipped / 2 inherited warnings, 23.45s.
Final source: 1,395 passed / 47 skipped / 2 inherited warnings, 28.64s.
Ruff: 65 inherited findings, zero new. Scripts pass Ruff and shell syntax checks.
The extra skipped module is the new opt-in Metal diagnostic suite.
The original 19 tracked user changes are unchanged; their diff hashes match in
`baseline/user-change-preservation.json`. Do not overwrite other starting files.

Next execution, from this repository:

```sh
scripts/qualification/smoke.sh
scripts/qualification/full.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
scripts/qualification/fused-full.sh --kernel-only
scripts/qualification/fused-correctness.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
```

The saved path identifies Qwen3-0.6B-8bit at revision
`11de96878523501bcaa86104e3c186de07ff9068`. Five other P1 recipe assets were
missing in the startup inventory; provide their actual local paths/conversions
before running all six. No automatic downloads are authorized by these scripts.
The additional four-model P0.1 track and its existing coverage remain separate.

When hardware output arrives, inspect the complete checksummed tree, exact source,
checkpoint hashes, JUnit identities, numerical comparisons and cleanup before
changing readiness. Known Qwen and single-query runtime fallbacks remain active.
For any fused mismatch, compare projected operands, scale/reduction ordering,
mask alignment and complete KV state against the fixed reference arithmetic.
Do not loosen tolerances or infer success from matching greedy IDs alone.

After numerical success, collect kernel and whole-model timing/memory comparisons
against the immutable baseline. Phase 1 still needs hybrid-compatible integration
and additional four-model coverage. Do not claim Phase 1 acceptance or enable a
new backend from operator-only results. Resume from these records; retain the
original baseline rather than moving it to the next HEAD.
