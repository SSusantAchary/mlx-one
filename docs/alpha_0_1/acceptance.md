# Alpha acceptance decision

## Qwen3 upstream native adoption — candidate, 20 October 2026

**Parallel port implemented; canonical model retained.** The candidate adopts pinned Apple mlx-lm v0.31.3 projections, QKNorm, RoPE, fused SDPA, SwiGLU, residuals and tied head as first-party native MLX code. Existing registry/default model, Engine/cache ownership and API paths are preserved. No production mlx-lm dependency was added. Candidate scope is unpadded contiguous reference B1 with dense KV; unsupported modes fail explicitly. See [qwen3-upstream-native-adoption.md](../qualification/archived-evidence.md).

Final current-source backend-free regression: **1703 passed / 48 skipped / 3 inherited warnings**. One newly introduced metadata-probe guard assumption failed the first full run and was corrected; its original failure receipt remains. Candidate Metal comparison, five-repeat frozen v1 quality, native APIs/lifecycle and matched performance are **NOT_RUN, awaiting the explicit authorization required by Phase 5**. A concrete bounded runner and seven existing-framework quality profiles are prepared. Earlier native/quality/parity receipts below retain their original source and do not qualify this candidate. No promotion, packaging or source-inclusion approval is inferred. Alpha **NO-GO**, version **0.1.0a1**; owner inclusion remains pending.

Native source SHA `280317b00742ccd32015a9c823cd303e2d8bde0598111a283c83424905407682`; broader validation SHA `3d453f4acfb73d08da393defc6519092f093419ff1decfab1e03d8903eb9a78c`. Evidence: `.venv/alpha_0_1/qwen3-adoption-20261020-1/`. Resume at the pending candidate Metal authorization; do not restart A0–A10.

## Independent mlx-lm comparison — 20 October 2026

**Decision D: INCONCLUSIVE_OR_EXECUTION_PENDING; all 14 authorized runs completed.** Current-source Qwen3 Q8 matches pinned mlx-lm 0.31.3 in every visible token, text and finish reason, including all 12 frozen v1 failures. Numerical parity **FAILS** the unchanged FP16 gate: 2/140 full-logit boundaries and 498/2352 KV components pass, despite 140/140 greedy tokens and all cache positions agreeing. Cached/fresh and reset logits reproduce exactly within each implementation. No native production defect is confirmed; production source and tolerances were preserved.

See [qwen3-mlx-lm-parity.md](../qualification/archived-evidence.md) for the 14-operation audit, exact upstream/source hashes, layer diagnostics, harness-only EOS bookkeeping repair and measured limitations. New regressions: 43 passed; diagnostic adapter 8 passed (7 pass/1 fail before its copy-return fix). Original Metal reports are preserved and the corrected visible-ID readout is derived on CPU from saved raw IDs. All workers exited within bounds. Five-repeat v1 quality remains 15/27, FAIL. Release **NO-GO**, version **0.1.0a1**; source inclusion remains explicitly pending and distributions NOT_RUN. Following entries retain their original scope and source.

## Independent text quality qualification — current result

**Release NO-GO. Qwen3 bounded task quality FAIL; coverage PASS.** Frozen protocol 1.0.0 executed all 27 cases five times: 75/135 case-repetitions passed (15/27 distinct cases), 12 distinct failures. Determinism, stream/generate agreement, native repeated parity, all seven lifecycle receipts and the existing completion deadline passed. No scoring threshold, prompt, gate or numerical tolerance was relaxed. See [release-readiness.md](release-readiness.md) for per-category results and exact failed outputs.

Development SHA `08826936f2d0bc8e3cc68cd48868976cece9bb5fb29ae9fc21e7ff8639f54c52`; broader validation SHA `a0e8934260dc9efa3b31bbc2c149b5792783f404f325d96687372b169f72dcb5`; native source-provenance SHA `65f462cfe1bfb2ec3c57c549dbf56d4a12ea5df684a2bb3116493b40e8264af7`. HEAD `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`, version `0.1.0a1`. Approved release-source SHA remains unavailable.

Current-source regression: **1680 passed, 47 skipped**, 3 warnings; focused tests 30 passed; changed-file Ruff and diff-check PASS. New evidence: `.venv/alpha_0_1/quality-gates-20261019-1/quality-run-1792434914820604000`; tests `.venv/alpha_0_1/quality-gates-20261019-1/backend-free/backend-free-1792434923804445000-579984c8`. Previous CLI/HTTP and numerical receipts remain bound to their original source; no earlier pass was rewritten as current-source evidence. No model/precision is ALPHA_SUPPORTED.

The five-repetition gap is resolved for the new runs. Remaining aggregate failures are measured quality failures; per-shard full-dataset coverage is pending while the aggregate scorer records complete bounded coverage. Performance comparison/reference and fairness remain pending; HTTP-load qualification NOT_RUN. Other checkpoints have only verified assets and prepared commands; none executed. Owner explicitly kept source inclusion pending, so wheel/sdist/isolated installation remain NOT_RUN. Exact 11-file owner review is in the ignored quality-gates evidence directory.

Earliest unfinished gate: diagnose the retained quality failures without rewriting v1; separately authorize a prepared checkpoint if pursuing an alternative. Preserve the owner's pending inclusion review. Version remains 0.1.0a1. Completed stages were not restarted.

## Previous source-bound state


## Authorized native resume — current state

**Qwen3 native B1 workflow PASS; alpha release NO-GO.** User explicitly authorized only the bounded Qwen3 workflow. Python, token streaming, CLI generation/streaming, real TCP HTTP chat, SSE completion, authentication, client disconnect, recovery and cleanup passed on Apple M4 / 32 GiB. Quality coverage and floating-weight native evidence remain pending. Packaging remains BLOCKED_ON_RELEASE_INCLUSION.

Development SHA `8c57a6aff778f210ea0d212d0648df75dd3b920d4f96d93355e7edab268fb7c0`; broader validation SHA `a9a3f23771a2a116bc3bfaf9953341163aeb2faf573f2346feef230fe11e557c`; native source-provenance SHA `554a1a98fd16e7cb6530b9fa8e81cfe3401034752f07e5d5808ad172f39bcdf4`. These are three different hashing scopes. HEAD remains `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`; version remains `0.1.0a1`. Approved release-source SHA is unavailable.

Only the A10 task worker and its focused tests changed in this resume. Native inference mathematics, model registries, execution engines, caches, task routers and numerical tolerances were reused. Final regression: **1671 passed, 47 skipped, 3 warnings**, 43.89s; focused alpha/task tests 90 passed; changed-file Ruff PASS. All prior active source remains present, only those two files differ from the prior A10 source manifest, and all 6494 historical files are hash-identical. Staged contents remain unchanged; Git status refreshed index stat metadata.

Measured Python load: 0.881s in a fresh process with warm filesystem caches. Four warm requests: TTFT 77.01–86.91ms, prefill 96.21–108.59 tokens/s, decode 25.41–26.64 tokens/s. Python peak MLX 729333940 bytes, peak sampled RSS 871219200 bytes; after unloading 8 active / 0 cached MLX bytes. No competitive threshold or speedup claim.

A10 now records actual startup, execution, cancellation, recovery, idempotent close, model weak-reference release, cache/lease cleanup and post-unload memory. The Qwen3 lifecycle receipt passed. Quality receipts count declared/observed/ground-truth cases and retain coverage_passed=null / NOT_RUN without an independently validated coverage protocol. Two factual continuation anchors passed separately; they do not establish broad model quality. The one-repetition task receipt retains aggregate **failed** (five_repetitions=false) with quality/coverage pending; this is not converted to qualification. Performance/fairness gates were not relaxed.

Final-source native evidence: `.venv/alpha_0_1/resume-release-blockers-20261019-1/final-native-1792433770647560000/result.json`; detailed task receipt `.venv/alpha_0_1/resume-release-blockers-20261019-1/task-lifecycle-1792433799685333000/native`. Final tests: `.venv/alpha_0_1/resume-release-blockers-20261019-1/final-validation/backend-free-1792433779259685000-8208d610`. Full outputs, token IDs, precision, commands, failures and exact resume: [native-qwen3-validation.md](native-qwen3-validation.md).

The private final inclusion proposal is `.venv/alpha_0_1/resume-release-blockers-20261019-1/proposed-release-manifest.final.private.json`: 324 candidates, 11 local-only (3 required, 8 owner decisions), 0 optional approved; all content hashes refreshed for this source. Original manifests and older-source native attempts remain unchanged. Wheel/sdist/isolated install NOT_RUN. No download, conversion, sweep, optimization, staging, commit, tag, push or publication occurred.

Exact next owner action: review and approve/reject the final private release-inclusion proposal by content hash. This authorizes neither publication nor another checkpoint. In parallel, A4/A10 still need independent quality/coverage fixtures and separately authorized native evidence for a discovered floating-weight checkpoint. Qwen2.5 Q8, LFM2 Q4 and all floating candidates remain native NOT_RUN / HARDWARE_PENDING.

## Earlier source-bound execution history

The following entries describe their original stage/source and pre-authorization state. Current status is the authorized resume above; prior receipts are preserved.

**NO-GO for an actual alpha release. Local implementation/contract work progressed through A0–A10; hardware and approved-distribution gates remain unfinished.** Version remains 0.1.0a1, unreleased by this task.

| Gate | Result | Evidence / blocker |
|---|---|---|
| A0 protection/source/ownership | PASS for local work | All 653 entry source/assets present; 15 scoped existing-file edits; 6494 historical hashed files unchanged; private preservation-final manifest |
| Native compute ownership / no inference wrappers | BACKEND_FREE_PASS | Source audit/import denial; original math and execution owner preserved |
| Reference defaults / experimental isolation | BACKEND_FREE_PASS | A1 79 tests; no optimization promoted |
| Precision metadata / pre-load safety | BACKEND_FREE_PASS | A2/A4–A7 and 77 final alpha contracts; real cells NOT_QUALIFIED |
| Lazy architecture/task/evidence registry | BACKEND_FREE_PASS | A3 81; existing 27 registrations and validators reused |
| Stable API/CLI/HTTP preservation | BACKEND_FREE_PASS | A8 68; 82 root exports unchanged; existing route scope retained |
| Full offline regression | PASS / native SKIP | 1666 passed, 47 skipped, 3 warnings; last unique final-validation receipt |
| Lint regression | PASS against baseline | 65 inherited diagnostics; 0 introduced |
| UI/resource/preserved workflows | PASS at tested scope | UI 7; Svelte check 0/0; A9 51 passed/7 skipped; no UI rebuild |
| Local editable install / entry point | PASS | a9-editable.log; no dependency downloads |
| Real pinned text correctness/quality/stream/lifecycle/memory | HARDWARE_PENDING | Core release blocker; no native run authorization |
| At least one real floating + one affine Q4/Q8 cell | NOT_QUALIFIED | Config metadata/tiny tests are insufficient |
| Task harness lifecycle/coverage receipt fields | INCOMPLETE | Existing worker leaves gates unpopulated; wrapper fails closed; must measure/extend before qualification |
| ASR / embedding / VLM optional checkpoint proof | HARDWARE_PENDING | Separate tracks; no all-modality claim |
| Approved release manifest | BLOCKED_ON_RELEASE_INCLUSION | Private review contains 324 estimated package inputs, 11 untracked; exact ownership approval absent |
| Wheel/sdist member audit + clean installed-wheel task tests | NOT_RUN / BLOCKED | Cannot package unapproved local source; editable pass is insufficient |
| Public Changelog source | BLOCKED_ON_RELEASE_INCLUSION | File exists locally untracked; not authorized for staging/publication |
| Publication | NOT_AUTHORIZED / NOT_RUN | No staging, commits, tags, pushes, package upload or artifact sync |

Stage implementation and test details: progress.md / execution-report.md and per-stage validation reports. Exact commands, receipt limitations and resume: qualification.md. Per-checkpoint/per-precision claims: support-matrix.md and JSON. No historical FAILED/DEFERRED/PENDING/NOT_RUN evidence converted to PASS.

Single next owner action: review and explicitly approve/reject exact content hashes in the ignored release-inclusion.pending.private.json. This does not authorize publication or native Metal work; those remain separate explicit actions.

## Resume checkpoint

Source identities match the prior 1666/47 baseline; no repeat test run or source edit. CPU checkpoint discovery PASS at header/tensor-layout/tokenizer/hash scope for all four requested tracks, including real local GPT-2 FP32 weights. Native correctness, quality, lifecycle, performance and Python/CLI/HTTP/SSE remain NOT_RUN/HARDWARE_PENDING because explicit native authorization has not yet been supplied. The reviewed old P1 command includes batching and B1/B2/B4; use the prepared supervised B1 CLI command first.

A9 proposal now classifies all 11 selected local-only inputs: 3 required, 8 requiring owner decisions; no optional inclusion approved. Wheel/sdist/clean-install still BLOCKED_ON_RELEASE_INCLUSION. A10 native lifecycle/coverage fields remain incomplete and were not fabricated. See [resume report](../qualification/archived-evidence.md). Release remains NO-GO. Exact next action: authorize/decline the bounded pinned Qwen3 M4 workflow.
