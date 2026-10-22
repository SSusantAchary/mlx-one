# A10 reproducible qualification and resume

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

Use existing `benchmarks/native_tasks.py` / `benchmarks/p1_native_qualification.py`. One thin additive wrapper exists at `scripts/qualification/alpha/run.py`; no second model/task runner or eight duplicated shell scripts. Every attempt gets a unique directory, source/file hashes, JSON/CSV readout and checksum manifest; stdout/stderr remain separate and pytest adds JUnit. Re-run only the unfinished independent task; earlier failures are never overwritten or resumed as passes. Artifact roots inside the checkout must be Git-ignored. No implicit downloads; native task requires explicit --allow-metal.

Backend-free (executed twice, both retained):

```sh
.venv/bin/python scripts/qualification/alpha/run.py backend-free \
  --artifact-root .venv/alpha_0_1/session-20261019/final-validation
```

Final result: **1666 passed / 47 skipped**, 36.93s, three dependency/API deprecation warnings. Alpha contracts: **77 passed**; focused qualification/contract suites: **125 passed**. Initial post-change 1663/47 receipt remains intact at its earlier source. Whole-tree Ruff: 65 inherited diagnostics, 0 introduced. Focused changed/new code lint passes. Editable install, CLI/import, UI and data-resource checks are documented separately; no real checkpoint/Metal/performance executed.

Owner-authorized reference parity command (NOT_RUN):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/p1_native_qualification.py correctness \
  --checkpoints qwen3-small --backends reference_dense \
  --checkpoint-paths scripts/qualification/checkpoint-paths.local.json \
  --output .venv/alpha_0_1/native-reference
```

Independent text/embedding/ASR/VLM task commands (NOT_RUN; select a separate pinned profile for each task):

```sh
.venv/bin/python benchmarks/native_tasks.py plan --profile "$PROFILE"
.venv/bin/python scripts/qualification/alpha/run.py task --profile "$PROFILE" \
  --allow-metal --artifact-root .venv/alpha_0_1/native-tasks
```

Task profiles use existing native-task-profile-v1: model_type, canonical task, model_id, immutable 40/64-hex revision, checkpoint_path, inputs_manifest and its SHA256. Alpha wrapper bounds B1, five repetitions, one warm-up, <=8 cases, <=600s. Existing preflight enforces actual M4/RSS/Metal availability and a process ceiling at most 75% RAM/recommended working set. Text settings.engine must explicitly select reference_dense; no speculative/paged optimization is required. Fixtures use checksum-declared assets and immutable expected token/text/vector/image/ASR anchors; missing goldens/thresholds remain pending. Speech execution_dtype stays native/None unless independently supported; weight format and activation/cache are separate.

Important existing runner limitation: task_lifecycle and task_quality_coverage are currently unpopulated by the worker. Simple anchors alone cannot complete these gates. This session exported the existing correctness-check set for the wrapper to reuse; no gate was relaxed. The wrapper rejects absent/non-PASS correctness, stale/wrong-source/profile and checksum failures even when the subprocess exits 0. It does not require competitive speedup: unrelated performance pending/failed fields retain their original aggregate status, separately from validated correctness checks. No automatic registry promotion or ALPHA_SUPPORTED claim occurs. Native lifecycle/cancel/cleanup and fixture coverage need measured, task-specific evidence before acceptance; this is a remaining A10 implementation/evidence gap, not only missing hardware.

Exact owner input needed: approve the private release inclusion manifest; explicitly authorize the bounded reference command; supply/confirm a pinned floating-weight checkpoint plus canonical ID, revision, local safetensors/config/tokenizer file hashes; supply separate ASR/embedding/VLM profiles and media/golden fixture hashes for any additional advertised task. The existing private path map has one entry; Qwen3-small config was read locally (Q8/group64/BF16 declaration only). Whisper Tiny pin was not resolved locally. No weights were downloaded or scanned as new qualification evidence.

Earliest unfinished gate: **A4 current-source native text task quality/stream/lifecycle/memory**, followed by A10 checkpoint/precision evidence and A9 approved-source wheel/sdist/install parity. Optional A5/A6/A7 qualification can proceed independently after its own asset/Metal opt-in. Do not repeat completed backend-free work without a regression/source change. To resume, read v5/progress, private source-entry and stage-ledger manifests, preservation-final and the last wrapper receipt; compare source hashes first. If source differs, retain old receipts and create a new invocation.
