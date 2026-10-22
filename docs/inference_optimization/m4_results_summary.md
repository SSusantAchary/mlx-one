> Historical review of the original three M4 runs. Current Phase 1 repair evidence, actual paired operator results, reference measurements, unresolved failures and rerun commands are in [phase1_repair/results.md](../qualification/archived-evidence.md). Original artifacts remain unchanged; the historical source below is not the current implementation source.

# M4 qualification results: Phases 0–1

Review date: **18 October 2026**, Asia/Kolkata.

**Phase 1 acceptance has not passed. Do not proceed to Phase 2 as an accepted
milestone or promote fused attention.** Qwen3-0.6B INT8 dense/reference parity
passed all six executed cases with zero reported logit/state error. Both fused
correctness runs stopped at an unavailable kernel configuration before reaching
the whole-model fused test. All five reference performance workers failed;
there are no valid inference timing pairs or speedup percentages.

The earlier [execution report](execution_report.md), [progress record](progress.md)
and [benchmark handoff](benchmarks.md) describe the state before these hardware
runs. Their blanket HARDWARE_PENDING statements are superseded here only for
the specific checks actually executed. Historical P0/P0.1 benchmarks are excluded.

## 1. Baseline, hardware and source identities

| Identity | Recorded value |
|---|---|
| Immutable implementation starting commit | `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902` |
| Starting working-source SHA-256 | `921448a4b6a6d05c1e742cb1c2fa7309192863f414db58ee43f2dd63089896b2` |
| Hardware-run Git revision | Same commit; dirty working tree |
| Hardware-run source SHA-256 | `ca00c382c993556d30acdb12473a4ec825af8f01c37b7159a484ae8a035ffd86` |
| Test/qualification-script SHA-256 | `832213d32fcd2723a5d8f36be2cb3b7c4f47782c71048c393ae44de615dde0d8` |
| Implementation file-map SHA-256 | `d63ad0a364048cbd1c626d8d9929233b8f3b5355b5dca9015a33cc2a24d64277` |
| Runtime source identity in dense parity | `source-sha256:11ba0eebf78c551beee04340b9d88a58c5d29a5519eadf450491bdb0512bfa28` |
| Observed hardware | Apple M4; 34,359,738,368 bytes RAM (32 GiB) |
| OS / Python / MLX | macOS 15.7.9 / Python 3.12.11 / MLX 0.32.2 |
| Other recorded packages | mlx-one 0.1.0a1; tokenizers 0.23.2; safetensors 0.8.0; huggingface-hub 1.30.0; uvicorn 0.52.4 |
| Worker RSS ceiling | 21,474,836,480 bytes (20 GiB) |
| Checkpoint | `mlx-community/Qwen3-0.6B-8bit` |
| Checkpoint revision | `11de96878523501bcaa86104e3c186de07ff9068` |
| Weights / activations / KV | INT8 affine, group size 64 / FP16 / FP16 |
| model.safetensors SHA-256 | `3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397` |

All three runs match [the final implementation source record](../qualification/archived-evidence.md)
and the current source/test/script identities. The runtime source identity hashes
the runtime tree using a different path convention and excludes benchmark files;
it is not the same digest definition as the qualification source hash. The
[starting environment](../qualification/archived-evidence.md) preserves the original dirty
baseline. There are no comparable original-baseline performance measurements
in these runs, so no before/after performance claim is possible.

## 2. Artifact review and run outcomes

All **161 files** under `qualification/results/inference_optimization/` were read.
This includes 123 JSON files, five JSONL files, seven CSV files, 18 logs, four
JUnit XML files and four Markdown files. All **25 integrity manifests / 121
checksum entries** verified. CSV records match their JSON sources, including
the failed/incomplete rows. The six-case dense correctness proof also passed
the current backend-specific validator against the local checkpoint hashes.
Checksummed failure evidence remains failure evidence.

The collection SHA-256 is
`eaadabf890349eba25fd6377b95cfc8ef5dacf9980d150c14d1c9dd323a185de`:
SHA-256 of the UTF-8, sorted, compact JSON mapping each path relative to
`qualification/results/inference_optimization` to its file-content SHA-256.
Raw artifacts were preserved without edits.

| Run | Start time, IST | Requested scope | Recorded outcome |
|---|---|---|---|
| [A: 1792335597241445000](../qualification/archived-evidence.md) | 20:29:57 | Phase 0 full; qwen3-small | Smoke PASS; dense correctness PASS; performance FAIL |
| [B: 1792338543660755000](../qualification/archived-evidence.md) | 21:19:03 | Phase 1 full; kernel-only | Fused smoke PASS; correctness FAIL; performance NOT_RUN |
| [C: 1792338567487306000](../qualification/archived-evidence.md) | 21:19:27 | Phase 1 correctness; qwen3-small requested | Correctness FAIL before checkpoint test; performance NOT_RUN |

Every recorded preflight verified Apple M4/32 GiB and actual Metal work on
`Device(gpu, 0)`. Preflight success is not model or performance qualification.

## 3. Final backend-free regression results

| Suite | Passed | Failed | Skipped | Warnings | Elapsed |
|---|---:|---:|---:|---:|---:|
| Starting implementation baseline | 1,301 | 0 | 46 | 2 | 22.44 s |
| Phase 0 boundary | 1,331 | 0 | 46 | 2 | 23.45 s |
| Final validated implementation | **1,395** | **0** | **47** | **2** | **28.64 s** |

Source: [final regression log](../qualification/archived-evidence.md), with the
earlier logs preserved under `baseline/`. This is the final available
implementation regression result, not a new regression run during this review.
The 47 skips are excluded from passing coverage; the extra skipped module is
the opt-in fused Metal suite. Ruff retained 65 inherited findings, zero new;
qualification-script lint/syntax and diff whitespace checks passed as recorded
in the implementation report.

## 4. Native fused attention correctness

| Cohort | Required operator/checkpoint cases | Executed PASS | Executed FAIL | Unexecuted | Skipped |
|---|---:|---:|---:|---:|---:|
| B, separate fused smoke | 1 | 1 | 0 | 0 | 0 |
| B, kernel correctness | 223 operators | 43 | 1 | 179 | 0 |
| C, checkpoint-scoped correctness | 223 operators + 1 model | 43 | 1 | 180 | 0 |

Each correctness run passed the smoke test and **42 identical-operand cases**.
These are the same covered configurations in both runs, not 84 distinct
configurations. Successful numerical cases cover D64, causal masks, B1/B2/B4,
FP16/BF16 single-query attention at histories 0/17, plus FP16 seven-query
attention at history 19 for MHA and GQA with two KV heads. Boolean/additive mask
cases, six long-prefill cases and the remaining configurations were not reached.
Run B additionally deselected the checkpoint test placeholder; it did not pass it.

For successful operand cases, the reference is `attention_reference` with stable
FP32 per-query arithmetic. The candidates actually call MLX SDPA with
`force_fused=True`, once with native input precision and once with FP32 operands,
and evaluate their outputs. No reference retry is performed. This verifies
forced-fused availability for the successful shapes; GPU kernel counts and
low-level dispatch traces are unavailable.

Both runs fail at:

| Parameter | Observed value |
|---|---|
| Test | `test_fused_kernel_identical_operands[causal-7-19-float16-8-1-1]` |
| Q shape | `[1, 8, 7, 64]` |
| K/V shape | `[1, 1, 26, 64]` |
| Mask / input precision | causal / FP16 |
| Candidate at failure | Native-precision forced-fused SDPA |
| GQA factor / query-length product | 8 / 56 |
| Reported vector-kernel limit | Product at most 32 for this selected path |
| Numerical error for failed call | Not measured; call raises before output |
| FP32 candidate for this case | NOT_RUN; native candidate fails first |

Exact exception:

```text
ValueError: [scaled_dot_product_attention] force_fused=True but no fused kernel is available: the vector attention kernel requires the query length times the GQA factor to be at most 32; got query length 7 and GQA factor 8.
```

Sources: [B worker log](../qualification/archived-evidence.md),
[C worker log](../qualification/archived-evidence.md),
[C JUnit](../qualification/archived-evidence.md).
JUnit reports one failure in each run. The corresponding JSON/CSV row still
says `NOT_RUN` because the exception occurred before its status update. This
review treats that row as an **executed availability failure**, not an unexecuted
test or a numerical tolerance failure. Both overall handoffs correctly report FAIL.

## 5. Whole-model Qwen3-0.6B INT8 parity

Run A compares `single_request_dense_fp16_kv` against
`experimental_dense_batch`, with **reference attention arithmetic**, at 32
forced continuation steps. These are real model/cache executions, not fused
attention qualification. Both page-size parameters are dense test configurations;
they do not show direct-paged GPU execution.

| Dense page parameter | Physical test batch | Max logit error | Mean logit error | Max component-state error | Greedy agreement | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 16 | B1 | 0 | 0 | 0 | 100% | PASS |
| 16 | B2 | 0 | 0 | 0 | 100% | PASS |
| 16 | B4 | 0 | 0 | 0 | 100% | PASS |
| 32 | B1 | 0 | 0 | 0 | 100% | PASS |
| 32 | B2 | 0 | 0 | 0 | 100% | PASS |
| 32 | B4 | 0 | 0 | 0 | 100% | PASS |

All six receipts report prefix isolation and seeded-RNG isolation true, and
cleanup active handles/reserved bytes both zero. FP16 atol/rtol remain 0.01/0.01.
Evidence: [dense correctness proof](../qualification/archived-evidence.md),
[dense JUnit](../qualification/archived-evidence.md)
and the six adjacent `parity-qwen3-small-False-{16,32}-{1,2,4}.json` files.

| Whole-model comparison | Actual attention dispatch evidence | Result |
|---|---|---|
| Dense single-request oracle vs dense batching | Receipt identifies reference attention; runtime reports experimental_dense_batch | PASS for the six cases above |
| Dense reference vs forced-fused Qwen3 | Checkpoint test never reached; no model fused-call/state receipt | **NOT_RUN** |

The current M4 artifacts establish neither passing nor failing **whole-model
fused** Qwen3 parity. Earlier recorded fused model failures are not substituted
for a result from this run.

## 6. Numerical error and tolerance comparisons

The following are the maxima over completed operand comparisons. “Largest mean
error” is the largest individual comparison's reported mean, not a pooled mean.
Both B and C report the same values.

| Original Q/K/V dtype | Actual candidate operands/backend | Comparisons per run | Max absolute error | Largest mean error | atol / rtol | Completed subset |
|---|---|---:|---:|---:|---|---|
| FP16 | Native FP16 forced-fused SDPA | 24 | 0.0009012222290039062 | 0.000052189428970450535 | 0.01 / 0.01 | PASS |
| FP16 | FP32 operands, forced-fused SDPA | 24 | 0.000000476837158203125 | 0.00000003862265884890803 | 0.01 / 0.01 | PASS |
| BF16 | Native BF16 forced-fused SDPA | 18 | 0.006932258605957031 | 0.00041275244439020753 | 0.05 / 0.01 | PASS |
| BF16 | FP32 operands, forced-fused SDPA | 18 | 0.00000035762786865234375 | 0.00000003226254818855523 | 0.05 / 0.01 | PASS |

Sources: [B numerical JSON](../qualification/archived-evidence.md)
and [C numerical JSON](../qualification/archived-evidence.md),
with matching adjacent CSV files. The reference outputs use the stable FP32
reference implementation. Acceptance is elementwise
`abs(actual-reference) <= atol + rtol*abs(reference)`, including finite-output checks.
The FP32-operand candidates retain their original input-dtype test tolerance;
they are not a separate FP32-origin qualification track. The frozen FP32-origin
tolerance remains 0.0001/0.0001, but that track was not independently exercised.

## 7. TTFT, prefill TPS, decode TPS and B1/B2/B4 throughput

All performance tasks in A requested `reference_dense`; none requested fused
execution. Each failed during the first cold traffic group. All **60 recorded
HTTP requests failed with 404**, zero completed, and none has a model metrics
payload. No warmup/measured repetition, prepared-history decode result or valid
request timing summary was produced. These failures are not zero-TPS measurements.

| Worker ID | Prompt / requested output | Requested serving slots | Planned mode | Actual inference dispatch | Failed / completed requests |
|---|---|---:|---|---|---:|
| `068b0b83417ef707c4e0` | 512 / 128 | B1 | reference_dense | Not reached; Metal initialization only | 12 / 0 |
| `6ee7f90d39151c2ff051` | 2048 / 128 | B1 | reference_dense | Not reached; Metal initialization only | 12 / 0 |
| `c44ad2eb2936984de0c8` | 4096 / 128 | B1 | reference_dense | Not reached; Metal initialization only | 12 / 0 |
| `19a1e20d394e209d768f` | 2048 / 128 | B2 | reference_dense | Not reached; Metal initialization only | 12 / 0 |
| `6a34670e0991079eea8c` | 2048 / 128 | B4 | reference_dense | Not reached; Metal initialization only | 12 / 0 |

B2/B4 here are requested cooperative serving slots, not evidence of a physical
B2/B4 forward. The dense correctness batch results above have no throughput
measurements.

| Requested comparison | Valid reference value | Valid optimized value | Speedup |
|---|---|---|---|
| TTFT | Not measured: reference workers failed | NOT_RUN | Not calculable |
| Prefill TPS | Not measured: reference workers failed | NOT_RUN | Not calculable |
| Decode TPS | Not measured: reference workers failed | NOT_RUN | Not calculable |
| B1 aggregate throughput | Not measured | NOT_RUN | Not calculable |
| B2 aggregate throughput | Not measured | NOT_RUN | Not calculable |
| B4 aggregate throughput | Not measured | NOT_RUN | Not calculable |
| Attention-kernel latency | No timed reference samples | Fused performance NOT_RUN | Not calculable |

Evidence: [performance outcomes](../qualification/archived-evidence.md),
each worker's `requests.jsonl`/`failure.json`/`worker.log`, and the
[empty top-level measurement export](../qualification/archived-evidence.md).
The [performance matrix](../qualification/archived-evidence.md)
is empty; its CSV is header-only. B/C CSVs contain numerical receipts, with
reference/fused timing fields empty. Pytest elapsed time is not kernel/model latency.

## 8. Peak MLX memory comparison and available load measurements

The only checkpoint MLX peaks recorded by the failed performance workers are
**load-stage** peaks. Sampled process RSS is for each failed worker's lifetime.
Neither metric represents a successful inference workload or an optimized pair.

| Reference worker: prompt / slots | Load seconds | Load-stage MLX peak, bytes | Sampled process RSS peak, bytes | Inference MLX peak |
|---|---:|---:|---:|---|
| [512 / B1](../qualification/archived-evidence.md) | 0.325818208 | 633,372,680 | [841,695,232](../qualification/archived-evidence.md) | Not measured |
| [2048 / B1](../qualification/archived-evidence.md) | 0.343733791 | 633,372,680 | [847,724,544](../qualification/archived-evidence.md) | Not measured |
| [4096 / B1](../qualification/archived-evidence.md) | 0.338305042 | 633,372,680 | [849,739,776](../qualification/archived-evidence.md) | Not measured |
| [2048 / B2](../qualification/archived-evidence.md) | 0.337214958 | 633,372,680 | [835,289,088](../qualification/archived-evidence.md) | Not measured |
| [2048 / B4](../qualification/archived-evidence.md) | 0.328568500 | 633,372,680 | [845,709,312](../qualification/archived-evidence.md) | Not measured |

Sources: the five worker IDs in section 7, under
`A/performance/Susants-MacBook-Air.local/qwen3-small/1792335765844576000/`,
using `load.json` and `supervision.json`. Load time is not TTFT or prefill time.
B's kernel correctness worker RSS peak was 96,403,456 bytes; C's was 95,879,168
bytes. Those workers failed before loading Qwen3 and are not comparable to a
whole-model reference memory peak. No reference-versus-optimized inference
memory reduction can be calculated.

## 9. Performance speedup percentages

**No performance speedup percentage is supported by this artifact set.** There
are zero valid paired performance measurements. Missing values are not 0% gains.
For future matched runs, report latency speedup as
`(reference_time / optimized_time - 1) * 100%` and throughput improvement as
`(optimized_TPS / reference_TPS - 1) * 100%`. Latency reduction percentage uses
`(1 - optimized_time / reference_time) * 100%` and must be labeled separately.

## 10. Exact unresolved failures

**Reference performance / model identity.** All five workers exit with:

```text
BaselineError: backend substitution or checkpoint identity mismatch
```

The five `checkpoint.json` files identify the loaded model by this local path:

```text
/Users/susant-achary/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068
```

The profiles and HTTP request bodies instead use
`mlx-community/Qwen3-0.6B-8bit`. From the recorded IDs and inspected code, the
likely common cause is a canonical-model-ID versus local-path mismatch:
the loader retains the local argument as `bundle.model_id`, the server rejects
a different requested model with 404, and `validate_identity` rejects a scope
whose model ID differs from the profile. This is an inference from artifacts
and code, not evidence of an actual substituted attention backend. The failed
workers did not save runtime/forward-dispatch receipts proving such substitution.

Relevant code: [loader](../../src/mlx_one/text/loading.py),
[HTTP model check](../../src/mlx_one/server/app.py),
[model metadata](../../src/mlx_one/server/model_manager.py),
[identity validator](../../src/mlx_one/inference_baseline.py).
Resolve the identity consistently across the profile, served alias and evidence
while retaining checkpoint hashes/revision validation. Do not bypass the gate.

**Fused kernel availability / incomplete coverage.** B/C both raise the exact
MLX exception in section 4. The current capability checks allow a short-query MQA
shape that this runtime's selected vector kernel cannot execute with force_fused.
Both commands use `-x`, so remaining cases and Qwen3's fused test are not reached.
The row-level JSON status also fails to record the raised exception explicitly.
Correct capability handling and failure classification, preserve the case, and
record unsupported configurations without calling them PASS. The observed
vector-path limit must not be generalized to untested long-query kernel paths.

**Missing acceptance evidence.** Whole-model fused logits/state/continuation,
boolean/additive mask coverage, long-prefill fused cases, comparative kernel/model
timings and inference memory comparisons remain NOT_RUN or HARDWARE_PENDING.
Additional model/hybrid coverage remains outstanding. Direct paged INT8 parity
remains DEFERRED; no paged tests ran here. No runtime fixes or new model/Metal
benchmarks were executed during this review.

## 11. Phase 1 acceptance gates and Phase 2 recommendation

| Gate | Evidence in latest M4 runs | Assessment |
|---|---|---|
| Final backend-free regression | 1,395 passed; 47 explicitly skipped | PASS for executed backend-free checks |
| M4/Metal preflight | Actual device work; correct chip/RAM | PASS |
| Qwen3 reference/dense batch state parity | Six cases; zero error | PASS for that scope |
| Complete supported fused operand coverage | 43 passed, 1 availability failure per correctness run; remaining cases unexecuted | **FAIL / incomplete** |
| Qwen3 whole-model fused parity and continuation | Test never reached | **NOT_RUN** |
| Qualified reference performance baseline | Five failed workers, 60 failed requests | **FAIL** |
| Fused speed improvement without unacceptable regressions | No valid paired timings | **NOT_RUN** |
| Inference memory comparison | Load-only peaks, no optimized pair | **NOT_RUN** |
| Remaining model/hybrid acceptance coverage | Not supplied by these runs | **NOT_RUN** |
| Overall Phase 1 acceptance | Required correctness and performance evidence absent | **NOT PASSED** |

**Recommendation: remain in Phase 1.** First repair the benchmark identity path
and the fused capability/failure-reporting gaps. Re-run affected backend-free
tests and regenerate source-bound hardware evidence after code changes. Then
complete fused operator coverage and run the actual Qwen3 checkpoint test.
Only after correctness passes should matched reference/fused kernel and
whole-model TTFT, prefill, decode, B1/B2/B4 and peak-memory comparisons be used
for acceptance. Keep reference defaults and known failing-route fallbacks;
do not promote a backend or advance the accepted milestone on this evidence.

Current reproduction commands, from the repository:

```sh
scripts/qualification/full.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
scripts/qualification/fused-full.sh --kernel-only
scripts/qualification/fused-correctness.sh --checkpoints qwen3-small --checkpoint-paths scripts/qualification/checkpoint-paths.local.json
```

The commands above reproduce the failed cohorts on the reviewed source; they
are not claims of successful qualification. Separate performance commands are
documented in [benchmarks.md](benchmarks.md) and require valid current correctness
receipts. Preserve A/B/C and write repaired runs to new output directories.
