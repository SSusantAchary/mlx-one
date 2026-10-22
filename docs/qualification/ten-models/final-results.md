> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# MLX-One 0.1.0a1 — numerical correction qualification

Cohort: `numerical-correction-20261020-1792496951619201000`. **Alpha: NO-GO.** Architectures implemented: **8/8**; exact checkpoints executed: **10/10**. [Machine-readable engineering summary](../archived-evidence.md). The previous report and all previous evidence are preserved in the original cohort and this run's `prior-reports/` snapshot.

Two proven Qwen3.5 defects are corrected: low-precision convolution product accumulation and promotion before the recurrent decay exponential. The folded-norm loader correction is preserved. No engine replacement, API rewrite, tolerance change, broad dtype policy change or experimental backend promotion occurred.

| ID / model | Reference oracle | Native full A/B | Cache state | Primary precision / numerical | Task quality | API | Qualified |
|---|---|---|---|---|---|---|---|
| M01 openai-community/gpt2 | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | float32 / FAIL_NUMERICAL | 2/2 | PASS | NO |
| M02 Qwen/Qwen2.5-0.5B-Instruct | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q8 / FAIL_NUMERICAL | 6/8 | PASS | NO |
| M03 Qwen/Qwen3-0.6B | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q8 / FAIL_NUMERICAL | 5/8 | PASS | NO |
| M04 LiquidAI/LFM2-350M | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 4/8 | PASS | NO |
| M05 Qwen/Qwen3.5-0.8B | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 5/8 | PASS | NO |
| M06 HuggingFaceTB/SmolLM2-1.7B-Instruct | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 5/8 | PASS | NO |
| M07 meta-llama/Llama-3.2-1B-Instruct | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 7/8 | PASS | NO |
| M08 Qwen/Qwen2.5-1.5B-Instruct | FAIL_NUMERICAL | FAIL | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 5/8 | PASS | NO |
| M09 google/gemma-3-1b-it | FAIL_NUMERICAL | PASS | FAIL_NUMERICAL | bfloat16 / FAIL_NUMERICAL | 7/8 | PASS | NO |
| M10 microsoft/Phi-3-mini-4k-instruct | FAIL_NUMERICAL | PASS | FAIL_NUMERICAL | affine_q4 / FAIL_NUMERICAL | 5/8 | PASS | NO |

Full A/B PASS is cross-implementation parity only. Cache state requires native prefix/chunk consistency and committed cross-implementation state parity. Reference oracle validation requires full/cache and chunk self-consistency. Numerical qualification also retains the original 17-boundary, forced-continuation and reset gates. A scope-specific PASS never overrides a failed checkpoint gate.

Gemma3/Phi3 fail the same canonical precision-level self-consistency gate in both implementations. Equal-input projections reproduce an early query-count-dependent difference; fixed-row diagnostic operators remove the tested discrepancy without changing cache semantics. Those substituted paths are diagnostic only; Phi3 additionally requires contiguous Q/K/V layouts in this control. Exact-weight sliding-window layer controls pass. Other families retain explicitly measured attention/projection and downstream state failures. [Root-cause analysis](../archived-evidence.md), [oracle comparisons](oracle-self-consistency.md), [quality requalification](generation-quality-requalification.md).

Python 10/10, CLI 10/10, HTTP completions 10/10, chat 9/9, completion SSE 10/10 and chat SSE 9/9 pass. Cancellation, restart, usage, stop handling, structured errors and cache/lease cleanup are in the raw API checks. Saved Q4/Q8 artifacts pass fresh strict-load Python/CLI/HTTP regressions. In-memory quantization still has no CLI selector; its CLI result remains UNSUPPORTED. Single-startup-model residency remains the policy.

Default: **1812 passed, 53 skipped, 3 warnings in 62.36s (0:01:02)**. Native: **326 passed, 2 warnings in 8.68s**. Ruff: **0 new**, 65 inherited. Fresh installed-wheel inference/HTTP/SSE smoke: **PASS**, with MLX-LM/Transformers imports rejected and no such packages installed. All 316 production Python files match the wheel.

Fresh entry measurements were 1,783 default tests passed / 47 skipped and 264 passed in the selected native baseline. Historical 1,779/51 and 285-native results remain historical; they are not reused as new measurements. The final native selection includes additional architecture contracts and new comparator/invariant tests. Exact final command lists are in the cohort's `regression-summary.json`.

No hardware extension was run because mathematical/state gates remain unresolved. Prior 6,750-cell coverage remains 142 hardware-execution PASS, 5,408 NOT_RUN and 1,200 UNSUPPORTED; these are historical measurements, not new qualification. Historical timeouts remain intact. Concurrent correctness probes are not performance benchmarks; process peaks are not used for memory-efficiency claims.

```text
Architectures implemented: 8/8
Exact checkpoints executed: 10/10
Reference oracles validated: 0/10
Stored-weight numerical PASS: 0/10
Primary-precision numerical PASS: 0/10
Cached-state PASS: 0/10
Generation-quality PASS: 1/10
Fully qualified checkpoints: 0/10
Python API: 10/10
CLI: 10/10
HTTP completions: 10/10
Chat: 9/9
SSE: 10/10 completion; 9/9 chat
Default tests: 1812 passed, 53 skipped, 3 warnings in 62.36s (0:01:02)
Native tests: 326 passed, 2 warnings in 8.68s
New Ruff findings: 0
Packaging: PASS
Alpha: NO-GO
```
