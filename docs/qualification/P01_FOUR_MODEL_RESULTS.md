# P0.1 four-model results

Status: **IN PROGRESS — not fully implemented or qualified**.

The frozen preintervention baseline is recorded at
`qualification/results/p01_math/baseline/1792231798413449000`. Reference timing
repetitions are not mathematical parity evidence. The previously completed
five-model P0 result applies to its own reference source/policy, not to P0.1.

| Model | Weights | Reference baseline | Optimized parity | Optimized performance | Lifecycle release gate |
|---|---|---|---|---|---|
| Qwen3-0.6B | INT8 | Completed; noise caveat | Head/vector 128 PASS; SDPA/matrix FAIL | NOT_RUN | State subset PASS; full pending |
| Qwen2.5-0.5B-Instruct | INT8 | Completed | Head/vector 128 PASS; SDPA unavailable | NOT_RUN | State subset PASS; full pending |
| LFM2-350M | 4-bit | Completed | Head 128 PASS | NOT_RUN | NOT_RUN |
| Qwen3.5-0.8B text | 4-bit | Completed; noise caveat | Head 128 PASS | NOT_RUN | NOT_RUN |

CPU candidate validation: 1,241 passed / 39 skipped on the earlier staged
snapshot, with two inherited warnings. A later targeted staged run passed 195
tests. Those receipts describe temporary CPU candidates, not an installed native
runtime or production hardware qualification. Final counts will be recorded
after the runtime source is installed and frozen again.

Pending native checks include ALL versus LAST_ONLY output shapes and logits,
stock SDPA/QMM dispatch and same-checkpoint tokens, full/cached/chunked component
state, streaming and cancellation, 20 load-close cycles, 100 generation cycles,
memory reconciliation, wheel smoke, and protected 8K performance.

No optimized scores are invented while these gates are incomplete. Final rows
must cite source digest, checkpoint revision, shape, policy, exact prompt and
forced/generated token fixture, source/evidence checksums and worker supervision.
Timings from instrumented runs remain separate from ordinary throughput.

## Installed Phase A validation

The installed candidate passed **1,262 backend-free tests**, with 50 skipped and
two inherited warnings. Scoped Ruff remains at 65 inherited findings, zero new.
Reference baseline: all 40 workers completed, but only 16/20 independent-worker
profiles met the declared 5% repeatability limit. Qwen3 and Qwen3.5 8K comparisons
remain pending a quieter matched repeat.

Native last-logits gates passed for both Qwen models at lengths 1, 2 and 128, in
two independent workers per shape. At 128 tokens, vocabulary output changed from
38,895,616 bytes to 303,872 bytes while hidden shapes and KV state were retained.
Qwen3 logits error was zero; Qwen2.5 maximum error was 0.0078125, within the
unchanged FP16 tolerance. This is not an end-to-end peak-memory result.

The initial SDPA experiment failed single-query decode state/logits parity while
greedy IDs agreed. It is retained under
`qualification/results/p01_math/phase-a-sdpa/1792238404864481000`. A failing CPU
gate regression was added, then the dispatcher was changed to select reference
attention for single-query/decode calls before cache mutation. The targeted CPU
repair run passed 82 tests; native revalidation passed all four workers (two per Qwen model) at 128 tokens. Tolerances were
not changed. Prior source evidence remains historical after this repair.

Native repaired SDPA artifacts:
`qualification/results/p01_math/phase-a-sdpa-repaired/1792238558593906000`.
The new evidence schema v3 additionally requires observed dispatch engagement;
older receipts remain readable diagnostics. Packed-QMM validation is now running.

## Packed projection findings

Full-state Qwen3 parity failed for both stock FP16 matrix QMM and a separately
recorded FP32-operand experiment (maximum logits errors 0.125 and 0.0387573).
Neither was promoted. Real projection probes showed exact vector outputs but
small matrix differences that accumulated through the model. Current dispatch
therefore retains rowwise reference projections for actual call rows greater
than one. The stock FP16 vector-only candidate subsequently passed both Qwen
checkpoints in two fresh workers each.
Failed artifacts remain in `phase-a-qmm` and `phase-a-qmm-fp32`; the independent
operator probe is in `qmm-accumulation-probe/1792238750141188000`.

## Nonuniform-prompt isolation

The head-only Qwen3 Engine state suite passed twice (D01–D10, E01–E10 and
optional P01). A supervised operator isolation run confirmed that vector-only
QMM also preserves native logits/state, while stock SDPA fails on a nonuniform
22-token prompt. FP32 transient attention operands did not remove that failure.
Both fused variants remain unavailable; passing repeated-input checks is not
a substitute for this broader state gate. Qwen2's nonuniform fused state gate
also failed. Reference attention plus head/vector execution passed all 21 state
cases in each of four fresh workers (two per Qwen model), recorded under
`phase-a-safe-state/1792240236670776000`. These state subsets do not establish
the full streaming/lifecycle or performance gates.

Evidence: `phase-a-last-only-state/1792239405077678000` and
`engine-state-probe/1792239517583211000`. The probe's field named
`engine_first_logit_comparison` compares token IDs and must not be interpreted
as a logits error; numerical conclusions use its `native.logits/components`,
actual dispatch events and recorded raw Engine logits.

## Current-source checkpoint regressions

The native opt-in run completed **6 passed, 1 failed** in 121.63 seconds.
All four target heads and both Qwen vector routes passed at 128 tokens in two
fresh workers each. Checksummed cohorts are under
`qualification/results/p01_math/checkpoints/` (timestamps beginning 17922412).
This is narrower than the complete length, chunk and lifecycle matrix.

GPT-2 LAST_ONLY preserved all 24 cached state fields exactly and passed the
logits/token comparisons. Its independent reference full-prefill versus cached
continuation failed two KV-key fields (layers 4 and 5, maximum absolute error
0.0234375); final logits and greedy IDs still passed. The unchanged FP16 tolerance
was applied elementwise. Artifact:
`checkpoints/gpt2-reference-reference/1792241255355943000/ee00227f728c6c26db8eda9f`.
This remains a failed regression gate, not an optimized speed result. GPT-2
expansion is stopped; reference arithmetic has not been altered to hide it.

## Latest local verification and ongoing work

The actual checkout passed **1,301 backend-free tests**, with **46 skipped** and
the same two dependency warnings. Scoped Ruff reports **65 inherited findings,
zero new**; diff checks pass. The wheel built and imported its Engine/math policy
without initializing MLX. Protected existing files were hash-checked: 116 present
files unchanged; six old UI asset deletions were already present before this work.

Qwen3's varied rendered-token manifest subsequently passed two workers each at
512 and 2048 tokens. Its 4096/8192 checks remain in progress in
`phase-a-varied-qwen3/1792241896814203000`.
The sequential shared-Qwen campaign is queued behind that cohort:
`active-shared-campaign/1792242056262828000/cohorts/1792242056865972000`.
It preserves parity, release, separate profiling and ten-repeat ABBA timing
boundaries. Execution failures and unexecuted dependencies remain explicit.
Neither an active campaign nor a successful timing worker is production
qualification. Later hybrid installation and final four-model performance
remain pending; no optimized 8K performance score is claimed here.
