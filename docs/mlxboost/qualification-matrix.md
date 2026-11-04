# MLXBoost outcome: PARTIAL / experimental, no promotion

One independently reversible host optimization is implemented and measured.
It removes impossible activation-calibration lookup work, not model operations.
The performance criterion is supported by repeated measurements. Under the
[2026-11-02 promotion policy](promotion-policy.md), the unchanged strict numerical
failures block model qualification but do not automatically block independent
promotion of this mathematically neutral optimization. Its current opt-in
EXPERIMENTAL deployment status is retained by this documentation edit. Default execution
remains `reference_dense`, alpha version remains 0.1.0a1, and no release status
has been upgraded.

| Checkpoint / precision / device | Same-partition pinned source parity | Reference/off/on state and exact output | Cached/fresh model qualification | Current optimization status |
| --- | --- | --- | --- | --- |
| Qwen2.5 Q8, FP16/KV FP16, M4 | PASS, 128 forced steps | PASS; 544 tensor comparisons, zero change | FAIL in native and pinned source | EXPERIMENTAL |
| Qwen3 Q8, BF16/KV BF16, M4 | PASS, 128 forced steps | PASS; 592 tensor comparisons, zero change | FAIL in native and pinned source | EXPERIMENTAL |
| SmolLM2 BF16/KV BF16, M4 | PASS, 128 forced steps | PASS; 544 tensor comparisons, zero change | FAIL in native and pinned source | EXPERIMENTAL |
| LFM2 Q4, FP16, M4 | PASS; 194 logits/KV/conv-carry comparisons | No optimization enabled | Broad self-consistency/quality not qualified | Baseline diagnostic only; optimization BLOCKED |
| Llama-3.2 Q4 | NOT MEASURED; matching Q4 artifact not resolved | NOT MEASURED | NOT QUALIFIED | BLOCKED |

Checkpoint revisions and full source/file hashes are in the packaged provenance,
run manifests and performance summary. The LFM2 cached Q4 revision is
`18dc72abf3b2337f9123cfd6eeeb58dfa7947066`; no new conversion was performed.
Its native diagnostic baseline is 254.73 tok/s, one process/two timed requests,
not an HTTP comparison or optimization speedup.

| Optimization / proposal | Evidence and disposition |
| --- | --- |
| `fast_decode` host lookup shortcut | Currently opt-in EXPERIMENTAL. 50 paired timed requests per flag per Qwen checkpoint: 4.77× / 4.50× median process gains, exact output agreement. Eligible for independent scoped promotion assessment; inherited numerical failures are model caveats, not an automatic veto. |
| Flag-off and `reference_dense` | Retained rollback/oracle; no new performance claim or model qualification. |
| Native quantized dispatch wrapper | Not implemented: existing source-faithful model already uses stock packed dispatch; no measured need for another wrapper. |
| Dense capacity / fused attention | Not implemented: existing dense cache already retains capacity; no demonstrated target bottleneck. Any change affecting numerical/state execution requires affected qualification. |
| Stream bridge / async evaluator | Not implemented: profile residual cost after the measured metadata fix. Demonstrate neutrality or qualify affected numerical/state behavior, plus API/cancellation behavior. |
| Custom QGEMV / residual-RMSNorm / compile | Entry gate NOT MET: no stock-kernel bottleneck proof; all switches disabled/unavailable with fallback. |
| Combined path | Only the single host flag exists; no unqualified multi-optimization combination. |

| Phase gate | Outcome |
| --- | --- |
| 0: source capture / source audit / original benchmark | PASS with retained source hashes and explicit metadata limits |
| 1: source math and functional fidelity | PARTIAL: source math passes identical-partition checks; cached/fresh and exact upstream output/termination qualification remain unresolved |
| 2: profiling | PASS for host attribution; no kernel-level GPU capture |
| 3: independent optimization / re-parity | Experimental host change passes bitwise change comparisons and measured HTTP ablations |
| 4: custom kernels / compilation | NOT ENTERED; prerequisites not met |
| 5: qualification / promotion | NO-GO for model/release promotion; isolated regression coverage, API/wheel evidence and repeated performance retained |

**Caveat:** Strict numerical failures remain FAIL. A neutral optimization can
qualify independently when measured comparisons show no introduced or worsened
failure. This policy edit changes no defaults or deployment status and issues no
new model qualification; see the [full caveat note](promotion-policy.md#caveat-note).

The monolithic regression hang predates this change and was independently
reproduced from unmodified HEAD. All regression cases were exercised in isolated
runs: 2192 passed, 79 skipped, 9 subtests passed. Explicit native primitive tests
add six passing opt-in cases. This is not a claim of a passing monolithic suite.
See [API/regression evidence](api-regression.md).

Read the evidence in this order:

1. [Source audit](source-audit.md) and [source diff map](source-diff-map.md).
2. [Baseline reproduction](baseline-reproduction.md) and [profile attribution](profiling-bottlenecks.md).
3. [Numerical gates](numerical-parity.md) and [exact output](output-parity.md).
4. [Independent ablation](optimization-ablation.md), [performance comparison](benchmark-comparison.md), and [machine-readable summary](performance-summary.json).
5. [API/wheel results](api-regression.md) and [deployment/rollback](deployment-and-rollback.md).

Further work, ranked by the observed prerequisites:

1. Trace shared cached/fresh numerical failures at embeddings, Q/K/V, RoPE,
   attention, MLP, residual and logits against both pinned implementations.
   Preserve the existing thresholds and distinguish source behavior from native
   defects before changing any model math.
2. Resolve exact upstream generation/output/EOS compatibility across the original
   and matched dependency stacks; the 0.32.2 native probe did not reproduce the
   0.32.4 HTTP SmolLM2 short EOS. Do not trim native text to manufacture parity.
3. Reprofile residual native/HTTP overhead independently of inherited model
   failures. Evaluate neutral host changes using change comparisons; head
   projection, normalization fusion or scheduling changes that affect numerical
   execution require affected numerical/state qualification.
4. Qualify longer-context attention and LFM2 hybrid state on valid pinned
   baselines before targeted attention or hybrid fusion experiments.
5. Speculative decode and continuous batching remain later, independent work,
   contingent on qualified B1 models and workload evidence.

Raw evidence remains under ignored `local/mlxboost/runs/`. No model weights were
copied into the benchmark artifacts. Missing thermal/power measurements,
long-context qualification, full activation traces, corpus quality and real
mid-prefill cancellation/backpressure coverage remain explicit limitations.
