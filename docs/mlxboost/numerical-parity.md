# Numerical gates and retained failures

The unchanged policy is `mlx_one.inference_baseline.TOLERANCES`: FP32
`atol=rtol=1e-4`, FP16 `atol=rtol=0.01`, BF16 `atol=0.05, rtol=0.01`.
Neither that file nor existing numerical tests were modified.

**Policy caveat (2026-11-02):** These strict gates remain diagnostics and
model-qualification requirements. An unchanged inherited failure does not
automatically block a mathematically neutral optimization with demonstrated
before/after numerical, state, output and functional equivalence. Optimization
promotion and model qualification are separate decisions under the
[promotion policy](promotion-policy.md). Failed diagnostics retain their failing
results and exit codes; affected numerical execution changes still require
qualification.

`benchmarks/mlxboost/qualify.py` loads the verified MLX-LM Git tree and native
MLX-One against the same MLX 0.32.2 backend and cached checkpoint. Git excludes
the generated `_version.py`; the harness supplies only this generated metadata
in memory. Implementation files are hash checked against the capture. No upstream
runtime is imported by production MLX-One.

The diagnostic uses a 128-token raw prompt and 128 source-forced steps. It
compares every step's full last-row logits, all layers' K/V at prefill and steps
16/128, and native cached versus fresh-full-prefix logits at those boundaries.
It retains errors, failing element counts, greedy decisions and raw logits. It
does not stop a forced numerical trace at EOS or count it as HTTP throughput.

| Checkpoint / exact execution precision | Source logits/KV gate | Forced greedy agreement | Cached/fresh gate |
| --- | --- | --- | --- |
| Qwen2.5 Q8, FP16/KV FP16 | PASS | 128/128 | FAIL at steps 16 and 128 |
| Qwen3 Q8, BF16/KV BF16 | PASS | 128/128 | FAIL at steps 16 and 128 |
| SmolLM2 BF16/KV BF16 | PASS | 128/128 | FAIL at steps 16 and 128 |

Worst cached/fresh absolute error at steps 16/128:

| Model | Step 16 / failing elements | Step 128 / failing elements |
| --- | --- | --- |
| Qwen2.5 | 0.029296875 / 671 | 0.0341796875 / 974 |
| Qwen3 | 0.34375 / 33764 | 0.1875 / 13642 |
| SmolLM2 | 0.3359375 / 12028 | 0.265625 / 9771 |

These are pre-optimization failures, preserved at
`local/mlxboost/runs/parity-baseline-{qwen2,qwen3,smollm2}-20261101/`.
Source agreement on identical partitions does not establish cached/fresh
self-consistency or equivalence to upstream's 127+1 generation prefill partition.
It does not resolve the alpha release's existing failures.

Post-change requalification at
`local/mlxboost/runs/source-requalification-20261101/qualify-*/` additionally
compares the pinned upstream's own cached and fresh calls. **The upstream also
fails at the same boundaries with the same maximum errors.** Source-to-native
agreement still passes. The harness exits 2 for these failed numerical gates;
they are not converted to passing tests. This narrows the next investigation to
shared execution/shape-dependent numerical behavior; it does not authorize
relaxing the policy or attributing the failure to a native port defect.

The independent flag comparison is `benchmarks/mlxboost/ablate.py`. It exercises
the public Engine with a live non-calibration registry, both `reference_dense`
and `mlxboost_b1` flag-off/on, checking every generated logit and selected full KV
states. Completed Qwen2.5 and Qwen3 comparisons are bit-identical (maximum absolute
error zero), including raw IDs, exact text, terminal ID, finish reason and counts.
Final per-model results are recorded in `optimization-ablation.md` and its raw
run directories. An optimization self-comparison PASS never changes the failed
baseline self-consistency gate to PASS.

No full embedding/Q/K/V/RoPE/MLP intermediate trace, long-context checkpoint
qualification, or full quality-corpus qualification was completed. The change
does not modify those operations. General model promotion remains NO-GO.
