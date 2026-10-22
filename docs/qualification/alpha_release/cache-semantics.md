> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Cache semantics

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

9/9 candidates pass the actual-model append and attention-visibility audits in both source and primary profiles. Coverage is 16/128/512/1024/2048 where supported (GPT-2 ends at 1024), whole prefill, N−1+1, thirds and blocks of 128. Every attention layer checks the expected absolute visibility matrix; every native state update compares logical incoming/history values exactly at FP16 storage, or architecture-owned convolution precision. These probes instrument evaluation boundaries and remain separate from the unmodified frozen gate. Analytical NumPy controls verify projection layout, RoPE positions, Q/K norms, masks, sliding eviction, FP16 representation, LFM convolution history and reset. No paging, quantized KV or batching certification follows from these dense/sliding/convolution results.

[model-records.json](../archived-evidence.md).
