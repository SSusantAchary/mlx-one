> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Cache and recurrent-state parity

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

| ID | Source cross-state components failing | Primary cross-state components failing | Source native full/decode max | Source native chunk max | Reset | Cache qualified |
|---|---:|---:|---:|---:|---|---|
| M01 | 0 | 0 | 0.002024173737 | 0.0002403259277 | PASS | NO |
| M02 | 0 | 0 | 0.28125 | 0.5625 | PASS | NO |
| M03 | 0 | 0 | 0.25 | 0.375 | PASS | NO |
| M04 | 0 | 0 | 0.171875 | 0.40625 | PASS | NO |
| M05 | 42 | 46 | 0.1875 | 0.234375 | PASS | NO |
| M06 | 0 | 0 | 0.34375 | 0.375 | PASS | NO |
| M07 | 0 | 0 | 0.1875 | 0.296875 | PASS | NO |
| M08 | 0 | 0 | 0.375 | 0.59375 | PASS | NO |
| M09 | 0 | 0 | 0.5 | 1.5625 | PASS | NO |
| M10 | 0 | 0 | 0.25 | 0.5 | PASS | NO |

Nine native/reference committed-state comparisons are exact for source and primary profiles. Full-versus-cached/chunk consistency is still required separately and fails canonical gates. The independent row controls remove the tested own discrepancies without changing cache semantics. This rules out a required cache-semantic change for those bounded control cases, not every possible future context.

Gemma3 layer window controls pass 511/512/513; Phi3 passes 2046/2047/2048. Local/global selection, retained length and absolute offsets remain distinct. Phi3's reference lacks its configured sliding window, so the beyond-window comparison explicitly supplies the HF-derived mask at the attention layer. It is not an unmodified full-model long-context oracle.

LFM2 now uses source Conv1d mathematics with its existing private convolution history and offset owner. Qwen3.5 keeps its verified FP32 convolution/decay fixes, source recurrence kernels and [B,H,K,V] owner with an explicit [B,H,V,K] boundary transpose. Exact first-layer FP64 state errors are 2.4782018415159257e-07 (source) and 3.0160777342302936e-07 (Q4); both pass FP32. Corrected convolution matches stock Conv1d exactly.

Prefix fingerprints include reference-first-v1 even without an explicit execution dtype. Qwen3.5 snapshots use qwen35-gated-delta-v2-reference-first; old transition snapshots are rejected, without deleting historical files. New cache-contract tests plus the existing cancellation, prefix restoration and compound snapshot tests verify this adapter.
