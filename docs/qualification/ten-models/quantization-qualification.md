> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Quantization qualification

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

Source-port equivalence is established before the derived profile diagnostics. Since source self-consistency remains unresolved, no primary profile is promoted past its source prerequisite. Q4/Q8 uses group 64 affine packed weights, standard MLX row/output-channel orientation, FP16 activations/scales and architecture-owned state. The native path dispatches stock QuantizedLinear/QuantizedEmbedding, not silent full dequantization. Packed payloads, scale/bias and dequantized weights match the reference in all eight equal-operand controls; Qwen3.5's actual preceding operands can differ and a common reference operand is explicitly replayed.

| ID | Profile | Max cross logit error | Cross-state failures | Canonical numerical/state qualified |
|---|---|---:|---:|---|
| M01 | stored | 0 | 0 | NO |
| M02 | affine_q8 | 0 | 0 | NO |
| M03 | affine_q8 | 0 | 0 | NO |
| M04 | affine_q4 | 0 | 0 | NO |
| M05 | affine_q4 | 0.08203125 | 46 | NO |
| M06 | affine_q4 | 0 | 0 | NO |
| M07 | affine_q4 | 0 | 0 | NO |
| M08 | affine_q4 | 0 | 0 | NO |
| M09 | stored | 0 | 0 | NO |
| M10 | affine_q4 | 0 | 0 | NO |

All 18 distinct stored/derived parameter profiles have zero unexplained matched-parameter differences. Qwen3.5 norms use explicitly verified representation mapping; vision/MTP arrays absent from its text comparator are excluded, and their REVIEW_MAPPING_SCOPE status is retained. Saved Qwen2 exact-source Q4 and Q8 artifacts pass strict reload, Python, CLI, HTTP, chat and SSE regression with pinned artifact digests. The initial mistyped Qwen3.5 module path is preserved as a failed probe; the corrected language_model.layers.1.linear_attn.in_proj_qkv replay passes payload controls.

Quantization-induced task loss is separate: M02 changes a correct-but-misformatted 42 to wrong 32; M05 changes correct 42 to wrong 50 and loses strict YES; M08 loses retrieval and multiturn fixture accuracy. Reference and native have the same scored outcomes in every selected profile.
