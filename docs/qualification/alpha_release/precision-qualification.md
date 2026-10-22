> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Precision qualification

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Source and primary transforms are reported independently. Stored FP32 GPT-2 and BF16 Gemma share their source/primary result, rather than being counted as two executions. Q4/Q8 use group size 64, quantize stored weights then cast floating tensors to FP16. Saved artifacts verify every tensor and packed payload byte on reload; their artifact revision is distinct from source revision. Public attention KV storage is FP16, including Gemma/Phi sliding KV. LFM convolution history remains BF16 for stored weights and FP16 for Q4. Excluded Qwen3.5 carries FP32 recurrent state with BF16/FP16 convolution history. Raw canonical caches retain activation precision and are recorded separately in model-records.json; they are not the public FP16 KV path. Hardware records include actual output-logit dtypes.

| Model/role | Profile | Short cross | Cross max abs | Original frozen outcome | Evidence |
|---|---|---|---|---|---|
| M01 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M01 primary | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M02 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M02 primary | affine_q8 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M03 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M03 primary | affine_q8 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M04 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M04 primary | affine_q4 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M05 source | stored | FAIL | 0.3125 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M05 primary | affine_q4 | FAIL | 0.08203125 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M06 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M06 primary | affine_q4 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M07 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M07 primary | affine_q4 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M08 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M08 primary | affine_q4 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M09 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M09 primary | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M10 source | stored | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
| M10 primary | affine_q4 | PASS | 0 | FAIL_NUMERICAL | [raw](../archived-evidence.md) |
