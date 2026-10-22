> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Numerical execution shapes

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

The original frozen gates are unchanged. Full and cached execution share logical inputs but can select different projection and SDPA reductions. Synthetic controls include conditioned/cancellation-sensitive cases, Q4/Q8 exact affine decoding in FP64, RMS normalization and fused/unfused GQA. A failing FP32-accumulation bound for quantized arithmetic is retained; the alternative low-precision affine bound is diagnostic and is not a release waiver. Trained-operand controls compare full and one-query SDPA against NumPy FP64.

Strict synthetic operation probes: 40/54 PASS; all 14 failures are quantized affine GEMM/GEMV probes against an assumed FP32 accumulation bound. This assumption is not a proven MLX kernel contract. Trained attention comparisons: 185/186 PASS. The failing witness is Gemma stored BF16, prefix 128, layer 0, full_query_last: max absolute error 0.13529943094275865, RMS 0.017463664188412793, one failing element. Both single-query view and contiguous forms pass the unchanged BF16 gate on the identical Q/K/V. No cache participates. This proves that full-forward is not automatically an accuracy oracle; it does not identify a universal root cause for all model drift. Three trained Q4/Q8 projection sets also pass identical-input, packed-payload, dequantization and native/reference controls.

| Model/profile | Context | Centered logit max | KL nats max | Top-1 changes | High-margin flips | Proposed own gate |
|---|---|---|---|---|---|---|
| M01 source | 16 | 0.0018348694 | 2.4622615e-07 | 0 | 0 | PASS |
| M01 source | 128 | 0.0020446777 | 1.3759219e-10 | 0 | 0 | PASS |
| M01 source | 512 | 0.0064468384 | 9.1734789e-09 | 0 | 0 | FAIL |
| M01 source | 1024 | 0.0034637451 | 7.7488053e-10 | 0 | 0 | PASS |
| M01 primary | 16 | 0.0018348694 | 2.4622615e-07 | 0 | 0 | PASS |
| M01 primary | 128 | 0.0020446777 | 1.3759219e-10 | 0 | 0 | PASS |
| M01 primary | 512 | 0.0064468384 | 9.1734789e-09 | 0 | 0 | FAIL |
| M01 primary | 1024 | 0.0034637451 | 7.7488053e-10 | 0 | 0 | PASS |
| M02 source | 16 | 0.77929688 | 0.00017742368 | 0 | 0 | FAIL |
| M02 source | 128 | 0.28320312 | 9.8130632e-07 | 0 | 0 | FAIL |
| M02 source | 512 | 0.21875 | 2.2028143e-05 | 0 | 0 | FAIL |
| M02 source | 1024 | 0.22753906 | 1.5605302e-06 | 0 | 0 | FAIL |
| M02 source | 2048 | 0.17578125 | 2.1829166e-06 | 0 | 0 | FAIL |
| M02 primary | 16 | 0.099609375 | 4.072264e-06 | 0 | 0 | FAIL |
| M02 primary | 128 | 0.048828125 | 2.783764e-08 | 0 | 0 | FAIL |
| M02 primary | 512 | 0.073730469 | 9.6474325e-07 | 0 | 0 | FAIL |
| M02 primary | 1024 | 0.099609375 | 2.2598977e-08 | 0 | 0 | FAIL |
| M02 primary | 2048 | 0.037109375 | 1.2542503e-08 | 0 | 0 | FAIL |
| M03 source | 16 | 0.25 | 0.00054276618 | 0 | 0 | FAIL |
| M03 source | 128 | 0.18884277 | 3.4613442e-07 | 0 | 0 | FAIL |
| M03 source | 512 | 0.234375 | 4.7693579e-05 | 0 | 0 | FAIL |
| M03 source | 1024 | 0.21875 | 1.410581e-07 | 0 | 0 | FAIL |
| M03 source | 2048 | 0.203125 | 2.0132644e-07 | 0 | 0 | FAIL |
| M03 primary | 16 | 0.033203125 | 1.5387864e-06 | 0 | 0 | FAIL |
| M03 primary | 128 | 0.034179688 | 3.2408461e-08 | 0 | 0 | FAIL |
| M03 primary | 512 | 0.033691406 | 2.8813398e-06 | 0 | 0 | FAIL |
| M03 primary | 1024 | 0.028320312 | 6.4901016e-09 | 0 | 0 | FAIL |
| M03 primary | 2048 | 0.03125 | 1.5655417e-08 | 0 | 0 | FAIL |
| M04 source | 16 | 0.4296875 | 0.0015282394 | 0 | 0 | FAIL |
| M04 source | 128 | 0.14111328 | 4.572211e-07 | 0 | 0 | FAIL |
| M04 source | 512 | 0.1328125 | 0.00064807498 | 0 | 0 | FAIL |
| M04 source | 1024 | 0.140625 | 1.1375169e-06 | 0 | 0 | FAIL |
| M04 source | 2048 | 0.15625 | 8.9115084e-05 | 0 | 0 | FAIL |
| M04 primary | 16 | 0.07421875 | 1.4186273e-05 | 0 | 0 | FAIL |
| M04 primary | 128 | 0.026367188 | 4.3162152e-08 | 0 | 0 | FAIL |
| M04 primary | 512 | 0.030273438 | 0.00013882109 | 0 | 0 | FAIL |
| M04 primary | 1024 | 0.022949219 | 4.2433093e-06 | 0 | 0 | PASS |
| M04 primary | 2048 | 0.026367188 | 4.1537599e-05 | 0 | 0 | FAIL |
| M06 source | 16 | 0.265625 | 0.0018447921 | 0 | 0 | FAIL |
| M06 source | 128 | 0.171875 | 8.5980626e-07 | 0 | 0 | FAIL |
| M06 source | 512 | 0.15625 | 3.6825601e-05 | 0 | 0 | FAIL |
| M06 source | 1024 | 0.25390625 | 1.4050954e-07 | 0 | 0 | FAIL |
| M06 source | 2048 | 0.234375 | 8.6601579e-08 | 0 | 0 | FAIL |
| M06 primary | 16 | 0.041992188 | 9.9324601e-05 | 0 | 0 | FAIL |
| M06 primary | 128 | 0.044921875 | 1.2923641e-08 | 0 | 0 | FAIL |
| M06 primary | 512 | 0.036865234 | 2.6490988e-07 | 0 | 0 | FAIL |
| M06 primary | 1024 | 0.046386719 | 1.7288972e-09 | 0 | 0 | FAIL |
| M06 primary | 2048 | 0.036132812 | 2.3389328e-09 | 0 | 0 | FAIL |
| M07 source | 16 | 0.140625 | 0.00059649371 | 0 | 0 | FAIL |
| M07 source | 128 | 0.1171875 | 4.0293981e-07 | 0 | 0 | FAIL |
| M07 source | 512 | 0.140625 | 0.00018902606 | 0 | 0 | FAIL |
| M07 source | 1024 | 0.12890625 | 3.9431074e-07 | 0 | 0 | FAIL |
| M07 source | 2048 | 0.140625 | 1.4286851e-05 | 0 | 0 | FAIL |
| M07 primary | 16 | 0.022460938 | 1.7211083e-05 | 0 | 0 | PASS |
| M07 primary | 128 | 0.027832031 | 6.9714311e-08 | 0 | 0 | FAIL |
| M07 primary | 512 | 0.029296875 | 4.8515035e-06 | 0 | 0 | FAIL |
| M07 primary | 1024 | 0.02734375 | 1.331157e-07 | 0 | 0 | FAIL |
| M07 primary | 2048 | 0.025390625 | 1.3790646e-08 | 0 | 0 | FAIL |
| M08 source | 16 | 0.60375977 | 0.00063271864 | 0 | 0 | FAIL |
| M08 source | 128 | 0.21142578 | 3.0549981e-06 | 0 | 0 | FAIL |
| M08 source | 512 | 0.24902344 | 3.4218104e-06 | 0 | 0 | FAIL |
| M08 source | 1024 | 0.203125 | 6.2701322e-07 | 0 | 0 | FAIL |
| M08 source | 2048 | 0.21875 | 9.3096927e-07 | 0 | 0 | FAIL |
| M08 primary | 16 | 0.067382812 | 2.8104408e-05 | 0 | 0 | FAIL |
| M08 primary | 128 | 0.03515625 | 1.3336934e-07 | 0 | 0 | FAIL |
| M08 primary | 512 | 0.03125 | 7.4990505e-08 | 0 | 0 | FAIL |
| M08 primary | 1024 | 0.03515625 | 6.1380345e-08 | 0 | 0 | FAIL |
| M08 primary | 2048 | 0.02734375 | 2.8850607e-08 | 0 | 0 | FAIL |
| M09 source | 16 | 0.671875 | 0.0090953279 | 0 | 0 | FAIL |
| M09 source | 128 | 0.28125 | 4.5441385e-08 | 0 | 0 | FAIL |
| M09 source | 512 | 0.28125 | 0.0018321779 | 0 | 0 | FAIL |
| M09 source | 1024 | 0.75 | 2.5618408e-08 | 0 | 0 | FAIL |
| M09 source | 2048 | 1 | 6.735607e-08 | 0 | 0 | FAIL |
| M09 primary | 16 | 0.671875 | 0.0090953279 | 0 | 0 | FAIL |
| M09 primary | 128 | 0.28125 | 4.5441385e-08 | 0 | 0 | FAIL |
| M09 primary | 512 | 0.28125 | 0.0018321779 | 0 | 0 | FAIL |
| M09 primary | 1024 | 0.75 | 2.5618408e-08 | 0 | 0 | FAIL |
| M09 primary | 2048 | 1 | 6.735607e-08 | 0 | 0 | FAIL |
| M10 source | 16 | 0.25 | 0.00099042813 | 0 | 0 | FAIL |
| M10 source | 128 | 0.25 | 1.0452593e-05 | 0 | 0 | FAIL |
| M10 source | 512 | 0.25 | 9.4582967e-06 | 0 | 0 | FAIL |
| M10 source | 1024 | 0.25 | 3.6931044e-06 | 0 | 0 | FAIL |
| M10 source | 2048 | 1 | 1.2467851e-06 | 0 | 0 | FAIL |
| M10 primary | 16 | 0.03125 | 1.3194855e-05 | 0 | 0 | FAIL |
| M10 primary | 128 | 0.0390625 | 1.5277133e-07 | 0 | 0 | FAIL |
| M10 primary | 512 | 0.03125 | 4.7693441e-08 | 0 | 0 | FAIL |
| M10 primary | 1024 | 0.03125 | 1.6081557e-07 | 0 | 0 | FAIL |
| M10 primary | 2048 | 0.03125 | 1.6865833e-07 | 0 | 0 | FAIL |

[trained-controls](../archived-evidence.md).
