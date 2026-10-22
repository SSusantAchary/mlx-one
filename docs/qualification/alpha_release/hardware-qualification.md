> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Hardware qualification

Cohort: `proof-20261021-1792544161047727000`. Tested source commit: `05a3b5f27f6a596bacb5e3e3c5199fc8e5948f51`.

Measured M4 characterization only; numerical qualification remains failed. Three warm repetitions after shape warmup, one model worker at a time, B1 reference_dense, prefix reuse disabled. Peak inference MLX resets after conversion/warmup. RSS is sampled during inference and includes resident model/allocator; lifetime conversion RSS is separately labeled. Input requests are 16/128/512/1024/2048/4096 with output 16; context 128 also has output budgets 64/128/256. Actual IDs, counts, EOS/finish reasons, raw repeats and post-request handles are retained.

| Model | Actual input / output (budget) | TTFT ms | Prefill tok/s | Decode tok/s | Peak MLX GiB | State MiB |
|---|---|---|---|---|---|---|
| M01 | 128 / 16 (16) | 59.77 | 3996.84 | 100.84 | 0.594 | 5.027 |
| M02 | 128 / 16 (16) | 84.37 | 2487.72 | 84.92 | 0.632 | 1.676 |
| M03 | 128 / 16 (16) | 114.52 | 1905.05 | 65.89 | 0.801 | 15.641 |
| M04 | 128 / 16 (16) | 67.59 | 3560.22 | 171.98 | 0.326 | 1.715 |
| M05 | 128 / 16 (16) | 230.36 | 704.73 | 37.31 | 0.598 | 20.309 |
| M06 | 128 / 16 (16) | 385.71 | 381.11 | 50.09 | 1.202 | 26.812 |
| M07 | 128 / 16 (16) | 224.97 | 740.52 | 73.49 | 0.886 | 4.469 |
| M08 | 128 / 16 (16) | 304.29 | 552.64 | 48.91 | 1.034 | 3.910 |
| M09 | 128 / 16 (16) | 200.52 | 987.58 | 28.22 | 2.028 | 3.631 |
| M10 | 129 / 16 (16) | 894.84 | 153.70 | 25.51 | 2.356 | 54.000 |

No cross-framework speed claim is made. [hardware](../archived-evidence.md).

Campaign totals: 90 requested cells, 86 measured, 4 unsupported, 258 measured repetitions. All measured cleanup checks pass. GPT-2 rejects 1024/2048/4096 input plus 16 output, Phi3 rejects 4096 plus 16. [Aggregate](../archived-evidence.md).
