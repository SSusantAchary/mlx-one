> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Hardware reference-dense characterization

Cohort: [reference-first-20261020-1792502347743318000](../archived-evidence.md). Frozen FP32 `(1e-4,1e-4)`, FP16 `(0.01,0.01)`, BF16 `(0.05,0.01)` tolerances are unchanged. Exact revisions, dtypes, token IDs and structured metrics are in the linked evidence.

Apple M4 / 32 GiB; macOS 15.7.9; Python 3.12.11; MLX 0.32.2. B1 reference_dense only, no reference model resident, no batching/paged/speculative promotion. Public Engine benchmarks use default FP16 KV; source weights/logits retain their stated precision, and Qwen3.5 recurrent state is FP32. These are separate from the raw-cache numerical profiles and the matched-storage quality profiles. Each cell has a matching-shape warmup and three measured requests, prefix reuse disabled. Requested contexts 16/128/512/1024/2048/4096 with output 16; unsupported context-limit cells remain explicit. M03 additionally measures context 128 and output budgets 64/128/256. This is characterization, not release-qualified performance while numerical gates fail.

| ID | Profile | Actual input / output (three repeats) | Median warm TTFT ms | Prefill TPS | Decode TPS | Peak inference MLX GiB | Sampled inference RSS GiB | Peak cache MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M01 | stored | [128, 128, 128] / [16, 16, 16] | 54.60 | 4872.05 | 105.74 | 0.594 | 0.662 | 5.027 |
| M02 | affine_q8 | [128, 128, 128] / [16, 16, 16] | 97.01 | 2456.33 | 73.61 | 0.632 | 0.294 | 1.676 |
| M03 | affine_q8 | [128, 128, 128] / [16, 16, 16] | 154.97 | 1481.08 | 61.12 | 0.816 | 0.413 | 15.641 |
| M04 | affine_q4 | [128, 128, 128] / [16, 16, 16] | 84.98 | 2756.44 | 156.51 | 0.326 | 0.273 | 1.715 |
| M05 | affine_q4 | [128, 128, 128] / [16, 16, 16] | 285.68 | 610.36 | 30.01 | 0.598 | 0.441 | 20.309 |
| M06 | affine_q4 | [128, 128, 128] / [16, 16, 16] | 393.07 | 380.38 | 38.21 | 1.202 | 0.424 | 26.812 |
| M07 | affine_q4 | [128, 128, 128] / [16, 16, 16] | 227.95 | 750.68 | 71.36 | 0.886 | 0.415 | 4.469 |
| M08 | affine_q4 | [128, 128, 128] / [16, 16, 16] | 378.14 | 418.68 | 41.90 | 1.034 | 0.316 | 3.910 |
| M09 | stored | [128, 128, 128] / [16, 16, 16] | 238.90 | 719.03 | 26.41 | 2.028 | 2.315 | 3.631 |
| M10 | affine_q4 | [129, 129, 129] / [16, 16, 16] | 996.73 | 137.14 | 23.94 | 2.362 | 0.290 | 54.000 |

Peak MLX memory resets after load/conversion/warmup and before each measured inference request. Reference model memory is absent. Load/conversion MLX and lifetime process RSS peaks are separately recorded. Inference RSS is sampled at 50ms intervals and includes the resident model/allocator; it is not an exact resettable process RSS high-water mark. Observed cache/state bytes are captured before cleanup. Every measured request records actual generated token IDs, EOS/finish reason, actual logits dtypes and active-handle/reservation cleanup. Early EOS is not replaced with fabricated output-token counts.

All measured contexts, raw repeats, load/materialization/conversion times, memory and unsupported cells: [hardware rollup](../archived-evidence.md) and [worker supervision](../archived-evidence.md).

The bounded 240-cell primary/B1 coverage map records 59 measured PASS cells, 16 context-limit UNSUPPORTED cells (including statically unsupported unexecuted combinations), and 165 NOT_RUN cells. The executed campaign requested 63 cells: 4 were unsupported and 177 measured repeats completed with zero handles/reservations. [Explicit coverage](../archived-evidence.md).
