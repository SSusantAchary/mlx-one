# Numerical validation

Frozen before new native runs, from the original `inference_baseline.TOLERANCES`:

| Dtype | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| FP32 | 0.0001 | 0.0001 |
| FP16 | 0.01 | 0.01 |
| BF16 | 0.05 | 0.01 |

Elementwise acceptance is `abs(actual-reference) <= atol + rtol*abs(reference)`.
Whole-model logits, complete component state, forced-token continuation, greedy
IDs, prefix restoration, principal isolation, seeded RNG and cleanup remain gates.
P1 cases retain page sizes 16/32 and B1/B2/B4 across all six pinned checkpoints.

New backend-free tests use explicitly simulated artifacts and NumPy tensors.
They verify admission and reference behavior, never establish Metal parity.
Native error values for this source: **HARDWARE_PENDING**.

Historical reports record Qwen SDPA nonuniform full-state failures, direct paged
INT8 parity failures and GPT-2 reference full-versus-cached KV failures. Those
reports apply to their recorded source snapshots. Their failed configurations
must not be promoted or silently inferred to pass on the new source. Existing
Qwen and single-query SDPA fallback rules remain enforced.
