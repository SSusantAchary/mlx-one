# Stock reference parity investigation

**OPEN — shape-dependent quantized arithmetic is localized; no complete fix or model qualification.** G03 remains unqualified. No reference precision, weight, model source, cache convention, attention path or tolerance was changed.

The original FP16 Qwen2.5-0.5B Q8 snapshot again gives cached/fresh maximum logit error **0.03515625**, outside existing `allclose(atol=.01,rtol=.01)`. Repeating the same cache path is bit-identical. Observing Python call/return frames without replacing modules or forcing tensors in callbacks gives the exact ordinary fresh output. Masks are original causal/full versus single-token None; offsets128→129 versus fresh129 and ordinary offset-based RoPE are recorded. Fixed token IDs and the forced token are in reference-repro-final/3e6ad8c757760f3960b2fd9c80826a02bbec5d33d296fc7b3c44e56d885acf39.json (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/reference-repro-final/3e6ad8c757760f3960b2fd9c80826a02bbec5d33d296fc7b3c44e56d885acf39.json`).

| Control | Bit-identical | Existing policy pass | Max absolute error |
| --- | --- | --- | --- |
| Cached path repeated | True | True | 0.0 |
| Observed fresh vs plain fresh | True | True | 0.0 |
| Cached vs fresh final logits | False | False | 0.03515625 |
| Cached vs fresh final hidden | False | False | 0.125 |
| 64-token chunks vs128-token prefill | False | False | 0.02734375 |
| Tokenwise prompt vs128-token prefill | False | False | 0.0234375 |

## First divergence

Embedding/first normalized input and Q-projection inputs agree exactly. First bitwise divergence is `model.layers.0.self_attn.q_proj.return`: max **0.0078125**. First policy failure is `model.layers.3.post_attention_layernorm.return`: max **0.02294921875**. Q/K/V, rotary outputs, attention inputs/results, residual sums, MLP inputs/projections, final hidden, logits and full/prefix/new-position KV state are traced. The JSON records maximum absolute/relative errors, representative coordinates and values, shape/dtype and exact/policy decisions. Cached and fresh greedy argmax agree for this step; that does not waive numeric/state failures.

The observer captures inputs at **call time** because MLX QuantizedLinear reassigns its local `x`;4D token axis is 2, not head axis 1. An early observer mislabeled return-time locals/4D axes; its preliminary capsule is retained but excluded, and its v1 source snapshot was not retained. Dedicated backend-free observer tests catch both errors. Final evidence above is from the corrected observer. Numeric trace inspection happens only after normal logits eval/synchronize; observed-vs-plain exactness controls scheduling effects.

## Minimal operation-level reproducer

Use original uint32 packed weights, scales/biases and the exact normalized vector; repeat that vector as rows of a contiguous matrix. [mlx `mlx/backend/metal/quantized.cpp:85`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/quantized.cpp#L85) sets the M4 vector limit 13 for K=N=896. M=1 uses qmv; M=2..12 uses qmv_wide; M>=13 enters matrix dispatch. The small-row path decodes/accumulates in FP32; the matrix tile loader materializes dequantized weights in T before MMA. See [mlx `mlx/backend/metal/kernels/quantized.h:758`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h#L758), [mlx `mlx/backend/metal/kernels/quantized.h:989`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h#L989), [mlx `mlx/backend/metal/kernels/quantized.h:1200`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h#L1200).

The standalone replay isolates this transition with **no model/cache/MLX-One import**. Separate learned projection-output bias is intentionally omitted; this is the affine operation, not a claim of reproducing every post-bias layer value. BF16 explicitly casts metadata/activations offline while preserving packed bytes; it is not qualification of the original FP16 checkpoint at another precision.

| Dtype | M | Max difference vs M1 | Bit-identical | Operation policy pass |
| --- | --- | --- | --- | --- |
| float16 | 1 | 0.0 | True | True |
| float16 | 2 | 6.103515625e-05 | False | True |
| float16 | 4 | 6.103515625e-05 | False | True |
| float16 | 6 | 6.103515625e-05 | False | True |
| float16 | 8 | 6.103515625e-05 | False | True |
| float16 | 12 | 6.103515625e-05 | False | True |
| float16 | 13 | 0.0078125 | False | True |
| float16 | 128 | 0.0078125 | False | True |
| bfloat16 | 1 | 0.0 | True | True |
| bfloat16 | 2 | 1.9073486328125e-06 | False | True |
| bfloat16 | 4 | 1.9073486328125e-06 | False | True |
| bfloat16 | 6 | 1.9073486328125e-06 | False | True |
| bfloat16 | 8 | 1.9073486328125e-06 | False | True |
| bfloat16 | 12 | 1.9073486328125e-06 | False | True |
| bfloat16 | 13 | 0.0625 | False | True |
| bfloat16 | 128 | 0.03125 | False | True |

minimal-stock-qmm.json (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/minimal-stock-qmm.json`). The standalone affine-only M13 replay differs from M1 by 0.0078125 in FP16 and 0.0625 in BF16. The full projection diagnostic, which also adds the learned output bias, records 0.125 in BF16 at M13; these are separate output scopes. Some operation comparisons remain inside their policy, while repeated model propagation exceeds the logits/state policy. A passed GEMV comparison does not establish whole-model equivalence.

Offline FP32 and storage-rounded dequantized small-projection oracles in reference-repro-final/3e6ad8c757760f3960b2fd9c80826a02bbec5d33d296fc7b3c44e56d885acf39.json (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/reference-repro-final/3e6ad8c757760f3960b2fd9c80826a02bbec5d33d296fc7b3c44e56d885acf39.json`) further distinguish accumulation from weight-storage rounding. Raw packed arrays/inputs/selected outputs: diagnostics/stock-q8-matrix-vector-inputs-final.npz (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/diagnostics/stock-q8-matrix-vector-inputs-final.npz`). No entire dense weight is allocated in inference or G03; only this offline small diagnostic dequantizes weights.

```bash
.venv/bin/python benchmarks/upstream/minimal_stock_qmm.py \
  --inputs /Users/susant-achary/dev_work/git_codes/c_base/mlx-one/local/mlxboost-upstream/20261116-1794812262942167000/diagnostics/stock-q8-matrix-vector-inputs-final.npz \
  --output <new-json> --execute-gpu
.venv/bin/python benchmarks/upstream/reference_repro.py --run <new-run> --execute-gpu
```

Normal GPU approval applies. Source and array hashes are retained. The raw replay is ready for a separately authorized upstream issue; no external message was sent.

## Confidence and remaining uncertainty

High confidence: the first projection arithmetic divergence is independent of KV/RoPE/cache mutation; identical-input matrix/vector differences and deterministic repeat controls establish it. This is consistent with the pinned backend's explicit dtype/reduction contracts and M13 dispatch transition. It does **not** establish the only cause of accumulated later-layer drift or prove every prefix/attention reduction equivalent. Cache state correctness and identical fresh logits are distinct questions.

No conclusive MLX-One cache-index bug or safe reference-preserving numeric fix was established. Routing prefill through repeated vectors, casting all weights/activations to FP32, changing stock dispatch or widening tolerances would change the protected baseline. None was adopted. Next: use the minimal capsule to determine an upstream reference-preserving remedy, then rerun original logits/state/fresh controls before model promotion.
