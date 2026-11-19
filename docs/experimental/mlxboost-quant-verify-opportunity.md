# Quantized small-row verification opportunity

**PROMISING FOR PROTOTYPE only in three exact wide-head scopes. No MTP/speculative/model qualification or production activation.** Here M denotes activation rows, distinct from the Apple M4 chip.

[mtplx `mtplx/verify_kernels.py:61`](https://github.com/youssofal/MTPLX/blob/eb580fce3fe4465bafb261420e6fed794515dd61/mtplx/verify_kernels.py#L61) provides plain-SIMD `vk_qmm_m4`/`vk_qmm_m6`, plus split-K alternatives. Packed MLX affine uint32 Q4/Q8, matching FP16/BF16 scales/biases, groups32/64/128 are compatible in tested cases. Msg requires K%64 and N%(4*NSG)==0 (NSG8 for M4 rows,4 for M6); split-K permits N%4. Public functions need external shape/layout guards: unsupported M=1/2/8, tail N129, group/alignment/layout cases were rejected before launch. M5 padding exists upstream but was not tested in this requested M1/2/4/6/8 matrix.

The experiment executed180 workload combinations,924 total profiles across this and real-weight studies, with 50 randomized synchronized warm samples/profile. The small synthetic matrix alone showed at most1.0363× for msg and1.0280× for split-K, below target. G03 and oMLX controls were also measured; no broad tuning sweep followed.

## Actual hidden-state validation

Original final normalized hidden states from the pinned128-token model prefill, original head weights/metadata, FP16/group 64. Five repeated interleaved50-trial comparisons per row. Each scope records281 actual marked GPU launches (preparation +5×(cold+5warmup+50));4-byte marker output is disclosed. No dense dequantized weight/workspace is allocated. Full original prefill logits are also compared as a **separate** shape-route control, not silently used as the M-specific operator oracle.

| Checkpoint | M | K×N | Max abs error | Five MLX/native speedups | Marked GPU runs | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| qwen25-q8 | 4 | 896×151936 | 0.00390625 | 1.0213, 1.0239, 1.0091, 1.0114, 1.0231 | 281 | NO 10% TARGET |
| qwen25-q8 | 6 | 896×151936 | 0.0078125 | 1.2426, 1.2267, 1.2225, 1.2347, 1.2265 | 281 | PROMISING FOR PROTOTYPE |
| llama-q4 | 4 | 2048×128256 | 0.00390625 | 1.1970, 1.1965, 1.1925, 1.1938, 1.1969 | 281 | PROMISING FOR PROTOTYPE |
| llama-q4 | 6 | 2048×128256 | 0.00390625 | 1.3114, 1.3108, 1.3109, 1.3070, 1.2884 | 281 | PROMISING FOR PROTOTYPE |

measurements/verifier-real-activation-repeats.csv (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/measurements/verifier-real-activation-repeats.csv`); verifier-validation-qwen25-q8/a0ca88f154d39b5a7c17eb57bbce4f42c334385d2af15f7070aec964eb7c05d2.json (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/verifier-validation-qwen25-q8/a0ca88f154d39b5a7c17eb57bbce4f42c334385d2af15f7070aec964eb7c05d2.json`); verifier-validation-llama-q4/3c0056d551c8ce25e62a6549a28d02a7b19709600e439db2aa1e3d8ca0e2ee60.json (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/verifier-validation-llama-q4/3c0056d551c8ce25e62a6549a28d02a7b19709600e439db2aa1e3d8ca0e2ee60.json`). All four operation comparisons pass unchanged policy; only three meet the 10% target in every stable repeat. BF16/group32/group128 wide-head performance has **not** been independently repeated; do not generalize these wins to all formats/dtypes.

## Why the narrow opportunity exists

MLX0.32.2 already has `qmv_wide`, so “stock lacks small-row specialization” is false. [mlx `mlx/backend/metal/quantized.cpp:544`](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/quantized.cpp#L544) caps vectors/threadgroup at 5. For M6 it launches two vector tiles of3, rereading the weight matrix across those tiles. This is one kernel dispatch with a larger grid, **not two GPU kernel launches**. MTPLX processes all 6 vectors in one pass with 24 explicit accumulators and coalesced pack-interleaved loads. Its four-output-column/SIMD design and multiple independent SIMD groups reduce huge-N threadgroup scheduling footprint. For Q8 M4, stock already reuses all four vectors and no 10% validation win remains. Occupancy/spill claims in upstream comments are not local hardware counter measurements.

Output memory is M*N*2 bytes, plus the 4-byte marker; weights stay packed. Small source/binary/shape changes require new evidence. Source analysis of [mtplx `mtplx/nax_verify.py:1`](https://github.com/youssofal/MTPLX/blob/eb580fce3fe4465bafb261420e6fed794515dd61/mtplx/nax_verify.py#L1) separates NAX m16 (G17/M5,macOS>=26.2,INCOMPATIBLE on this M4) from its plain SIMD fallback. No NAX instruction was run.

## Integration conditions, not permission to activate

MLX-One already has greedy MTP/external draft and reversible target-state machinery in `engine/speculation.py`; it must not be duplicated. Native MTP currently requires complete Qwen3.5 predictor weights; none of the four selected snapshots supplies them. A compatible drafter, acceptance rates, rollback/EOS behavior, complete logits/state parity, unchanged A0 fresh controls and >=5% repeated end-to-end decode gain are still missing. Operator M6 acceleration does not accelerate B1 by itself.

A future guarded MLX Metal wrapper should initially target Q8 K896/N151936/M6 and Q4 K2048/N128256/M4 or6, FP16/group 64 only; preserve stock fallback for every miss and keep offline human review/default-off. Use existing F04 evidence mechanisms, not online tuning or a self-approving benchmark registry. No production registration is added here.

Credit: **MTPLX by Youssof Altoukhi**, [pinned repository](https://github.com/youssofal/MTPLX/tree/eb580fce3fe4465bafb261420e6fed794515dd61). The CLI prints “Powered by MTPLX”. Apache-2.0 LICENSE and full NOTICE are preserved under licenses (local-only: `local/mlxboost-upstream/20261116-1794812262942167000/licenses`); NOTICE's in-product attribution statement and dflash-derived NAX provenance must be reviewed/retained for future product reuse. This study executes isolated shaders, not the MTPLX server.
