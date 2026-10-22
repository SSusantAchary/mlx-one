# Packed execution design

Production modules are under `src/mlx_one/quantization/turboquant/`.

* `config.py`: backend-free strict `TQ3Config` and physical `TQ3Storage` description.
* `loading.py`: compressed buffer validation, module replacement and explicit format exceptions.
* `linear.py`: online block Hadamard, bias, shape restoration and dense dispatch.
* `experts.py`: shared/per-routing inputs, GPU sorting/reordering and sparse dispatch.
* `dispatch.py`: deterministic rules and opt-in `collect_dispatch()` counters using context-local state.
* `kernels/tq3_qmv.py`: fused B1 projection. Eight independent 32-lane SIMD groups per threadgroup;
  lanes read neighboring packed words, decode ten codes in registers, accumulate FP32 and reduce
  with `simd_sum`. Only output elements are written.
* `kernels/tq3_qmm.py`: 16-token by 64-output tile, four word partitions/output,
  cooperative activation staging and FP32 accumulators. No full dense weight materialization.
* `kernels/tq3_expert_qmv.py`: fused selected-bank GEMV with GPU expert IDs. Shared inputs are
  addressed directly without repeating vectors; down projections use a separate row per routing.
* `kernels/tq3_gather_qmm.py`: fixed-capacity GPU tile construction for sorted expert segments,
  padded with -1. Tiled packed arithmetic writes only routed outputs. Unsorted inputs are sorted
  and restored on GPU. Only routed activations, not expert weights, are expanded.

The GEMM kernels adapt pinned Apache-2.0 TurboQuant kernels. Attribution and modification
notices are retained in source and packaged `LICENSE.txt`/`NOTICE.txt`. GEMV kernels are independent
coalesced implementations. Activation staging was changed from hardcoded half to template T,
preserving FP32 input accuracy; BF16 inputs promote through the existing rotation semantics.
The installed `mx.fast.metal_kernel` signature was inspected on MLX 0.32.2. It accepts name,
input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs and compile_options.
Calls use template, grid, threadgroup, output_shapes and output_dtypes. Hadamard is
`mx.hadamard_transform`, not a `mx.fast` method.

## Measured deterministic dispatch

Dense: one row -> GEMV; two rows and O<=2880 -> two GEMVs; otherwise packed GEMM.
Sparse: use tiled gather from 128 routings when O>=512, otherwise from 512 routings.
Below this use sparse GEMV, including small unsorted prefill. The rule is deterministic and has
boundary tests. `evidence/dispatch-sweep.json` includes sorting cost and the supporting grid.
For actual GPT expert dimensions the observed sparse crossover is between 64 and 128 routings:
64 routes: GEMV 5.339 ms / tiled 6.822 ms; 128: 10.324 / 8.319 ms;
256: 20.302 / 8.539 ms. Unknown shapes use conservative packed paths; rules are not a universal optimum.

Online rotation and attention still use MLX operators. Routing remains top four with softmax over
selected logits; capped gate/up activation, down projection and expert-weighted sum are unchanged.
No CPU routing reads, per-token expert loops, full-bank dequantization, or full-model dense caches
exist. Two-row dense dispatch has two bounded kernel launches. Large prefill uses bounded tiled
scratch rather than all-model decompression; routed activation memory still scales with tokens.
Per-module Python dispatch, sorting and repeated gate/up sorting remain optimization opportunities.
Opt-in counters are most useful for direct execution on the same thread; synchronized layer profiling
is a benchmark-only pass and is excluded from throughput samples.
