# G02: activation-only SwiGLU design and source study

Status: **DESIGN ONLY — NOT IMPLEMENTED / NOT ACTIVATED.** This document grants no execution, registration, patching, promotion, or numerical-policy exception. G03 has separate qualification gates. No G02 GPU measurements or parity tests were run for this study. A future implementation requires a separate assignment after G03 qualification.

## Problem and eligible graph

The narrow candidate is the elementwise activation between the existing gate/up projections and down projection:

```text
h[B,T,D] ─ gate projection ─ g[B,T,I] ─ SiLU ─ multiply ─ a[B,T,I] ─ down projection
         └ up projection ── u[B,T,I] ──────────┘
a = SiLU(g) * u; SiLU(g) = g * sigmoid(g)
```

Decode normally has B=1,T=1; prefill changes T and may change the winning launch geometry. Both projections, their quantization, optional bias, down projection, residual, normalization, attention, cache and model outputs remain owned by the existing model. Quantized affine Q4/Q8 weights are not activation inputs: G03 returns floating projection outputs; G02 would consume those, without changing packed encodings or allocating dequantized weight matrices. Combining G02 with a quantized projection would be another independently qualified design.

The intended result is one floating activation tensor of exactly the reference shape and dtype. No input mutation, alias-dependent writes, hidden conversion, device transfer, concatenation, routing change, or new full-size scratch allocation is permitted. Input storage must remain live under MLX's lazy stream semantics. Zero elements, broadcasting, mixed dtypes and unsupported strides need explicit contracts; initially the safe choice is unchanged reference execution.

**The authoritative baseline already compiles this graph.** Shared `reference_activations.swiglu` is decorated with `mx.compile(shapeless=True)`, and Qwen3's separate upstream candidate also uses a compiled function. The standard baseline is that actual callable with the active execution profile, not separately evaluated eager SiLU and multiply. Source operation count does not prove GPU launch count or materialized intermediates. An approved Metal capture or profiling run must establish those facts before arguing that custom fusion saves work. The separate shared `transformer.SwiGLU` helper uses policy-aware projections and an eager-looking expression; preserve that path's own baseline identity rather than treating all MLPs as the same entry point.

## Mathematical and rounding contracts

The standard callable evaluates `nn.silu(gate) * up`; MLX v0.32.2 implements SiLU as `x * mx.sigmoid(x)`. Its sigmoid Metal helper uses an absolute-value exponential and a sign-dependent result. A naive `1/(1+exp(-x))`, `fast::exp`, a changed multiply association, or unconditional FP32 promotion is not a demonstrated replacement. The mathematical identity alone does not establish FP16/BF16 equality. See [MLX v0.32.2 activation source](https://github.com/ml-explore/mlx/blob/v0.32.2/python/mlx/nn/layers/activations.py).

Let R_T denote rounding to the activation dtype. An eager materialized expression can introduce intermediate R_T operations; compiler fusion can change which intermediates materialize. Do not infer a round-after-every-operation contract from Python syntax. Compare with the unchanged compiled reference and inspect its lowering before selecting the candidate's exact cast sequence. Record compilation flags, sigmoid implementation and output dtype in the future math-contract identity. The existing `precise_swiglu(h,g,u)` is a **different explicit contract**: cast g and u to FP32, apply SiLU and multiplication there, then cast the result to h.dtype. It must never silently select a standard-dtype kernel. No claim is made that the families below call this precise helper.

The local family paths contain no explicit FP32 cast inside their standard SwiGLU call. Projection output dtype depends on the checkpoint, MLX projection and existing inference policy; it must be observed and matched, not guessed from quantization bit width.

## Model eligibility and test-design matrix

All rows are source-inspected; **G02 kernel, logits, state and generation qualification are untested for every row**. Existing model tests are available regression anchors, not evidence that a nonexistent G02 kernel passed them.

| Family / exact path | Reference activation and projection layout | Initial research eligibility / risks | Required independent evidence |
|---|---|---|---|
| Qwen2 / Qwen2.5 `qwen2/reference_math.py:89–97` | Independent bias-free gate/up Linear; compiled standard SiLU(g)*u; down projection unchanged | Shape-compatible, but HOLD for fidelity. Pinned ZMLX excludes Qwen `swiglu_mlp`; do not auto-enable from a standalone pass | Each checkpoint and precision separately; full/last logits, forced tokens, greedy continuation, cache state and fresh-vs-cached controls |
| Qwen3 public reference `qwen3/reference_math.py:90–98` | Same independent projections and shared compiled standard helper | HOLD for the same Qwen risk; QK norm and attention are outside scope | Public model route separately from upstream candidate; exact dispatch identity and per-layer activations |
| Qwen3 upstream candidate `upstream_model.py:69–77`, `upstream_math.py:19–21` | Independent projections; separate local compiled swiglu copied from the recorded MLX-LM v0.31.3 lineage | HOLD; do not merge provenance or qualification with the public path | Existing upstream-candidate profile, checkpoint and cache tests; no changed route selection |
| Llama `llama/reference_math.py:102–110` | Independent gate/up/down; configured MLP bias preserved; SiLU-only config guard | Best first standard-graph study after permission; bias belongs to projections, not activation | Both accepted bias configurations where checkpoint-supported; Q4 reference and floating projection tensors |
| LFM2 `lfm2/reference_math.py:114–135` | w1 and w3 are independent gate/up; w2 is down. Hidden width is adjusted then rounded to `multiple_of`, with the optional `ffn_dim_multiplier` applied first | Standard graph candidate; hybrid short-convolution/state and hidden-width rules stay unchanged | Exact configured intermediate width; attention and shortconv layers; recurrent state and continuation |
| Gemma3 text `gemma3/model.py:107–115` | Independent projections but `gelu_approx(g)*u`, not SiLU; config accepts tanh GELU variants | **NO-GO generic G02.** A future GeGLU operation requires a separate contract and qualification | Negative dispatch/fallback test; no activation substitution |
| GPT-OSS `shared/sparse.py:63–80`, `gpt_oss/model.py:130–139` | Selected experts; g=min(g,L), u=clip(u,−L,L); `(u+1)*(g*sigmoid(1.702*g))`, default L=7 | **NO-GO generic G02.** Alpha, clipping, +1, gather, expert bias/routing are essential | Negative generic dispatch test; any future specialized experiment must preserve selected-expert semantics |
| Alternate `shared/transformer.py:232–245` | Separate policy-aware `project` calls and `(g*sigmoid(g))*u` | Separate baseline/profile; do not replace globally because other models may use it | Trace projection output dtype, path identity, eager/compiled caller and execution policy |

GPT-OSS checkpoint `gate_up_proj` is interleaved, but the weight sanitizer splits even/odd rows into contiguous **separate weights**. This is not evidence that runtime activations are concatenated. Its existing optional `mlxboost/gptoss_kernels.py` activation is a distinct MXFP4 experiment with its own clipping/rounding; this study neither changes it nor reuses its eligibility as generic SwiGLU proof.

A future concatenated `[... ,2I]` activation interface would need an explicit encoding (`[gate|up]` versus alternating values), offsets, strides and lifetime. The inspected dense family paths provide two separate projections. Allocating a concatenated tensor just to feed a custom activation adds traffic and is outside the initial design.

## Reference implementation comparison

| Source actually inspected | Useful engineering constraint | Transfer limit |
|---|---|---|
| MLX v0.32.2 Python SiLU, installed 0.32.2 Metal sigmoid header, compiled local MLX-LM activation port | MLX-compatible expression, stable sigmoid handling, explicit precise mode, existing compiler baseline | Installed header is hashed separately; its bytes must not be represented as a fetched tag file unless matched. GPU lowering and launch count are unmeasured |
| Captured MLX-LM commit `93e6d4a6447b3080861985947f0820e76fcb957d` activations/Qwen2/Qwen3/Llama/LFM2 | Independent gate/up projections, standard compiled and precise helper contracts | Does not establish current upstream-main behavior or compatibility of another model revision |
| Qwen3 upstream candidate header commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, version v0.31.3 | A second concrete lineage must retain its own identity | Header-recorded adaptation, not a newly fetched repository checkout |
| Pinned ZMLX `d8cd0d88d4299ca6821d706f9d5b4a3520d688f5` `transformer.py:20–123,882–1009` | `swiglu(x)` accepts `[... ,2D]` split halves; `swiglu2(gate,up)` is the matching separate-input idea. Cached custom forward plus VJP, same output dtype, row-contiguous wrapper, optional discovered-kernel registry | Same-shape guard is not a complete dtype/layout/device contract. `msl.py:20–24` uses a T-valued sigmoid intermediate and T constants despite the broad FP32-compute deprecation message. Inspect the actual specialization and test rounding; do not adopt the message as a numerical guarantee |
| Same pinned ZMLX patch selector and `patterns/swiglu_mlp.py:22–62,180–324` | Qwen SwiGLU fidelity exclusion; independent gate/up or w1/w3 supported. Quantized fused projection mode defaults off; normal patch replaces only activation with `swiglu2`, leaving down projection separate | Structural activation detection defaults to true when unknown, which MLX-One must not adopt. The optional Q4/Q8 projection-fusion branches are outside G02. Source-reported 200-token validation was not reproduced here; family heuristics and overrides are not MLX-One promotion rules |
| Same pinned ZMLX `docs/BENCHMARKS.md:3–20,43–51,107–128`, `docs/TOUR.md:31–66` | Unpatched model baseline, greedy token comparison, isolated per-variant subprocesses, AB/BA runs and raw reproducibility capsules are useful methodology | Historical examples use other hardware, MLX 0.30.4 or modified MLX, and older ZMLX commits. No timing or claimed guarantee is transferred to MLX-One 0.32.2 |
| MLX v0.32.2 extension CMake/bindings example | Existing MLX module/stream/lifetime conventions if native bridge is justified later | No reason yet to add a new native binding: Python `mx.fast.metal_kernel` can test an activation-only hypothesis first |

The exact ZMLX leaf-kernel/pattern hashes and source spans are recorded in the companion source map. The separate-input shader uses a flat element grid and the concatenated shader uses row/column indexing. `metal.py:211–360` passes the exact grid using dispatchThreads semantics, chooses a threadgroup up to512, and delegates row-contiguity enforcement to MLX. Its telemetry expressly records implicit-copy risk; an environment pattern can disable that enforcement. MLX-One must use explicit capability guards and cannot infer safe arbitrary strides or free layout conversion. Neither path has been executed or qualified here. ZMLX's catalog or a patch selector comment alone does not establish a kernel's complete precision/stride coverage, physical launch count, temporary allocations, MLX 0.32.2 compatibility or model speedup. MLX and MLX-LM are MIT. The pinned ZMLX LICENSE is MIT; retain its copyright and license notice for any later substantial reuse. No ZMLX, GGML or llama.cpp runtime dependency is proposed. This study reuses no production kernel code. ZMLX construction timing measures Python callable construction, not completed Metal compilation; its verbose execution timing evaluates/synchronizes and measures wall time. Neither counter should be mislabeled GPU command duration. Catalog prose claims broad prefill guards, while the inspected generic `swiglu2` activation has no token-count cutoff; guards in optional projection/MoE branches cannot be generalized to every activation path.

## Future interface sketches (Markdown only)

```python
# Proposed interface, NOT implemented or registered.
activation_swiglu(
    gate, up, *,
    contract="compiled_standard_v1",  # precise_fp32_v1 is a separate identity
    stream=None,
)
# Metadata guard + exact qualified registry hit -> selected candidate.
# Otherwise -> unchanged original compiled callable, with the original arguments.
```

Prototype with `mx.fast.metal_kernel` only after a separate assignment, explicit opt-in and approved GPU execution. Initially require equal nonempty shape/dtype, supported device, FP16 or BF16, valid row-major contiguous input spans, and known-safe index range. Reject broadcasting, mixed devices/dtypes, overlapping write aliases and arbitrary strided views through the reference path. Distinguish malformed calls (preserve reference errors) from valid unsupported requests (fallback). Noncontiguous inputs should not be silently copied for a nominal win. Threadgroups and vector width are tuning dimensions, with element-count bounds and a scalar masked tail. Grad/JVP/VJP/vmap are not inferred from inference correctness; preserve the original path unless separately implemented and qualified.

A future C++ sketch is justified only if an approved prototype shows a reproducible benefit and F01 integration improves dispatch/build behavior:

```cpp
// Interface sketch only; no implementation or ABI commitment.
array swiglu_activation(const array& gate, const array& up,
                        MathContract contract, StreamOrDevice stream);
```

It must use the accepted F01 MLX ABI and lifetime rules, not a foreign allocator or separate command queue. Output cannot be evaluated on the host to decide dispatch. Output layout, stream and math contract enter F02's capability checks; do not reuse G03's operation IDs.

## Numerical hazards and proposed parity matrix

| Dimension | Required cases and failure signal |
|---|---|
| Precision/order | FP16 and BF16 separately; FP32 as an oracle diagnostic, not a substituted model path; compiled standard versus explicitly precise helper separate; signed zero, subnormal/flush behavior, rounding midpoints and large magnitudes |
| Sigmoid/product | Positive/negative saturation, values near zero, half overflow, products near dtype bounds; finite outputs and NaN/Inf masks compared with reference, rather than filtering exceptional values away |
| Shapes/tails | B=1 decode; B>1 negative dispatch initially; T=1,2,128,4096 and admitted32768 prefill; actual model I plus 1,31,32,33,127,128,129,255,256,257 tail cases; empty/zero-rank/mismatched shapes |
| Layout | Contiguous exact buffers, transposed views, sliced last axes, row padding, offset views, overlapping read inputs, non-unit/zero strides. Unsupported legal views must produce unchanged reference output and a recorded fallback reason |
| Dtype/device | FP16/BF16 matched inputs, mixed inputs, unsupported integer/complex types, CPU or unavailable Metal; no accidental conversion or dispatch |
| Model accumulation | Per-layer activation/down output, complete logits (not last-only hiding failures), top-token identity, forced-token continuation, deterministic greedy sequence, cache/state snapshots and cached-versus-fresh reference control |
| Isolation | G02 alone, G03 alone, both only after each passes independently; no eligibility inherited across checkpoints, precisions or profiles |

Use the existing model/profile numerical qualification policy unchanged and identify its exact policy hash in artifacts. G03's current FP16 `(atol,rtol)=(1e-2,1e-2)` and BF16 `(5e-2,1e-2)` kernel limits are not blanket permission for a future activation or logits test. Preserve any tighter existing model test. Report maximum absolute/relative error, error distribution, finite masks, top-token mismatches, first divergent layer/token and output hashes. Never loosen tolerances, shorten deterministic sequences or choose a friendlier input to admit a candidate. A failed cached-versus-fresh reference control blocks attribution and model promotion until its cause is resolved. A standalone tensor pass cannot override a model exclusion or logits/state failure.

## Future measurement campaign and low-memory priorities

First qualify on physical M4 32 GiB. Then repeat on physical 8,16,24 GiB Apple Silicon configurations; a memory cap on M4 32 GiB is useful stress evidence but does not prove another machine's performance. Record chip/GPU family, core count, memory, OS, MLX/Metal/Xcode versions, compiler flags, checkpoint/tokenizer hashes, power/thermal conditions, peak memory, swapping and rejected cases. Keep full-logit qualification and current admission budgets. Explicitly skip contexts exceeding a model's maximum or the memory budget; do not change the correctness path to fit a benchmark.

Microbenchmarks should compare unchanged compiled standard, explicitly precise mode where used, diagnostic eager graph, and future custom candidate using the same already evaluated gate/up tensors. Avoid projection, initialization and host-read work in an activation-only timing. Record first-use compilation separately from process-cold and warm latency; synchronize before and after each wall-time interval and distinguish GPU command duration from synchronized wall time. Use F03's raw randomized A/B samples, at least 50 warm trials over independent runs, robust medians/dispersion and exact timing scope; capture launch count and allocations rather than assuming them. Include negative stride/dtype/shape cases proving native work was not submitted. Retain slow variants as rejected evidence, not registry winners.

Model campaign starts with the requested Qwen2.5-0.5B Q8, Qwen3-0.6B Q8, Llama-3.2-1B Q4, Qwen2.5-1.5B Q4 and validated SmolLM2-1.7B Q4 if available. Check actual source eligibility for SmolLM2 before admitting it; no family inference from model size. Qwen cases remain held for fidelity until full independent qualification. Add LFM2 only with a validated checkpoint; Gemma3/GPT-OSS are generic-dispatch negative controls. For each admitted checkpoint, compare default/reference with only the future G02 flag changed, batch1, contexts128/4k/32k where supported, identical prompt IDs/seed/cache/configuration and at least five randomized process-paired repeats. Measure prefill/decode tokens/sec, TTFT, cold/warm latency, peak MLX memory plus process footprint/RSS with their distinct scopes, output/state parity, and actual native dispatch counters. Include all-fallback and forced rejection runs. G02+G03 combinations are a separate later evidence row.

**UNVERIFIED hypothesis:** an activation-only kernel may reduce launch/temporary overhead if the current compiler leaves multiple passes, with useful memory traffic tending toward two reads and one write per activation element. The existing compiler may already achieve that traffic and one launch, leaving no gain or a custom-kernel regression. Small decode shapes may be dispatch-bound; large prefill may be bandwidth-bound. These are hypotheses, not measured bottlenecks or promised speedups. The user's G03 10% kernel/5% reproducible decode targets do not themselves qualify G02; a future acceptance assignment must state G02 thresholds and still require model correctness and no unacceptable memory regression.

## Phase 0 dependencies and recommendation

F01 supplies optional extension, stream/ABI and packaging mechanics. Reuse it only after a native bridge is justified. F02 must default off, select only exact supported metadata with qualified performance evidence and return the unchanged model reference on every miss. F03 must distinguish actual kernel dispatch, GPU time and end-to-end behavior. F04 must record offline immutable correctness/performance evidence keyed by operation, math contract, source/binary/compiler/MLX version, hardware, dtype, layout, shape and model profile; never online-tune during inference or borrow a G03 registry proof. Source/binary changes invalidate qualification. Standard wheels must remain free of a required native extension; future native wheels need their own clean-environment tests, enabled/disabled parity and missing-library fallback. Packaging success is not GPU qualification.

Ranked recommendation:

1. **GO for source/profiling research only:** establish the true current compiled activation launch/memory baseline and resolve reference controls first. No new code is authorized here.
2. **Conditional future GO:** activation-only, separate-input, contiguous standard SiLU on a single eligible Llama checkpoint, then LFM2. Require a separate implementation assignment, numerical qualification and measured benefit.
3. **HOLD:** Qwen2/2.5 and Qwen3 despite matching algebra; pinned ZMLX fidelity exclusions require full per-profile model evidence. An elementwise allclose result is insufficient.
4. **NO-GO for initial scope:** concatenated/interleaved activation APIs, arbitrary-stride native handling with silent copies, precise-mode substitution, monolithic MLP/projection fusion, Gemma3 generic SwiGLU and GPT-OSS generic SwiGLU.
5. **NO-GO for production promotion now:** no G02 implementation, dispatch, wheel, GPU parity, timing or model acceleration evidence exists from this design task.

Source lines, hashes, exact captured revisions and unavailable references are in [mlxboost-g02-source-map.json](mlxboost-g02-source-map.json). Backend-free validation of that JSON and source-line bounds checks documentation integrity only; it is not kernel or model qualification.
