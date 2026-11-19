# G01 design: residual addition and RMSNorm

**Status: DESIGN ONLY — NOT IMPLEMENTED / NOT ACTIVATED.** No G01 kernel, model adapter, operation registration, environment switch or build change is supplied by this study. No G01 GPU test or performance measurement was run. G03 qualification remains a separate prerequisite; this document does not change its result. Source inspection date: 2026-11-15. Exact local file hashes and provenance limitations are in [the source map](mlxboost-g01-source-map.json).

## Decision and problem statement

Recommend a **conditional GO for a later, isolated experiment**, beginning with a two-output residual-add→FFN-RMSNorm boundary in Llama. Recommend **NO-GO for production implementation or activation in this assignment**. First complete G03's independent correctness, reference cached/fresh parity, performance and distribution gates. Standalone algebraic similarity is insufficient evidence to override a family fidelity exclusion.

The prospective operation computes a rounded residual sum and normalizes that sum. Its purpose would be to remove one dispatch and one subsequent read of the summed state when the implementation can retain the values while computing RMS. This is an **unverified hypothesis**: MLX already supplies specialized RMSNorm kernels, compiled graphs may eliminate adjacent work, and the two-output contract still writes both tensors. Decode may be sensitive to launch overhead; large prefill may be sensitive to bandwidth, occupancy and reduction cost. Neither a kernel win nor an inference win is established here.

Let `N_i(v)` denote the exact existing `nn.RMSNorm` operation with that layer's weight and epsilon, `A` attention, and `F` the feedforward module. The common local pre-norm block is:

```text
r = A(N_1(x), mask, cache, position_ids)
h = x + r
z = N_2(h)
out = h + F(z)
```

Only `(h = x + r; z = N_2(h))` is a candidate boundary. Replacing the block with `N(x + A(x))`, normalizing the final residual output, or normalizing a branch before the existing addition changes the model. `h` remains live for the final residual addition, so the normalized-only interface `y = N(x + residual)` cannot directly replace this boundary. Folding a final block addition into the next block's input norm would require explicitly carrying both residual and normalized values across layers and preserving final-norm handling; defer that wider API change.

Inputs have hidden width `H`, normally `[B,L,H]`; independent rows number `B*L`. Batch-1 decode has `B=L=1`. Prefill has many rows and cannot inherit decode qualification. Quantized linear layers still produce floating activation tensors; G01 has no packed-weight input and must not dequantize, repack or own quantized weights.

## Exact local family eligibility

All rows below are **source-reviewed, GPU-untested for G01**. Epsilon values shown are config defaults, not substitutes for each loaded checkpoint's value. Source-map line spans identify the actual implementations. Candidate scope must record public/reference-first versus separate experimental implementations explicitly.

| Family and source | Eligible pattern and reference equation | Needed outputs | Dtype/order hazards | Expected benefit hypothesis | Required test coverage | Risks | Do not support in initial experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2 / Qwen2.5, `language/qwen2/reference_math.py:100–122` | `r=A(N1(x)); h=x+r; out=h+F(N2(h))`; narrow `h,N2(h)` boundary, checkpoint `rms_norm_eps` (default `1e-6`) | Both `h` and `N2(h)` | Preserve addition rounding before RMS, FP32 RMS reduction and weight multiplication order | Possibly fewer launches/reads; unmeasured | Both Qwen2.5 pinned Q4/Q8 profiles, FP16/BF16, full logits, every decode step, cached/fresh, deterministic tokens | ZMLX excludes Qwen residual fusion for fidelity; independent local proof required even if tensor tests pass | Whole block replacement, blanket Qwen-family matching, attention/RoPE/cache changes, production patching; defer model prototype while exclusion is unresolved |
| Qwen3, `language/qwen3/reference_math.py:105–123` | Same narrow equation; checkpoint `rms_norm_eps` (default `1e-6`) | Both | Same; Q/K per-head RMSNorm at `43–44` is a different graph and shape | Decode launch reduction is only a hypothesis | Qwen3-0.6B Q8, full prefill/decode and Q/K values unchanged; separately test public and candidate routes if either is proposed | Qwen fidelity exclusion; candidate-only `qwen3/upstream_model.py:87–117` is not interchangeable with public reference-first model | Q/K norm fusion, all-model monkeypatches, treating candidate tests as public runtime qualification |
| Llama, `language/llama/reference_math.py:114–136` | Same narrow equation, checkpoint epsilon (default `1e-6`) | Both | Addition must materialize in the activation dtype; reassociation/fused multiply-add can alter results | Best initial bounded prototype because the inspected graph directly matches; no speed prediction | Pinned Llama-3.2-1B Q4; FP16/BF16, logits, cache offsets and state, full-context versus incremental, deterministic generation | Existing fast RMSNorm may win; larger widths increase registers/reduction overhead | Attention, KV-cache, projection or checkpoint-format changes; normalized-only replacement at this boundary |
| LFM2, `language/lfm2/reference_math.py:138–179` | `r=Operator(Nop(x)); h=x+r; out=h+FFN(Nffn(h))`; Operator is attention or ShortConv depending on layer. Candidate only `h,Nffn(h)`, checkpoint `norm_eps` (default `1e-5`) | Both | Same sum/RMS rounding; do not borrow Qwen epsilon. Stateful convolution ownership is independent of fusion | Potential local read/launch saving; hybrid-model effect unmeasured | Attention and convolution layers separately, reset/chunk/full/cached state, all masks/positions, FP16/BF16, logits and generation | A correct norm result does not prove recurrent/convolution state parity | Fusion across operator/conv/gating, Q/K per-head norms (`64–65`), whole hybrid-model approval from an attention-only result |
| Gemma3-text, `language/gemma3/model.py:12–28,118–132` | `r=A(Nin(x)); h=AddGemma(x,Npost(r)); out=AddGemma(h,Npostff(F(Npreff(h))))`. A distinct `AddGemma→Npreff` boundary exists, but **excluded from generic initial scope** | Both if a future family-specific boundary is investigated | Effective norm weight `1+w`; FP16 addition promotes operands to FP32, clips to ±FP16 max, casts back. Post-norm precedes each addition; ordinary `x+r` is wrong | No generic benefit claim; family-specific two-output boundary could be studied later | Separate clipping/cancellation/weight-offset tests plus complete Gemma3-text cache/logit/token validation | Offset-weight arithmetic and clipping are semantic requirements; checkpoint attention layouts also differ | Generic pre-norm rewrite, removing clip, normalizing raw attention branch instead of post-norm branch, sharing qualification with Llama |
| GPT-OSS / other accepted families | No eligible graph established by this G01 study | Unknown until exact source review | Do not infer compatibility from RMSNorm class names | None established | Separate graph and fidelity study first | ZMLX also excludes GPT-OSS residual_norm for fidelity | All unreviewed families, MoE boundaries, multimodal wrappers and recurrent families by name matching |

Final model norms are separate consumers after the layer loop (Qwen2 `model.py:65`, Qwen3 `model.py:72`, Llama `model.py:66`, LFM2 `model.py:82`, Gemma3 `model.py:163`). They are not residual-add sites by themselves. Norms inside attention operate over head dimensions and are outside this proposal.

## MLX and ZMLX comparison

The pinned baseline is MLX **v0.32.2**, recorded locally as revision `1f8e74e3f12f31365464a6867c6579f0e9b29d85`. Its specialized Metal RMSNorm uses FP32 reduction, epsilon inside reciprocal square root, and a low-precision normalization cast before multiplication by the weight. Single-row and looped implementations cover different widths; changing their reduction tree may change results. Therefore computing the entire normalization and weight product in FP32 before a single final cast is not a sufficient compatibility specification. See [the pinned RMSNorm kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/rms_norm.metal). The installed Python `nn.RMSNorm` forwards to `mx.fast.rms_norm`; its installed source is hashed separately from upstream captures.

Use these distinct baselines in any future experiment:

1. **Existing eager graph:** compute `h=x+r`, then `mx.fast.rms_norm(h,w,eps)` and return both. Keep actual loaded tensor dtypes and epsilon.
2. **Compiled existing graph:** the same two-output function under the supported MLX compilation mode; record the compilation configuration and whether it is already used by the tested runtime. Validate its parity independently. Do not silently substitute a compiled microbenchmark for the public model baseline.
3. **Specialized pinned stock RMSNorm:** profile the normal operation, including width/row-dependent dispatch. Compare total add+norm latency and observed dispatches, not the custom fused kernel against an artificial slow RMS implementation.
4. **Explicit diagnostic custom prototype**, only after a separate implementation assignment. This is not evidence or a production route today.

Pinned MLX-LM revision `93e6d4a6447b3080861985947f0820e76fcb957d` captures the same pre-norm/add/pre-norm/add pattern in Qwen2, Qwen3 and Llama, and attention/conv→add→FFN-norm in LFM2. Local MLX-One code is still the actual runtime authority: it explicitly carries position IDs and owns its reference-first cache contracts. Upstream similarity does not authorize swapping model classes.

An additional local archive contains `references_libs/mlx-lm-main/mlx_lm/kernels/norm_add_norm.py`. It illustrates returning the residual stream and a following normalized output, but computes a **post-normalized branch and weighted FP32 residual stream** before optional pre-norm. That stream/output contract differs from the materialized low-precision sum above. The archive lacks a verifiable upstream commit; the source map records a content hash and unknown revision. It is a design comparison, not pinned MLX v0.32.2 runtime code or an eligible drop-in. Do not import the reference archive.

ZMLX revision `d8cd0d88d4299ca6821d706f9d5b4a3520d688f5` provides an important negative reference. Its inspected patch index excludes `residual_norm` for Qwen and GPT-OSS due to model fidelity, warns that norm replacements can lose against MLX fast operations, and excludes residual_norm from default reversible smart-patching experiments because of prior regressions. These are upstream source claims, **not measurements on this Mac or evidence that our unimplemented G01 passes/fails**. Keep per-family exclusions until a separate exact graph implementation satisfies unchanged tensor and model gates. See [the pinned ZMLX patch policy](https://github.com/Hmbown/ZMLX/blob/d8cd0d88d4299ca6821d706f9d5b4a3520d688f5/src/zmlx/patch/__init__.py).

The exact pinned leaves were subsequently captured through the parent's approved source retrieval and inspected. Their names conceal important semantic differences:

| Inspected reference | Equation/output and launch/memory behavior | Compatibility finding |
| --- | --- | --- |
| ZMLX `kernels/transformer.py:239–298,374–468`, `rmsnorm_residual` | Returns `(normalized, updated_residual)` in the activation dtype. One threadgroup per row; width-strided loops and threadgroup reduction; writes the sum output, then reads that output for normalization. Sum-square accumulation uses the **unrounded FP32 sum**, while normalized numerator reads the rounded sum. Weight product is cast only at final output. Wrapper validates rank, same shape and weight width; wraps a custom VJP. | Not the exact materialized-add→pinned-fast-RMSNorm equation. Do not copy its normalization denominator or cast sequence. One launch does not imply no residual reread. Its derivatives also need independent validation if adapted. |
| ZMLX `kernels/transformer.py:472–550`, `fused_add_rmsnorm` | Normalized-only output. Computes FP32 sum for reduction and recomputes that sum for output; no returned residual buffer. Same one-threadgroup-per-row reduction. Wrapper has fewer explicit shape guards than the two-output version. | Sum is unrounded before normalization; output contract is insufficient for the current block boundary. Not a safe drop-in. Row-contiguity handling in its helper does not prove arbitrary-layout zero-copy support. |
| ZMLX `kernels/norms.py:641–703,786–871`, actual `add_rms_norm` used by patch | Caches the **unrounded FP32 sum** in thread-local registers, computes RMS and normalized output from those values, writes a separately rounded residual output. One group per row, EPT width-dependent register array, SIMD/tree reduction from `msl.py:43–60`, final-only output cast after weight multiplication. | Explicit semantic mismatch with the materialized low-precision sum and pinned MLX cast order. Its “improved precision” cannot be adopted as a correctness improvement under the existing model contract. Register pressure is width-dependent. The docstring describes a second input reread, but actual kernel caches registers; source is the authority. This identifies a concrete risk, not a proven cause of upstream model failures. |
| ZMLX `patch/patterns/residual_norm.py:22–140` | Matches broad attribute names and RMSNorm classes; wraps the whole block; calls `kernels.norms.add_rms_norm`, **not** the transformer catalog function; forwards mask/cache through compatibility retries. Mutates the module class. | Do not import this patching strategy. It does not establish Gemma clipping/offset-weight semantics, local `position_ids` forwarding, hybrid state safety or scoped MLX-One model identity restoration. |
| ZMLX `docs/BENCHMARKS.md:1–57` | Repeated greedy token fidelity, raw run capsules with revision/environment, isolated subprocesses and AB/BA ordering for pressure-sensitive models. | Useful methodology, not local performance evidence or proof all wrappers have the same shape guard. Retain MLX-One's stronger unchanged acceptance gates. |

Both kernel helpers request row contiguity. `metal.py:113–119,240–257,375–418` forwards the effective setting to `mx.fast.metal_kernel` and supports environment-based overrides and copy-risk telemetry; it does not establish zero-copy eligibility for arbitrary views. A future adaptation must account for copies in latency/memory and preserve its own strict layout admission. The literal source accepts a dtype template but does not independently prove FP16/BF16 model qualification. Exact leaf-source hashes and line spans are in the source map. Upstream API compatibility with the pinned MLX wheel remains a future check, not a claim from reading Python code.

MLX, MLX-LM and the inspected ZMLX capture use MIT licenses. Future adapted code must preserve the relevant copyright/license notices and document the source revision. This study adds original documentation; it copies no production kernel. Newer upstream custom-kernel helpers do not establish ABI or packaging support in MLX v0.32.2.

## Numerical and ownership contract for a future experiment

The semantic oracle is **the exact existing graph**, evaluated with the same inputs, loaded weights, dtype, device, stream and MLX version. Let `T` be the matching floating activation dtype and `round_T` the reference materialization:

```text
h_ref = reference_add(x, residual)        # exact original promotion/rounding
y_ref = mx.fast.rms_norm(h_ref, w, eps)   # exact pinned dispatch/casts
```

For the initial homogeneous FP16/BF16 case, the fused kernel must round the sum into `T` *before* its values enter the RMS reduction, even if it keeps those rounded values in registers. It must return that same `h_ref`. Retaining an unrounded FP32 sum for RMS changes the graph. The RMS accumulator is FP32, epsilon is added to the mean square before reciprocal square root, and weight multiplication must preserve the pinned operation's cast point. Record the actual weight dtype: a matching-dtype Metal template does not prove mixed activation/weight dtype semantics. Reject mixed dtypes until a prototype independently satisfies the pinned backend conversion and output rules described below. Never silently recast checkpoint weights to enlarge eligibility.

Pinned `mlx/fast.cpp:53–120` further establishes `out_type=result_type(input,weight)`: weights and primitive input are converted to that type, while fallback normalization accumulates in FP32, casts the normalized value to `out_type`, then multiplies weight. Thus a loaded FP32 norm weight with FP16 activation can change output type; never assume norm output matches activation solely from the model architecture. Initial matching-dtype eligibility deliberately avoids that additional conversion contract. Preserve unsupported mixed-dtype execution through the existing graph rather than converting parameters.

The scalar equation does not dictate a bit-identical reduction tree. Normalized outputs must satisfy the unchanged numerical policy and downstream model gates. The sum output has no changed reduction and should match the reference sum exactly for finite identical-dtype inputs; use an exact check there to catch lost materialization. Nonfinite characterization is separate from acceptance: existing finite-required policies remain in force. Epsilon must be finite and positive for the proposed experimental contract; unsupported values must use the existing reference semantics or produce an explicit strict failure, without changing public MLX-One behavior.

First prototype scope: matching shapes and dtypes, contiguous row-major floating arrays, contiguous one-dimensional weight of length `H`, nonzero width, supported Metal device, out-of-place results. Validate shapes, strides, integer-size bounds and launch geometry before launch. A vectorized load must mask the final lane group; divisibility is not implied by a model name. Unsupported views, broadcast residuals, transposes, different row strides and mixed dtypes take the existing reference path or produce the existing typed strict-mode failure. Do not allocate a hidden contiguous copy merely to claim fast-path coverage; copy-based candidates require separate latency/memory qualification.

Do not donate input storage, mutate `x`, `residual` or weights, or alias outputs in the initial implementation. Identical input objects and overlapping read-only input views must either produce correct out-of-place results or be explicitly rejected before launch. Both output arrays must remain valid under lazy evaluation, temporary Python inputs, multiple streams and delayed evaluation. Keep no raw borrowed pointers beyond MLX primitive ownership. A custom primitive must obey the current stream/device contract. Gradients/vmap are outside inference scope: future experimental API must explicitly reject unsupported transforms or implement and qualify their rules; silently returning incorrect derivatives is unacceptable.

Gemma's clipped FP16 add and offset norm weights require a separate semantic version and family-specific tests if proposed later. Do not add an unchecked `gemma=True` switch to a generic kernel.

## Interface sketches — documentation only

These sketches do not define a shipped API or change any existing inference interface.

```python
# A future local Python Metal experiment, after separate assignment.
def experimental_add_rms_norm(
    x, residual, weight, eps, *, return_sum=True, stream=None
):
    # Default proposed shape: return (summed_state, normalized_state).
    # No model mutation; validate explicit inference-only semantic contract.
    ...

# Separate normalized-only variant, if graph use-count proves the sum is dead.
def experimental_add_rms_norm_value(x, residual, weight, eps, *, stream=None):
    ...
```

Use two separately identified output contracts in qualification keys; a normalized-only result cannot be used where downstream code needs the sum. `return_sum=False` is not a free optimization for current local block boundaries. Python `mx.fast.metal_kernel` can explore launch shapes with no packaging promise. Only if that prototype beats the compiled/eager baseline and model gates should a future C++ bridge be considered:

```cpp
// Design sketch only; ownership/stream signature must follow the pinned bridge.
std::pair<mlx::core::array, mlx::core::array> add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    mlx::core::StreamOrDevice stream);
```

An eventual dispatch request would describe operation semantic version, output contract, device capabilities, MLX/native ABI and source digest, input/output shape/dtype/strides, norm-weight contract, epsilon, row count, model family/route, compiler options and variant. No registry entry or operation identifier is added by this document. Keep explicit experimental forcing distinct from qualification approval. Never persist approvals from the benchmark that selects its own candidate.

## Proposed numerical matrix and failure policy

Every item below is **planned / NOT RUN for G01**. Existing `tests/models/test_local_reference_math.py`, `test_reference_first_cache_contracts.py` and `test_numerical_invariants_native.py` protect portions of current models; they are not G01 qualification. Existing G03 tests likewise confer no G01 coverage.

| Dimension | Future cases | Acceptance and evidence |
| --- | --- | --- |
| Precision | FP16, BF16 independently; FP32 as a separately supported or reference-only scope; matching/mixed weight dtypes | Exact output dtype/shape; FP32 `(atol,rtol)=(1e-4,1e-4)`, FP16 `(1e-2,1e-2)`, BF16 `(5e-2,1e-2)` from frozen `inference_baseline.TOLERANCES`; no new wider tolerance |
| Intermediate sum | Random, cancellation, near-rounding-boundary values; input equal to residual | Exact finite `h` versus reference separate add; explicitly catch using an FP32 unrounded sum for normalization |
| RMS and weights | Near-zero/constant rows, large/small magnitudes, alternating signs, zero/negative/nonunit weights, checkpoint epsilon, default `1e-6` and `1e-5` | Check normalized output using frozen comparison helper; record first failure, max error and changed elements, not only aggregate cosine similarity |
| Widths and rows | H=`1,31,32,33,63,64,65,127,128,129`; actual checkpoint widths and large looped widths; rows `1,2,128,4096` when admitted | Tail bounds, all values written, no OOB; decode and prefill scopes separate; real configuration governs model dimensions |
| Layout/shape | Contiguous `[H]`, `[1,H]`, `[B,L,H]`; slices, transposes, nonunit last stride, padded rows, broadcasts, mismatched/empty/huge shapes | Accept only proven layouts, otherwise explicit fallback/typed strict rejection. No implicit copies in measured candidate |
| Extremes/invalids | FP16 overflow/cancellation, BF16 exponent range, subnormals, signed zero, NaN/Inf, zero/negative/nonfinite epsilon, wrong weight length/dtype/device | Characterize reference behavior; preserve finite-required qualification. Invalid contracts fail before launch; no saturation introduced for ordinary families |
| Ownership/runtime | Input unchanged, same object as both inputs, overlapping views, delayed `mx.eval`, temporary inputs, concurrent streams, exceptions/cancellation, unsupported autodiff | No mutation/lifetime bugs; typed rejection where unsupported; adapter/function restoration and request isolation |
| Model scope | Llama first; Qwen2/Qwen3 only after exclusion investigation; LFM2 only after attention+conv state matrix; Gemma-specific design separate | Compare full logits at each step, deterministic greedy IDs, outputs/cache state, fresh versus cached/chunked execution, public API identities and cancellation cleanup |

Use `mlx_one.ops.numerical_trace.compare_tensors` for policy-defined comparisons. The frozen rule is `abs(candidate-reference) <= atol + rtol*abs(reference)` with finite values required. Integer tokens and metadata compare exactly. Do not substitute a more permissive RMS-specific tolerance or use FP32 replay to approve a different FP16 runtime graph. Microbenchmark success cannot override a token/state failure.

Current reference cached/fresh failure A0 must be resolved and reproduced on the relevant reference scope before model-level G01 qualification. Preserve historical failure evidence. If reference parity remains unresolved, record model approval as blocked even when custom-vs-reference incremental outputs agree. Any G01 tensor, token, state, ownership or memory failure excludes that exact variant/family scope; report the failure and keep the reference path. Do not change the baseline or tests to make a candidate pass.

## Proposed performance campaign and low-memory coverage

This is a future plan, **not a performance report**. No latency, dispatch count, token rate or memory saving has been measured for G01.

Microbenchmarks must evaluate both outputs, warm each compiled candidate and baseline, synchronize GPU timing boundaries, and run at least 50 paired trials with randomized/interleaved order where safe. Store every sample, median/variability, exact command/environment, source and wheel hashes, device, compiler/fast-math options, launch geometry, actual dispatch counters and allocation measurements. Separate cold compilation from warm execution. Compare eager add+fast norm, the same compiled graph, and each prototype across real row/width/dtype/epsilon/layout tuples. Include rejected geometries and slower results. A two-output kernel must never be timed while laziness removes its residual output from the baseline or candidate.

Run end-to-end reference and explicit diagnostic prototype in fresh processes with identical immutable checkpoint revisions, prompts/token IDs, precision, seeds, generation counts, cache policy and admission budget. Reuse G03's available Qwen2.5-0.5B Q8, Qwen3-0.6B Q8, Llama-3.2-1B Q4 and Qwen2.5-1.5B Q4 snapshots only in eligible scopes; quantization belongs to the projections, not this operator. Do not download a new model for this study. Capture prefill t/s, decode t/s, TTFT, total latency, cold versus warm startup, peak MLX allocation, process RSS and system pressure separately. Verify actual native/prototype dispatch counts; zero execution cannot support attribution.

Contexts `128,4096,32768` are separate scopes where supported and admitted. Keep the existing 8 GiB working-memory admission rule and full-logit reference evaluation. Mark unsupported contexts `SKIPPED — admission rule` with the actual reason; never shrink prompts or remove full logits silently. At least five repeated paired model comparisons are needed, with all token/logit/state checks. The G03 planning targets (10% kernel and 5% decode improvement, with each paired repeat satisfying the model requirement) can inform a later proposal, but do not automatically approve G01; declare the G01 decision rule before measurement. No promotion based on a selected median hiding failed repetitions.

| Hardware tier | Coverage priority and constraint | Permitted inference from results |
| --- | --- | --- |
| M4 32 GiB host | First actual device; run strict 8 GiB working budget as applicable; document power/thermal state and OS pressure | Qualifies only the measured M4/device/config scope; an allocation cap does not emulate an 8 GiB physical Mac |
| 8 GiB physical Mac | Highest low-resource follow-up; one model/process, smallest admitted contexts first, quantify swap/pressure and memory headroom | Requires real hardware; may be memory-bound before this small activation fusion matters |
| 16 GiB physical Mac | Next representative capacity; same paired decode and admitted prefill scopes | Do not extrapolate across GPU generations/core counts |
| 24 GiB physical Mac | Check width/context crossover and sustained memory/thermal effects | Independent device/config evidence |
| Other 32 GiB devices | Distinguish GPU family, cores and memory bandwidth from capacity alone | No blanket “Apple Silicon” speed claim |

No new persistent tensor cache is justified. Two outputs require roughly the same materialized activation storage as the current graph; the hypothesized benefit is reduced reread/launch work, not removal of model weights or KV state. Peak memory can still grow if fusion extends output lifetimes or register pressure causes spills. At long contexts full-logit/reference costs and KV memory may dominate. Record allocator and process/system metrics with their actual meanings instead of calling MLX allocator peak total unified memory.

## Foundation, distribution and ranked next choices

| Dependency | Requirement for any future implementation |
| --- | --- |
| F01 bridge | Keep optional native build and pinned MLX C++/Metal ABI; reuse existing ownership/stream patterns only after primitive review; standard wheel must install and operate without G01 |
| F02 dispatcher | Default-off; exact shape/layout/dtype/device/family and semantic checks; missing/invalid evidence falls back to existing graph; preserve strict typed failures; no global model monkeypatch |
| F03 profiling | Real synchronized GPU measurements and executed-call counters; reference/eager/compiled paths labeled separately; model metrics and correctness separate from kernel timing |
| F04 registry | Immutable offline evidence keyed by exact semantic/source/device/MLX tuple; checksum and numerical/performance gates; no first-request tuning or benchmark self-approval |
| Packaging/API | Independent standard/native wheel installs from wheel files in clean environments, no editable/worktree library dependency; preserve Python/CLI/HTTP/SSE and model/module identities; no production API redesign |

Ranked research choices after a separate assignment:

1. **GO for measurement design:** establish eager versus compiled stock baselines with both outputs materialized. This can reveal that no custom kernel is needed.
2. **Conditional GO for a bounded Llama two-output prototype:** only after G03/A0 closure and a separately authorized implementation assignment, preserving the now-captured exact pinned backend semantics. Inference scope first, one explicit boundary and no model-wide automatic patching.
3. **Defer Qwen2/Qwen3 prototype eligibility:** investigate the pinned ZMLX fidelity exclusions and qualify exact local graphs; similar equations do not cancel that evidence.
4. **Defer LFM2 whole-model scope:** a boundary prototype may be mathematically eligible, but hybrid state/cache coverage is mandatory.
5. **NO-GO for generic Gemma3, normalized-only substitution in these blocks, global norm replacement or broader family inference.** Each requires a different contract or graph-use proof.

Stop at this design. **G01 remains NOT IMPLEMENTED / NOT ACTIVATED; its numerical and performance gates are NOT RUN.**
