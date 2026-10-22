> Historical source-bound report. Current checkpoint support follows the [functional admission policy](../../model-admission.md); raw artifacts are [archived locally](../archived-evidence.md). Numerical research outcomes below are preserved historical observations, not current admission gates.

# Ten-checkpoint native architecture comparison

This continues the preserved [migration comparison](../archived-evidence.md)
and [earlier qualification](../first-10-models.md). The current cohort is
`m4-32gb/ten-model-qualification-20261020-1792475268786913000`. Registration,
mathematical implementation, exact-checkpoint parity and release qualification
are separate claims. There are eight families and ten checkpoint identities.

## References and ownership

The primary mathematical references are the supplied, read-only
`references_libs/mlx-lm-main/mlx_lm/models/` files. The independent architecture
references are the supplied
`references_libs/transformers-main/src/transformers/models/` files. Source hashes
are recorded with the cohort. No online source retrieval or runtime reference
imports are required. PyTorch execution was not available; Transformers was
inspected as source, not executed as a numerical oracle. The isolated development
importer disables bytecode writes and never imports the installed `mlx_lm` package.

| Family / targets | Local classes and files | Native implementation | Mathematical / weight contract | State / outputs | Tests and unresolved scope |
|---|---|---|---|---|---|
| GPT-2 / M01 | MLX-LM `gpt2.py`: `ModelArgs`, `GPT2Model`, `Attention`, `MLP`; Transformers `gpt2/configuration_gpt2.py`, `modeling_gpt2.py` | `models/language/gpt2/{config,model,weights}.py` | Learned token + absolute position embeddings; LayerNorm; QKV packed Q/K/V; scaled causal attention; tanh GELU; two residual additions; tied embedding head. HF Conv1D matrices transpose into MLX Linear layout. | Dense KV offsets advance learned positions; native `ModelOutput` carries logits, hidden and optional cache/states. | Existing native decode/projection tests plus exact full, incremental, chunked and reset traces. FP32 decoding gate remains unresolved; first attention-core differences precede late MLP amplification. No claim that GELU caused them. |
| Qwen2 / M02, M08 | MLX-LM `qwen2.py`: `ModelArgs`, `Model`, `Attention`; Transformers `qwen2/{configuration,modeling}_qwen2.py` | `models/language/qwen2/` + `models/shared/transformer.py` | Config-driven biased Q/K/V, bias-free output; GQA; standard nontraditional RoPE; RMSNorm and SwiGLU; tied or explicit head. Strict canonical keys and shapes; no duplicate M08 architecture. | Dense per-layer KV, absolute rotary offset and full-history padding mask. | Exact 896/1536 hidden widths and 24/28-layer models are tested separately. Low-precision attention operands now accumulate in FP32 and return the operand dtype; explicit stable INT8 policy remains distinct. Reference parity remains a checkpoint gate. |
| Qwen3 / M03 | MLX-LM `qwen3.py`: `ModelArgs`, `Model`, `Attention`; Transformers `qwen3/{configuration,modeling}_qwen3.py` | `models/language/qwen3/` + shared transformer | Bias-free Q/K/V, per-head Q/K RMSNorm, GQA, RoPE, SwiGLU, tied head; head width is independently configured (128 for the exact target). | Dense KV and shared lifecycle. Existing experimental upstream adapter is preserved and is not selected by the cohort runner. | Q/K norm and shared attention regressions, frozen checkpoint traces, actual tokenizer template and thinking-disabled chat checks. Matching tokens never replaces the numerical gate. Historical FP16 evidence is retained. |
| LFM2 / M04 | MLX-LM `lfm2.py`: `ModelArgs`, `Model`, attention and short-convolution blocks; Transformers `lfm2/{configuration,modeling}_lfm2.py` | `models/language/lfm2/`, `models/shared/hybrid.py` | Configured mix of attention and gated causal convolution; B/C/x projections, channelwise kernels, Q/K normalization, residuals and gated MLP. HF depthwise kernel layouts are translated explicitly. | Attention KV plus private convolution history, not interchangeable with ordinary KV. Actual stored BF16 logits/state and explicit FP16 profiles are recorded independently. | Full/chunked/incremental/state reset and cancellation checks; BF16 exact-checkpoint gate remains independent of tiny-model correctness. |
| Qwen3.5 / M05 | MLX-LM `qwen3_5.py`: `TextModelArgs`, `GatedDeltaNet`, `DecoderLayer`, `Model`; helpers `qwen3_next.py`, `gated_delta.py`; Transformers `qwen3_5/{configuration,modeling}_qwen3_5.py` | `models/vision_language/qwen3_5/` | Mixed full attention and gated-delta layers; partial rotary, gated Q/output, offset RMSNorm, depthwise causal convolution, L2-normalized Q/K and recurrent delta update. Previous publisher-folded norm translation repair is retained. | FP32 recurrent state plus convolution and KV components. Reference recurrent layout `[B,H,V,K]` is transposed to native `[B,H,K,V]` for comparisons. | Text-only exact checkpoint qualification; no vision promotion. Norm rounding, recurrent reductions and chunked consistency remain separately measurable. BF16 recurrent output is cast back at its defined boundary, not reported as FP32 merely because accumulation uses FP32. |
| Llama / M06, M07 | MLX-LM `llama.py`: `ModelArgs`, `Model`, attention/blocks; `rope_utils.py`: `Llama3RoPE`; Transformers `llama/{configuration,modeling}_llama.py`, `modeling_rope_utils.py` | `models/language/llama/` + shared transformer | Config-driven RMSNorm, GQA/MHA, SwiGLU, tied or untied head. SmolLM2 uses standard theta 130000; Llama3.2 uses theta 500000 and three frequency bands, factor 32, original context 8192. Llama3 scaling is not applied as a blanket position divisor. | Dense KV and logical positions across prefill/decode. | Isolated low/mid/high frequency tests at positions 0/4096/8192/32768, standard rotary regressions and both exact checkpoints. Ordinary chat messages preserve absence of `tool_calls` while optional attribute lookups remain compatible with Qwen templates. |
| Gemma3 text / M09 | MLX-LM `gemma3_text.py`: `ModelArgs`, `Attention`, `RMSNorm`, `TransformerBlock`, `Model`; Transformers `gemma3/{configuration,modeling}_gemma3.py` | New `models/language/gemma3/{config,model,weights}.py`, `models/shared/decoder.py`, `core/sliding_cache.py` | Scaled token embeddings; four offset RMSNorms per block; per-head Q/K offset norms; tanh GELU gating; alternating local/global rotary bases and causal masks; residual ordering and FP16 clipping; tied head; optional softcaps. Native MLX SDPA implements the attention contraction. | Local window 512, every sixth block global for the exact checkpoint. Window caches retain a bounded tail and absolute offset; chunk updates return the necessary history before retention. Quantized sliding KV and batching are explicitly unsupported. | Tiny full/incremental/chunk/reset, normalization, window boundaries, exact strict weights and native APIs. Stored BF16 reference logits match exactly in the initial trace; frozen native full-versus-chunk/decode consistency still fails, as does the comparator's own consistency. This is not an exact-checkpoint qualification pass. |
| Phi3 / M10 | MLX-LM `phi3.py`: `ModelArgs`, `Attention`, `MLP`, `Phi3Model`, `Model`; `rope_utils.py`: `SuScaledRoPE`; Transformers `phi3/{configuration,modeling}_phi3.py` | New `models/language/phi3/{config,model,weights}.py`, shared decoder and sliding cache | Packed QKV in Q/K/V order; packed gate/up in gate/up order; RMSNorm and SiLU; standard/linear/LongRoPE, optional partial rotation; residuals; untied head. Arbitrary explicit positions are handled in LongRoPE. | Exact target context 4096 and configured sliding window 2047. The provided MLX-LM reference ignores this window, so beyond-window correctness uses independent native boundary/receptive-field tests; short reference comparisons do not exercise that discrepancy. | Packed shape contracts, tiny reference parity, full/chunk/incremental/reset and outside-window perturbation tests; exact target load/API campaign. No 8K extension or unsupported sliding-KV quantization claim. |

## Shared runtime contracts

New families return the existing `ModelOutput` contract and register lazy loaders
and strict weight contracts, increasing registrations from 27 to 29. Existing
multimodal registrations are retained. Python, CLI and HTTP continue through the
existing generation service, scheduler, DeviceExecutor and cache ownership.
No HTTP-only model execution implementation was introduced.

`SlidingKVCache` exposes explicit quantization and batching restrictions to the
cache planner. Engine configuration preserves these family-owned caches rather
than silently replacing them with full-history storage. Configuration rejects
unsupported mathematics. Neither new family is promoted into optimized actor,
paged-attention or batching backends by these tests.

The decoded stop matcher now detects stop strings inside a token and across
token boundaries. It withholds only a possible stop prefix and closes the inner
iterator on a match, preserving resource cleanup. Tests exercise both boundaries
and verify that a generated partial-token match remains accounted for as a
generated token.

## Numerical interpretation

The cohort retains the existing precision-specific absolute/relative thresholds
from `mlx_one.inference_baseline.TOLERANCES` unchanged. Module hooks and projection
input snapshots are process-local diagnostics and restore patched methods.
Reports distinguish the first nonzero difference, first threshold violation,
state comparison, native self-consistency, reference self-consistency, forced
continuation and independently generated greedy sequences. Unmatched operations
are listed; a matched module boundary does not prove every internal operation
was independently traced.

The BF16 Qwen3.5 offset-norm implementation evaluates the HF normalization with
FP32 weight addition, whereas the supplied MLX comparator folds `1 + weight` in
checkpoint precision during sanitization. The first norm boundary differs. This
is a reference representation difference to investigate, not permission to
change tolerances or classify all downstream differences as harmless rounding.

The exact-source first-norm replay now measures this boundary directly: folding
in BF16 changes 6,139 of 16,384 output elements, with maximum error 0.03125 and
mean error 0.002009706455282867. Replaying the unrounded FP32 offset addition
matches the native BF16 result exactly. See the cohort's
`primitive-calls/M05-calls.json`. The isolated norm boundary still passes its
unchanged BF16 component threshold; downstream model/state gates continue to
fail. This evidence identifies the initial representation difference without
classifying the complete model discrepancy as benign.

The primary quantized traces additionally intercept gathered packed projections
that bypass module `__call__`. This closes a coverage gap in ordinary module
hooks; the original central traces explicitly retain their unmatched operation
lists. Primitive audits separately record selected-row embedding dequantization,
packed projection operands, observed state bytes and a full-16 versus
prefill-15-plus-one-token check for both stored and primary profiles.

## Numerical correction phase

The subsequent cohort `numerical-correction-20261020-1792496951619201000` preserves all eight architectures and changes only Qwen3.5 convolution accumulation and recurrent decay precision. See [current root-cause analysis](../archived-evidence.md) and [oracle self-consistency](oracle-self-consistency.md). Native-only Qwen3.5 vision and MTP side branches remain outside the local text comparator; no MTP, speculative, batched or paged backend is promoted. The original version of this report is retained in the new cohort's `prior-reports/`.
