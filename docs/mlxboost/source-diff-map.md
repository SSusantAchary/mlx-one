# Source-to-native execution audit

| Boundary | Pinned MLX-LM symbol | MLX-One symbol | Finding / classification |
| --- | --- | --- | --- |
| Public HTTP | generate_step and server orchestration | GenerationScheduler → DeviceExecutor → GenerationEngine → _stream_bundle_impl | API_ONLY; profile host handoffs separately |
| Incremental input | generate_step::_step | text/generation.py::_stream_bundle_impl | NO_DIFF: prompt once, subsequent input length one; persistent request cache |
| Evaluation | async_eval followed by item | eval(output.logits) followed by argmax.item | PROFILING_CANDIDATE: two evaluation boundaries; no async experiment without state parity |
| Greedy | sample_utils::greedy_sampler | text/sampling.py::select_token | argmax; upstream subtracts common logsumexp first; near-tie numerical behavior must be tested |
| Detokenization | BPEStreamingDetokenizer | _decode_piece / HFTokenizerAdapter | intentional raw-byte contract differs: upstream removes one initial space; do not trim native output to manufacture parity |
| Dense cache | cache.py::KVCache.update_and_fetch | core/cache.py::KVCache.update / FullKVComponent | capacity 256, axis 2; growth copy/concatenate differs; boundary tests required |
| Packed projections | QuantizedLinear → quantized_matmul | existing nn.QuantizedLinear modules | NO_DIFF; no repeated full-weight dequantization found |
| Attention | models/base.py::scaled_dot_product_attention | models/shared/reference_math.py::scaled_dot_product_attention | stock fast SDPA; explicit storage-to-query dtype cast is native cache API extension |
| Qwen2 | models/qwen2.py::Attention/TransformerBlock | models/language/qwen2/reference_math.py | same projection/RoPE/GQA/norm/residual order for contiguous B1; position/paged extensions excluded |
| Qwen3 | models/qwen3.py::Attention/TransformerBlock | models/language/qwen3/reference_math.py | same Q/K norm, RoPE, SDPA and residual order; checkpoint numerical gate remains necessary |
| SwiGLU | models/activations.py::swiglu | models/shared/reference_activations.py::swiglu | existing shapeless compile is inherited source math; no new compilation experiment |
| SmolLM2 | config model_type=llama, models/llama.py | native llama registration | BF16 diagnostic; not Qwen; exact configuration inspected |
| Memory admission | no equivalent | execution_forecast / transient_reservation / check_host_capacity | native lifecycle contract; profile before attempting to reduce overhead |
| LFM2 | models/lfm2.py::ShortConv and Attention | existing native LFM2 model | independent convolution state, OUT_OF_SCOPE until valid local Q4 baseline |

The reference implementation and `inference_baseline.TOLERANCES` are frozen.
No source inference/runtime replacement is permitted. Component and checkpoint
tests determine whether an apparent NO_DIFF actually passes the numerical gate.
