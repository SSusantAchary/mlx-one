# Qwen3.5

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen3_5`. The dense multimodal architecture mixes gated-delta
linear attention with full attention and includes learned/interpolated vision
positions, spatial merging, and multimodal RoPE. Native text loading and
generation are implemented; multimodal checkpoint/task qualification remains
architecture-level and synthetic.

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) is a publisher
checkpoint reference; 0.8B/2B variants are candidates. The presence of image/video
capabilities in the registry does not qualify the public vision task path.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Image/video patches                         Text token IDs
             ↓                                      ↓
┌──────────── Vision encoder ───────────┐     Token embeddings
│ Learned/interpolated positions        │             │
│ Vision Transformer → spatial merger   │             │
└──────────────────┬────────────────────┘             │
                   └──────────────┬──────────────────┘
                       Replace visual-token slots
                                  ↓
┌────────────────── Dense hybrid text decoder ────────────────────┐
│ RMSNorm → Gated DeltaNet OR full GQA + multimodal RoPE → add    │
│           conv/recurrent state     attention KV                 │
│ RMSNorm → gated feed-forward                         → add      │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final norm → LM head → logits
    └→ Optional MTP modules, only with complete compatible tensors
```

## Papers and upstream references

Component paper: [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).
Checkpoint architecture reference:
[Qwen3.5-0.8B official model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).

A standalone dense-Qwen3.5 technical report was not verified for this guide.
The component paper is not a Qwen3.5 release report; Qwen3.5-Omni is a different
model. The image branch is architectural; the implemented public workflow is
text generation, and task/checkpoint qualification remains as described below.

## Installation and assets

Use Python 3.10+ and `python -m pip install -e .` from the repository root.
Execution requires Apple Silicon/Metal. Keep the checkpoint's nested `config.json`,
matching text and vision safetensors weights/index, `tokenizer.json`, tokenizer
metadata, and chat template. The text loader still validates the full multimodal
weight contract; text-only usage does not permit dropping required vision tensors.
Quantized bundles must retain their matching MLX quantization metadata.

## Text completion and chat

These are candidate usage examples, not checkpoint qualification results. They
may download weights; pin an immutable revision for reproducible execution.

```bash
mlx-one inspect Qwen/Qwen3.5-0.8B
mlx-one generate Qwen/Qwen3.5-0.8B "The advantages of unified memory are" \
  --max-tokens 64 --temperature 0 --stream
```

The CLI requests raw completion. For chat, use the template from the checkpoint:

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model("Qwen/Qwen3.5-0.8B")
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Describe unified memory briefly."}],
    options=TextGenerationOptions(max_tokens=64),
    template_options={"enable_thinking": False},
):
    print(chunk.text, end="", flush=True)
```

The thinking option acts only when implemented by the loaded template. Optional
MTP tensors enable an implementation path; incomplete MTP tensors are rejected.
Their presence does not establish speculative-decoding parity or speedup for a
checkpoint. Ordinary text generation remains the starting point.

## Vision task boundary

The registry's loader is `load_text_model`, not a dedicated Qwen3.5 VLM loader.
Although the generic vision loader checks for a loader entry, it currently builds
`QwenImageProcessor`, which was implemented for Qwen2-VL. That is insufficient
evidence for Qwen3.5 processor compatibility. No image/video inference recipe is
qualified here; do not substitute Qwen2-VL preprocessing and assume parity.

## Hybrid cache and troubleshooting

Attention KV, convolution history, and linear/recurrent states must stay aligned
within each request. Hybrid prefix restore and block KV are unsupported. The
[M4 APC results](../../../../../docs/cache-benchmark-m4-32gb.md) apply to their
Qwen2.5 baseline, not this hybrid family.

Preserve `layer_types`, linear-attention dimensions, vision position embeddings,
and complete tensor sets. Missing vision weights, partial MTP assets, or lost
nested config fields are snapshot/config issues rather than a reason to disable
strict loading. Check memory with short contexts before attempting long inputs.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_qwen_multimodal_architectures.py),
[synthetic Metal tests](../../../../../tests/test_native_qwen_multimodal_mlx.py), and
[text runtime tests](../../../../../tests/test_text_runtime.py).

```bash
pytest -q tests/test_native_qwen_multimodal_architectures.py tests/test_text_runtime.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_multimodal_mlx.py
```

See the [cache matrix](../../../../../docs/cache-runtime.md) and
[qualification levels](../../../../../README.md#capability-levels).
