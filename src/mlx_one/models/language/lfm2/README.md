# LFM2 and LFM2.5

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `lfm2`. The decoder mixes short-convolution layers and attention
layers with gated feed-forward blocks. Dense generation means the ordinary
backend for this hybrid topology; its cache still contains both convolution state
and attention KV. Native loading, chat, completion, sampling, and streaming exist.

[LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M) is an upstream
checkpoint reference. Other LFM2/LFM2.5 variants need exact-checkpoint
qualification; the shared family name alone is insufficient. The local M4 cache
report exercised `mlx-community/LFM2-350M-4bit` for native loading and block
rejection, while its dense/parallel/cancellation matrix remains pending.

## Installation and assets

Install `python -m pip install -e .` from the repository root in Python 3.10+.
Execution requires Apple Silicon/Metal. Preserve `config.json`, complete
safetensors weights/index, `tokenizer.json`, tokenizer metadata, and chat template.
Keep the config's layer layout, convolution-state length, and quantization metadata.

## CLI and chat examples

These candidate examples may download weights. Pin an immutable revision for
reproducibility. CLI generation requests raw completion text.

```bash
mlx-one inspect LiquidAI/LFM2-350M
mlx-one generate LiquidAI/LFM2-350M "The advantages of unified memory are" \
  --max-tokens 32 --temperature 0 --stream
```

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model("LiquidAI/LFM2-350M")
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Describe unified memory in one sentence."}],
    options=TextGenerationOptions(max_tokens=32),
):
    print(chunk.text, end="", flush=True)
```

The checkpoint template controls chat formatting. The ChatML fallback does not
establish checkpoint-specific prompting or reasoning behavior.

## Hybrid cache safety

Every request must own its convolution and attention states together. Block KV
and automatic hybrid prefix restoration are unsupported; restoring only the
attention prefix would lose convolution history. Weight quantization (for example,
4-bit weights) does not change this topology or imply quantized convolution state.

The [M4/32 GB report](../../../../../docs/cache-benchmark-m4-32gb.md) records an
explicit `CacheTopologyUnsupported` block rejection. This is a topology-safety
result, not a fallback or a completed dense-generation benchmark. Long-context,
parallel, and cancellation results remain pending there. Use dense mode and
small contexts until the intended workload has measured evidence.

## Troubleshooting and validation

A config/weight error may indicate a variant with a different convolution layout
or MoE routing. Do not reinterpret every layer as attention. Use the separate
[MoE guide](../lfm2_moe/README.md) for sparse expert models.

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_lfm_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_lfm_mlx.py).

```bash
pytest -q tests/test_native_lfm_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_lfm_mlx.py
```

See [cache runtime](../../../../../docs/cache-runtime.md) and the
[candidate catalog](../../../../../docs/model-catalog.md) before publishing support claims.
