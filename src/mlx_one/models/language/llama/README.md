# Llama, MiniCPM5, and SmolLM2

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `llama`. The native decoder implements GQA, explicit attention
head dimensions, RoPE, RMSNorm, and gated feed-forward layers. Configuration,
weight mapping, quantized loading, multi-EOS handling, and cached execution have
tests. A shared registry type does not qualify every Llama-compatible checkpoint.

## Installation and assets

Use Python 3.10+, Apple Silicon, and Metal for execution. From the repository
root, run `python -m pip install -e .`. A complete bundle needs `config.json`,
safetensors weights (and their index when sharded), `tokenizer.json`, and matching
tokenizer metadata. Chat uses the bundle's `chat_template.jinja` or
`tokenizer_config.json` template. MLX quantized weights must retain their matching
quantization configuration; model-weight precision and cache precision are separate.

## MiniCPM5

The pinned integration gate covers these publisher MLX checkpoints:

| Checkpoint | Revision in the gate |
| --- | --- |
| [openbmb/MiniCPM5-1B-MLX](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) | `9879b18bf2928355fcdf4287635388a3665a40cb` |
| [openbmb/MiniCPM5-2B-MLX](https://huggingface.co/openbmb/MiniCPM5-2B-MLX) | `8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3` |

The test asserts 4-bit affine weights, group size 64, multiple EOS IDs, template
rendering, and one-token generation. Its configured 128K context is not a measured
safe context on every Mac. An available integration gate is not a recorded passing run.

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model(
    "openbmb/MiniCPM5-1B-MLX",
    revision="9879b18bf2928355fcdf4287635388a3665a40cb",
)
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Explain unified memory briefly."}],
    options=TextGenerationOptions(max_tokens=64),
    template_options={"enable_thinking": False},
):
    print(chunk.text, end="", flush=True)
```

`enable_thinking` is a checkpoint-template option here. Rendering a thinking
prompt does not establish reasoning quality. Use `True` only when the loaded
checkpoint template supports it and allow enough output tokens for the answer.

## SmolLM2

[HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
uses the Llama path. The catalog pins
`31b70e2e869a7173562077fd711b654946d38674` for recorded M4 LoRA-training and
adapter-reload evidence. Its tiny held-out quality gate scored `1.0 → 0.0` and
failed. This evidence is scoped to that workload, not general inference quality.

```bash
mlx-one inspect HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --revision 31b70e2e869a7173562077fd711b654946d38674
```

For chat, load this checkpoint with `load_text_model` and use `stream_chat` as
above, omitting MiniCPM5's `template_options`. Raw `mlx-one generate MODEL PROMPT`
performs completion; it does not automatically format an instruction conversation.

## Cache and troubleshooting

Dense attention KV is the baseline; block KV is restricted to dense Qwen2/Qwen2.5.
Preserve the checkpoint's head dimension and tokenizer/EOS metadata. Wrong
dimensions or unknown tensors should be resolved against the exact config rather
than ignored. A missing tokenizer or shard requires a complete matching snapshot.
Use the checkpoint template instead of copying chat markers from another family.

## Implementation and validation

See [configuration](config.py), [model](model.py), and [weights](weights.py), plus
[architecture tests](../../../../../tests/test_native_llama_architecture.py),
[synthetic Metal tests](../../../../../tests/test_native_llama_mlx.py), and the
[MiniCPM5 integration gate](../../../../../tests/test_native_minicpm5_integration.py).

```bash
pytest -q tests/test_native_llama_architecture.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_llama_mlx.py
MLX_ONE_RUN_MINICPM5_INTEGRATION=1 pytest -q tests/test_native_minicpm5_integration.py
```

The last command may download pinned checkpoints. See the
[candidate catalog](../../../../../docs/model-catalog.md) and
[SmolLM2 evidence](../../../../../qualification/results/smollm2-1.7b-lora-m4-air-32gb.json).
