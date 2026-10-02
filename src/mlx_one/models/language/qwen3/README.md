# Qwen3 Dense

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen3`. The dense decoder adds bias-free attention, Q/K
normalization, and explicit head dimensions to the shared GQA/RoPE stack.
Native text loading, cache, completion, chat, sampling, and streaming are
implemented. [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is a
checkpoint candidate; synthetic validation does not qualify real-model quality
or throughput. Qwen3 MoE checkpoints are outside this dense implementation.

## Installation and assets

Use Python 3.10+ and `python -m pip install -e .` from the repository root.
Model execution needs Apple Silicon/Metal. Supply the matching `config.json`,
complete safetensors weights/index, `tokenizer.json`, tokenizer metadata, and chat
template. Keep explicit `head_dim` from the checkpoint; it need not equal
`hidden_size / num_attention_heads`.

## Completion and chat

```bash
mlx-one inspect Qwen/Qwen3-0.6B
mlx-one generate Qwen/Qwen3-0.6B "The advantages of unified memory are" \
  --max-tokens 64 --temperature 0 --stream
```

The CLI uses a raw completion prompt. For instruct chat, let the checkpoint
template format the messages. This candidate example may download the model;
add an immutable `revision` to `load_text_model` for a reproducible run.

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model("Qwen/Qwen3-0.6B")
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "What is unified memory?"}],
    options=TextGenerationOptions(max_tokens=64),
    template_options={"enable_thinking": False},
):
    print(chunk.text, end="", flush=True)
```

`enable_thinking` is passed to the loaded template. It only affects templates
that implement it; the ChatML fallback has no thinking switch. Enabling thinking
needs a larger generation budget and does not constitute reasoning qualification.

## Cache and troubleshooting

Use dense KV. The current experimental block backend is restricted to dense
Qwen2/Qwen2.5, so Qwen3 does not inherit its support. Neither prefix reuse nor
long-context performance is qualified by these examples. Reduce context/output
size when memory admission fails; the config's maximum is not a safe-memory promise.
Unexpected tensor shapes often indicate a MoE checkpoint or a lost explicit
head dimension. Use the separate [embedding](../../embeddings/qwen3_embedding/README.md)
and [reranker](../../embeddings/qwen3_reranker/README.md) guides for retrieval tasks.

## Implementation and validation

See [configuration](config.py), [model](model.py), and [weights](weights.py).
[Backend-free tests](../../../../../tests/test_native_qwen_architectures.py) and
[Metal tests](../../../../../tests/test_native_qwen_mlx.py) cover the native families.

```bash
pytest -q tests/test_native_qwen_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_mlx.py
```

Qualification definitions and candidates are in the
[root README](../../../../../README.md#capability-levels) and
[model catalog](../../../../../docs/model-catalog.md).
