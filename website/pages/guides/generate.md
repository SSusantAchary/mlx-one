# Generate text

Start with a checkpoint that has recorded execution evidence in the
[inventory](../../../docs/benchmarks/inventory.md). Exact revisions, precision and
runtime scope matter independently of model-family implementation.

## Generate with Python

```python
from mlx_one import Engine

with Engine(model="openai-community/gpt2") as engine:
    result = engine.generate("The future of local AI is", max_tokens=64)
    print(result.text)
```

The context manager loads the model and releases its resources when you leave
the block. A complete local snapshot path also works. Use `revision` to pin
checkpoint identity and `offline=True` to require local or cached assets.

## Stream tokens

```python
from mlx_one import Engine

with Engine(model="LiquidAI/LFM2-350M") as engine:
    for chunk in engine.stream("Name two advantages of MLX.", max_tokens=64):
        print(chunk.text, end="", flush=True)
```

## Use the CLI

```bash
mlx-one generate openai-community/gpt2 "The future of local AI is" \
  --max-tokens 64 --temperature 0.8 --top-p 0.95 --seed 0
```

Add `--stream` for incremental text, `--json-output` for structured results,
or `--offline` when the complete checkpoint is already available locally.
Model downloads occur only when an online model repository is explicitly used.

For chat, expose an instruction-tuned checkpoint through the
[HTTP server](serve.md), where tokenizer chat templates format the messages.
GPT-2 is a completion model, rather than an instruction-following chat example.

## Next steps

- [Python and CLI reference](../../../docs/api/index.md)
- [Checkpoint revisions and precision](../../../docs/benchmarks/inventory.md)
- [Text measurements](../../../docs/benchmarks/text-performance.md)
- [Runtime experiments and optimized modes](../../../docs/experimental/inference-runtime.md)
