# GPT-2

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `gpt2`. The native decoder implements learned absolute positions,
fused QKV attention, GELU feed-forward layers, byte-level BPE, and per-layer KV
caches. Configuration profiles cover 124M, 355M, 774M, and 1.5B variants.

[openai-community/gpt2](https://huggingface.co/openai-community/gpt2) is the
124M checkpoint candidate. Architecture and synthetic execution tests exist;
this guide does not establish real-checkpoint, performance, or quality qualification.

## Installation and assets

Use Python 3.10+ and an Apple Silicon Mac with working Metal for execution.
From the repository root, install with `python -m pip install -e .`.
The loader requires `config.json` and complete safetensors weights. Multiple
shards require `model.safetensors.index.json`. It uses `vocab.json` plus
`merges.txt` when present, otherwise `tokenizer.json`. Pickle weights are rejected.

## Completion examples

GPT-2 is a completion model. Pass ordinary text rather than instruction/chat
markers. Run shell examples from the repository root with the environment active.
These candidate examples may download weights; pin `revision` for reproducible runs.

```bash
mlx-one inspect openai-community/gpt2
mlx-one generate openai-community/gpt2 "Once upon a time" \
  --max-tokens 32 --temperature 0 --stream
```

```python
from mlx_one import TextGenerationOptions, generate

result = generate(
    "openai-community/gpt2",
    "Once upon a time",
    options=TextGenerationOptions(max_tokens=32, temperature=0.0),
)
print(result.text)
```

## Settings, cache, and troubleshooting

- Keep input plus output inside the checkpoint's learned-position limit; changing
  a context option does not create longer position embeddings.
- Sampling uses `TextGenerationOptions` (`temperature`, `top_k`, `top_p`, `seed`,
  and `stop`). Zero temperature selects greedy decoding.
- Dense KV is the baseline. Experimental block KV is restricted to Qwen2/Qwen2.5;
  do not infer APC qualification for GPT-2 from its cache implementation.
- A missing safetensors or tokenizer error means the snapshot is incomplete or
  uses an unsupported format. A `.bin` checkpoint alone is insufficient.
- Native GPT-2 training is explicitly rejected by the current training gate.

## Implementation and validation

See [configuration](config.py), [model](model.py), and [weight contract](weights.py).
[Backend-free tests](../../../../../tests/test_native_gpt2_architecture.py) cover
profiles, weight mapping, tokenizer, loading policy, and CLI contracts.
[Metal tests](../../../../../tests/test_native_gpt2_mlx.py) exercise tiny models:

```bash
pytest -q tests/test_native_gpt2_architecture.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_gpt2_mlx.py
```

The Metal gate uses synthetic weights. Consult the
[capability levels](../../../../../README.md#capability-levels) before making a support claim.
