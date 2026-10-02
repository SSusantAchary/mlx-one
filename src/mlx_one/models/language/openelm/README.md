# OpenELM

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `openelm`. The native decoder uses layer-specific attention-head
counts and feed-forward widths, fused QKV, GQA, RoPE, and RMSNorm.
[apple/OpenELM-270M](https://huggingface.co/apple/OpenELM-270M) is an upstream
reference; 270M–3B variants are catalog candidates. Native loading/generation and
synthetic tests exist, but each exact checkpoint release needs integration qualification.

## Installation and assets

Install `python -m pip install -e .` from the repository root using Python 3.10+.
Execution needs Apple Silicon/Metal. Supply the actual OpenELM `config.json` and
complete safetensors weights/index. The text loader requires `tokenizer.json`;
an external compatible tokenizer can be supplied through `tokenizer_source`.
The current adapter does not load a SentencePiece `.model` alone. Obtain the
publisher-recommended tokenizer assets, including any required access authorization.

## Inspection and completion

```bash
mlx-one inspect apple/OpenELM-270M
```

This example uses two prepared local directories; replace both paths. The
checkpoint must satisfy the native weight contract, and tokenizer IDs must match
the checkpoint vocabulary. It is not a recorded integration result.

```python
from mlx_one import TextGenerationOptions, generate, load_text_model

bundle = load_text_model(
    "/path/to/OpenELM-270M",
    tokenizer_source="/path/to/compatible-tokenizer",
    offline=True,
)
result = generate(
    bundle,
    "The advantages of running models locally include",
    options=TextGenerationOptions(max_tokens=32),
)
print(result.text)
```

Base models take completion text. Use `stream_chat` only with an appropriate
instruction checkpoint/template; a generic chat fallback is not evidence that a
base model was instruction tuned. The `generate` CLI currently has no
`--tokenizer-source` option; use the Python loader when tokenizer assets are separate.

## Settings, cache, and troubleshooting

- Preserve per-layer dimensions from the config. Treating OpenELM as a uniform
  Llama layout can produce invalid QKV and FFN shapes.
- Dense KV is the baseline; block KV is restricted to dense Qwen2/Qwen2.5.
- Read context length from the checkpoint and qualify memory independently.
- Missing `tokenizer.json` requires compatible exported tokenizer assets, not
  remote-code execution. Verify BOS/EOS IDs before interpreting output.
- Native config support does not qualify checkpoint training, parity, or quality.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_openelm_architecture.py), and
[synthetic Metal tests](../../../../../tests/test_native_openelm_mlx.py).

```bash
pytest -q tests/test_native_openelm_architecture.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_openelm_mlx.py
```

See the [candidate catalog](../../../../../docs/model-catalog.md) for release qualification.
