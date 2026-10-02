# Qwen2-MoE

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen2_moe`. The native decoder combines GQA/RoPE attention with
top-k routed experts, a shared expert, and router outputs. It has a registered
native text loader and generation path, but qualification currently covers
architecture/synthetic contracts rather than an exact real checkpoint.

[Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) is an
upstream family reference, not a qualified mlx-one checkpoint. Its active-parameter
name must not be used as its total memory footprint. Verify `model_type`, complete
expert weights, total parameters, and the local weight contract before attempting a load.

## Installation and assets

Use Python 3.10+ and install `python -m pip install -e .` from the repository root.
Execution requires Apple Silicon/Metal. The experimental native loader expects
`config.json` with `model_type=qwen2_moe`, complete safetensors weights/index,
`tokenizer.json`, and matching tokenizer/chat-template metadata. Expert tensor
mapping and shapes must satisfy [weights.py](weights.py).

## Inspection and experimental usage

Inspect metadata without loading weights:

```bash
mlx-one inspect Qwen/Qwen1.5-MoE-A2.7B
```

The following requires a prepared local checkpoint that passes the native contract;
replace the path. It is a usage example, not a verified upstream-model recipe.

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model("/path/to/qwen2-moe", offline=True)
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Explain sparse expert routing."}],
    options=TextGenerationOptions(max_tokens=32),
):
    print(chunk.text, end="", flush=True)
```

Use the checkpoint template for chat. Raw `mlx-one generate MODEL PROMPT` accepts
completion text; it does not automatically render an instruct conversation.

## Limits and troubleshooting

- Check routed/shared expert counts, top-k settings, and expert tensor stacking
  against the actual config. Unknown or incomplete weights are not safe to ignore.
- Dense attention caches are implemented. Block KV is restricted to dense
  Qwen2/Qwen2.5; Qwen2-MoE is outside that runtime boundary.
- Total expert weights determine storage requirements even when only a subset
  executes per token. Initial qualification policy limits candidates by total
  parameters, not active parameters.
- Real-checkpoint parity, memory, performance, and quality remain unqualified.

## Implementation and validation

See [configuration](config.py), [model](model.py),
[architecture tests](../../../../../tests/test_native_qwen_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_qwen_mlx.py).

```bash
pytest -q tests/test_native_qwen_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_mlx.py
```

See the [family backlog](../../../../../model_list.txt) and
[cache runtime](../../../../../docs/cache-runtime.md) for qualification boundaries.
