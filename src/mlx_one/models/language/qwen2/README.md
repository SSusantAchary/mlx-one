# Qwen2, Qwen2.5, and Qwen2.5-Coder

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen2`. These families share the dense decoder path: GQA,
RoPE, RMSNorm, biased QKV projections, and gated feed-forward layers. Native
loading, completion, chat, sampling, and streaming are implemented. Qualification
is per checkpoint, precision, operation, and workload.

## Checkpoints and evidence

[Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
at `2e1fd397ee46e1388853d2af2c993145b0f1098a` has recorded M4 LoRA-training
and adapter-reload evidence. Its tiny held-out quality gate scored `0.0 → 0.0`
and failed; the result is not a quality qualification.

Qwen2.5 0.5B/1.5B/3B and additional Coder sizes remain candidates in the
[catalog](../../../../../docs/model-catalog.md). The Coder variant uses the same
architecture, with its own weights and tokenizer/template assets.

## Installation and assets

From the repository root, install `python -m pip install -e .` in a Python 3.10+
environment. Execution requires Apple Silicon and Metal. Preserve `config.json`,
complete safetensors weights/index, `tokenizer.json`, tokenizer metadata, and the
checkpoint chat template. Quantized checkpoints also need matching MLX
quantization metadata. Base and instruct checkpoints have different prompting needs.

## CLI and Python examples

The CLI below requests raw completion; an instruct conversation should use the
Python chat example. Both are usage examples, not new qualification results.

```bash
mlx-one inspect Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a
mlx-one generate Qwen/Qwen2.5-Coder-1.5B-Instruct "def fibonacci(n):" \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --max-tokens 64 --temperature 0 --stream
```

```python
from mlx_one import TextGenerationOptions, load_text_model, stream_chat

bundle = load_text_model(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
)
for chunk in stream_chat(
    bundle,
    [{"role": "user", "content": "Write a Python function that adds two numbers."}],
    options=TextGenerationOptions(max_tokens=64, temperature=0.0),
):
    print(chunk.text, end="", flush=True)
```

The chat layer reads the checkpoint template; its fallback uses ChatML for this
registry type. Prefer matching publisher assets over hand-written role markers.

## Cache and memory limits

Dense KV is the default. Experimental block KV is limited to unquantized,
text-only Qwen2/Qwen2.5, batch size one, and f16 KV. An HTTP 200 block smoke
result does not prove token parity or shared-prefix execution.

The [M4/32 GB report](../../../../../docs/cache-benchmark-m4-32gb.md) records
approximately 28.2 GB peak MLX memory at 16K input for its unquantized Qwen
baseline and a clean failure at approximately 32K. Its APC trial produced four
misses and zero reused tokens because the published chat-prompt snapshot did not
match the desired common prefix. Dense and block APC remain unqualified in that
report. These measurements must not be transferred to the 1.5B Coder training run.

Use small initial contexts and consult the
[cache runtime matrix](../../../../../docs/cache-runtime.md). Weight quantization
does not automatically quantize KV storage. Resolve shape/tokenizer mismatches
against the exact checkpoint; do not skip strict weight validation.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_qwen_architectures.py),
[Metal tests](../../../../../tests/test_native_qwen_mlx.py), and
[Qwen training evidence](../../../../../qualification/results/qwen2.5-coder-1.5b-lora-m4-air-32gb.json).

```bash
pytest -q tests/test_native_qwen_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_mlx.py
```
