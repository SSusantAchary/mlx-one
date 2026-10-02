# Qwen2.5-VL

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen2_5_vl`. The native architecture implements windowed/full
vision attention, a biased SwiGLU vision tower, spatial merging, multimodal RoPE,
and a Qwen text decoder. Configuration, strict tensor contracts, and synthetic
execution have tests. Current support is architecture-level: no native task loader
is registered for this model type.

[Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
is an upstream checkpoint candidate. Complete checkpoint loading, processor
parity, image/video task integration, training, and quality remain unqualified.

## Installation and assets

Use Python 3.10+ and install `python -m pip install -e '.[vision]'` from the
repository root. Synthetic execution needs Apple Silicon/Metal. Future
checkpoint integration needs complete text/vision weights, nested configuration,
matching tokenizer/template assets, and the exact image/video processor contract.
Installing the vision extra does not add the missing task loader.

## Available inspection and configuration examples

Metadata inspection does not load model weights:

```bash
mlx-one inspect Qwen/Qwen2.5-VL-3B-Instruct
```

For a local publisher configuration, validate the nested schema without allocating
the model. Replace the path with a complete local config; incompatible configurations
raise `ConfigError`.

```python
import json
from pathlib import Path

from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig

payload = json.loads(Path("/path/to/Qwen2.5-VL/config.json").read_text())
config = Qwen2_5_VLConfig.from_dict(payload)
print(config.model_type, config.vision_config.fullatt_block_indexes)
```

## Input contract and limitations

The model expects tokenized text, image/video patch tensors, grids, and matching
multimodal positions. The configuration checks special-token IDs, tower widths,
vision patch/window sizes, and full-attention layer indexes. A public image-chat
recipe must wait for a compatible task loader and processor integration.

`load_vlm_model` rejects this registry entry because it lacks a task loader.
The `generate` CLI is text-only; passing `--image` is unsupported. Do not rename
the checkpoint's `model_type` to Qwen2-VL to bypass this boundary.

Model-level cache tests do not qualify server prefix reuse or block KV. Block
KV/APC remains limited to dense text Qwen2/Qwen2.5. Unknown weights or nested
config errors must be resolved against the actual architecture.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[architecture tests](../../../../../tests/test_native_qwen_multimodal_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_qwen_multimodal_mlx.py).

```bash
pytest -q tests/test_native_qwen_multimodal_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_multimodal_mlx.py
```

The [family backlog](../../../../../model_list.txt) tracks candidates.
See [capability levels](../../../../../README.md#capability-levels) for promotion criteria.
