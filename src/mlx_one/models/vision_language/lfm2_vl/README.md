# LFM2-VL and LFM2.5-VL

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `lfm2_vl`. The native architecture contains a vision tower,
pixel-unshuffle/projector, image-token feature insertion, and an LFM2 hybrid
language decoder. Nested configuration and synthetic execution have tests.
No native checkpoint task loader, registry weight sanitizer/contract, or complete
image-chat processor workflow is registered for this entry.

[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) is an
upstream candidate reference. Publisher checkpoint/processor mapping and real
task parity remain unqualified; architecture tests are not a loadability claim.

## Installation and assets

Use Python 3.10+ and `python -m pip install -e '.[vision]'` from the repository
root. Synthetic execution requires Apple Silicon/Metal. Future task integration
needs text/vision config and weights, the actual image processor, special-token
IDs, and matching tokenizer/template assets. The vision extra supplies image
dependencies but does not provide the missing checkpoint loader.

## Available inspection and configuration examples

```bash
mlx-one inspect LiquidAI/LFM2.5-VL-1.6B
```

Validate a compatible local nested config without initializing the model;
replace the path. Upstream fields may need explicit mapping before this schema
accepts them, which is part of the pending integration work.

```python
import json
from pathlib import Path

from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig

payload = json.loads(Path("/path/to/lfm2-vl/config.json").read_text())
config = Lfm2VLConfig.from_dict(payload)
print(config.model_type, config.vision_config.projection_dim)
```

## Input contract, cache, and limitations

The vision projection must match the text hidden width. Patch-grid/downsampling
dimensions and image-token positions must match the image features. There is no
validated public prompt/image preparation recipe for this family yet.

`load_vlm_model` rejects this registry entry because it has no task loader.
Do not use Qwen's processor or rename `model_type` as a workaround. The `generate`
CLI takes text prompts and does not provide an image option.

The language backbone contains convolution state and attention KV. Block KV and
hybrid prefix restore are unsupported; vision architecture coverage does not
establish server APC, memory, training, or quality qualification. A projection
width or image-grid error requires matching the nested schema to the intended
checkpoint, not silently reshaping weights.

## Implementation and validation

See [configuration](config.py), [model](model.py),
[architecture tests](../../../../../tests/test_native_lfm_architectures.py), and
[synthetic Metal tests](../../../../../tests/test_native_lfm_mlx.py).

```bash
pytest -q tests/test_native_lfm_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_lfm_mlx.py
```

See the [family backlog](../../../../../model_list.txt) and
[qualification levels](../../../../../README.md#capability-levels).
