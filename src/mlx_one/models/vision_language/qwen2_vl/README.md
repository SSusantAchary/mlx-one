# Qwen2-VL 2B

[Model guide index](../../../../../README.md#model-specific-documentation)

## Status and exact targets

**CHECKPOINT_SMOKE_PASSED** for the pinned primary checkpoints below on Apple
M4 with 32 GiB unified memory. Real Python/CLI/live HTTP/SSE, multi-turn,
changed-image and cancellation checks passed. Direct-model performance is measured
separately using pinned HF documentation images; see the
[hardware report](../../../../../docs/validation/vlm-hf-2026-10-24/report.md).
These bounded smokes are experimental, not general accuracy or long-context qualification.

- `mlx-community/Qwen2-VL-2B-Instruct-4bit` (Q4) at `01af461cdb9574acc09084a0ef94e216e142b085`.

## Model-specific implementation

The pinned 2B checkpoint has a 32-block, width-1280 vision Transformer and
width-1536 text backbone. Native static-image preparation uses patch size 14,
temporal patch size 2 (a duplicated still frame), merge factor 2, checkpoint
CLIP normalization, patch-aligned dynamic resolution and exact image start/end
markers. MLX OTHWI Conv3D tensors are transposed into processor CTHW patch order.
Visual features replace exactly the expanded image token slots. Three-axis
multimodal RoPE deltas continue through cached language decode.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Images / video frame patches                 Text token IDs
             ↓                                      ↓
┌──────────── Vision encoder ───────────┐     Token embeddings
│ Temporal/spatial patch embedding      │             │
│ Vision Transformer + vision RoPE      │             │
│ Spatial patch merger / projection     │             │
└──────────────────┬────────────────────┘             │
                   └──────────────┬──────────────────┘
                       Replace visual-token slots
                                  ↓
             Qwen decoder + multimodal RoPE (time/height/width)
                                  ↓
                         LM head → token logits
```

## Installation and native image inference

Install `python -m pip install -e '.[vision,server]'` from the repository root.
Python 3.12 and Apple Silicon/Metal are the manual test target. Keep the complete
checkpoint, matching tokenizer, processor metadata and template together; strict
loading rejects missing or mismatched tensors. Packed affine Q4/Q8 remain packed;
only modules with actual scale tensors are quantized, preserving floating vision
or projector modules. No MLX-VLM, Transformers or PyTorch runtime is imported.

After the pinned manual download in the [runbook](../../../../../docs/VLM_TESTING.md):

```bash
mlx-one vlm --model ~/mlx-one-models/vlm/qwen2-vl-2b \
  --offline --image benchmarks/vlm/images/VLM-S1-1.png \
  --prompt 'Describe the shapes and colors.' --max-tokens 64 --stream
```

```python
from pathlib import Path
from mlx_one.vision import VisionLanguageEngine
from mlx_one.text import TextGenerationOptions

with VisionLanguageEngine(Path('~/mlx-one-models/vlm/qwen2-vl-2b').expanduser(),
                          offline=True) as engine:
    result = engine.generate('Describe the shapes and colors.',
        images=['benchmarks/vlm/images/VLM-S1-1.png'],
        options=TextGenerationOptions(max_tokens=64))
    print(result.text, result.token_ids, result.image_metadata)
```

For ordered interleaved images and multi-turn chat, use `messages=` as documented
in the runbook. HTTP uses OpenAI `image_url` base64 data URLs, the existing
positional `serve MODEL` command, aliases, authentication and SSE. Remote image
fetching, video, animation, audio, training, VLM tool calls and speculative decode
are unsupported. Source image count/pixels, processed token/memory budgets and
checkpoint context limits are enforced. Large image grids can exceed M4 memory
before the configured language context is reached.

## Request state and validation

Image encoding/fusion runs once per prefill, followed by the existing native
cached generation loop. Each request owns its image features and full model
state. Closing/cancelling a stream or encountering an exception releases state;
there is no cross-request image-prefix reuse, paged VLM cache or VLM continuous
batching claim. Attention KV defaults to FP16; hybrid recurrent state preserves
its model-specific policy.

```bash
python -m pytest -q --no-cov tests/test_vlm_checkpoint_metadata.py tests/test_vlm_benchmark.py
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest -q --no-cov tests/test_vlm_pipeline_native.py tests/test_task_vlm_native.py
```

See [manual acceptance checks](../../../../../docs/VLM_TESTING.md),
[benchmark methodology and pending records](../../../../../docs/benchmarks/vlm/README.md),
[reference attribution](../UPSTREAM.md), [configuration](config.py),
[native model](model.py) and [strict weight mapping](weights.py).
Any new checkpoint revision requires its own image-dependent smoke and output
review before inheriting support. The measured primary revisions above passed
the bounded checks; this does not establish general OCR accuracy.
