# LFM2-VL and LFM2.5-VL

[Model guide index](../../../../../README.md#model-specific-documentation)

## Status and exact targets

**CHECKPOINT_SMOKE_PASSED** for the pinned primary checkpoints below on Apple
M4 with 32 GiB unified memory. Real Python/CLI/live HTTP/SSE, multi-turn,
changed-image and cancellation checks passed. Direct-model performance is measured
separately using pinned HF documentation images; see the
[hardware report](../../../../../docs/validation/vlm-hf-2026-10-24/report.md).
These bounded smokes are experimental, not general accuracy or long-context qualification.

- `mlx-community/LFM2-VL-450M-8bit` (Q8) at `2422c0686f1f1c6267640bc1f069ff4e8b06b8d0`.
- `LiquidAI/LFM2.5-VL-3B-MLX-4bit` (Q4) at `7fad79d26092c1145aa8dde873896a07d3e96a3b`.
- Later target `mlx-community/LFM2-VL-3B-4bit` has its own pinned metadata fixture.

**Additional alternative checkpoints remain metadata-only:** their listed pins
were inspected but were not executed in this five-primary hardware campaign.

## Model-specific implementation

Both targets reuse the existing LFM2 convolution/full-attention hybrid language
backbone. They use SigLIP2 NaFlex, HWC flattened patches, patch size 16,
bilinear antialiased resizing of the learned position table, tanh-GELU vision
blocks and downsample-factor-2 pixel unshuffle. Padded patch slots never enter
attention: each view is sliced to its actual spatial length. Odd raw patch grids
are padded during pixel unshuffle, and visual-token expansion uses the matching
ceil-divided count. A two-layer GELU projector replaces the old synthetic-only
single-layer contract for these actual SigLIP2 configurations.

| Target | Vision tower | Projector | Processor distinction |
|---|---|---|---|
| LFM2-VL 450M Q8 | 12 blocks, width 768 | width 2560, input LayerNorm | Published bilinear resampling; dynamic tiling up to 10 tiles; checkpoint disables thumbnail |
| LFM2.5-VL 3B Q4 | 27 blocks, width 1152 | width 2048, no input LayerNorm | Nested processor configuration, default bicubic; checkpoint disables splitting and thumbnail |

The processor reads both `preprocessor_config.json` and `processor_config.json`,
including nested `image_processor` values. Per-checkpoint flags win: it does not
force Liquid 3B tiling on. As in the official LFM processor, packed-patch resizing
is applied even when older SigLIP metadata contains `do_resize: false`. Row/column,
thumbnail and image-boundary special tokens preserve view order. Legacy explicit
MLX-One synthetic processor fixtures remain supported but are not substitutes for
publisher checkpoints. The additional 3B LFM2-VL Q4 checkpoint has separate header
and processor metadata for later manual tests.

**LFM2.5-VL 3B effective context is capped at 32,768 tokens** despite its text
configuration advertising a larger maximum; no 128K VLM support is claimed.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Image patch grids                            Text token IDs
       ↓                                            ↓
┌────────── Native vision tower ──────────┐    Token embeddings
│ Patch + interpolated position embeds    │            │
│ Vision Transformer                      │            │
│ Pixel unshuffle → norm (if set)         │            │
│ Projection to text hidden width         │            │
└──────────────────┬──────────────────────┘            │
                   └──────────────┬───────────────────┘
                       Replace image-token slots
                                  ↓
                   LFM2 convolution/attention hybrid decoder
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
mlx-one vlm --model ~/mlx-one-models/vlm/lfm2-vl-450m \
  --offline --image benchmarks/vlm/images/VLM-S1-1.png \
  --prompt 'Describe the shapes and colors.' --max-tokens 64 --stream
```

```python
from pathlib import Path
from mlx_one.vision import VisionLanguageEngine
from mlx_one.text import TextGenerationOptions

with VisionLanguageEngine(Path('~/mlx-one-models/vlm/lfm2-vl-450m').expanduser(),
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
