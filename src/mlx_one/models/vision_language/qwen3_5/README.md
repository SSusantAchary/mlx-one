# Qwen3.5 0.8B

[Model guide index](../../../../../README.md#model-specific-documentation)

## Status and exact targets

**CHECKPOINT_SMOKE_PASSED** for the pinned primary checkpoints below on Apple
M4 with 32 GiB unified memory. Real Python/CLI/live HTTP/SSE, multi-turn,
changed-image and cancellation checks passed. Direct-model performance is measured
separately using pinned HF documentation images; see the
[hardware report](../../../../../docs/validation/vlm-hf-2026-10-24/report.md).
These bounded smokes are experimental, not general accuracy or long-context qualification.

- `mlx-community/Qwen3.5-0.8B-4bit` (Q4) at `da28692b5f139cb0ec58a356b437486b7dac7462`.
- Alternative `mlx-community/Qwen3.5-0.8B-MLX-4bit` has its own pinned metadata fixture.

**Additional alternative checkpoints remain metadata-only:** their listed pins
were inspected but were not executed in this five-primary hardware campaign.

## Model-specific implementation

The dedicated Qwen3.5 processor reads this checkpoint's patch size 16,
temporal patch size 2, merge factor 2, 0.5 mean/std normalization and dynamic
resolution. Learned/interpolated vision positions, tanh-GELU blocks and the spatial
merger feed the existing gated-delta/full-attention language backbone. Cached
decode retains attention KV, convolution history, recurrent state and multimodal
RoPE deltas together. The two MLX checkpoints have separately pinned metadata and
headers; matching architecture names do not substitute for converter validation.
MLX folded offset norms and convolution layouts use the existing Qwen3.5 sanitizer.
Declared-but-absent optional MTP modules are not instantiated; partial MTP tensor
sets fail loading. VLM speculative decoding is outside this milestone.

Historical text-only evidence for `Qwen/Qwen3.5-0.8B` remains separate: its
stored-source short smoke does not establish either targeted MLX VLM Q4 checkpoint
or long-context functionality.

## Model Architecture

The diagram follows the [native implementation](model.py). Layer counts, widths,
head counts, and input sizes come from the checkpoint configuration; this is an
architecture map, not an additional support or qualification claim.

```text
Image/video patches                         Text token IDs
             ↓                                      ↓
┌──────────── Vision encoder ───────────┐     Token embeddings
│ Learned/interpolated positions        │             │
│ Vision Transformer → spatial merger   │             │
└──────────────────┬────────────────────┘             │
                   └──────────────┬──────────────────┘
                       Replace visual-token slots
                                  ↓
┌────────────────── Dense hybrid text decoder ────────────────────┐
│ RMSNorm → Gated DeltaNet OR full GQA + multimodal RoPE → add    │
│           conv/recurrent state     attention KV                 │
│ RMSNorm → gated feed-forward                         → add      │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final norm → LM head → logits
    └→ Optional MTP modules, only with complete compatible tensors
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
mlx-one vlm --model ~/mlx-one-models/vlm/qwen3.5-0.8b \
  --offline --image benchmarks/vlm/images/VLM-S1-1.png \
  --prompt 'Describe the shapes and colors.' --max-tokens 64 --stream
```

```python
from pathlib import Path
from mlx_one.vision import VisionLanguageEngine
from mlx_one.text import TextGenerationOptions

with VisionLanguageEngine(Path('~/mlx-one-models/vlm/qwen3.5-0.8b').expanduser(),
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
