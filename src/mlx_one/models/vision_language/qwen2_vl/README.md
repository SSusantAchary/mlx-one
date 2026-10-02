# Qwen2-VL

[Model guide index](../../../../../README.md#model-specific-documentation)

## Overview and status

Registry type: `qwen2_vl`. The native architecture combines a vision Transformer,
3D patch projection, patch merger, GQA text decoder, multimodal RoPE, and image/video
token insertion. A native loader and static-image streaming API are implemented.
[Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) is a
candidate; exact checkpoint, processor parity, hardware, and quality qualification
are still required. Architectural video support does not establish a video-file workflow.

## Installation and assets

From the repository root, install `python -m pip install -e '.[vision]'` with
Python 3.10+. Execution requires Apple Silicon/Metal; the vision extra supplies
Pillow. Keep nested `config.json`, complete text/vision safetensors weights/index,
`tokenizer.json`, tokenizer metadata, and the checkpoint chat template.
The native image processor uses its implemented resize/normalization rules;
publisher processor assets still need parity verification for the exact checkpoint.

## Image prompting and Python example

`stream_vlm` takes a text prompt containing one literal `<image>` placeholder
per supplied image. Inputs are local paths or Pillow images, in placeholder order.
It does not download image URLs or decode PDF/video files.

This candidate example may download weights. Supply your own `image.jpg` and pin
an immutable `revision` in `load_vlm_model` for reproducibility.

```python
from mlx_one import TextGenerationOptions
from mlx_one.vision import load_vlm_model, stream_vlm

bundle = load_vlm_model("Qwen/Qwen2-VL-2B-Instruct")
for chunk in stream_vlm(
    bundle,
    "<image>\nDescribe this image.",
    ["image.jpg"],
    options=TextGenerationOptions(max_tokens=64),
):
    print(chunk.text, end="", flush=True)
```

This is the low-level placeholder input contract. `stream_vlm` does not apply the
loaded chat template automatically. Publisher chat formatting must preserve
the placeholder count and requires checkpoint-specific verification.

## CLI inspection

```bash
mlx-one inspect Qwen/Qwen2-VL-2B-Instruct
```

The `generate` CLI is text-only and has no `--image` flag. Use the vision Python
API above; do not transfer `mlx_vlm.generate` options from upstream documentation.

## Cache, limits, and troubleshooting

- Multimodal position IDs and the vision grid must match the inserted visual tokens.
- A placeholder-count error means the prompt and image list disagree.
- A missing Pillow error requires the vision extra. Paths must point to images
  Pillow can decode; URL strings are not local image paths.
- Larger image grids increase visual tokens and prefill memory. Start with one
  modest image and a short output; model context limits are not host-memory guarantees.
- Cached autoregressive decoding exists, but managed block KV/APC is restricted to
  dense text Qwen2/Qwen2.5. Qwen2-VL does not inherit that qualification.

## Implementation and validation

See [configuration](config.py), [model](model.py), [weights](weights.py),
[image processing](../../../vision/processing.py), and
[vision streaming](../../../vision/generation.py).
The Qwen suite includes architecture and synthetic execution coverage:

```bash
pytest -q tests/test_native_qwen_architectures.py
MLX_ONE_RUN_MLX_TESTS=1 pytest -q tests/test_native_qwen_mlx.py
```

See [backend-free tests](../../../../../tests/test_native_qwen_architectures.py),
[Metal tests](../../../../../tests/test_native_qwen_mlx.py), and
[qualification levels](../../../../../README.md#capability-levels).
