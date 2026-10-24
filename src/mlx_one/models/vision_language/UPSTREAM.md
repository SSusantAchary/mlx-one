# Native VLM reference provenance

The checkpoint layout mappings, Liquid dynamic tiling/NaFlex patch ordering,
SigLIP2 position resizing and two-layer projector were checked against
[MLX-VLM commit a4b586936d08eef2812235f7e4bbb6de56914cc0](https://github.com/Blaizzy/mlx-vlm/tree/a4b586936d08eef2812235f7e4bbb6de56914cc0/mlx_vlm/models).
Adapted processor algorithms retain the [upstream MIT license](UPSTREAM_LICENSE),
Copyright © 2025 Prince Canuma. The files affected are
`vision/lfm_processing.py`, `models/shared/vlm_weights.py`, and the Liquid config,
model and weight sanitizer. Existing Qwen native models and language backbones
are reused, with checkpoint layout corrections in their existing sanitizers.

Mathematical references also include the official Hugging Face Transformers
Qwen2-VL, Qwen2.5-VL, Qwen3.5 and LFM2-VL model/processor implementations.
No reference model library is imported at inference time. Pillow, NumPy,
standalone tokenizers, Jinja and native MLX are the processing/runtime utilities.

The seven metadata-only checkpoint fixtures in `tests/fixtures/vlm_checkpoints`
record immutable Hub revisions, retrieved asset SHA-256 hashes, safetensors
header shapes/dtypes, and publisher weight LFS SHA-256 identities. No tensor
payloads were retrieved. The benchmark plan repeats the five primary pins and
checks actual downloaded files against these content hashes before execution.

A deliberate difference from the inspected MLX-VLM processor: MLX-One respects
checkpoint-declared Liquid tiling/thumbnail flags, including LFM2.5-VL-3B's
nested `do_image_splitting: false`, rather than overriding them globally.
