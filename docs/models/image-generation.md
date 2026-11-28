# Native text-to-image generation

MLX-One runs the FLUX.2 Klein 4B and Z-Image Turbo denoisers, Qwen3 prompt
conditioning and VAE decoding with Apple MLX. The implementation includes
BF16 loading and affine Q4/Q8 denoiser profiles. Both families passed small
Metal pipeline tests. Full-checkpoint generation and performance qualification
are pending. [Qualification record](../qualification/image-generation/README.md)

| Model ID | Pinned checkpoint revision | Default API steps | Denoiser calls |
| --- | --- | ---: | ---: |
| `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | 4 | 4 |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` | 9 | 8 |

## Download and run published Q4 weights

MLX-One also accepts these two published MFLUX-layout bundles directly with the
native image engine. MLX-One validates the pinned assets, maps the stored tensor
names, and uses the packed weights as published. The runtime uses Apple MLX
and needs no MFLUX installation or local conversion.

| Model ID | Pinned bundle revision |
| --- | --- |
| `mflux-community/flux2-klein-4b-mflux-q4` | `794cd159538149ad9830848508c31f0ea7088e58` |
| `mflux-community/z-image-turbo-mflux-q4` | `f427e257d8e6ffa03edd4d9ac554a05809da456c` |

```bash
Q4_MODEL=mflux-community/flux2-klein-4b-mflux-q4
Q4_REVISION=794cd159538149ad9830848508c31f0ea7088e58
Q4_DIR="$HOME/models/flux2-klein-4b-mflux-q4"

hf download "$Q4_MODEL" --revision "$Q4_REVISION" --local-dir "$Q4_DIR"
mlx-one image generate "$Q4_MODEL" --checkpoint "$Q4_DIR" --offline \
  --prompt "A botanical scientific illustration" --width 768 --height 512 \
  --seed 42 --output image.png
```

The model ID selects Q4 automatically. The same ID works in Python and with
the image server. Omit `--checkpoint` and `--offline` to let MLX-One resolve
the pinned snapshot on first use. In this profile, the text embedding and linear
projections, the denoiser linear projections, and the VAE's decoder mid-block
attention projections use Q4. The remaining weights use BF16. Published shard
identities and the canonical configuration are recorded in the package.
[Q4 artifacts](../../src/mlx_one/data/image_published_q4.json)

## Use a manually downloaded checkpoint

Install the existing image-codec extra from this checkout.

```bash
pip install -e '.[vision]'
mlx-one image list
```

Download the repository's Diffusers component layout at the exact revision in
the table. The required files live under `transformer/`, `text_encoder/`,
`vae/`, `scheduler/` and `tokenizer/`, with `model_index.json` at the root.
FLUX's top-level single-file checkpoint can be omitted.
The repository includes the exact filenames and config hashes.
[Checkpoint artifacts](../../src/mlx_one/data/image_checkpoints.json)

```bash
MODEL=black-forest-labs/FLUX.2-klein-4B
CHECKPOINT=/path/to/local/flux2-klein-4b

mlx-one image generate "$MODEL" --checkpoint "$CHECKPOINT" --offline \
  --quantization 4 --prompt "A botanical scientific illustration" \
  --width 768 --height 512 --seed 42 --output image.png
```

An explicitly requested Q4 or Q8 profile packs the denoiser's attention and
feed-forward linear weights in MLX affine groups of 64 along the input axis.
The text encoder, VAE, and remaining denoiser weights use BF16. You can export
a reusable packed checkpoint to avoid repacking the weights after an unload.

```bash
mlx-one image convert "$MODEL" "$CHECKPOINT" /path/to/flux2-klein-q4 --quantization 4
```

Pass the converted directory with the matching `--quantization`. The loader
checks its manifest, shard hashes and native parameter layouts. Python and CLI
can also resolve the pinned Hugging Face snapshot by model ID when the
checkpoint argument is omitted. `--offline` limits that lookup to the local cache.

## Python

```python
from mlx_one import ImageEngine

with ImageEngine(
    "black-forest-labs/FLUX.2-klein-4B",
    checkpoint="/path/to/local/flux2-klein-4b",
    quantization=4,
    offline=True,
) as engine:
    result = engine.generate(
        "A botanical scientific illustration", width=768, height=512, seed=42,
    )
    result.save("image.png")
    print(result.metadata)
```

Metadata includes the checkpoint revision, effective seed, dimensions, requested
steps, executed denoiser calls, precision, stage times and memory telemetry.
Quantized components load on demand and can stay resident during the configured
idle window. BF16 runs release each component after its stage. The image engine
uses MLX-One's shared device executor and checks unified-memory headroom before
loading weights.

## OpenAI-compatible HTTP

```bash
pip install -e '.[server,vision]'
mlx-one image serve "$MODEL" --checkpoint "$CHECKPOINT" --offline --quantization 4
```

The server binds to `127.0.0.1:8080` by default. To configure an image sidecar
alongside an existing text model, use `mlx-one serve TEXT_MODEL --image-model
"$MODEL" --image-checkpoint "$CHECKPOINT" --image-quantization 4`. In mixed
mode, use the existing text server port. The image request runs on the shared
device executor.

```python
from openai import OpenAI

with OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local") as client:
    result = client.images.generate(
        model="black-forest-labs/FLUX.2-klein-4B",
        prompt="A botanical scientific illustration",
        size="768x512",
        n=1,
        response_format="b64_json",
        extra_body={"seed": 42},
    )
```

Use the exact image model ID when the server is explicitly configured for a
checkpoint. Image entries join `/v1/models` when the configured precision
passes qualification. Requests accept one PNG, dimensions divisible by 16
from 64 to 1536 pixels per side, and up to 1,048,576 pixels in total.
The default is 512×512. Prompt text is limited to 8,192 characters and the
checkpoint tokenizer truncates at 512 tokens. The local `steps` extension
accepts up to 20 steps. Editing and streaming are outside the current
text-to-image API.

The [DeepGradient smoke script](../../examples/native_tasks/deepgradient_image.py)
writes a 768×512 editorial PNG to a caller-selected path. The
[SDK example](../../examples/native_tasks/image_sdk.py) writes the same PNG
format from an HTTP response.
