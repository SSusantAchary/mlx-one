# IMG-01 qualification record

Starting commit `d99d26a8f3a6aa3e13b7babe87e15acfa8e62f21`. The checkout
contained unrelated changes before IMG-01. The 53 targeted CLI, server,
registry and Qwen architecture tests passed before image changes. The normal
tool sandbox withheld Metal. The host exposed Apple M4 and 32 GiB unified
memory (Mac16,12) in the Metal test session, running macOS 15.8.3, Python 3.12.11 and
MLX 0.32.2.

| Gate | FLUX.2 Klein 4B | Z-Image Turbo |
| --- | --- | --- |
| G0 pinned config, schedule and strict tensor contract | Passed | Passed |
| G1 BF16 parity | Partial. Tiny denoiser and official VAE fixtures passed. Full trace pending. | Partial. Tiny denoiser fixture passed. Full trace and official VAE pending. |
| G2 affine precision and fixed-prompt image quality | Small native BF16/Q4/Q8 fixtures passed. Real checkpoint Q4/Q8 pending. | Small native BF16/Q4/Q8 fixtures passed. Real checkpoint Q4/Q8 pending. |
| G3 API, CLI, HTTP and lifecycle | Small Metal fixture and SDK checks passed. Exact checkpoint and GPT-OSS handoff pending. | Small Metal fixture and SDK checks passed. Exact checkpoint and GPT-OSS handoff pending. |
| G4 M4 benchmarks and editorial PNG | Pending local checkpoint. | Pending local checkpoint. |

Checkpoint transfers were stopped at the user's request for manual downloads.
No full-checkpoint generation latency, memory result or editorial PNG is
recorded. No 8–24 GB device is qualified. Both image entries remain out of the
HTTP qualified model catalog.

## Reproduction

```bash
MLX_ONE_RUN_MLX_TESTS=1 MLX_ONE_RUN_IMAGE_SDK_TESTS=1 \
  python -m pytest --no-cov -q tests/test_image_generation.py
python benchmarks/native_image.py MODEL_ID CHECKPOINT_DIRECTORY NEW_RESULTS_DIRECTORY
python examples/native_tasks/deepgradient_image.py output.png --checkpoint CHECKPOINT_DIRECTORY
```

Both source checkpoint revisions and their config and tensor-header digests
are recorded in the package. [Pinned model metadata](../../models/image-generation.md)
[Artifact digests](../../../src/mlx_one/data/image_checkpoints.json)

The Z-Image publisher's schedule starts at sigma one and includes terminal
zero in the default nine nodes. MLX-One performs the eight nonzero Euler
updates with guidance disabled. The current Diffusers development revision
has changed its default sigma construction. Its newer schedule and the
MFLUX default Z-Image schedule are unsuitable as comparisons for this
specific nine-node pipeline. Use identical injected timesteps for component
comparisons. [Publisher pipeline](https://github.com/Tongyi-MAI/Z-Image/blob/26f23eda626ffadda020b04ff79488e1d72004cd/src/zimage/pipeline.py)
[Diffusers v0.36.0](https://github.com/huggingface/diffusers/blob/9380e58821092ad05e618fa8a259030136ba43c0/src/diffusers/pipelines/z_image/pipeline_z_image.py)

The OpenAI SDK is needed for the optional SDK test. The fixture checks used
OpenAI SDK 2.54.0. Python, CLI and SDK output PNGs matched byte for byte
for each tiny Q4 pipeline. The 74 targeted tests and isolated wheel install
passed. The wheel included the pinned artifact catalog and source notices.
Fixed-input reference metrics are recorded in the machine-readable results.
[Results](results.json)

Source pins and file-level attribution are retained with the implementation.
Apple MLX was used at the existing package pin. No serving code was copied
from Rapid-MLX, mlx-openai-server or Mfluxible.
[Native image notices](../../../src/mlx_one/models/image_generation/NOTICE.md)

## Published Q4 checkpoint loader

Both published Q4 revisions passed header-to-module checks for every component.
For each family, seven prompts produced the same token IDs and padding masks
as the corresponding official tokenizer assets. Native Metal tests exercised
packed fixture weights through Python, CLI and HTTP and rejected a modified
shard. The image and server regression run passed 78 tests. Full published
weights were left for manual download, and generation with those exact weights
remains pending.
[Published Q4 identities](../../../src/mlx_one/data/image_published_q4.json)
