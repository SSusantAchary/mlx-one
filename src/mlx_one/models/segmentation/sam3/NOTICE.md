# SAM3 native port notices

The architecture and configuration ports in this directory are derived from
Hugging Face Transformers, revision
`6133195dcb027feb7502379acd13750e167eb6c5`:

- `src/transformers/models/sam3/`
- `src/transformers/models/sam3_tracker/`
- `src/transformers/models/sam3_tracker_video/`
- `src/transformers/models/sam3_video/`

Copyright 2025 Meta AI and The HuggingFace Team. All rights reserved.
Upstream source is licensed under the Apache License, Version 2.0, as is
mlx-one; see the repository [LICENSE](../../../../../LICENSE).
[Pinned reference source](https://github.com/huggingface/transformers/tree/6133195dcb027feb7502379acd13750e167eb6c5/src/transformers/models/sam3).

Changes include native MLX kernels and tensor semantics, backend-free configuration,
strict composite safetensors remapping, and host-owned video histories. Training,
framework integration decorators, and PyTorch runtime dependencies are omitted.
The API, processor, session management, and CLI are mlx-one implementations.

The independent uint8 image-resize implementation was checked against the pinned
reference processor and PyTorch's fixed-coefficient antialiased bilinear algorithm
at `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`. It rounds each separable pass before
normalization, matching torchvision v2 pixels. The older float-only approximation
is retained in historical failed evidence; production does not import PyTorch.

The `facebook/sam3` **checkpoint is subject to its own SAM license and gated
access terms**, not mlx-one's Apache-2.0 license. Acceptance and authorized access
are required. Review the [publisher checkpoint card and license](https://huggingface.co/facebook/sam3)
before downloading, using, or redistributing weights. No checkpoint weights are
distributed with this port. SAM3.1 is outside this implementation's scope.
