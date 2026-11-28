# Image generation source notices

MLX-One uses Apache-2.0. The checkpoint licenses and third-party source licenses
apply separately.

- The implementation of interleaved rotary positions, transformer blocks and VAE
  decoding in `math.py`, `flux2.py`, `zimage.py`, and `vae.py` follows MFLUX,
  copyright 2026 Filip Strand under MIT. The original license is included here.
  [Pinned MFLUX source](https://github.com/mflux-community/mflux/tree/86245d339a8d957c57da5d97b2914f0aefcbc048)
- The same model files and `image/schedule.py` follow the pipeline contracts
  and checkpoint parameter layouts in Hugging Face Diffusers, Apache-2.0.
  Copyright 2025 Black Forest Labs, the HuggingFace Team, the InstantX Team
  and the Alibaba Z-Image Team. Copyright 2026 the HuggingFace Team.
  The repository's included LICENSE contains Apache-2.0.
  [Pinned Diffusers source](https://github.com/huggingface/diffusers/tree/509b7658e8d99a811050f26ce4ed0a40b65ccfee)
- Z-Image's default terminal node and eight evaluations follow the publisher's
  Apache-2.0 scheduler and the nine-node Diffusers release behavior.
  [Tongyi-MAI source](https://github.com/Tongyi-MAI/Z-Image/tree/26f23eda626ffadda020b04ff79488e1d72004cd)
  [Diffusers v0.36.0](https://github.com/huggingface/diffusers/tree/9380e58821092ad05e618fa8a259030136ba43c0)
- `conditioning.py` uses MLX-One's existing Qwen3 primitives, whose MIT
  attribution is kept alongside those sources.
  [Qwen3 notice](../language/qwen3/UPSTREAM_LICENSE)

The Apache-2.0 FLUX.2 Klein 4B and Z-Image Turbo checkpoint identities are pinned
in the image registry. This distribution contains no checkpoint weights.

MLX-One can consume the pinned MFLUX-formatted Q4 conversions by MFlux Community.
The package records their artifact identities and embeds the original model
configuration. MFLUX is a validation reference and checkpoint publisher.
[FLUX.2 Klein 4B Q4](https://huggingface.co/mflux-community/flux2-klein-4b-mflux-q4)
[Z-Image Turbo Q4](https://huggingface.co/mflux-community/z-image-turbo-mflux-q4)
