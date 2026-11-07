Native Depth Anything and DINOv2 adapt Hugging Face Transformers v4.57.1,
commit 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc, modeling_depth_anything.py,
modeling_dinov2.py and the checkpoint-selected DPT image processor. Copyright
2024 HuggingFace Inc. team (Depth Anything); 2023 Meta Platforms, Inc. and
affiliates and the HuggingFace Inc. team (DINOv2). Apache License 2.0; see
models/UPSTREAM_LICENSE. Modifications: MLX-owned modules/interpolation,
existing native attention/MLP reuse, exact checkpoint mapping and bounded
host image preparation. Source frameworks are isolated references only.
The selected V2 Small checkpoint declares Apache-2.0 separately; weights are
not packaged. This candidate predicts relative depth, without metric units.
Intermediate numerical gates remain unresolved; no public depth task API
or release qualification is implied by the raw model helper.
