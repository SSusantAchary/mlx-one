# Native OCR source provenance

The model-specific MLX layers in this directory are local source ports, not wrappers around
an installed MLX ecosystem package. No `mlx-vlm`, `mlx-lm`, Transformers, PyTorch,
PaddlePaddle or ONNX Runtime is imported by the OCR runtime.

Native reference: [mlx-vlm revision e9af376626b04ba0cdad23b1476fd7a4a583d721](https://github.com/Blaizzy/mlx-vlm/tree/e9af376626b04ba0cdad23b1476fd7a4a583d721/mlx_vlm/models).
Copyright © 2025 Prince Canuma; MIT license reproduced in `UPSTREAM_LICENSE`.
`provenance.json` identifies each initial reference file and its SHA-256.
The LightOn projector also derives from that revision's `mistral3/mistral3.py`;
its vision layers derive from `pixtral/vision.py`.

TrOCR was implemented against [Hugging Face TrOCR](https://github.com/huggingface/transformers/tree/main/src/transformers/models/trocr)
and ViT mathematics, Copyright The Hugging Face Inc. team, Apache-2.0
(the repository's root LICENSE reproduces Apache-2.0). Its tensor layout was checked against
the immutable Microsoft checkpoint headers recorded under `docs/benchmarks/ocr/metadata`.
DeepSeek uses the pinned original checkpoint’s plain OCR prompt, QuickGELU CLIP feed-forward
activation and encoder normalization constants. LightOn uses its OCR-specific image-token
expansion without generic Pixtral row separators.

GOT tokenizer conventions follow the checkpoint's Alibaba/StepFun tokenization source;
the vocabulary remains a separately downloaded model asset.

Local changes replace external shared helpers, processors and caches; remove automatic
Transformers processor registration; integrate existing Qwen2/Qwen3 decoders and selected
expert dispatch; verify tensor mappings; support tiny configurable fixtures; materialize
Pixtral rotary constants before thread handoff; correct positional interpolation layout and antialiased resizing;
retain image encoding for blank inputs; and enforce task-specific output parsing.
The source commits of referenced model repositories are in the OCR inventory.

The checkpoint licenses are separate from this code license. Review each pinned model card,
especially where model-card license metadata is absent. No model weights are redistributed.
