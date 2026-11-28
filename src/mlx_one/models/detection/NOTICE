Native DETR adapts modeling_detr.py at Transformers commit
8cb5963cc22174954e7dca2c0a3320b7dc2f4edc, copyright 2021 Facebook AI
Research and the HuggingFace Inc. team, Apache License 2.0.
The ResNet50 bottleneck/stride layout was audited against timm 1.0.22
models/resnet.py, copyright Ross Wightman, Apache License 2.0.
See models/UPSTREAM_LICENSE. Modifications: native MLX modules and checkpoint
mapping, existing native attention projection reuse, bounded image processing
and development-only layer tracing. DETR remains blocked by numerical gates;
no detector task API or release qualification is implied by its native loader.
Model weights are not packaged; COCO images/annotations remain local, with
per-photo rights review pending. No SAM3, OCR or MLXBoost code is changed.
