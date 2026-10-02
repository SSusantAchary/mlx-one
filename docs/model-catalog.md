# Text Model Candidate Catalog

This catalog is derived from the maintainer's `model_list.txt`. A row is a
discovery candidate, not a support claim. Only exact revisions with linked run
evidence may be promoted in the capability registry.

For current native workflows, checkpoint assets, and family-specific examples,
see the [model guide index](../README.md#model-specific-documentation). This catalog
tracks qualification candidates; it is not the inventory of implemented architectures.

## Active qualification targets

| Model | Revision | Architecture | Operations | Status |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | Qwen2 | LoRA train, export reload | Hardware verified |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | Llama | LoRA train, export reload | Hardware verified |
| `openbmb/MiniCPM5-1B-MLX` | `9879b18bf2928355fcdf4287635388a3665a40cb` | Llama | 4-bit load, chat, reasoning, generation | Pinned integration gate |
| `openbmb/MiniCPM5-2B-MLX` | `8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3` | Llama | 4-bit load, chat, reasoning, generation | Pinned integration gate |

Their quality-evaluation entries remain candidates. The Qwen tiny held-out suite
scored `0.0 → 0.0`; SmolLM2 scored `1.0 → 0.0`. Both gates failed, so neither
model has a quality-verified claim. See the checksummed [Qwen result](../qualification/results/qwen2.5-coder-1.5b-lora-m4-air-32gb.json)
and [SmolLM2 result](../qualification/results/smollm2-1.7b-lora-m4-air-32gb.json).

## Subsequent text candidates

| Wave | Families from `model_list.txt` | Intended qualification |
| --- | --- | --- |
| A | Qwen2.5 0.5B/1.5B/3B, Qwen2.5-Coder 0.5B/3B, Qwen3 0.6B/1.7B | SFT, LoRA, QLoRA, templates, parity |
| A | Llama 3.2 1B/3B, Gemma 3 270M/1B, SmolLM2 135M/360M | Additional Llama-family and small-model regression |
| B | SmolLM3 3B, LFM2/LFM2.5 small variants, Phi-2/Phi-3-mini | Architecture-specific loading and training |
| B | DeepSeek-Coder 1.3B, R1-Distill-Qwen 1.5B, StarCoder2-3B, CodeGemma 2B | Coding and reasoning evaluation |
| C | OPT, MobileLLM, OpenELM, Pythia, GPT-2, GLM-Edge small variants | Research and compatibility regression |

Before onboarding any candidate, resolve its canonical publisher, immutable
revision, complete parameter count, license, tokenizer or processor, upstream
backend support, and safe hardware workload. The VLM, embedding, and audio rows
in `model_list.txt` track separate qualification targets. Existing native loaders
and synthetic architecture tests do not complete those task/quality gates; their
current workflow boundaries are documented in the model guides linked above.

## Segmentation qualification candidate

| Model | Revision | Operations | Status |
| --- | --- | --- | --- |
| `facebook/sam3` | `3c879f39826c281e95690f02c7821c4de09afae7` | Concept/interactive segmentation, automatic-mask candidate, offline/streaming tracking | Partial M4/32 GB fixture evidence; full qualification pending |

The two 32-frame single-object comparisons pass the pinned CPU-reference gates;
they do not qualify multi-object tracking, occlusion, corrections, or all image
prompts. See the [SAM3 guide](../src/mlx_one/models/segmentation/sam3/README.md) and
[measured results](sam3-qualification-m4-32gb.md). Checkpoint license/access terms
are separate from the repository's Apache-2.0 license.
