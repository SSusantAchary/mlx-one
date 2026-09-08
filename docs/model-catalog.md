# Text Model Candidate Catalog

This catalog is derived from the maintainer's `model_list.txt`. A row is a
discovery candidate, not a support claim. Only exact revisions with linked run
evidence may be promoted in the capability registry.

## Active qualification targets

| Model | Revision | Architecture | Operations | Status |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | Qwen2 | LoRA train, export reload | Hardware verified |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | Llama | LoRA train, export reload | Hardware verified |

Their quality-evaluation entries remain candidates. The Qwen tiny held-out suite
scored `0.0 → 0.0`; SmolLM2 scored `1.0 → 0.0`. Both gates failed, so neither
model has a quality-verified claim. See the checksummed [Qwen result](../qualification/results/qwen2.5-coder-1.5b-lora-m4-air-32gb.json)
and [SmolLM2 result](../qualification/results/smollm2-1.7b-lora-m4-air-32gb.json).

## Subsequent text candidates

| Wave | Families from `model_list.txt` | Intended qualification |
| --- | --- | --- |
| A | Qwen2.5 0.5B/1.5B/3B, Qwen2.5-Coder 0.5B/3B, Qwen3 0.6B/1.7B | SFT, LoRA, QLoRA, templates, parity |
| A | Llama 3.2 1B/3B, Gemma 3 270M/1B, SmolLM2 135M/360M | Second-family and small-model regression |
| B | SmolLM3 3B, LFM2/LFM2.5 small variants, Phi-2/Phi-3-mini | Architecture-specific loading and training |
| B | DeepSeek-Coder 1.3B, R1-Distill-Qwen 1.5B, StarCoder2-3B, CodeGemma 2B | Coding and reasoning evaluation |
| C | OPT, MobileLLM, OpenELM, Pythia, GPT-2, GLM-Edge small variants | Research and compatibility regression |

Before onboarding any candidate, resolve its canonical publisher, immutable
revision, complete parameter count, license, tokenizer or processor, upstream
backend support, and safe hardware workload. The VLM, embedding, and audio rows
in `model_list.txt` remain deferred until the text qualification gate passes.
