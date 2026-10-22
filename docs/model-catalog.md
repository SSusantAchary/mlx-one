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
model has a quality-verified claim. See the checksummed [Qwen result](qualification/archived-evidence.md)
and [SmolLM2 result](qualification/archived-evidence.md).

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

## Requested native audio qualification matrix

The audio plan contains twelve checkpoints, with original VibeVoice 1.5B as
multi-speaker TTS rather than ASR. The first three have scoped English short-form
evidence, and Qwen ASR 0.6B adds passing English/French offline evidence. Qwen
ASR/alignment have native implementations, but their remaining checkpoint,
language and streaming gates are incomplete. Parakeet v2/v3 now have native
offline implementations; full qualification is WIP. Nemotron, Voxtral and
VibeVoice inference remain unimplemented. This table does not promote an
unqualified candidate into a general quality/streaming support claim.

| Stage | Checkpoint | Revision / status |
| --- | --- | --- |
| 1 | `distil-whisper/distil-large-v3` | `8031d2e6ce6631b7fc45629dddfc00271116d981`; recorded M4 English short-form gates pass |
| 2 | `moonshine-ai/moonshine-tiny` | `390624ed33d594443aa4aa221f5b9f283b545b5a`; recorded M4 English short-form gates pass |
| 2 | `moonshine-ai/moonshine-base` | `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf`; recorded M4 English short-form gates pass |
| 3 | `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; native ASR/streaming; English/French offline gates pass, streaming diagnostic fails; other gates pending |
| 3 | `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5`; native ASR/streaming; English 100-clip WER/CER match reference, but exact anchors fail (15/16); other languages/streaming pending |
| 3 | `Qwen/Qwen3-ForcedAligner-0.6B` | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`; native alignment, never ASR; English timestamp gate fails (800 ms versus 80-ms limit) |
| 4 | `nvidia/parakeet-tdt-0.6b-v2` | `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; native offline + isolated conversion implemented; English 100-clip/16-anchor converted-reference gates pass; **WIP** original NeMo/timestamp/full qualification |
| 4 | `nvidia/parakeet-tdt-0.6b-v3` | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; native offline implemented; English 100/16 anchors and French/German 25-clip gates pass; **WIP** timestamp/full qualification |
| 4 | `nvidia/nemotron-3.5-asr-streaming-0.6b` | `ea30d66debe3740a08b573244286791d423d6b3e`; canonical tensor/asset contract frozen; **WIP / not implemented** native cached streaming |
| 5 | `mistralai/Voxtral-Mini-3B-2507` | `3060fe34b35ba5d44202ce9ff3c097642914f8f3`; indexed tensor/Tekken/processor contract frozen; **WIP / not implemented** native transcription |
| 5 | `mistralai/Voxtral-Mini-4B-Realtime-2602` | `2769294da9567371363522aac9bbcfdd19447add`; canonical tensor/Tekken/processor contract frozen; **WIP / not implemented** native cached streaming; exception to initial text 3B tier |
| 6 | `microsoft/VibeVoice-1.5B` | `c00898d257e6b46004e3e2866a47534085fb685a`; indexed tensor contract and original TTS source/tokenizer pinned; **WIP / not implemented** native synthesis, parity and publisher-fixture permission/intended-use evidence |

Target revision pins alone identify qualification candidates, not runnable
loaders. All six Stage 4–6 targets now have checksummed tensor/asset contracts;
only Parakeet has a native inference implementation among Stage 4 targets.
As of 2026-10-04, failed and missing
qualification gates remain **WIP** but no longer block later implementation.
Development proceeds through Stage 4, Stage 5 and Stage 6; a checkpoint is not
qualified until its applicable unchanged gates pass. The
[failed numerical candidate](audio-qualification-m4-32gb.md#stage-three-closure-investigation--rejected-numerical-candidate)
is explicit benchmark-only tooling and does not replace production defaults.

See [audio evidence](audio-qualification-m4-32gb.md),
[Whisper/Distil guide](../src/mlx_one/models/audio/whisper/README.md) and
[Moonshine guide](../src/mlx_one/models/audio/moonshine/README.md), and
[Qwen ASR/alignment guide](../src/mlx_one/models/audio/qwen3_asr/README.md) and
[Parakeet guide](../src/mlx_one/models/audio/parakeet/README.md). The maintainer's
`model_list.txt` is locally ignored by Git; this tracked table retains the audio
matrix for repository readers without changing historical text results or roadmap.
