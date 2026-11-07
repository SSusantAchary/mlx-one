# Models and native architecture coverage

This index combines the former README family guides and native component tables.
It describes architecture and available workflow boundaries, not blanket checkpoint
qualification. Model-specific guides remain beside their implementations.

## Checkpoint compatibility and admission

- [Authoritative text checkpoint inventory](../benchmarks/inventory.md) is generated
  from the existing JSON inventory and support manifest; exact revisions, primary
  precision, functional status and benchmark gaps are not copied into this index.
- [Functional admission policy](../model-admission.md) defines checkpoint gates;
  [candidate catalog](../model-catalog.md) tracks the separate qualification backlog.
- [VLM methods and coverage](../benchmarks/vlm/README.md),
  [audio coverage](../benchmarks/audio/coverage.md) and
  [OCR methods](../benchmarks/ocr/README.md) retain modality-specific compatibility.
- [Experimental tasks](../experimental/native-tasks.md) records shared task admission,
  blocked candidates and preserved historical failures. [Repository summary](../REPOSITORY_SUMMARY.md)
  describes package ownership and dependency boundaries.

## Reading the evidence

| Description | Meaning |
| --- | --- |
| Architecture implemented | Native config/model/cache/registry/weight contract and synthetic execution exist; no blanket checkpoint claim |
| Checkpoint loadable | A particular revision and precision loaded; useful generation, parity and task quality require separate evidence |
| Functional smoke passed | The exact checkpoint executed the recorded native workflow within disclosed limits |
| Reference parity passed | Only the named operation, trace, source revision and tolerance passed; not automatic end-to-end qualification |
| Benchmark measured | The recorded workload has timing and hardware evidence, independent of functional admission |
| Fully qualified | All applicable task/checkpoint/release gates passed; no such promotion is made by this index |

The policy names `ARCHITECTURE_IMPLEMENTED`, `MODEL_SMOKE_TESTED`,
`EXPERIMENTAL_SUPPORTED` and `USER_VALIDATED` have the definitions in the admission
policy. Legacy “Integration-tested”, “Hardware-verified” and “Quality-verified”
claims remain scoped to their task contract, checksummed hardware workload and
pinned quality protocol respectively. Candidates still need integration evidence.
These descriptions are not interchangeable. Each claim needs model revision,
operation, precision, workload, software and hardware scope.

Family summaries below retain earlier evidence boundaries. Dated reports may
contain later bounded measurements, especially audio; those measurements alone do
not resolve broader quality gates. See [known limitations](../experimental/known-limitations.md).

## Text generation guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| GPT-2 124M / 355M / 774M / 1.5B (`gpt2`) | Learned positions, fused QKV, byte BPE, safe loading, cache, generation and streaming | Completion, sampling, streaming | 124M bounded functional smoke; larger checkpoints remain candidates | [Docs](../../src/mlx_one/models/language/gpt2/README.md) |
| Llama / MiniCPM5 / SmolLM2 (`llama`) | GQA, explicit head dimensions, RoPE, 4-bit MLX loading, multi-EOS generation | Completion, template chat, quantized loading | MiniCPM5 pinned gate; SmolLM2 M4 LoRA/reload evidence, quality gate failed | [Docs](../../src/mlx_one/models/language/llama/README.md) |
| Qwen2 / Qwen2.5 / Qwen2.5-Coder (`qwen2`) | Dense Transformer, GQA, RoPE, cache | Completion, template chat, streaming | Coder M4 LoRA/reload evidence, quality gate failed; cache qualification partial | [Docs](../../src/mlx_one/models/language/qwen2/README.md) |
| Qwen3 dense (`qwen3`) | Bias-free attention, Q/K norm, explicit head dimensions | Completion, template chat, streaming | 0.6B bounded functional smoke; other checkpoints remain candidates | [Docs](../../src/mlx_one/models/language/qwen3/README.md) |
| Qwen2-MoE (`qwen2_moe`) | Top-k experts, shared expert, router outputs | Experimental native text loading/generation | Architecture/synthetic; checkpoint unqualified | [Docs](../../src/mlx_one/models/language/qwen2_moe/README.md) |
| OpenELM 270M–3B (`openelm`) | Layer-wise heads/FFN widths, fused QKV, GQA | Completion with compatible tokenizer assets | Candidate; verify exact release | [Docs](../../src/mlx_one/models/language/openelm/README.md) |
| LFM2 / LFM2.5 (`lfm2`) | Hybrid convolution/attention decoder | Hybrid text generation and chat | Per checkpoint; M4 block-rejection safety result | [Docs](../../src/mlx_one/models/language/lfm2/README.md) |
| LFM2-MoE (`lfm2_moe`) | Hybrid decoder and sparse expert routing | Experimental hybrid text loading/generation | Architecture/synthetic tests; LFM2.5-24B-A2B full-checkpoint smoke pending | [Docs](../../src/mlx_one/models/language/lfm2_moe/README.md) |
| Gemma3 text (`gemma3_text`) | Native text decoder and sliding attention | Completion/chat, bounded sliding state | 1B bounded functional smoke; numerical/answer-quality limitations disclosed | [Docs](../../src/mlx_one/models/language/gemma3/README.md) |
| Phi3 (`phi3`) | Native decoder and sliding attention | Completion/chat, bounded sliding state | Mini-4K bounded functional smoke; wider context/precision not qualified | [Status](../model-admission.md) |
| GPT-OSS (`gpt_oss`) | Native MoE, MXFP4 and packed TQ3 projections | Experimental native MXFP4 and TQ3 inference | Architecture/kernel tests; both 20B full-checkpoint runs pending | [Docs](../native-moe/manual-inference.md) |

## Vision-language guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Qwen2-VL (`qwen2_vl`) | Vision Transformer, 3D patches, merger, multimodal RoPE, image/video token insertion | Native Python, image CLI, HTTP chat and SSE | Pinned Q4 checkpoint smoke passed on M4; experimental; static images only | [Docs](../../src/mlx_one/models/vision_language/qwen2_vl/README.md) |
| Qwen2.5-VL (`qwen2_5_vl`) | Window/full vision attention, biased SwiGLU vision tower, merger, multimodal RoPE | Native Python, image CLI, HTTP chat and SSE | Pinned Q4 checkpoint smoke passed on M4; experimental; static images only | [Docs](../../src/mlx_one/models/vision_language/qwen2_5_vl/README.md) |
| Qwen3.5 0.8B / 2B (`qwen3_5`) | Hybrid gated-delta/full-attention text tower, learned/interpolated vision positions, multimodal RoPE | Native text and static-image generation, CLI, HTTP/SSE | Targeted 0.8B MLX Q4 image checkpoint smoke passed; alternative revisions unqualified | [Docs](../../src/mlx_one/models/vision_language/qwen3_5/README.md) |
| LFM2-VL / LFM2.5-VL (`lfm2_vl`) | SigLIP2 NaFlex, pixel unshuffle/projector, hybrid language model | Native SigLIP2 NaFlex + hybrid decoder, Python/CLI/HTTP/SSE | Pinned Q8 450M and Q4 3B checkpoint smokes passed on M4; experimental | [Docs](../../src/mlx_one/models/vision_language/lfm2_vl/README.md) |

## Retrieval guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| BERT / all-MiniLM-L6-v2 (`bert`) | BERT encoder, mean pooling, normalization, cosine | Single-vector embeddings and similarity | Candidate; synthetic validation | [Docs](../../src/mlx_one/models/embeddings/bert/README.md) |
| MPNet / all-mpnet-base-v2 (`mpnet`) | MPNet relative positions, mean pooling, normalization, cosine | Single-vector embeddings and similarity | Candidate; synthetic validation | [Docs](../../src/mlx_one/models/embeddings/mpnet/README.md) |
| Qwen3 Embedding (`qwen3_embedding`) | Final-token pooling, instructed queries, Matryoshka dimensions, normalization, cosine | Instructed queries, dimensions, normalized vectors | 0.6B candidate; synthetic validation | [Docs](../../src/mlx_one/models/embeddings/qwen3_embedding/README.md) |
| Qwen3 Reranker (`qwen3_reranker`) | Official pair prompt, yes/no logit scoring, probabilities, stable ranking | Pair scores, probabilities, stable ranking | 0.6B candidate; synthetic validation | [Docs](../../src/mlx_one/models/embeddings/qwen3_reranker/README.md) |
| LFM2 ColBERT (`lfm2_colbert`) | Token embeddings, masks, late-interaction MaxSim | Lower-level token embeddings and masked MaxSim | Candidate; generic single-vector `embed` incompatible | [Docs](../../src/mlx_one/models/embeddings/lfm2_colbert/README.md) |

## Audio guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Whisper / Distil-Whisper (`whisper`) | Encoder-decoder, safe loading, log-Mel, BPE, decoding, language, timestamps, WER/CER | Transcription, language detection, timestamps, WER/CER | Tiny/turbo smoke gates; Distil-large-v3 M4 English short-form gates pass; broader qualification pending | [Docs](../../src/mlx_one/models/audio/whisper/README.md) |
| Moonshine Tiny / Base (`moonshine`) | Raw-waveform convolutions, GroupNorm, partial interleaved RoPE, encoder/cached decoder, tokenizer, greedy transcription | Native offline English transcription, Python/CLI | Both pinned checkpoints pass M4 English short-form gates; streaming/long-form unqualified | [Docs](../../src/mlx_one/models/audio/moonshine/README.md) |
| Qwen3-ASR 0.6B / 1.7B (`qwen3_asr`) | Chunked Conv2D/audio encoder, Qwen3 decoder, strict BF16 loading, context/language prompts, accumulated-utterance streaming | Native transcription and bounded accumulated-utterance streaming, Python/CLI | 0.6B English/French offline gates pass; **WIP**: 1.7B English WER/CER match reference but only 15/16 exact anchors pass; streaming parity and remaining language gates unresolved | [Docs](../../src/mlx_one/models/audio/qwen3_asr/README.md) |
| Qwen3-ForcedAligner 0.6B (`qwen3_forced_aligner`) | Separate 5,000-class timestamp head, known-text units and boundary reconstruction | Native known-text alignment, Python/CLI; not ASR | Float32 components pass; **WIP**: English timestamp differences reach 800 ms against an 80-ms limit; remaining qualification unresolved | [Docs](../../src/mlx_one/models/audio/qwen3_asr/README.md) |
| Parakeet TDT v2 / v3 (`parakeet_tdt`) | Float32 FastConformer, LSTM predictor, joint token/duration heads, greedy offline transcription, strict safetensors loading | Native offline transcription, Python/CLI; isolated v2 archive conversion | Float32 components, both English 100-clip/16-anchor gates and v3 French/German 25-clip gates pass; **WIP** original v2 NeMo, timestamp and broader qualification | [Docs](../../src/mlx_one/models/audio/parakeet/README.md) |
| LFM2.5 Audio (`lfm2_audio`) | Native checkpoint loader, Conformer/LFM2, Depthformer and waveform codec | Native ASR, TTS, speech-to-speech, Python/CLI/HTTP and ordered PCM streaming | Source/tiny native tests; Q4 checkpoint and speech quality validation pending | [Docs](../../src/mlx_one/models/audio/lfm2_audio/README.md) |

## Segmentation guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Meta SAM3 (`sam3`, `sam3_tracker`, `sam3_tracker_video`, `sam3_video`) | `sam3`: ViT, feature pyramid, CLIP text/geometry encoders, DETR, presence scoring, masks; `sam3_tracker`: Point/box/mask prompts, two-way decoder, multimask prediction, refinement; `sam3_tracker_video`: Memory encoding/attention, object pointers, temporal state; `sam3_video`: Shared image backbone, detector/tracker association, object lifecycle, offline/streaming sessions | Concept/interactive images, experimental automatic masks, composite offline/streaming tracking | M4/32 GB single-object 32-frame fixture passes CPU-reference gates; full qualification pending; standalone temporal tracker architecture-only | [Docs](../../src/mlx_one/models/segmentation/sam3/README.md) |

## OCR and documents

Native source paths cover TrOCR, GOT-OCR 2.0, Florence-2, PaddleOCR-VL 1.5,
LightOnOCR 2 and DeepSeek-OCR. [OCR implementation and APIs](../OCR_TESTING.md)
describes line crops, page transcription, region workflows and limitations.
[OCR performance](../benchmarks/ocr/performance.md) retains the separate page
and handwritten-line protocols, failures and truncations; experimental
measurements are not full task qualification.

## Additional experimental tasks

The current tree also implements bounded DistilBERT sentiment, BERT NER, CLIP
zero-shot image classification and Silero VAD profiles. They use the shared
registry/Python/CLI/HTTP paths within the exact admitted scopes recorded in the
[native-task evidence](../experimental/native-tasks.md) and
[API contracts](../native_tasks/api-contracts.md). ViT, SigLIP2, DETR, document/video
QA and depth candidates have separate blocked/helper status. Architecture source
alone does not expose a public task API. RT-DETR inference is not implemented.

## Historical component-table notes

The previous component table recorded GPT-2 124M bounded smoke (larger sizes are
candidates), Gemma3 1B and Phi-3 Mini-4K bounded smoke, and LFM2.5-24B full-checkpoint
smoke pending. These remain scoped by the generated checkpoint inventory.
Qwen3.5’s targeted image-Q4 smoke and source-text smoke do not qualify its primary
converted text-Q4 profile. GPT-OSS’s older local HTTP run cannot identify today’s
pinned inventory targets; see [HTTP provenance](../benchmarks/http-performance.md).

SAM3's standalone `sam3_tracker_video` is architecture-only and participates in
the composite video workflow; fixture-qualified composites do not independently
qualify every prompt, refinement or automatic mask mode. SAM3 uses the dedicated
segmentation API and CLI, rather than the text `Engine`.
