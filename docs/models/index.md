# Models and native architecture coverage

**45 native architecture types across seven modalities.** The current source
includes ten additional verified checkpoint profiles for T5, ViT, Kokoro, BGE-M3,
Nomic and multilingual BERT/XLM-R retrieval. See the
[model workflows](batch-1-sub-6b/README.md) for Python, CLI and HTTP examples,
and the [validation report](batch-1-sub-6b/REPORT.md) for recorded checks and limits.

The [support inventory](../../status_models_testing.md) keeps the additional
source profiles distinct from the dated 40-type **0.1.1** release approval.
Version **0.1.2** is approved for publication with these workflows.
[Current release and publication record](../releases/0.1.2.md).
[0.1.1 release record](../releases/0.1.1.md).

## Checkpoint compatibility and admission

Support requires model/state tests, native real-checkpoint execution and working
relevant interfaces, with regression and distribution checks. Accuracy and
performance measurements keep their own workloads and limits. There is no
blanket experimental label for the supported primary workflows.

- [Supported checkpoints and known limits](../../status_models_testing.md).
- [Functional admission policy](../model-admission.md) and
  [machine-readable catalog](../../src/mlx_one/data/model_support_release_20261108.json).
- [Measured performance](../benchmarks/README.md) preserves original protocols,
  hardware and quality diagnostics.
- [Historical text inventory](../benchmarks/inventory.md),
  [candidate catalog](../model-catalog.md) and
  [optional experiments](../experimental/README.md) retain their distinct scopes.

## Reading the evidence

**Supported** identifies a passing selected real-checkpoint workflow with model,
interface and regression checks. Accuracy and performance measurements retain
their own datasets, input bounds and hardware. Historical labels describe their
original profiles and remain separate from the current support inventory.

A supported model type does not admit every checkpoint size, precision, context,
optional task or hardware configuration. Model-family guides explain native
components; the linked current inventory identifies the tested primary operations.

## Text generation guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| GPT-2 124M / 355M / 774M / 1.5B (`gpt2`) | Learned positions, fused QKV, byte BPE, safe loading, cache, generation and streaming | Completion, sampling, streaming | Supported: `openai-community/gpt2`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/gpt2/README.md) |
| T5 (`t5`) | Encoder-decoder, relative attention, tied embeddings, strict loading | Bounded greedy text-to-text generation and teacher forcing | Supported: `google-t5/t5-small`, FP32, 512 encoder tokens; source checkout | [Docs](../../src/mlx_one/models/language/t5/README.md) |
| Llama / MiniCPM5 / SmolLM2 (`llama`) | GQA, explicit head dimensions, RoPE, 4-bit MLX loading, multi-EOS generation | Completion, template chat, quantized loading | Supported: `HuggingFaceTB/SmolLM2-1.7B-Instruct`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/llama/README.md) |
| Qwen2 / Qwen2.5 / Qwen2.5-Coder (`qwen2`) | Dense Transformer, GQA, RoPE, cache | Completion, template chat, streaming | Supported: `Qwen/Qwen2.5-0.5B-Instruct`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/qwen2/README.md) |
| Qwen3 dense (`qwen3`) | Bias-free attention, Q/K norm, explicit head dimensions | Completion, template chat, streaming | Supported: `Qwen/Qwen3-0.6B`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/qwen3/README.md) |
| Qwen2-MoE (`qwen2_moe`) | Top-k experts, shared expert, router outputs | Native quantized text generation | Supported: `mlx-community/Qwen1.5-MoE-A2.7B-4bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/qwen2_moe/README.md) |
| OpenELM 270M–3B (`openelm`) | Layer-wise heads/FFN widths, fused QKV, GQA | Completion with compatible tokenizer assets | Supported: `apple/OpenELM-270M`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/openelm/README.md) |
| LFM2 / LFM2.5 (`lfm2`) | Hybrid convolution/attention decoder | Hybrid text generation and chat | Supported: `LiquidAI/LFM2-350M`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/lfm2/README.md) |
| LFM2-MoE (`lfm2_moe`) | Hybrid decoder and sparse expert routing | Native hybrid text generation | Supported: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/lfm2_moe/README.md) |
| Gemma3 text (`gemma3_text`) | Native text decoder and sliding attention | Completion/chat, bounded sliding state | Supported: `google/gemma-3-1b-it`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/language/gemma3/README.md) |
| Phi3 (`phi3`) | Native decoder and sliding attention | Completion/chat, bounded sliding state | Supported: `microsoft/Phi-3-mini-4k-instruct`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Status](../model-admission.md) |
| GPT-OSS (`gpt_oss`) | Native MoE, MXFP4 and packed TQ3 projections | Native MXFP4 text generation; TQ3 remains a separate profile | Supported: `mlx-community/gpt-oss-20b-MXFP4-Q4`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../native-moe/manual-inference.md) |

## Vision-language guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Qwen2-VL (`qwen2_vl`) | Vision Transformer, 3D patches, merger, multimodal RoPE, image/video token insertion | Native Python, image CLI, HTTP chat and SSE | Supported: `mlx-community/Qwen2-VL-2B-Instruct-4bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/vision_language/qwen2_vl/README.md) |
| Qwen2.5-VL (`qwen2_5_vl`) | Window/full vision attention, biased SwiGLU vision tower, merger, multimodal RoPE | Native Python, image CLI, HTTP chat and SSE | Supported: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/vision_language/qwen2_5_vl/README.md) |
| Qwen3.5 0.8B / 2B (`qwen3_5`) | Hybrid gated-delta/full-attention text tower, learned/interpolated vision positions, multimodal RoPE | Native text and static-image generation, CLI, HTTP/SSE | Supported: `mlx-community/Qwen3.5-0.8B-4bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/vision_language/qwen3_5/README.md) |
| LFM2-VL / LFM2.5-VL (`lfm2_vl`) | SigLIP2 NaFlex, pixel unshuffle/projector, hybrid language model | Native SigLIP2 NaFlex + hybrid decoder, Python/CLI/HTTP/SSE | Supported: `mlx-community/LFM2-VL-450M-8bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/vision_language/lfm2_vl/README.md) |

## Retrieval guides

Additional BERT profiles cover BGE-small, multilingual MiniLM, multilingual E5
and a trained cross-encoder. Their pooling, prefixes and limits are documented
in the [retrieval workflows](batch-1-sub-6b/README.md#dense-retrieval-and-paired-reranking).

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| BERT / all-MiniLM-L6-v2 (`bert`) | BERT encoder, mean pooling, normalization, cosine | Single-vector embeddings and similarity | Supported: `sentence-transformers/all-MiniLM-L6-v2`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/embeddings/bert/README.md) |
| XLM-R / BGE-M3 / BGE Reranker (`xlm-roberta`) | Multilingual encoder, dense/sparse/ColBERT heads and scalar pair head | Dense, sparse, token and hybrid retrieval; paired reranking | FP32; reference/performance scope through 512 tokens; extra retrieval heads use Python/CLI | [Docs](../../src/mlx_one/models/embeddings/xlm_roberta/README.md) |
| Nomic Embed (`nomic_bert`) | Rotary encoder, mean pooling and Matryoshka profiles | Normalized full vectors and explicit 64/128/256/512/768-D profiles | Supported: `nomic-ai/nomic-embed-text-v1.5`, FP32; through 512 tokens | [Docs](../../src/mlx_one/models/embeddings/nomic_bert/README.md) |
| MPNet / all-mpnet-base-v2 (`mpnet`) | MPNet relative positions, mean pooling, normalization, cosine | Single-vector embeddings and similarity | Supported: `sentence-transformers/all-mpnet-base-v2`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/embeddings/mpnet/README.md) |
| Qwen3 Embedding (`qwen3_embedding`) | Final-token pooling, instructed queries, Matryoshka dimensions, normalization, cosine | Instructed queries, dimensions, normalized vectors | Supported: `Qwen/Qwen3-Embedding-0.6B`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/embeddings/qwen3_embedding/README.md) |
| Qwen3 Reranker (`qwen3_reranker`) | Official pair prompt, yes/no logit scoring, probabilities, stable ranking | Pair scores, probabilities, stable ranking | Supported: `Qwen/Qwen3-Reranker-0.6B`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/embeddings/qwen3_reranker/README.md) |
| LFM2 ColBERT (`lfm2_colbert`) | Token embeddings, masks, late-interaction MaxSim | Lower-level token embeddings and masked MaxSim | Supported: `LiquidAI/LFM2-ColBERT-350M`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/embeddings/lfm2_colbert/README.md) |

## Audio guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Whisper / Distil-Whisper (`whisper`) | Encoder-decoder, safe loading, log-Mel, BPE, decoding, language, timestamps, WER/CER | Transcription, language detection, timestamps, WER/CER | Supported: `openai/whisper-tiny`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/audio/whisper/README.md) |
| Kokoro (`kokoro`) | Native acoustic model, duration/pitch/noise prediction and waveform synthesis | 24 kHz mono English speech, Python/CLI/Engine/HTTP | Supported: `hexgrad/Kokoro-82M`, `af_heart`, up to 510 phonemes / 30 seconds | [Docs](../../src/mlx_one/models/audio/kokoro/README.md) |
| Moonshine Tiny / Base (`moonshine`) | Raw-waveform convolutions, GroupNorm, partial interleaved RoPE, encoder/cached decoder, tokenizer, greedy transcription | Native offline English transcription, Python/CLI | Supported: `UsefulSensors/moonshine-tiny`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/audio/moonshine/README.md) |
| Qwen3-ASR 0.6B / 1.7B (`qwen3_asr`) | Chunked Conv2D/audio encoder, Qwen3 decoder, strict BF16 loading, context/language prompts, accumulated-utterance streaming | Native transcription and bounded accumulated-utterance streaming, Python/CLI | Supported: `mlx-community/Qwen3-ASR-0.6B-8bit`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/audio/qwen3_asr/README.md) |
| Qwen3-ForcedAligner 0.6B (`qwen3_forced_aligner`) | Separate 5,000-class timestamp head, known-text units and boundary reconstruction | Native known-text alignment, Python/CLI; not ASR | Supported known-text alignment; TIMIT mean error 29.472 ms, p95 68.1875 ms; [record](../../status_models_testing.md) | [Docs](../../src/mlx_one/models/audio/qwen3_asr/README.md) |
| Parakeet TDT v2 / v3 (`parakeet_tdt`) | Float32 FastConformer, LSTM predictor, joint token/duration heads, greedy offline transcription, strict safetensors loading | Native offline transcription, Python/CLI; isolated v2 archive conversion | Supported: `mlx-community/parakeet-tdt-0.6b-v3`; [exact operations and evidence](../../status_models_testing.md#current-supported-models) | [Docs](../../src/mlx_one/models/audio/parakeet/README.md) |
| LFM2.5 Audio (`lfm2_audio`) | Native checkpoint loader, Conformer/LFM2, Depthformer and waveform codec | Native ASR, TTS, speech-to-speech, Python/CLI/HTTP and ordered PCM streaming | Supported pinned Q8 ASR/TTS/STS and applicable interfaces; Q4 accuracy diagnostics retained | [Docs](../../src/mlx_one/models/audio/lfm2_audio/README.md) |
| Silero VAD (`silero_vad`) | Native recurrent voice-activity detection | Python/CLI/HTTP VAD | Supported speech/silence and state controls | [Audio guide](../AUDIO_TESTING.md) |

## Segmentation guides

| Family / registry type | Native components | Available workflow | Evidence boundary | Guide |
| --- | --- | --- | --- | --- |
| Meta SAM3 (`sam3`, `sam3_tracker`, `sam3_tracker_video`, `sam3_video`) | `sam3`: ViT, feature pyramid, CLIP text/geometry encoders, DETR, presence scoring, masks; `sam3_tracker`: Point/box/mask prompts, two-way decoder, multimask prediction, refinement; `sam3_tracker_video`: Memory encoding/attention, object pointers, temporal state; `sam3_video`: Shared image backbone, detector/tracker association, object lifecycle, offline/streaming sessions | Concept/point images and native composite/standalone offline/streaming video workflows | All four registered SAM3 types supported for recorded workflows; [exact evidence](../../status_models_testing.md) | [Docs](../../src/mlx_one/models/segmentation/sam3/README.md) |

## OCR and documents

Native source paths cover TrOCR, GOT-OCR 2.0, Florence-2, PaddleOCR-VL 1.5,
LightOnOCR 2 and DeepSeek-OCR. [OCR implementation and APIs](../OCR_TESTING.md)
describes line crops, page transcription, region workflows and limitations.
[OCR performance](../benchmarks/ocr/performance.md) retains the separate page
and handwritten-line protocols, failures and truncations; experimental
measurements are not full task qualification.

<a id="additional-experimental-tasks"></a>

## Classification guides

ViT supervised images, DistilBERT sentiment, BERT NER and CLIP zero-shot
classification have passing primary-workflow records. Silero VAD is counted under audio.

| Model type | Selected checkpoint | Primary workflow | Guide |
| --- | --- | --- | --- |
| `distilbert` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | text-classification | [API guide](../api/index.md) |
| `bert_token_classification` | `dslim/bert-base-NER` | token-classification | [API guide](../api/index.md) |
| `clip` | `openai/clip-vit-base-patch32` | zero-shot-image-classification | [API guide](../api/index.md) |
| `vit` | `google/vit-base-patch16-224` | supervised 224×224 RGB classification; Python/CLI/Engine/HTTP | [ViT workflow and diagnostic limit](batch-1-sub-6b/README.md#vit-supervised-image-classification) |

SigLIP2, DETR, document/video QA and depth candidates retain their original
blocked/helper status. RT-DETR inference is not implemented. These candidates
are outside the supported primary-workflow inventory.
