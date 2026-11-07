# Architecture

## Implemented text execution path

The native text inference path is:

```text
Python Engine / HTTP + SSE / CLI
                 ↓
Prepared prompt + identity + resolved execution plan
                 ↓
Reference execution or InferenceActor + BatchScheduler
                 ↓
Native model adapter → dense / paged / hybrid runner
                 ↓
CacheManager → transactional ownership, prefix index, page arena, hybrid state
                 ↓
MLX + Metal + unified memory
```

The actor owns request state, sampling, cancellation and execution barriers.
Token budgets and chunked prefill coordinate active requests. Cache reservations
and live memory-pressure telemetry govern admission; matching evidence governs
qualified backend selection before cache storage is constructed. Native model
implementations remain within mlx-one.

## What is implemented

- Native MLX architectures across language, vision-language, embeddings,
  audio-language, ASR, and segmentation.
- Shared attention, GQA, RoPE, normalization, feed-forward, MoE, vision, hybrid
  convolution, masking, and KV-cache primitives.
- Lazy metadata-driven model registration without initializing Metal during
  lightweight package imports.
- Strict configuration validation and deterministic weight contracts that reject
  unknown or shape-incompatible tensors.
- Safe local and pinned Hugging Face artifact inspection.
- Native Whisper loading, audio preprocessing, tokenization, decoding, language
  detection, segment timestamps, word timestamps, and WER/CER evaluation.
- Distil-Whisper large-v3 through the native Whisper loader, with pinned M4
  English short-form transcript/segment gates and safe precision-variant selection.
  See [audio qualification](../audio-qualification-m4-32gb.md).
- Native original Moonshine Tiny/Base waveform encoders and cached greedy
  transcription, with pinned M4 English short-form gates.
- Native Qwen3-ASR 0.6B/1.7B, separate known-text ForcedAligner, and bounded
  accumulated-utterance streaming through Python/CLI. Qwen checkpoint qualification
  is incomplete; 0.6B has scoped English/French offline passing evidence.
- Native offline Parakeet TDT v2/v3, strict float32 loading and isolated v2 NeMo
  conversion. Both English transcript gates and v3 French/German gates pass;
  original v2 NeMo parity, timestamps and broader qualification remain WIP.
- Unified native text loading, tokenizer/chat templates, affine 4-bit/8-bit checkpoints,
  cached greedy and seeded sampling, stop sequences, token streaming, and serving
  for GPT-2, Qwen2/Qwen3, LFM2, OpenELM, and text-only Qwen3.5 families.
- Native Qwen3 embedding and reranking with byte-level BPE, safe loading,
  padding-aware batching, Matryoshka dimensions, normalization, cosine similarity,
  yes/no pair scoring, and stable ranking.
- Native Meta SAM3 float32 loading, text/box concept segmentation, interactive
  point/box/mask prediction, experimental automatic masks, and stateful offline
  and streaming video tracking through Python and CLI. Qualification is partial;
  see the [SAM3 model guide](../../src/mlx_one/models/segmentation/sam3/README.md) and
  [M4/32 GB results](../sam3-qualification-m4-32gb.md).
- Dataset validation, memory planning, LoRA/QLoRA SFT workflows, adapter export,
  evaluation, comparison, benchmarking, and evidence records.
- Backend-free tests plus opt-in Metal and real-checkpoint integration gates.
- A context-managed `mlx_one.Engine` with streaming, asynchronous submission,
  cooperative cancellation, runtime capability inspection, and explicit cleanup.

## Architecture principles

- Configuration determines architecture; model-name conditionals do not.
- Shared operations are implemented once and reused across families.
- Family-specific classes stay inside their model packages.
- Weight mapping and sanitization are explicit and deterministic.
- Unsupported settings and unknown tensors fail visibly.
- Forward execution, generation, processing, and training remain separate layers.
- Training and inference converge on the same native model implementation.
- Lightweight inspection and registry imports do not initialize Metal.
- Qualification is based on reproducible evidence, never inference from a model
  name or parameter count.

## Visual architecture and planning

[Full-resolution artwork](../../mlx-one_arch.png) preserves the project’s existing
visual identity. MTP, CPT and RL labels in the artwork include planned workflows;
see the [inference-engine roadmap](inference-engine.md) and
[development roadmap](../roadmap.md). Existing bounded Qwen3.5 MTP support does
not establish general multi-token training or speculative coverage.

The [repository summary](../REPOSITORY_SUMMARY.md) describes package boundaries.
The [runtime experiments](../experimental/inference-runtime.md) distinguish
source components from measured and qualified execution. The
[MLXBoost ADR](mlxboost-module-boundary.md) describes current
ownership and a proposed future distribution, without claiming that split exists.
