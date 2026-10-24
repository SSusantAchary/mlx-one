<p align="center">
  <img src="mlx-one.png" alt="mlx-one" width="1000">
</p>

<h1 align="center">mlx-one</h1>

<p align="center"><strong>Train locally. Prove it on Apple Silicon. Scale when needed.</strong></p>

<p align="center">
  A unified, native MLX stack for language, vision-language, embeddings,
  audio-language, speech, and segmentation models on Apple Silicon.
</p>

<p align="center">
  <a href="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml"><img src="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 License"></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha status">
</p>

[**Quickstart**](#quick-start) | [**Performance**](#model-performance) | [**Installation**](#installation) | [**Architecture**](#architecture) | [**Repository summary**](docs/REPOSITORY_SUMMARY.md) | [**Model guides**](#model-specific-documentation) | [**Documentation**](docs/ui.md) | [**Examples**](#cli)

> [!IMPORTANT]
> mlx-one is under active development. Native architecture support and model
> qualification are tracked separately. A family marked Architecture ✅ has a
> validated config, MLX model structure, registry entry, strict weight contract,
> and synthetic execution tests. It does not automatically mean every checkpoint,
> task, precision, or device has been qualified.
>
> The 0.1.0a1 development candidate uses the [functional model-support policy](docs/model-admission.md).
> Checkpoint validation and publication approval remain separate; see the
> [current checkpoint inventory](docs/benchmarks/inventory.md) for exact revisions and limits.

## Why mlx-one?

MLX is an excellent foundation for machine learning on Apple Silicon, but model
workflows are often split across separate packages and incompatible task APIs.
mlx-one is building those pieces as one coherent stack:

The project owns its supported model math directly. Hugging Face supplies
configuration, tokenizer/processor assets and safe `safetensors` checkpoints.

<!-- model-performance:start -->
## Model Performance

Actual **direct-model inference** measurements; experimental, batch size 1. Python Engine, CLI and HTTP serving throughput are separate measurement types and are not represented by these numbers. Parameters are nominal model sizes.

Configured context windows come from the pinned checkpoint configs; they are separate from benchmark input lengths and do not establish full-window validation. See the [config sources](docs/benchmarks/inventory.md).

**Historical 128-input / 16-output protocol.** These archived runs are not the new standardized 128-input / 128-output results. Source and primary precision measurements are listed separately; “source” means the stored weights.

**historical-128-16** — Apple M4, 32 GB; macOS-15.7.9-arm64-arm-64bit; Python 3.12.11; MLX 0.32.2 / Metal 0.32.2. Backend: native MLX diagnostic dense; full sequence logits; production_fp16 KV. Actual input: 128 tokens; actual output: 16 tokens per repetition. 1 shape warmup; 3 measured repetitions; per-column median. FP16 KV; fresh state, no prefix reuse. TTFT includes prefill and greedy selection of the first token. Peak MLX resets before each request after load/conversion/warmup; it includes resident model allocations.

| Model name | Parameters | Configured context (tokens) | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Time to first token (ms) | Peak MLX memory (GiB) |
|---|---|---:|---|---:|---:|---:|---:|
| [openai-community/gpt2](docs/benchmarks/historical/M01-stored-performance.json) | 124M | 1,024 | FP32 | **6722.29** | **118.04** | 19.04 | 0.594 |
| [Qwen/Qwen2.5-0.5B-Instruct](docs/benchmarks/historical/M02-affine_q8-performance.json) | 0.5B | 32,768 | Affine Q8 (group 64) | **2001.96** | **83.48** | 63.94 | 0.632 |
| [Qwen/Qwen2.5-0.5B-Instruct](docs/benchmarks/historical/M02-stored-performance.json) | 0.5B | 32,768 | BF16 (source) | **1801.11** | **50.49** | 71.07 | 1.068 |
| [Qwen/Qwen3-0.6B](docs/benchmarks/historical/M03-affine_q8-performance.json) | 0.6B | 40,960 | Affine Q8 (group 64) | **1275.02** | **65.47** | 100.39 | 0.816 |
| [Qwen/Qwen3-0.6B](docs/benchmarks/historical/M03-stored-performance.json) | 0.6B | 40,960 | BF16 (source) | **892.22** | **33.95** | 143.46 | 1.312 |
| [LiquidAI/LFM2-350M](docs/benchmarks/historical/M04-affine_q4-performance.json) | 350M | 128,000 | Affine Q4 (group 64) | **3180.66** | **185.81** | 40.24 | 0.313 |
| [LiquidAI/LFM2-350M](docs/benchmarks/historical/M04-stored-performance.json) | 350M | 128,000 | BF16 (source) | **3169.32** | **91.87** | 40.39 | 0.783 |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](docs/benchmarks/historical/M06-affine_q4-performance.json) | 1.7B | 8,192 | Affine Q4 (group 64) | **384.06** | **55.69** | 333.28 | 1.202 |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](docs/benchmarks/historical/M06-stored-performance.json) | 1.7B | 8,192 | BF16 (source) | **309.63** | **19.21** | 413.39 | 3.390 |
| [meta-llama/Llama-3.2-1B-Instruct](docs/benchmarks/historical/M07-affine_q4-performance.json) | 1B | 131,072 | Affine Q4 (group 64) | **374.73** | **65.36** | 341.58 | 0.886 |
| [meta-llama/Llama-3.2-1B-Instruct](docs/benchmarks/historical/M07-stored-performance.json) | 1B | 131,072 | BF16 (source) | **572.81** | **29.75** | 223.46 | 2.455 |
| [Qwen/Qwen2.5-1.5B-Instruct](docs/benchmarks/historical/M08-affine_q4-performance.json) | 1.5B | 32,768 | Affine Q4 (group 64) | **411.23** | **53.52** | 311.26 | 1.034 |
| [Qwen/Qwen2.5-1.5B-Instruct](docs/benchmarks/historical/M08-stored-performance.json) | 1.5B | 32,768 | BF16 (source) | **375.86** | **21.46** | 340.55 | 3.044 |
| [google/gemma-3-1b-it](docs/benchmarks/historical/M09-stored-performance.json) | 1B | 32,768 | BF16 | **675.56** | **32.17** | 189.47 | 2.028 |
| [microsoft/Phi-3-mini-4k-instruct](docs/benchmarks/historical/M10-affine_q4-performance.json) | 3.8B | 4,096 | Affine Q4 (group 64) | **159.64** | **29.92** | 801.79 | 2.357 |
| [microsoft/Phi-3-mini-4k-instruct](docs/benchmarks/historical/M10-stored-performance.json) | 3.8B | 4,096 | BF16 (source) | **126.13** | **10.87** | 1014.81 | 7.256 |

**standard-128-128** — Apple M4, 32 GB; macOS-15.8.1-arm64-arm-64bit; Python 3.12.11; MLX 0.32.2 / Metal 0.32.2. Backend: native MLX direct dense; last_only logits; production_fp16 KV. Actual input: 128 tokens; actual output: 128 tokens per repetition. 1 shape warmup; 3 measured repetitions; per-column median. FP16 KV; fresh state, no prefix reuse. TTFT includes prefill and greedy selection of the first token. Peak MLX resets before each request after load/conversion/warmup; it includes resident model allocations.

| Model name | Parameters | Configured context (tokens) | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Time to first token (ms) | Peak MLX memory (GiB) |
|---|---|---:|---|---:|---:|---:|---:|
| [openai-community/gpt2](docs/benchmarks/runs/M01-direct-128-128.json) | 124M | 1,024 | FP32 | **4057.67** | **96.48** | 31.55 | 0.570 |

**Standardized 128/128 coverage: 1 / 13 primary configurations.** Remaining runs are pending; see the coverage report.

**Missing primary-precision performance:** `Qwen/Qwen3.5-0.8B`, `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, `mlx-community/gpt-oss-20b-MXFP4-Q4`, `manjunathshiva/gpt-oss-20b-tq3`. No speed or memory figures are inferred for these checkpoints.

See the [methodology and manual commands](docs/benchmarks/README.md), [checkpoint/precision inventory](docs/benchmarks/inventory.md), and [coverage report](docs/benchmarks/coverage.md). Full samples, output token IDs, immutable revisions, source hashes and hardware evidence live in [docs/benchmarks](docs/benchmarks/).
<!-- model-performance:end -->

### GPT-OSS native HTTP serving

Measured on Apple M4 with 32 GB unified memory, Python 3.12, using the native MLX-One server on 2026-10-23. One warmup followed by five sequential tool-call requests; 175 prompt tokens and 91 generated tokens per request. Throughput and TTFT are per-column medians; memory is the observed peak. This serving workload uses a separate protocol from the direct-model tables above.

| Model name | Parameters | Configured context (tokens) | Quantization/precision | Prefill tokens/sec | Decode tokens/sec | Native first-token time (ms) | Peak MLX memory (GiB) |
|---|---|---:|---|---:|---:|---:|---:|
| [mlx-community/gpt-oss-20b-MXFP4-Q4](docs/validation/gpt-oss-kilo-2026-10-23.json) | 20B / 3.6B active | 131,072 | Mixed MXFP4 / affine Q4–Q8 | **75.95** | **18.96** | 2355.38 | 10.608 |

Native first-token time includes hidden analysis; the median first actionable tool delta arrived at **7.127 seconds**, after argument validation. The configured 131,072-token context was not tested at its full length. See the [full measurements and methodology](docs/GPT_OSS_KILO_COMPATIBILITY.md#measured-performance). Standardized direct-model 128/128 coverage remains unchanged.

<!-- MLX-ONE-VLM-BENCHMARK-START -->
## Vision-Language Model Performance

Actual **direct-model native MLX/Metal** inference on pinned checkpoint revisions. These are image-inference measurements; Python Engine, CLI and HTTP serving overhead are separate. Functional and quality limits are recorded with the evidence.

**VLM-HF-S1 v1:** one 512 × 512 padded image from [Hugging Face documentation-images](https://huggingface.co/datasets/huggingface/documentation-images), with dataset revision and image hashes pinned in the [workload manifest](benchmarks/vlm/hf-images.json). Batch 1, fixed prompt, greedy generation up to 128 emitted tokens; one warmup and five repetitions, per-column median. EOS is respected and actual output counts determine speed. Each checkpoint retains its own resize/tiling. TTFT includes image decoding/preprocessing; prefill combines vision encoding, projection and language prefill. “Vision processing” means image decoding/preprocessing only.

Q4/Q8 denote the published checkpoint format; vision or projector modules may retain floating-point weights.

**Measured hardware:** Apple M4, 32 GiB; macOS-15.8.1-arm64-arm-64bit; MLX 0.32.2; Python 3.12.11. Separate execution cohort.

| Model | Precision | Images | Vision processing (ms) | Prefill TPS | Decode TPS | TTFT (ms) | Peak MLX memory (GiB) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [Qwen2-VL 2B](docs/benchmarks/vlm/results/V01-VLM-HF-S1.json) | Q4 | 1 | 6.05 | **279.68** | **67.28** | 1290.28 | 1.755 | Measured; experimental |
| [Qwen2.5-VL 3B](docs/benchmarks/vlm/results/V02-VLM-HF-S1.json) | Q4 | 1 | 6.05 | **215.86** | **41.19** | 1670.08 | 3.164 | Measured; experimental |
| [Qwen3.5 0.8B](docs/benchmarks/vlm/results/V03-VLM-HF-S1.json) | Q4 | 1 | 4.81 | **717.75** | **68.78** | 401.16 | 1.129 | Measured; experimental |
| [LFM2-VL 450M](docs/benchmarks/vlm/results/V04-VLM-HF-S1.json) | Q8 | 1 | 7.16 | **1739.05** | **154.33** | 169.97 | 1.163 | Measured; experimental |
| [LFM2.5-VL 3B](docs/benchmarks/vlm/results/V05-VLM-HF-S1.json) | Q4 | 1 | 7.37 | **300.73** | **52.01** | 942.38 | 2.801 | Measured; experimental |

**Actual fused input / emitted output tokens per repetition:** Qwen2-VL 2B: 359 / 112; Qwen2.5-VL 3B: 359 / 79; Qwen3.5 0.8B: 284 / 106; LFM2-VL 450M: 282 / 128; LFM2.5-VL 3B: 281 / 101.


High-resolution (1024 × 1024), four-image and OCR workloads remain separate. The HF image protocol is not pooled with the earlier synthetic-image protocol. See the [methodology and coverage](docs/benchmarks/vlm/README.md), [pinned checkpoint inventory](docs/benchmarks/vlm/benchmark_plan.json), and [manual download/inference/benchmark runbook](docs/VLM_TESTING.md). Text-model measurements above use different protocols. See the [checkpoint smoke and cleanup report](docs/validation/vlm-hf-2026-10-24/report.md).
<!-- MLX-ONE-VLM-BENCHMARK-END -->

<!-- MLX-ONE-AUDIO-BENCHMARK-START -->
## Audio Model Performance

Native audio support is experimental. Values below are validated measurements; “—” means unavailable. Recognition uses one pinned English LibriSpeech utterance (9.9 seconds, padded to 10); its WER is a single-clip check, not corpus accuracy. See the linked raw evidence.

Protocol: batch 1, one warmup, five repetitions, median; actual audio durations. ASR uses a fixed 10-second clip (ASR-S1); TTS uses fixed 30-word text (TTS-S1). RTF is processing seconds divided by input audio seconds for ASR, or generated audio seconds for synthesis. Lower RTF is faster. Hardware and MLX version are listed for each measured cohort. Direct-model, Python Engine and full file-pipeline runs remain separate.

### Speech Recognition Performance

| Model | Precision | Audio (s) | Latency (ms) | RTF ↓ | Audio sec/sec ↑ | WER (%) | Peak MLX (GiB) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Whisper Tiny](docs/benchmarks/audio/results/hf-2026-10-24/A01-ASR-S1.json) | FP32 | 10.000 | 2144.869 | 0.214 | **4.662** | 4.167 | 0.485 | Measured; experimental |
| [Whisper Large v3 Turbo](docs/benchmarks/audio/results/hf-2026-10-24/A02-ASR-S1.json) | FP16 | 10.000 | 7666.962 | 0.767 | **1.304** | 4.167 | 2.694 | Measured; experimental |
| [Distil-Whisper Large v3](docs/benchmarks/audio/results/hf-2026-10-24/A03-ASR-S1.json) | FP16 | 10.000 | 5433.400 | 0.543 | **1.840** | 0.000 | 2.596 | Measured; experimental |
| [Moonshine Tiny](docs/benchmarks/audio/results/hf-2026-10-24/A04-ASR-S1.json) | FP32 | 10.000 | 117.038 | 0.012 | **85.443** | 4.167 | 0.191 | Measured; experimental |
| [Moonshine Base](docs/benchmarks/audio/results/hf-2026-10-24/A05-ASR-S1.json) | FP32 | 10.000 | 153.511 | 0.015 | **65.142** | 12.500 | 0.423 | Measured; experimental |
| [Qwen3-ASR 0.6B](docs/benchmarks/audio/results/hf-2026-10-24/A06-ASR-S1.json) | Q8 / BF16 | 10.000 | 528.726 | 0.053 | **18.913** | 0.000 | 1.281 | Measured; experimental |
| [Qwen3-ASR 1.7B](docs/benchmarks/audio/results/hf-2026-10-24/A07-ASR-S1.json) | Q8 / BF16 | 10.000 | 1017.307 | 0.102 | **9.830** | 0.000 | 2.656 | Measured; experimental |
| [Parakeet TDT v2](docs/benchmarks/audio/results/hf-2026-10-24/A08-ASR-S1.json) | FP32 | 10.000 | 169.265 | 0.017 | **59.079** | 0.000 | 2.383 | Measured; WIP |
| [Parakeet TDT v3](docs/benchmarks/audio/results/hf-2026-10-24/A09-ASR-S1.json) | FP32 | 10.000 | 174.019 | 0.017 | **57.465** | 0.000 | 2.417 | Measured; experimental |
| [LFM2.5 Audio 1.5B](docs/benchmarks/audio/results/hf-2026-10-24/A10-ASR-S1.json) | Q4 / BF16 | 10.000 | 547.659 | 0.055 | **18.260** | 4.167 | 1.472 | Measured; experimental |

Measured on Apple M4, 32 GiB; MLX 0.32.2; native MLX/Metal; direct-model-prepared-pcm; ASR-S1.

### Forced Alignment Performance

| Model | Precision | Audio (s) | Latency (ms) | Mean boundary error (ms) | Peak MLX (GiB) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen3-ForcedAligner 0.6B](docs/benchmarks/audio/results/hf-2026-10-24/A11-ALIGN-S1.json) | Q8 / BF16 | 10.000 | 189.892 | — | 1.546 | Measured; WIP |

Measured on Apple M4, 32 GiB; MLX 0.32.2; native MLX/Metal; direct-model-prepared-pcm; ALIGN-S1.

### Speech Generation Performance

| Model | Precision | Task | First audio (ms) | Output (s) | RTF ↓ | Audio sec/sec ↑ | Peak MLX (GiB) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [LFM2.5 Audio 1.5B](docs/benchmarks/audio/results/hf-2026-10-24/A12-TTS-S1.json) | Q4 / BF16 | synthesize | 328.986 | 12.320 | 0.312 | **3.207** | 1.850 | Measured; experimental |
| [LFM2.5 Audio 1.5B](docs/benchmarks/audio/results/hf-2026-10-24/A13-STS-S1.json) | Q4 / BF16 | speech-to-speech | — | — | — | — | — | Failed; WIP |

Measured on Apple M4, 32 GiB; MLX 0.32.2; native MLX/Metal; direct-model-prepared-pcm; TTS-S1.

### Speech-to-speech: separate short-question workload

**STS-HF-Q1:** a different 2.04-second spoken question from the same pinned dataset, without padding; greedy generation, one warmup, five repetitions, median. This completed case does not replace the failed 10-second STS-S1 run.

| Model | First audio (ms) | Output (s) | RTF ↓ | Audio sec/sec ↑ | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [LFM2.5 Audio 1.5B Q4](docs/benchmarks/audio/results/hf-2026-10-24/A13-STS-HF-Q1-direct-model-prepared-pcm.json) | 426.127 | 15.360 | 0.349 | **2.867** | 2.016 |

Apple M4, 32 GiB; MLX 0.32.2; native MLX/Metal; direct-model-prepared-pcm. Experimental; speech quality requires listening review.

Qwen alignment remains WIP because historical timestamp discrepancies are unresolved; boundary errors need independently annotated reference timings. LFM speech-to-speech exceeded its output bound on this fixture; its sampled diagnostic also had unstable waveform amplitude. No speed is published for that failed task. TTS timings do not establish perceptual quality. Qwen ASR streaming reprocesses a bounded accumulated utterance; it is not a causal streaming encoder.

[Methodology and coverage](docs/benchmarks/audio/README.md) · [Manual download, Python/CLI/HTTP and benchmark commands](docs/AUDIO_TESTING.md)
<!-- MLX-ONE-AUDIO-BENCHMARK-END -->

## Architecture

<p align="center">
  <a href="mlx-one_arch.png">
    <img src="mlx-one_arch.png" alt="mlx-one API, inference and training runtimes, native execution layer, and Apple MLX architecture" width="1100">
  </a>
</p>

Click the diagram to view it at full resolution. It shows current and planned
architecture: MTP, CPT and RL labels do not establish implemented or qualified
workflows. Checkpoint, task, precision and device validation are tracked separately.

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

## Native inference engine

mlx-one now exposes a lifecycle-managed Python engine while retaining the
existing `generate()` and `stream_generate()` functions:

```python
from mlx_one import Engine

with Engine(model="LiquidAI/LFM2-350M") as engine:
    result = engine.generate("Explain unified memory briefly.", max_tokens=128)
    print(result.text)

    for chunk in engine.stream("Name two advantages of MLX."):
        print(chunk.text, end="", flush=True)
```

The runtime includes typed sequence/runner contracts, isolated request RNG,
bounded output queues, transactional caches, byte-bounded prefix sharing and
explicit execution modes. HTTP prefix namespaces derive from authenticated
principals; anonymous requests remain private. Python callers can supply an
explicit cache namespace.

| Mode | Execution and availability |
|---|---|
| `reference_dense` | Existing single-request dense execution; cooperative server slots use separate B=1 forwards. |
| `experimental_block` | Materialized block-cache path for supported unquantized Qwen2/Qwen2.5; experimental. |
| `experimental_dense_batch` | Actor-driven dense batching for supported native adapters; experimental and not production-qualified. |
| `paged` | Direct page-backed attention using the segmented arena and existing Metal kernels; supported contracts only, qualification pending. |
| `experimental_hybrid_batch` | Native LFM2 attention and request-local convolution state; qualification pending. |
| `auto` | Opt-in selection from matching performance-qualified evidence; otherwise reference dense with a reported reason. |

Omitting the mode preserves existing behavior. The default active-sequence limit
remains one; larger batches require explicit configuration. Unsupported explicit
optimized modes raise capability errors. Native affine INT8 loading and floating
FP16 conversion preserve packed weight payloads; optimized P1 profiles use FP16 KV.
Quantized paged KV and 128K-class admission remain evidence-gated.

See the [repository summary](docs/REPOSITORY_SUMMARY.md) for current validation,
layout and benchmark status, and the
[native inference-engine roadmap](docs/architecture/inference-engine.md) for broader plans.

## Cache runtime qualification

Cache and optimized execution paths remain experimental; dense B1 is the bounded
reference scope. Historical P1 results do not certify other checkpoints, longer
contexts or optimized backends. Model admission now follows the
[functional support policy](docs/model-admission.md).

For the recorded workloads, numerical investigations, limitations and development
commands, see the [P1 execution report](docs/p1_execution_report.md),
[P1 qualification matrix](docs/p1_qualification_matrix.md) and
[M4 cache report](docs/cache-benchmark-m4-32gb.md).

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
  See [audio qualification](docs/audio-qualification-m4-32gb.md).
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
  see the [SAM3 model guide](src/mlx_one/models/segmentation/sam3/README.md) and
  [M4/32 GB results](docs/sam3-qualification-m4-32gb.md).
- Dataset validation, memory planning, LoRA/QLoRA SFT workflows, adapter export,
  evaluation, comparison, benchmarking, and evidence records.
- Backend-free tests plus opt-in Metal and real-checkpoint integration gates.
- A context-managed `mlx_one.Engine` with streaming, asynchronous submission,
  cooperative cancellation, runtime capability inspection, and explicit cleanup.

## Installation

mlx-one requires Python 3.10 or newer and an Apple Silicon Mac with macOS 15 or
later for model execution. The current MLX/Metal pair is pinned to 0.32.2;
Python 3.12.11 is the tested development environment.

```bash
git clone https://github.com/SSusantAchary/mlx-one.git
cd mlx-one
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install development tools with:

```bash
python -m pip install -e ".[dev]"
```

The native engine and server do not require `mlx-lm`. Install
`mlx-one[legacy-mlx-lm]` only for legacy compatibility workflows that explicitly
request that bridge.

For SAM3 image processing, install the existing vision extra:

```bash
python -m pip install -e ".[vision]"
```

Meta's [SAM3 checkpoint](https://huggingface.co/facebook/sam3) requires authorized
gated access and acceptance of its separate checkpoint license. Authenticate with
`hf auth login`, or use an authorized complete local snapshot with `--offline`.
SAM3 runs in float32; quantized checkpoints and SAM3.1 are unsupported.

FFmpeg is required when the Whisper/Moonshine/Qwen/Parakeet API receives an audio file; encoded SAM3
videos require both FFmpeg and ffprobe. Decoded mono 16-kHz waveforms, SAM3 frame
directories, and Python RGB frame iterators need no video/audio decoder.

## Quick start

### Native server and Web UI

Load one supported text model and start the bundled local UI:

```bash
python -m pip install -e ".[server]"
mlx-one serve Qwen/Qwen3-0.6B --inference-mode reference_dense
```

<p align="center">
  <a href="mlx_One_chat_ui.png">
    <img src="mlx_One_chat_ui.png" alt="mlx-one native chat Web UI" width="900">
  </a>
</p>

Click the thumbnail to open the full-size UI screenshot.

To expose a shorter model name and protect `/v1/*` with a Bearer key:

```bash
mlx-one serve LiquidAI/LFM2-350M \
  --host 127.0.0.1 \
  --port 8181 \
  --alias lfm2-local \
  --api-key local-secret
```

To enable the UI microphone and audio-file transcription, load a private native Whisper sidecar:

```bash
mlx-one serve CHAT_MODEL \
  --transcription-model WHISPER_MODEL \
  --host 127.0.0.1 \
  --port 8181
```

The Whisper model is loaded only when this flag is supplied. Recordings stay local, are
transcribed through MLX, and the editable transcript is inserted into the composer without being
sent automatically. FFmpeg is required for browser audio decoding.

Enter `local-secret` in the UI's **API key** field and select **Connect**. Terminal clients must
send the same key and use the alias as the request model:

```bash
curl http://127.0.0.1:8181/v1/models \
  -H 'Authorization: Bearer local-secret'

curl -N http://127.0.0.1:8181/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm2-local","messages":[{"role":"user","content":"Explain MLX in one sentence."}],"stream":true}'
```

Authentication is disabled when neither `--api-key` nor `MLX_ONE_API_KEY` is configured. A
`401 Unauthorized` response from `/v1/*` means the Bearer key is missing or invalid; `/health`
and the static UI remain public.

The native server also supports model aliases, optional Bearer authentication, effective context
limits, request deadlines, cooperative parallel slots, prompt/context caching, context shifting,
reasoning output, quantized KV caches, and Qwen3.5 MTP speculative decoding. Run
`mlx-one serve --help` for the complete flag list.

Cache Runtime V1 keeps dense KV as the default and exposes the Qwen2/Qwen2.5 block backend as an
opt-in experimental path:

```bash
mlx-one serve MODEL --cache-backend block --kv-block-size 32 \
  --kv-cache-budget-mib 8192 --prefix-cache-mib 512
```

The block backend currently requires unquantized f16 Qwen2/Qwen2.5 text generation with batch size
one. Unsupported combinations fail explicitly. Runtime cache ownership, memory, reservation, block,
and prefix statistics are included in `GET /v1/runtime`.

> [!WARNING]
> M4/32 GB qualification is incomplete. An unquantized Qwen2.5-0.5B dense run completed through
> 16K input tokens but failed at approximately 32K under the current memory budget. Dense snapshot
> APC did not reuse divergent chat-prompt suffixes, so block APC, block sharing, and dense/block
> parity are not qualified. Keep dense as the recommended backend; use block mode only for
> development testing. Hybrid LFM models reject block KV and APC intentionally. See the measured
> [cache qualification report](docs/cache-benchmark-m4-32gb.md) and
> [runtime support matrix](docs/cache-runtime.md).

With the default port, open `http://127.0.0.1:8888` (or `http://127.0.0.1:8181`
for the explicit-port examples above). The same process exposes an OpenAI-compatible
API at `/v1`, including streaming chat completions. Model execution remains inside
mlx-one's native model, tokenizer, sampling, generation, and MLX runtime. See
[the Web UI and server guide](docs/ui.md) for API and development details.

### Inspect, plan, and run tasks

Check the machine and inspect a model without loading its weights:

```bash
mlx-one doctor

mlx-one inspect Qwen/Qwen2.5-1.5B
```

Build a workload-aware memory plan:

```bash
mlx-one plan inference \
  --model Qwen/Qwen2.5-1.5B \
  --hardware m4-air-32gb \
  --precision bf16 \
  --context-length 2048
```

Generate text through the native GPT-2 stack:

```bash
mlx-one generate openai-community/gpt2 "The future of local AI is" \
  --max-tokens 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 0
```

Use `--stream` for incremental text, `--json-output` for a structured result,
or `--offline` with an existing local/cache copy. Model downloads occur only
when a user explicitly supplies an online repository without `--offline`.

Transcribe audio through the native Whisper stack:

```bash
mlx-one transcribe openai/whisper-tiny recording.m4a \
  --language en \
  --word-timestamps \
  --json-output
```

Omit `--language` for automatic detection. Add `--task translate` for
speech-to-English translation or `--offline` to require local/cached assets.

Create document or instructed query embeddings:

```bash
mlx-one embed Qwen/Qwen3-Embedding-0.6B "A document to index" \
  --dimensions 512 \
  --json-output

mlx-one embed Qwen/Qwen3-Embedding-0.6B "local Apple Silicon training" \
  --input-type query
```

Rerank candidate documents while preserving their original indexes:

```bash
mlx-one rerank Qwen/Qwen3-Reranker-0.6B \
  "Which Macs support MLX?" \
  "MLX is designed for Apple silicon." \
  "CUDA targets NVIDIA GPUs." \
  --top-k 1 \
  --json-output
```

For reproducible evaluation and qualification, add `--revision` with an exact
model commit. Raw commit hashes are kept in integration tests and evidence
records rather than introductory examples.

## Model-Specific Documentation

Each registered model type has a guide beside its implementation, with checkpoint
assets, model architecture diagrams, paper references, input formats, examples,
settings, troubleshooting, and validation links.
The organization follows [mlx-vlm's model guides](https://github.com/Blaizzy/mlx-vlm/tree/main#model-specific-documentation);
the workflows below describe mlx-one's current implementation.

An available loader or integration gate is not a passing qualification result.
Examples marked as candidates require verification for the exact checkpoint and
workload. See [capability levels](#capability-levels) and the
[qualification candidate catalog](docs/model-catalog.md).

### Text generation guides

| Family | Available workflow | Qualification summary | Guide |
| --- | --- | --- | --- |
| GPT-2 | Completion, sampling, streaming | 124M bounded functional smoke; larger checkpoints remain candidates | [Docs](src/mlx_one/models/language/gpt2/README.md) |
| Llama / MiniCPM5 / SmolLM2 | Completion, template chat, quantized loading | MiniCPM5 pinned gate; SmolLM2 M4 LoRA/reload evidence, quality gate failed | [Docs](src/mlx_one/models/language/llama/README.md) |
| Qwen2 / Qwen2.5 / Qwen2.5-Coder | Completion, template chat, streaming | Coder M4 LoRA/reload evidence, quality gate failed; cache qualification partial | [Docs](src/mlx_one/models/language/qwen2/README.md) |
| Qwen3 dense | Completion, template chat, streaming | 0.6B bounded functional smoke; other checkpoints remain candidates | [Docs](src/mlx_one/models/language/qwen3/README.md) |
| Qwen2-MoE | Experimental native text loading/generation | Architecture/synthetic; checkpoint unqualified | [Docs](src/mlx_one/models/language/qwen2_moe/README.md) |
| OpenELM | Completion with compatible tokenizer assets | Candidate; verify exact release | [Docs](src/mlx_one/models/language/openelm/README.md) |
| LFM2 / LFM2.5 | Hybrid text generation and chat | Per checkpoint; M4 block-rejection safety result | [Docs](src/mlx_one/models/language/lfm2/README.md) |
| LFM2-MoE | Experimental hybrid text loading/generation | Architecture/synthetic tests; LFM2.5-24B-A2B full-checkpoint smoke pending | [Docs](src/mlx_one/models/language/lfm2_moe/README.md) |
| Gemma3 text | Completion/chat, bounded sliding state | 1B bounded functional smoke; numerical/answer-quality limitations disclosed | [Docs](src/mlx_one/models/language/gemma3/README.md) |
| Phi3 | Completion/chat, bounded sliding state | Mini-4K bounded functional smoke; wider context/precision not qualified | [Status](docs/model-admission.md) |
| GPT-OSS | Experimental native MXFP4 and TQ3 inference | Architecture/kernel tests; both 20B full-checkpoint runs pending | [Docs](docs/native-moe/manual-inference.md) |

### Vision-language guides

| Family | Available workflow | Qualification summary | Guide |
| --- | --- | --- | --- |
| Qwen2-VL | Native Python, image CLI, HTTP chat and SSE | Pinned Q4 checkpoint smoke passed on M4; experimental; static images only | [Docs](src/mlx_one/models/vision_language/qwen2_vl/README.md) |
| Qwen2.5-VL | Native Python, image CLI, HTTP chat and SSE | Pinned Q4 checkpoint smoke passed on M4; experimental; static images only | [Docs](src/mlx_one/models/vision_language/qwen2_5_vl/README.md) |
| Qwen3.5 | Native text and static-image generation, CLI, HTTP/SSE | Targeted 0.8B MLX Q4 image checkpoint smoke passed; alternative revisions unqualified | [Docs](src/mlx_one/models/vision_language/qwen3_5/README.md) |
| LFM2-VL / LFM2.5-VL | Native SigLIP2 NaFlex + hybrid decoder, Python/CLI/HTTP/SSE | Pinned Q8 450M and Q4 3B checkpoint smokes passed on M4; experimental | [Docs](src/mlx_one/models/vision_language/lfm2_vl/README.md) |

### Retrieval guides

| Family | Available workflow | Qualification summary | Guide |
| --- | --- | --- | --- |
| BERT / all-MiniLM-L6-v2 | Single-vector embeddings and similarity | Candidate; synthetic validation | [Docs](src/mlx_one/models/embeddings/bert/README.md) |
| MPNet / all-mpnet-base-v2 | Single-vector embeddings and similarity | Candidate; synthetic validation | [Docs](src/mlx_one/models/embeddings/mpnet/README.md) |
| Qwen3 Embedding | Instructed queries, dimensions, normalized vectors | 0.6B candidate; synthetic validation | [Docs](src/mlx_one/models/embeddings/qwen3_embedding/README.md) |
| Qwen3 Reranker | Pair scores, probabilities, stable ranking | 0.6B candidate; synthetic validation | [Docs](src/mlx_one/models/embeddings/qwen3_reranker/README.md) |
| LFM2 ColBERT | Lower-level token embeddings and masked MaxSim | Candidate; generic single-vector `embed` incompatible | [Docs](src/mlx_one/models/embeddings/lfm2_colbert/README.md) |

### Audio guides

| Family | Available workflow | Qualification summary | Guide |
| --- | --- | --- | --- |
| Whisper / Distil-Whisper | Transcription, language detection, timestamps, WER/CER | Tiny/turbo smoke gates; Distil-large-v3 M4 English short-form gates pass; broader qualification pending | [Docs](src/mlx_one/models/audio/whisper/README.md) |
| Moonshine Tiny / Base | Native offline English transcription, Python/CLI | Both pinned checkpoints pass M4 English short-form gates; streaming/long-form unqualified | [Docs](src/mlx_one/models/audio/moonshine/README.md) |
| Qwen3-ASR 0.6B / 1.7B | Native transcription and bounded accumulated-utterance streaming, Python/CLI | 0.6B English/French offline gates pass; **WIP**: 1.7B English WER/CER match reference but only 15/16 exact anchors pass; streaming parity and remaining language gates unresolved | [Docs](src/mlx_one/models/audio/qwen3_asr/README.md) |
| Qwen3-ForcedAligner 0.6B | Native known-text alignment, Python/CLI; not ASR | Float32 components pass; **WIP**: English timestamp differences reach 800 ms against an 80-ms limit; remaining qualification unresolved | [Docs](src/mlx_one/models/audio/qwen3_asr/README.md) |
| Parakeet TDT v2 / v3 | Native offline transcription, Python/CLI; isolated v2 archive conversion | Float32 components, both English 100-clip/16-anchor gates and v3 French/German 25-clip gates pass; **WIP** original v2 NeMo, timestamp and broader qualification | [Docs](src/mlx_one/models/audio/parakeet/README.md) |
| LFM2.5 Audio | Native ASR, TTS, speech-to-speech, Python/CLI/HTTP and ordered PCM streaming | Source/tiny native tests; Q4 checkpoint and speech quality validation pending | [Docs](src/mlx_one/models/audio/lfm2_audio/README.md) |

### Audio implementation progress — WIP

The [native audio runbook](docs/AUDIO_TESTING.md) documents the current source
implementation, exact manual checkpoints and remaining limitations. Standardized
measurements remain pending; historical investigations below are retained evidence.

As of 2026-10-04, unresolved qualification gates remain **WIP** and no longer
block implementation of later families. This changes development sequencing,
not acceptance thresholds: WIP does not mean qualified, and unavailable tasks
are not advertised as working loaders.

| Stage | Checkpoints | Implementation / qualification status |
| --- | --- | --- |
| 3 | Qwen3-ASR 0.6B / 1.7B; ForcedAligner 0.6B | Implemented; **WIP** exact anchors, alignment timestamps, streaming parity and missing language/corpus gates |
| 4 | Parakeet TDT v2 / v3; Nemotron 3.5 ASR | Parakeet offline implemented; **WIP** full qualification. Nemotron contract frozen; native cached streaming not implemented |
| 5 | Voxtral Mini 3B; Voxtral Realtime 4B | Checkpoint/tensor/processor contracts frozen; **WIP / not implemented** native offline/streaming transcription and qualification |
| 6 | Original VibeVoice 1.5B multi-speaker TTS | Checkpoint contract, original source and supplemental tokenizer pinned; **WIP / not implemented** native synthesis, reference parity and fixture permission evidence |

See the [audio report](docs/audio-qualification-m4-32gb.md) for the numerical
investigations, checkpoint limitations and remaining validation work.

### Segmentation guides

| Family | Available workflow | Qualification summary | Guide |
| --- | --- | --- | --- |
| Meta SAM3 (`sam3`, `sam3_tracker`, `sam3_tracker_video`, `sam3_video`) | Concept/interactive images, experimental automatic masks, composite offline/streaming tracking | M4/32 GB single-object 32-frame fixture passes CPU-reference gates; full qualification pending; standalone temporal tracker architecture-only | [Docs](src/mlx_one/models/segmentation/sam3/README.md) |

The [SAM3 qualification report](docs/sam3-qualification-m4-32gb.md) records exact
checkpoint/reference revisions, memory measurements, preserved failures, and
remaining gates. These fixture passes do not qualify all segmentation workflows.

<a id="native-model-coverage"></a>

## Supported Models

For checkpoint assets, input formats, and usage limitations, see the
[model-specific guides](#model-specific-documentation). The tables below summarize
implemented components; architecture coverage does not guarantee an end-to-end task.

### Language models

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `gpt2` | GPT-2 124M, 355M, 774M, 1.5B | Learned positions, fused QKV, byte BPE, safe loading, cache, generation and streaming | ✅ | 124M bounded functional smoke; larger checkpoints remain candidates |
| `llama` | Llama, MiniCPM5 1B/2B | GQA, explicit head dimensions, RoPE, 4-bit MLX loading, multi-EOS generation | ✅ | MiniCPM5 pinned integration gate |
| `qwen2` | Qwen2, Qwen2.5, Qwen2.5-Coder | Dense Transformer, GQA, RoPE, cache | ✅ | Per checkpoint |
| `qwen3` | Qwen3 dense | Bias-free attention, Q/K norm, explicit head dimensions | ✅ | 0.6B bounded functional smoke; other checkpoints remain candidates |
| `qwen2_moe` | Qwen2-MoE | Top-k experts, shared expert, router outputs | ✅ | Architecture only |
| `openelm` | OpenELM 270M–3B | Layer-wise heads/FFN widths, fused QKV, GQA | ✅ | Verify release |
| `lfm2` | LFM2 and LFM2.5 | Hybrid convolution/attention decoder | ✅ | Per checkpoint |
| `lfm2_moe` | LFM2 / LFM2.5 MoE | Hybrid decoder and sparse expert routing | ✅ | Architecture/synthetic tests; LFM2.5-24B full-checkpoint smoke pending |
| `gemma3_text` | Gemma3 1B | Native text decoder and sliding attention | ✅ | Bounded functional smoke; known numerical/answer-quality limitations |
| `phi3` | Phi-3 Mini-4K | Native decoder and sliding attention | ✅ | Bounded functional smoke; wider context/precision not qualified |
| `gpt_oss` | GPT-OSS 20B | Native MoE, MXFP4 and packed TQ3 projections | ✅ | Architecture/kernel tests; full-checkpoint smoke pending |

### Vision-language models

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `qwen2_vl` | Qwen2-VL | Vision Transformer, 3D patches, merger, multimodal RoPE, image/video token insertion | ✅ | Verify exact checkpoint and processor |
| `qwen2_5_vl` | Qwen2.5-VL-3B | Window/full vision attention, biased SwiGLU vision tower, merger, multimodal RoPE | ✅ | Pinned static-image checkpoint smoke passed on M4 |
| `qwen3_5` | Qwen3.5 0.8B/2B | Hybrid gated-delta/full-attention text tower, learned/interpolated vision positions, multimodal RoPE | ✅ | Text source smoke retained; targeted MLX VLM Q4 checkpoint smoke passed on M4 |
| `lfm2_vl` | LFM2-VL / LFM2.5-VL | SigLIP2 NaFlex, pixel unshuffle/projector, hybrid language model | ✅ | Pinned Q8 450M / Q4 3B static-image checkpoint smokes passed on M4 |

### Embeddings and retrieval

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `bert` | all-MiniLM-L6-v2 | BERT encoder, mean pooling, normalization, cosine | ✅ | Candidate |
| `mpnet` | all-mpnet-base-v2 | MPNet relative positions, mean pooling, normalization, cosine | ✅ | Candidate |
| `qwen3_embedding` | Qwen3-Embedding-0.6B | Final-token pooling, instructed queries, Matryoshka dimensions, normalization, cosine | ✅ | Candidate; synthetic validation only |
| `qwen3_reranker` | Qwen3-Reranker-0.6B | Official pair prompt, yes/no logit scoring, probabilities, stable ranking | ✅ | Candidate; synthetic validation only |
| `lfm2_colbert` | LFM2/LFM2.5 ColBERT | Token embeddings, masks, late-interaction MaxSim | ✅ | Candidate |

### Audio and speech

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `whisper` | tiny, base, small, medium, large-v3, turbo, Distil-large-v3 | Encoder-decoder, safe loading, log-Mel, BPE, decoding, language, timestamps, WER/CER | ✅ | Tiny/turbo smoke gates; [Distil M4 English short-form evidence](docs/audio-qualification-m4-32gb.md); broader qualification pending |
| `moonshine` | Original Tiny 27M / Base 61M | Raw-waveform convolutions, GroupNorm, partial interleaved RoPE, encoder/cached decoder, tokenizer, greedy transcription | ✅ | [M4 English short-form evidence](docs/audio-qualification-m4-32gb.md#moonshine-tinybase-stage-two-results); streaming/long-form unqualified |
| `qwen3_asr` | Qwen3-ASR 0.6B / 1.7B | Chunked Conv2D/audio encoder, Qwen3 decoder, strict BF16 loading, context/language prompts, accumulated-utterance streaming | ✅ | [0.6B offline passes and remaining failures](docs/audio-qualification-m4-32gb.md#additional-stage-three-checkpoint-runs-and-failures); **WIP** full qualification |
| `qwen3_forced_aligner` | Qwen3-ForcedAligner 0.6B | Separate 5,000-class timestamp head, known-text units and boundary reconstruction | ✅ | Alignment only; **WIP** timestamp parity (800 ms versus 80-ms limit) |
| `parakeet_tdt` | Parakeet TDT 0.6B v2 / v3 | Float32 FastConformer, LSTM predictor, joint token/duration heads, greedy offline transcription, strict safetensors loading | ✅ | [Scoped M4 evidence](docs/audio-qualification-m4-32gb.md#parakeet-stage-four-progress--2026-10-04); **WIP** full qualification; offline-only |
| `lfm2_audio` | LFM2.5-Audio | Native checkpoint loader, Conformer/LFM2, Depthformer and waveform codec | ✅ | Source implementation; real checkpoint smoke and speech quality pending |

### Segmentation and tracking

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `sam3` | Meta SAM3 concept detector | ViT, feature pyramid, CLIP text/geometry encoders, DETR, presence scoring, masks | Native | Pinned text-prompt image fixture evidence; box exemplars and full qualification pending |
| `sam3_tracker` | Meta SAM3 interactive image model | Point/box/mask prompts, two-way decoder, multimask prediction, refinement | Native | Pinned positive-point fixture passes; refinement and automatic-mask parity pending |
| `sam3_tracker_video` | Meta SAM3 temporal tracker | Memory encoding/attention, object pointers, temporal state | Native | Standalone architecture only; used by the composite video workflow |
| `sam3_video` | Meta SAM3 composite video model | Shared image backbone, detector/tracker association, object lifecycle, offline/streaming sessions | Native | Separate 32-frame single-object CPU-reference comparisons pass on M4/32 GB; natural multi-object clips unqualified |

Use [`facebook/sam3`](https://huggingface.co/facebook/sam3) through the dedicated
segmentation API or `segment`/`track` commands, not text generation or the text
`Engine`. Native components do not imply full task qualification. The measured
video fixtures achieved minimum mask IoU 0.99998, maximum score difference
0.00007511, exact boxes, and stable object ID `0`, with approximately 8.41 GB peak
MLX allocation. These are controlled single-truck clips, not general tracking
quality or isolated performance benchmarks; see the
[qualification report](docs/sam3-qualification-m4-32gb.md).

The detailed family backlog and exact status are maintained in
[model catalog](docs/model-catalog.md). Package ownership and dependency boundaries
are defined in [repository summary](docs/REPOSITORY_SUMMARY.md).

### Text checkpoint status

The existing text checkpoint scope is dense B1 with FP16 KV. These are bounded
functional results, not a general quality or maximum-context guarantee. Exact
revisions, precision limits and the three planned large checkpoints are recorded
in the [checkpoint inventory](docs/benchmarks/inventory.md).

| Checkpoint | Primary precision | Status |
| --- | --- | --- |
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) | float32 | Experimental; bounded functional smoke |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) | affine_q8 | Experimental; bounded functional smoke |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | affine_q8 | Experimental; bounded functional smoke |
| [LiquidAI/LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M/tree/0b6ea934a94b880d1048cbcba45a0d81ca721eb0) | affine_q4 | Experimental; bounded functional smoke |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | affine_q4 | Experimental; primary Q4 unqualified |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) | affine_q4 | Experimental; bounded functional smoke |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/9213176726f574b556790deb65791e0c5aa438b6) | affine_q4 | Experimental; bounded functional smoke |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | affine_q4 | Experimental; bounded functional smoke |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it/tree/dcc83ea841ab6100d6b47a070329e1ba4cf78752) | bfloat16 | Experimental; bounded functional smoke |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/f39ac1d28e925b323eae81227eaba4464caced4e) | affine_q4 | Experimental; bounded functional smoke |


## Capability levels

mlx-one deliberately avoids turning one successful test into a broad support
claim.

| Level | Meaning |
| --- | --- |
| Architecture ✅ | Config, model math, registry, weight contract, and synthetic tests pass |
| Candidate | The family or checkpoint still needs full integration evidence |
| Integration-tested | An exact checkpoint revision passes its task contract |
| Hardware-verified | A checksummed workload passes on a recorded Apple Silicon profile |
| Quality-verified | A pinned evaluation protocol passes its quality threshold |

Every evidence claim is scoped to a model revision, operation, precision,
workload, software environment, and hardware profile.

## CLI

### Inspect and plan

```bash
mlx-one inspect MODEL [--revision REVISION] [--offline] [--json-output]
mlx-one hardware list
mlx-one hardware detect --json-output
mlx-one plan inference --model MODEL --hardware PROFILE
mlx-one plan train --model MODEL --hardware PROFILE --method auto
```

Inspection is metadata-only: it does not execute remote model code or open
pickle checkpoints.

### Generate text

```bash
mlx-one generate MODEL PROMPT \
  [--revision REVISION] [--offline] [--cache-dir DIRECTORY] \
  [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] \
  [--seed N] [--stop TEXT] [--stream] [--json-output]
```

GPT-2 uses the native path for direct generation, evaluation, and inference
benchmarking. Native GPT-2 training and adapters are intentionally rejected
until their own implementation and validation gates are complete.

### Embed and rerank

```bash
mlx-one embed MODEL TEXT... \
  [--input-type query|document] [--instruction TEXT] [--dimensions N] \
  [--max-length N] [--batch-size N] [--offline] [--json-output]

mlx-one rerank MODEL QUERY DOCUMENT... \
  [--instruction TEXT] [--top-k N] [--max-length N] [--batch-size N] \
  [--offline] [--json-output]
```

Both commands use the native Qwen3 tokenizer and strict safetensors loader.
Document embeddings are unprefixed; query embeddings use the documented Qwen3
instruction format. Reranking returns the original document index, raw yes/no
logit difference, and two-class probability.

### Segment images and track video

```bash
mlx-one segment facebook/sam3 image.jpg --mode concept --text "person" \
  --output results-concept
mlx-one segment facebook/sam3 image.jpg --mode interactive \
  --prompts-json prompts.json --output results-interactive
mlx-one segment facebook/sam3 image.jpg --mode automatic \
  --points-per-side 32 --output results-automatic

mlx-one track facebook/sam3 video.mp4 --text "person" --output results-offline
mlx-one track facebook/sam3 frames/ --streaming --text "person" --output results-stream
```

Both commands accept `--revision`, `--offline`, and `--cache-dir`. The publisher's
SAM3 checkpoint defaults to the pinned revision recorded in the qualification
report. An interactive `prompts.json` can contain:

```json
{"points": [[100, 120]], "point_labels": [1]}
```

Coordinates must lie within the original image; labels `1` and `0` mean positive
and negative points. The [SAM3 guide](src/mlx_one/models/segmentation/sam3/README.md)
documents box exemplars, refinement masks, and timed video prompt JSON for
corrections/removal. Exports contain mask PNGs and a JSON manifest with scores,
pixel-space XYXY boxes, identifiers, dimensions, and checkpoint provenance.
Existing output directories are never overwritten.

Streaming processes frames chronologically and rejects retroactive corrections;
offline sessions retain frames for replay. Streaming omits future-dependent
hotstart removal, so its outputs are qualified separately. Automatic generation
currently uses a single-image point grid and mask-IoU NMS, not a qualified
crop-layer pipeline. These workflows remain experimental; no SAM3 training,
HTTP endpoints, or Web UI integration is provided.

### Validate training data

```bash
mlx-one data validate \
  --dataset examples/text-sft/train.jsonl \
  --layout prompt-completion
```

Supported layouts are `instruction`, `messages`, `prompt-completion`, and
`text`. A tokenizer can be supplied for truncation and loss-mask previews.

### Train and export

```bash
mlx-one train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --dataset examples/text-sft/train.jsonl \
  --config examples/text-sft/train.yaml

mlx-one export \
  --checkpoint runs/qwen-coder-sft/adapters/mlx-one-checkpoint.json \
  --output artifacts/qwen-coder-sft
```

Training is experimental. Start with the planner and pin the model revision.

### Evaluate, compare, and benchmark

```bash
mlx-one evaluate \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --dataset examples/text-sft/eval.jsonl \
  --predictions examples/text-sft/predictions-smoke.json \
  --runs-dir runs \
  --json-output

mlx-one compare runs/base/result.json runs/candidate/result.json \
  --gate examples/text-sft/quality-gate.yaml \
  --format markdown

mlx-one benchmark inference \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --prompts examples/text-sft/prompts.json
```

ASR evaluation uses the pinned `whisper-wer-v1` and `whisper-cer-v1` profiles.

### Evidence and registry

```bash
mlx-one registry list
mlx-one registry validate registry.json
mlx-one evidence publish --result runs/example/result.json --output evidence.json
```

## Python API

Inspect and plan without initializing Metal:

```python
from mlx_one import (
    Precision,
    WorkloadKind,
    WorkloadSpec,
    estimate_memory,
    inspect_model,
    load_hardware_profile,
)

model = inspect_model("Qwen/Qwen2.5-1.5B").model
hardware = load_hardware_profile("m4-air-32gb")
workload = WorkloadSpec(
    kind=WorkloadKind.INFERENCE,
    precision=Precision.BF16,
    context_length=2048,
)

estimate = estimate_memory(model, hardware, workload)
print(estimate.to_json())
```

Transcribe a file or mono 16-kHz waveform:

```python
from mlx_one import WhisperDecodeOptions, transcribe

result = transcribe(
    "openai/whisper-tiny",
    "recording.wav",
    language="en",
    word_timestamps=True,
    options=WhisperDecodeOptions(seed=0),
)

print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
```

Embed and rerank through typed native retrieval results:

```python
from mlx_one import embed, rerank

query = embed(
    "Qwen/Qwen3-Embedding-0.6B",
    "How does MLX use unified memory?",
    input_type="query",
    dimensions=512,
)

ranking = rerank(
    "Qwen/Qwen3-Reranker-0.6B",
    "How does MLX use unified memory?",
    ["MLX arrays share CPU and GPU memory.", "A recipe for sourdough."],
    top_k=1,
)

print(query.embeddings[0])
print(ranking.items[0].index, ranking.items[0].score)
```

Segment an image with Meta SAM3 through the lazy segmentation namespace:

```python
from mlx_one.segmentation import load_segmentation_model, segment_image

model = load_segmentation_model("facebook/sam3", task="concept")
result = segment_image(model, "image.jpg", text="person")
print(result.masks.shape, result.scores, result.boxes)
```

For interactive masks, load with `task="interactive"` and call
`predict_masks(model, image, points=..., point_labels=...)`. Interactive
`predicted_iou` is distinct from concept confidence `scores`; refinement uses
`low_res_logits`. All masks are original-resolution boolean arrays.

Track lexically ordered, fixed-dimension frames with a session-owned state:

```python
from mlx_one.segmentation import load_segmentation_model

model = load_segmentation_model("facebook/sam3", task="video", offline=True)
with model.video_session("frames/") as session:
    prompt_id = session.add_prompt("person")
    for result in session.propagate():
        print(result.frame_index, result.object_ids, result.prompt_ids)
```

`offline=True` requires a complete cached checkpoint. Use zero-padded frame
filenames. Python RGB iterators and local encoded-video paths are also accepted.
For live Python frames, use `model.video_session(streaming=True)` and
`session.process_frame(frame)`. Context exit releases session resources without
unloading another session's shared model. SAM3 temporal memory is separate from
the text dense/APC/block cache runtime. See the
[model guide](src/mlx_one/models/segmentation/sam3/README.md) for prompt validation,
corrections, cancellation, and current qualification limits.

Train through the typed SFT contract:

```python
from mlx_one import SFTTrainer, TrainConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    revision="PINNED_REVISION",
    train_dataset="examples/text-sft/train.jsonl",
    args=TrainConfig(
        output_dir="runs/qwen-coder-sft",
        max_seq_length=512,
        method="lora",
        max_steps=10,
    ),
)

result = trainer.train()
print(result.to_json())
```

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

## Validation

The current repository gates include:

- Backend-free tests covering schemas, configs, registries, weight contracts,
  tokenization, processing, evaluation, loading policy, and CLI behavior.
- Opt-in Apple Silicon/Metal tests covering native model execution, shapes, caches,
  masks, multimodal feature insertion, embeddings, reranking, MoE routing, and ASR
  components.
- Pinned real-checkpoint smoke gates for `openai/whisper-tiny` and
  `openai/whisper-large-v3-turbo`.
- Whisper log-Mel comparison within `1e-5` against the pinned reference path.
- SAM3 network-free prompt/export/lifecycle checks, strict pinned checkpoint
  loading, and separately recorded 32-frame offline/streaming CPU-reference
  fixture comparisons. Natural multi-object tracking qualification is pending.
- Ruff, source/wheel builds, and package metadata checks.

These counts describe the current development tree and will change as coverage
expands.

Run backend-free checks:

```bash
python scripts/release/check_lint.py
pytest -q
python -m build --no-isolation
python -m twine check dist/*
```

Run native MLX execution tests on an Apple Silicon host:

```bash
MLX_ONE_RUN_MLX_TESTS=1 pytest -q
```

Run pinned Whisper integration gates:

```bash
MLX_ONE_RUN_WHISPER_INTEGRATION=1 pytest -q \
  tests/test_native_whisper_integration.py
```

Ordinary tests are network-free. Integration tests may download only the pinned
assets needed by their explicit gate.

## Citation

If you use `mlx-one` in your work, please cite it as:

```bibtex
@software{mlx_one2026,
author = {Achary, S. Susant},
title  = {mlx-one: Unified Native MLX Stack for Apple Silicon},
year   = {2026},
url    = {https://github.com/SSusantAchary/mlx-one},
note   = {Version 0.1.0a1. Native GPU-accelerated machine learning workflows on Apple Silicon}
}
```

See [CITATION.bib](CITATION.bib) for the repository citation file.

## Roadmap

The implementation proceeds by evidence-backed vertical slices:

1. Complete checkpoint loading, generation, and parity qualification for native
   language families.
2. Expand the native retrieval APIs beyond the initial Qwen3 embedding and
   reranking vertical slice and qualify exact checkpoint revisions.
3. Add complete image processing, generation, OCR/VQA evaluation, and training
   paths for native VLMs.
4. Qualify Whisper revisions and expand native ASR, alignment, audio-language,
   and TTS coverage.
5. Add native quantization, advanced training, export, and community evidence
   workflows.

## Contributing

Contributions should include the smallest relevant config, synthetic execution,
weight-contract, integration, and documentation updates. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Report security
issues through [SECURITY.md](SECURITY.md), not a public issue.

Released under the [Apache License 2.0](LICENSE).
