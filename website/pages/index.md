---
title: Home
---

# MLX-One

**One stack. Native MLX. Built for Apple Silicon.** MLX-One brings text generation,
vision-language inference, audio, OCR, retrieval, classification and segmentation into one native
MLX library. Load a model, run it from Python or the CLI, and serve a local chat UI
and HTTP API. Native model implementations share checkpoint loading, execution and
developer tooling on Apple Silicon.

**45 native model types across seven modalities.** Generate text with T5 and
causal language models, classify images with ViT, synthesize speech with Kokoro,
and build search with BGE-M3, Nomic, multilingual E5 and paired rerankers.
**50 verified checkpoint profiles** cover these architectures. All ten additional
workflows meet the technical release criteria for their documented tasks.
[Explore model workflows and examples](../../docs/models/batch-1-sub-6b/README.md) ·
[Search verified checkpoints](insights/index.md#supported-model-explorer).

<!-- insights:summary -->

The latest workflows are available from this source checkout. From the repository
root, install the server, image-processing and optional English speech extras:

```bash
python -m pip install -e '.[server,vision,kokoro]'
```

```bash
# Inspect a checkpoint without loading its weights.
mlx-one inspect openai-community/gpt2

# Generate text through the native MLX runtime.
mlx-one generate openai-community/gpt2 "The future of local AI is" \
  --max-tokens 64 --temperature 0.8 --seed 0

# Start the local chat UI and an OpenAI-compatible HTTP API.
mlx-one serve LiquidAI/LFM2-350M --alias lfm2-local \
  --host 127.0.0.1 --port 8888 --inference-mode reference_dense
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Supported models](insights/index.md#supported-model-explorer){ .md-button }
[How it works](reference/architecture.md){ .md-button }

## What it makes possible

<!-- insights:modalities -->

[Explore model coverage, benchmarks and development progress](insights/index.md){ .md-button }

Each architecture type is counted once under its primary modality. Checkpoint
sizes, revisions and quantizations share a type. The regression count records the
Python 3.12 / Apple M4 run: 2,599 passed, 9 subtests and 95 skips; 54 final targeted
checks also passed. The core regression matrix passed on Python 3.10–3.13.
[Validation scope and results](../../docs/models/batch-1-sub-6b/REPORT.md).

| Workflow | Native capabilities | Start here |
| --- | --- | --- |
| Text generation | Cached causal decoding, sampling and streaming; T5 encoder-decoder text-to-text | [Generate text](guides/generate.md) · [T5](../../docs/models/batch-1-sub-6b/README.md#t5-encoder-decoder-text-generation) |
| Local applications | Python APIs, CLI workflows and HTTP/SSE serving | [Serve an API and chat UI](guides/serve.md) |
| Vision-language | Supported static-image and text workflows | [Vision-language guide](guides/vision.md) |
| Audio and speech | ASR, alignment, voice activity detection, LFM speech and Kokoro English synthesis | [Audio guide](guides/audio.md) · [Kokoro](../../docs/models/batch-1-sub-6b/README.md#kokoro-native-english-speech) |
| Documents | Line recognition and implemented page-model OCR workflows | [OCR guide](guides/ocr.md) |
| Retrieval | Dense embeddings, learned sparse weights, ColBERT token vectors, hybrid scores and paired reranking | [Embeddings and reranking](guides/retrieval.md) |
| Segmentation and tracking | SAM3 image prompts and native offline/streaming video tracking | [SAM3 guide](models/segmentation/sam3.md) |
| Classification | Sentiment, named entities, CLIP zero-shot images and supervised ViT images | [Classification guides](models/index.md#classification-guides) |

The [model explorer](insights/index.md#supported-model-explorer) records exact checkpoints and operations.
[Performance reports](performance/index.md) retain measured workloads, and
[known quality and scope limits](models/support.md#known-quality-and-scope-limitations)
remain visible alongside support.

## Release record

**0.1.2 was approved for publication on 10 November 2026**, including the ten
additional workflows and all 45 model types in their documented scope.
[Release notes](releases/0.1.2.md) ·
[Publication state](../../docs/releases/0.1.2.json).
The previously published [0.1.1 release](releases/0.1.1.md) covers the original
40-type cohort.

```bash
python -m pip install --upgrade "mlx-one[server,vision,kokoro]==0.1.2"
```

Use the source installation above while the 0.1.2 artifacts are being uploaded.

<p align="center">
  <a href="../../status_models_testing.md">
    <img src="../../docs/assets/mlx-one-supported-model-ecosystem.png" alt="MLX-One ecosystem illustration of the original 40-type 0.1.1 release cohort" width="1200" loading="lazy">
  </a>
</p>

The illustration records the original 40-type release cohort. The modality cards
above show the current 45-type registry.

## Find a guide

- **New here?** [Get started](getting-started.md): install the library, generate
  your first response and start the local server.
- **Choosing a checkpoint?** [Choose a model](models/index.md) for model-family
  guides, input formats and implementation scope.
- **Building an application?** Use the [Python and CLI reference](reference/api.md)
  or [serve an HTTP API](guides/serve.md).
- **Working with images or documents?** See [vision-language](guides/vision.md)
  and [OCR](guides/ocr.md).
- **Working with sound?** [Audio and speech](guides/audio.md) covers the available
  transcription, alignment and speech-generation paths.
- **Building search?** Start with [embeddings and reranking](guides/retrieval.md).
- **Segmenting or tracking?** Use the [dedicated SAM3 workflows](models/segmentation/sam3.md).
- **Planning memory?** [Inspect and plan](guides/memory.md) before loading weights.
- **Looking for measurements?** [All-modality performance](performance/index.md)
  includes text, VLM, audio, OCR, HTTP and task latency, with original evidence.

## In one paragraph

MLX-One implements native model workflows on Apple's MLX framework. Family
packages own model equations, weight contracts and processing; shared runtimes
manage generation, request lifecycle and caches. Python, CLI and HTTP interfaces
connect applications to that execution stack. Dense reference execution remains
the default, and optional [MLXBoost](experimental/mlxboost.md) acceleration has its
own experimental scope. See [architecture](reference/architecture.md) for the
implemented components and [experimental documentation](experimental/index.md)
for investigations, known failures and remaining gates.

Apache-2.0 · [GitHub](https://github.com/SSusantAchary/mlx-one) ·
[PyPI](https://pypi.org/project/mlx-one/0.1.2/) ·
[Citation and license](reference/citation.md)
