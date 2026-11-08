---
title: Home
---

# MLX-One

**One stack. Native MLX. Built for Apple Silicon.** MLX-One brings text generation,
vision-language inference, audio, OCR, embeddings and segmentation into one native
MLX library. Load a model, run it from Python or the CLI, and serve a local chat UI
and HTTP API. Native model implementations share checkpoint loading, execution and
developer tooling on Apple Silicon.

**0.1.1 approved for release · 8 November 2026.** All registered model types have
passing primary-workflow records. [Release notes](releases/0.1.1.md) ·
[Supported models and known limits](models/support.md).

<!-- insights:summary -->

The installation examples target **0.1.1**. Publication is pending; until it is
available on PyPI, use the [source installation](getting-started.md#source-installation-for-development).

```bash
python -m pip install "mlx-one==0.1.1"
```

```bash
# Inspect a checkpoint without loading its weights.
mlx-one inspect openai-community/gpt2

# Generate text through the native MLX runtime.
mlx-one generate openai-community/gpt2 "The future of local AI is" \
  --max-tokens 64 --temperature 0.8 --seed 0

# Start the local chat UI and an OpenAI-compatible HTTP API.
python -m pip install "mlx-one[server]==0.1.1"
mlx-one serve LiquidAI/LFM2-350M --alias lfm2-local \
  --host 127.0.0.1 --port 8888 --inference-mode reference_dense
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Supported models](models/support.md){ .md-button }
[How it works](reference/architecture.md){ .md-button }

## What it makes possible

<!-- insights:modalities -->

[Explore model coverage, benchmarks and development progress](insights/index.md){ .md-button }

Counts come from the verified support catalog and native registry on each build.
Each type has a pinned checkpoint, precision, passing task record and applicable
interface checks. Checkpoint sizes and quantizations share a model type.

| Workflow | Native capabilities | Start here |
| --- | --- | --- |
| Text generation | Cached decoding, sampling, chat templates and token streaming | [Generate text](guides/generate.md) |
| Local applications | Python APIs, CLI workflows and HTTP/SSE serving | [Serve an API and chat UI](guides/serve.md) |
| Vision-language | Supported static-image and text workflows | [Vision-language guide](guides/vision.md) |
| Audio and speech | ASR, known-text alignment, voice activity detection and Q8 speech generation | [Audio guide](guides/audio.md) |
| Documents | Line recognition and implemented page-model OCR workflows | [OCR guide](guides/ocr.md) |
| Retrieval | Embeddings, similarity and Qwen3 reranking | [Embeddings and reranking](guides/retrieval.md) |
| Segmentation and tracking | SAM3 image prompts and native offline/streaming video tracking | [SAM3 guide](models/segmentation/sam3.md) |
| Classification | Sentiment, named entities and zero-shot image classification | [Classification guides](models/index.md#classification-guides) |

The [support inventory](models/support.md) records exact checkpoints and operations.
[Performance reports](performance/index.md) retain measured workloads, and
[known quality and scope limits](models/support.md#known-quality-and-scope-limitations)
remain visible alongside support.

## Find a guide

- **New here?** [Get started](getting-started.md): install from PyPI, generate
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
[PyPI](https://pypi.org/project/mlx-one/0.1.1/) ·
[Citation and license](reference/citation.md)
