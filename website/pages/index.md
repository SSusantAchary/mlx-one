---
title: Home
---

# MLX-One

**One stack. Native MLX. Built for Apple Silicon.** MLX-One brings text generation,
vision-language inference, audio, OCR, embeddings and segmentation into one native
MLX library. Load a model, run it from Python or the CLI, and serve a local chat UI
and HTTP API. Native model implementations share checkpoint loading, execution and
developer tooling on Apple Silicon.

```bash
python -m pip install "mlx-one==0.1.0a1"
```

```bash
# Inspect a checkpoint without loading its weights.
mlx-one inspect openai-community/gpt2

# Generate text through the native MLX runtime.
mlx-one generate openai-community/gpt2 "The future of local AI is" \
  --max-tokens 64 --temperature 0.8 --seed 0

# Start the local chat UI and an OpenAI-compatible HTTP API.
python -m pip install "mlx-one[server]==0.1.0a1"
mlx-one serve LiquidAI/LFM2-350M --alias lfm2-local \
  --host 127.0.0.1 --port 8888 --inference-mode reference_dense
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Choose a model](models/index.md){ .md-button }
[How it works](reference/architecture.md){ .md-button }

## What it makes possible

| Workflow | Native capabilities | Start here |
| --- | --- | --- |
| Text generation | Cached decoding, sampling, chat templates and token streaming | [Generate text](guides/generate.md) |
| Local applications | Python APIs, CLI workflows and HTTP/SSE serving | [Serve an API and chat UI](guides/serve.md) |
| Vision-language | Supported static-image and text workflows | [Vision-language guide](guides/vision.md) |
| Audio and speech | ASR, separate known-text alignment and experimental speech generation | [Audio guide](guides/audio.md) |
| Documents | Line recognition and implemented page-model OCR workflows | [OCR guide](guides/ocr.md) |
| Retrieval | Embeddings, similarity and Qwen3 reranking | [Embeddings and reranking](guides/retrieval.md) |
| Segmentation and tracking | SAM3 image prompts and experimental video tracking | [SAM3 guide](models/segmentation/sam3.md) |

Exact checkpoint and interface coverage varies. The [model index](models/index.md)
and [checkpoint inventory](reference/checkpoints.md) record those boundaries;
[performance reports](performance/index.md) keep measured workloads separate from
broader qualification claims.

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
[PyPI](https://pypi.org/project/mlx-one/0.1.0a1/) ·
[Citation and license](reference/citation.md)
