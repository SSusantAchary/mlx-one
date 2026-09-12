<p align="center">
  <img src="mlx-one.png" alt="mlx-one" width="1000">
</p>

<h1 align="center">mlx-one</h1>

<p align="center"><strong>Train locally. Prove it on Apple Silicon. Scale when needed.</strong></p>

<p align="center">
  A unified, native MLX stack for language, vision-language, embeddings,
  audio-language, and speech models on Apple Silicon.
</p>

<p align="center">
  <a href="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml"><img src="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha status">
</p>

> [!IMPORTANT]
> mlx-one is under active development. Native architecture support and model
> qualification are tracked separately. A family marked Architecture ✅ has a
> validated config, MLX model structure, registry entry, strict weight contract,
> and synthetic execution tests. It does not automatically mean every checkpoint,
> task, precision, or device has been qualified.

## Why mlx-one?

MLX is an excellent foundation for machine learning on Apple Silicon, but model
workflows are often split across separate packages and incompatible task APIs.
mlx-one is building those pieces as one coherent stack:

```text
Python API + CLI
       ↓
Tasks: generation · embeddings · VLM · ASR · training
       ↓
Native model families + shared processors + safe loading
       ↓
Evaluation · benchmarking · evidence · qualification
       ↓
MLX on Apple Silicon
```

The project owns its supported model math directly. Hugging Face is used as the
artifact ecosystem for configuration, tokenizer/processor assets, and safe
`safetensors` checkpoints—not as a remote-code execution runtime.

## What is implemented

- Native MLX architectures across language, vision-language, embeddings,
  audio-language, and ASR.
- Shared attention, GQA, RoPE, normalization, feed-forward, MoE, vision, hybrid
  convolution, masking, and KV-cache primitives.
- Lazy metadata-driven model registration without initializing Metal during
  lightweight package imports.
- Strict configuration validation and deterministic weight contracts that reject
  unknown or shape-incompatible tensors.
- Safe local and pinned Hugging Face artifact inspection.
- Native Whisper loading, audio preprocessing, tokenization, decoding, language
  detection, segment timestamps, word timestamps, and WER/CER evaluation.
- Unified native text loading, tokenizer/chat templates, MLX 4-bit checkpoints,
  cached greedy and seeded sampling, stop sequences, token streaming, and serving
  for GPT-2, Qwen2/Qwen3, LFM2, OpenELM, and text-only Qwen3.5 families.
- Native Qwen3 embedding and reranking with byte-level BPE, safe loading,
  padding-aware batching, Matryoshka dimensions, normalization, cosine similarity,
  yes/no pair scoring, and stable ranking.
- Dataset validation, memory planning, LoRA/QLoRA SFT workflows, adapter export,
  evaluation, comparison, benchmarking, and evidence records.
- Backend-free tests plus opt-in Metal and real-checkpoint integration gates.

## Installation

mlx-one requires Python 3.10 or newer and an Apple Silicon Mac for model
execution.

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

FFmpeg is required only when the Whisper API receives an audio file. Already
decoded mono 16-kHz waveforms can be passed directly from Python.

## Quick start

### Native server and Web UI

Load one supported text model and start the bundled local UI:

```bash
mlx-one serve mlx-community/Qwen3.5-0.8B-4bit
```

The native server also supports model aliases, optional Bearer authentication, effective context
limits, request deadlines, cooperative parallel slots, prompt/context caching, context shifting,
reasoning output, quantized KV caches, and Qwen3.5 MTP speculative decoding. Run
`mlx-one serve --help` for the complete flag list.

Then open `http://127.0.0.1:8080`. The same process exposes an OpenAI-compatible API at
`http://127.0.0.1:8080/v1`, including streaming chat completions. Model execution remains inside
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

## Native model coverage

### Language models

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `gpt2` | GPT-2 124M, 355M, 774M, 1.5B | Learned positions, fused QKV, byte BPE, safe loading, cache, generation and streaming | ✅ | Candidate; synthetic validation only |
| `qwen2` | Qwen2, Qwen2.5, Qwen2.5-Coder | Dense Transformer, GQA, RoPE, cache | ✅ | Per checkpoint |
| `qwen3` | Qwen3 dense | Bias-free attention, Q/K norm, explicit head dimensions | ✅ | Candidate |
| `qwen2_moe` | Qwen2-MoE | Top-k experts, shared expert, router outputs | ✅ | Architecture only |
| `openelm` | OpenELM 270M–3B | Layer-wise heads/FFN widths, fused QKV, GQA | ✅ | Verify release |
| `lfm2` | LFM2 and LFM2.5 | Hybrid convolution/attention decoder | ✅ | Per checkpoint |
| `lfm2_moe` | LFM2 MoE | Hybrid decoder and sparse expert routing | ✅ | Excluded from ≤3B qualification |

### Vision-language models

| Registry type | Family | Native components | Architecture | Qualification |
| --- | --- | --- | :---: | --- |
| `qwen2_vl` | Qwen2-VL | Vision Transformer, 3D patches, merger, multimodal RoPE, image/video token insertion | ✅ | Verify exact checkpoint and processor |
| `qwen2_5_vl` | Qwen2.5-VL-3B | Window/full vision attention, biased SwiGLU vision tower, merger, multimodal RoPE | ✅ | Architecture only; verify checkpoint and processor |
| `qwen3_5` | Qwen3.5 0.8B/2B | Hybrid gated-delta/full-attention text tower, learned/interpolated vision positions, multimodal RoPE | ✅ | Architecture only; synthetic validation |
| `lfm2_vl` | LFM2.5-VL | Vision tower, pixel unshuffle/projector, hybrid language model | ✅ | Verify exact checkpoint and processor |

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
| `whisper` | tiny, base, small, medium, large-v3, turbo | Encoder-decoder, safe loading, log-Mel, BPE, decoding, language, timestamps, WER/CER | ✅ | Candidate; pinned tiny/turbo smoke gates pass |
| `lfm2_audio` | LFM2.5-Audio | Audio encoder, Conformer, Depthformer, detokenizer, feature insertion | ✅ | Verify processor and codec weights |

The detailed family backlog and exact status are maintained in
[model_list.txt](model_list.txt). Package ownership and dependency boundaries
are defined in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

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
- Ruff, source/wheel builds, and package metadata checks.

These counts describe the current development tree and will change as coverage
expands.

Run backend-free checks:

```bash
ruff check .
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

See [mlxone_roadmap.md](mlxone_roadmap.md) for milestone exit criteria and
[mlx_one_full.md](mlx_one_full.md) for the complete product plan.

## Contributing

Contributions should include the smallest relevant config, synthetic execution,
weight-contract, integration, and documentation updates. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Report security
issues through [SECURITY.md](SECURITY.md), not a public issue.

Released under the [MIT License](LICENSE).
