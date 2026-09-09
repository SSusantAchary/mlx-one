<p align="center">
  <img src="mlx-one.png" alt="mlx-one" width="1000">
</p>

<h1 align="center">mlx-one</h1>

<p align="center"><strong>Train locally. Prove it on Apple Silicon. Scale when needed.</strong></p>

<p align="center">
  A local-first CLI and Python package for inspecting, planning, training,
  evaluating, benchmarking, and qualifying MLX model workflows.
</p>

<p align="center">
  <a href="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml"><img src="https://github.com/SSusantAchary/mlx-one/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha status">
</p>

> [!IMPORTANT]
> The released package is `v0.1.0a1`. Current `main` contains experimental
> evaluation, benchmarking, MLX SFT workflows, and native architecture work.
> An architecture checkmark means synthetic structure tests pass; it does not
> mean that a real checkpoint, training run, or hardware profile is qualified.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Inspect a model](#inspect-a-model)
  - [Plan for Apple Silicon](#plan-for-apple-silicon)
  - [Validate a dataset](#validate-a-dataset)
  - [Fine-tune](#fine-tune)
  - [Evaluate and compare](#evaluate-and-compare)
  - [Benchmark](#benchmark)
- [Python API](#python-api)
- [Native architectures](#native-architectures)
- [Optional backends](#optional-backends)
- [Evidence and qualification](#evidence-and-qualification)
- [Reference M4 results](#reference-m4-results)
- [Project status](#project-status)
- [Development](#development)

## Installation

mlx-one requires Python 3.10 or newer. Install the current development version
from source:

```bash
git clone https://github.com/SSusantAchary/mlx-one.git
cd mlx-one
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development tools and tests:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Check the host, inspect a model without loading its weights, and create a
metadata-only memory plan:

```bash
mlx-one doctor

mlx-one inspect Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323

mlx-one plan inference \
  --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 \
  --hardware m4-air-32gb \
  --context-length 2048 \
  --batch-size 1
```

These commands do not initialize MLX or materialize model weights.

```text
Discover → Inspect → Plan → Train → Convert → Verify → Benchmark → Gate → Ship
```

## Usage

### Inspect a model

Inspect a Hugging Face repository or local model directory without executing
remote code or opening pickle-based weights:

```bash
mlx-one inspect mlx-community/Qwen3-0.6B-4bit
mlx-one inspect mlx-community/Qwen3-0.6B-4bit --revision main --json-output
mlx-one inspect ./local-model --output inspection.json
mlx-one inspect organization/cached-model --offline
```

Inspection reports model identity, transformer dimensions, formats, dtypes,
quantization, tokenizer or processor metadata, license information, access
restrictions, and cautious backend hints. Backend hints are candidates, not
compatibility claims.

### Plan for Apple Silicon

List bundled hardware profiles or detect the local machine:

```bash
mlx-one hardware list
mlx-one hardware show m4-air-32gb
mlx-one hardware detect --json-output
```

Estimate an inference or training workload without loading the model:

```bash
mlx-one plan inference \
  --model Qwen/Qwen2.5-1.5B \
  --hardware m4-air-32gb \
  --precision bf16 \
  --context-length 2048

mlx-one plan train \
  --model Qwen/Qwen2.5-3B \
  --hardware m4-air-32gb \
  --method auto \
  --context-length 2048 \
  --output plan.json
```

Plans include lower, central, and upper memory estimates, fit status,
confidence, assumptions, warnings, and calibration references. Estimates are
guidance, not proof that a workload will fit.

### Validate a dataset

The text pipeline accepts local JSON and JSONL records:

| Layout | Required fields |
| --- | --- |
| `instruction` | `instruction`, optional `input`, and `output` or `response` |
| `messages` | OpenAI-style `messages`, ending with an assistant response |
| `prompt-completion` | `prompt` and `completion` |
| `text` | `text` |

```bash
mlx-one data validate \
  --dataset examples/text-sft/train.jsonl \
  --layout prompt-completion
```

Add `--tokenizer MODEL` to preview token counts, truncation, and response-only
loss masks. Private record content is not printed.

### Fine-tune

> [!WARNING]
> Training on `main` is experimental. Run it locally on Apple Silicon, begin
> with the planner, and use a pinned model revision.

Run LoRA SFT with the included example configuration:

```bash
mlx-one train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset examples/text-sft/train.jsonl \
  --config examples/text-sft/train.yaml
```

The worker preserves an atomic run result, adapter checksum, base-model
ancestry, and checkpoint manifest. Export a validated adapter bundle with:

```bash
mlx-one export \
  --checkpoint runs/qwen-coder-sft/adapters/mlx-one-checkpoint.json \
  --output artifacts/qwen-coder-sft
```

Run the complete reference lifecycle as a resumable workflow:

```bash
mlx-one workflow text-sft \
  --config examples/text-sft/qwen2.5-coder-1.5b.workflow.yaml \
  --resume
```

See the [text SFT example](examples/text-sft/README.md) for configuration,
evaluation, and quality-gate files.

### Evaluate and compare

Evaluate precomputed predictions without loading a model:

```bash
mlx-one evaluate \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --dataset examples/text-sft/eval.jsonl \
  --predictions examples/text-sft/predictions-smoke.json \
  --runs-dir runs \
  --run-id scoring-smoke \
  --json-output
```

Omit `--predictions` to generate with the isolated MLX worker. Add `--adapter`
to evaluate a fine-tuned adapter.

```bash
mlx-one compare \
  runs/base/result.json \
  runs/candidate/result.json \
  --gate examples/text-sft/quality-gate.yaml \
  --format markdown \
  --output comparison.md
```

Comparisons require matching profiles, dataset revisions, and generation
settings. Incompatible runs can be inspected but cannot pass qualification.

### Benchmark

Measure load time, prompt throughput, decode throughput, wall time, and peak
Metal memory in an isolated process:

```bash
mlx-one benchmark inference \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --prompts examples/text-sft/prompts.json \
  --max-tokens 32 \
  --repeats 3 \
  --runs-dir runs
```

Create a calibration plan without running its workloads:

```bash
mlx-one calibrate text \
  --matrix m4-air-32gb-text-v1 \
  --output calibration-results \
  --dry-run
```

Remove `--dry-run` only on matching reference hardware.

## Python API

Inspect a model and estimate memory:

```python
from mlx_one import (
    Precision,
    WorkloadKind,
    WorkloadSpec,
    estimate_memory,
    inspect_model,
    load_hardware_profile,
)

model = inspect_model(
    "Qwen/Qwen2.5-1.5B",
    revision="8faed761d45a263340a0528343f099c05c9a4323",
).model
hardware = load_hardware_profile("m4-air-32gb")
workload = WorkloadSpec(
    kind=WorkloadKind.INFERENCE,
    precision=Precision.BF16,
    context_length=2048,
)

estimate = estimate_memory(model, hardware, workload)
print(estimate.to_json())
```

Use the native training contract:

```python
from mlx_one import TrainConfig
from mlx_one.training import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
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

The supported Unsloth-shaped compatibility facade maps to the same MLX
primitives. Unsupported arguments fail instead of being silently ignored.

```python
from mlx_one import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
    max_seq_length=512,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
```

## Native architectures

Current `main` contains internal, architecture-first MLX implementations for:

| Registry type | Family coverage | Architecture | Checkpoint qualification |
| --- | --- | :---: | --- |
| `qwen2` | Qwen2, Qwen2.5, Qwen2.5-Coder | ✅ | Separate per model and revision |
| `qwen3` | Qwen3 dense | ✅ | Candidate |
| `qwen2_moe` | Qwen2-MoE | ✅ | Unqualified |
| `qwen2_vl` | Qwen2-VL text and vision stack | ✅ | Verify exact revision and processor |
| `openelm` | OpenELM 270M, 450M, 1.1B, and 3B | ✅ | Verify exact release |

These implementations are validated with tiny random configurations, cache and
shape tests, registry tests, and strict synthetic weight contracts. They remain
internal in this phase: real checkpoint loading, tokenizer and processor wiring,
generation routing, parity testing, quantization, and training integration are
separate qualification work.

See [model_list.txt](model_list.txt) for the complete architecture tracker and
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for package ownership and the
target native layout.

## Optional backends

MLX-LM is the default text runtime. Install only the additional modality
integrations you need:

```bash
python -m pip install -e ".[vlm]"         # mlx-vlm
python -m pip install -e ".[audio]"       # mlx-audio
python -m pip install -e ".[embeddings]"  # mlx-embeddings
python -m pip install -e ".[multimodal]"  # all modality backends
python -m pip install -e ".[tracking]"    # optional MLflow integration
```

Quote extras in `zsh` so square brackets are not expanded as a glob.

## Evidence and qualification

mlx-one records support for an exact model revision, operation, precision,
workload, runtime, and hardware profile—not for a model name alone.

| State | Meaning |
| --- | --- |
| `candidate` | Metadata suggests a possible integration; not tested |
| `upstream-documented` | The execution backend documents support |
| `integration-tested` | The mlx-one adapter contract passed |
| `hardware-verified` | A checksummed device run passed its safety policy |
| `quality-verified` | A pinned evaluation protocol passed |
| `unsupported` | The exact operation is known not to work |
| `deprecated` | Previously supported evidence is no longer current |

```bash
mlx-one registry list
mlx-one registry list --model Qwen/Qwen2.5-1.5B --operation benchmark
mlx-one registry validate registry.json
```

Measured and estimated metrics remain distinct. Failed, interrupted, skipped,
or safety-invalidated runs cannot be promoted to verified capability entries.
See the [candidate model catalog](docs/model-catalog.md).

## Reference M4 results

The first public calibration matrix used an M4 MacBook Air with 32 GiB unified
memory and a 24 GiB MLX process budget. The table below reports measured
2048-token device performance at batch size 1; it is not a model-quality score.

| Model | BF16 inference | 4-bit inference | BF16 LoRA | 4-bit QLoRA |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | 88.7 tok/s · 1.49 GiB | 233.2 tok/s · 1.02 GiB | 894.3 tok/s · 7.54 GiB | 798.5 tok/s · 6.89 GiB |
| Qwen2.5 1.5B | 30.5 tok/s · 3.38 GiB | 93.0 tok/s · 1.51 GiB | 393.4 tok/s · 11.17 GiB | 333.9 tok/s · 9.11 GiB |
| Qwen2.5 3B | 15.3 tok/s · 6.20 GiB | 50.3 tok/s · 2.26 GiB | 176.5 tok/s · 15.62 GiB | 157.8 tok/s · 11.49 GiB |

Training and inference throughput measure different work and must not be
compared with each other. All 2048-token matrix workloads completed. At 4096
tokens, 0.5B and 1.5B BF16 LoRA runs were invalidated by swap safety, and both
3B training workers failed.

See the [calibration methodology](docs/calibration.md) and
[published result records](calibration/results/m4-air-32gb-text-v1).

The first end-to-end LoRA qualifications also verified adapter reload and a
deterministic inference smoke test:

| Model | Training loss | Throughput | Peak memory | Support state |
| --- | ---: | ---: | ---: | --- |
| [Qwen2.5-Coder 1.5B Instruct](qualification/results/qwen2.5-coder-1.5b-lora-m4-air-32gb.json) | 6.522 → 0.004 | 25.18 tok/s | 3.51 GiB | Hardware-verified |
| [SmolLM2 1.7B Instruct](qualification/results/smollm2-1.7b-lora-m4-air-32gb.json) | 6.771 → 0.005 | 22.85 tok/s | 3.82 GiB | Hardware-verified |

Their two-sample held-out coding comparisons did not pass the quality gate, so
neither model is quality-verified.

## Project status

| Milestone | Outcome | Status |
| --- | --- | --- |
| M0 / `v0.1.0a1` | Diagnostics, inspection, schemas, planning, calibration | ✅ released |
| M1 | Evidence registry, local runs, text evaluation, comparison | 🚧 experimental on `main` |
| M2 | Owned MLX SFT, LoRA, QLoRA, checkpoints, export | 🚧 LoRA path hardware-verified; quality gate pending |
| Native models | Qwen2, Qwen3, Qwen2-MoE, Qwen2-VL, OpenELM structures | 🚧 architecture-only |
| M3 | CUDA source portability and conversion parity | 📋 planned |
| M4–M6 | Embeddings, retrieval, VLM, OCR, ASR, and TTS workflows | 📋 planned |
| M7+ | Optimization, advanced training, release qualification | 📋 planned |

The current 3B boundary reflects the maintainer's M4 32 GiB validation target,
not a package API limit. See the [full roadmap](docs/roadmap.md).

## Development

Run the backend-free quality gates:

```bash
ruff check .
pytest -q
python -m build --no-isolation
python -m twine check dist/*
```

Model execution and hardware tests are opt-in and require Apple Silicon with
Metal access. Ordinary package imports and backend-free tests do not initialize
Metal or access the network.

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request. Report vulnerabilities through
[SECURITY.md](SECURITY.md), not a public issue.

Released under the [MIT License](LICENSE).
