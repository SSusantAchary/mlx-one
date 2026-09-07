# mlx-one

Train anywhere. Prove it on Apple Silicon.

`mlx-one` is building an open, model-size-agnostic lifecycle for training through
pluggable backends, converting models to MLX, evaluating text, vision-language,
ASR, and TTS models, comparing quality and device performance, and packaging the
evidence needed to ship a model.

> Status: `v0.1.0a1` foundation. Diagnostics, metadata-only inspection, schemas,
> and model-loading utilities are available; evaluation, comparison, merging, and
> release bundles are milestone work and are not advertised as complete.

## Development installation

```bash
git clone https://github.com/SSusantAchary/mlx-one.git
cd mlx-one
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional model backends are installed independently:

```bash
python -m pip install -e ".[vlm]"         # mlx-vlm
python -m pip install -e ".[audio]"       # mlx-audio
python -m pip install -e ".[embeddings]"  # mlx-embeddings
python -m pip install -e ".[multimodal]"  # all modality backends
python -m pip install -e ".[tracking]"    # optional local MLflow integration
```

## Check your Mac

Run diagnostics without importing MLX into the CLI process:

```bash
mlx-one doctor
mlx-one doctor --json-output
```

The report covers the host architecture, chip, unified memory, Python version,
Metal availability, and installed MLX modality backends. MLX model execution is
supported on Apple Silicon macOS. Model size is not an eligibility restriction;
compatibility evidence is attached to the exact model, configuration, workload,
and hardware used for each verified run.

## Inspect model metadata

Inspect a Hugging Face repository or local model directory without loading model
weights, importing a model runtime, or executing remote code:

```bash
mlx-one inspect mlx-community/Qwen3-0.6B-4bit
mlx-one inspect mlx-community/Qwen3-0.6B-4bit --revision main --json-output
mlx-one inspect ./local-model --output inspection.json
mlx-one inspect organization/cached-model --offline
```

Inspection reports the resolved revision, modality, architecture, parameter-count
provenance, weight format/size/dtype, quantization, tokenizer/processor, license,
access restrictions, remote-code requirements, cautious backend hints, and any
missing metadata. Backend hints identify candidates only; they are not verified
compatibility claims.

## Plan hardware fit

Inspect the bundled M4 MacBook Air profile or detect the current machine without
collecting serial numbers, hardware UUIDs, usernames, or private paths:

```bash
mlx-one hardware list
mlx-one hardware show m4-air-32gb
mlx-one hardware detect --json-output
```

Generate metadata-only inference and training plans. Planning does not load model
weights or initialize MLX:

```bash
mlx-one plan inference --model Qwen/Qwen2.5-1.5B \
  --hardware m4-air-32gb --context-length 2048 --batch-size 1

mlx-one plan train --model Qwen/Qwen2.5-3B \
  --hardware m4-air-32gb --method auto --context-length 2048 \
  --output plan.json
```

The estimator reports a lower, central, and upper peak-memory range, the process
budget, fit status, confidence, assumptions, and warnings. `auto` considers full
tuning, BF16 LoRA, then 4-bit QLoRA. Estimates are planning guidance, not verified
compatibility or a guarantee that a workload will fit.

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
print(estimate_memory(model, hardware, workload).to_json())
```

## Calibrate the estimator

The bundled `m4-air-32gb-text-v1` matrix pins Qwen2.5 0.5B, 1.5B, and 3B
revisions. It covers BF16/4-bit inference and BF16 LoRA/4-bit QLoRA at contexts
512, 1024, 2048, and 4096. Validate its 48 cells without downloading weights:

```bash
mlx-one calibrate text --matrix m4-air-32gb-text-v1 \
  --output calibration-results/ --dry-run
```

Run it from a local Apple Silicon terminal with Metal available:

```bash
mlx-one calibrate text --matrix m4-air-32gb-text-v1 \
  --output calibration-results/ --resume
```

The first real run downloads and converts all three checkpoints and therefore
requires substantial disk space and time. Model artifacts stay under the user
cache and are never added to results. Each workload runs in an isolated process;
unsafe cells are skipped at preflight, and execution stops when swap growth or
system memory pressure crosses the safety policy. See
[the calibration methodology](docs/calibration.md) before publishing results.

The maintainer's first through-3B run is stored under
[`calibration/results/m4-air-32gb-text-v1`](calibration/results/m4-air-32gb-text-v1).
All 24 inference cells completed. Across the full matrix, 44 cells completed, two
4,096-token BF16 LoRA cells were invalidated for excessive swap, and the 3B
LoRA/QLoRA 4,096-token workers failed. Failures remain part of the evidence.

### Measured model scores and compatibility

The following are **device-performance scores**, not model-quality benchmark
scores. They were measured on the reference M4 MacBook Air with 32 GiB unified
memory and a 24 GiB MLX process budget. Every score below uses batch size 1 and a
2,048-token context; inference scores are warm decode throughput and training
scores are warm optimizer-step throughput.

| Model | BF16 inference | 4-bit inference | BF16 LoRA | 4-bit QLoRA |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | ✅ 88.7 tok/s · 1.49 GiB | ✅ 233.2 tok/s · 1.02 GiB | ✅ 894.3 tok/s · 7.54 GiB | ✅ 798.5 tok/s · 6.89 GiB |
| Qwen2.5 1.5B | ✅ 30.5 tok/s · 3.38 GiB | ✅ 93.0 tok/s · 1.51 GiB | ✅ 393.4 tok/s · 11.17 GiB | ✅ 333.9 tok/s · 9.11 GiB |
| Qwen2.5 3B | ✅ 15.3 tok/s · 6.20 GiB | ✅ 50.3 tok/s · 2.26 GiB | ✅ 176.5 tok/s · 15.62 GiB | ✅ 157.8 tok/s · 11.49 GiB |

`tok/s · GiB` means measured throughput followed by peak Metal memory. Training
and inference throughput measure different work and should not be compared with
each other.

The longest-context compatibility results show where additional infrastructure
or lower-memory settings are required:

| Model | 4,096-token inference | 4,096-token BF16 LoRA | 4,096-token 4-bit QLoRA | Infrastructure guidance |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B | ✅ BF16 and 4-bit | ⚠️ Invalid: swap limit | ✅ 16.99 GiB peak | M4 32 GB verified; use 2,048 tokens for the safer BF16 LoRA path |
| Qwen2.5 1.5B | ✅ BF16 and 4-bit | ⚠️ Invalid: swap limit | ✅ 20.44 GiB peak | M4 32 GB verified; 4,096-token QLoRA runs close to the 24 GiB process budget |
| Qwen2.5 3B | ✅ BF16 and 4-bit | ❌ Worker failed | ❌ Worker failed | M4 32 GB verified for training through 2,048 tokens; reduce context/activation memory or use larger training infrastructure for 4,096 tokens |

Legend: ✅ completed measurement; ⚠️ completed but invalidated by the safety
policy; ❌ failed on the reference workload. These statuses apply only to the
pinned revisions and exact workload above, not every model with the same
parameter count. Other Apple Silicon, NVIDIA, AMD, and TPU configurations can be
added as contributor evidence without changing the model-size-agnostic API.

Model-quality evaluation scores are not available yet. They will be published
with pinned datasets and evaluator versions when the M1 text evaluation workflow
lands; until then, `mlx-one` does not present performance measurements as quality.

## Current Python utilities

```python
from mlx_one.utils.model_loader import (
    load_audio_model,
    load_embedding_model,
    load_model,
    load_vlm_model,
)

model, tokenizer = load_model("mlx-community/Qwen3-0.6B-4bit")
vlm, processor = load_vlm_model("mlx-community/Qwen2-VL-2B-Instruct-4bit")
asr = load_audio_model("mlx-community/parakeet-tdt-0.6b-v3", task="stt")
embedding_model, tokenizer = load_embedding_model(
    "mlx-community/all-MiniLM-L6-v2-4bit"
)
```

Optional loaders raise an actionable error identifying the required package extra
when their backend is missing.

## Versioned schemas

The CLI and Python API share backend-free, versioned records for model identity,
hardware, run configuration, results, and compatibility evidence:

```python
from mlx_one import HardwareSpec, Modality, ModelSpec, Operation, RunSpec, inspect_model

model = ModelSpec(
    model_id="organization/model",
    revision="exact-model-revision",
    modality=Modality.TEXT,
    parameter_count=8_000_000_000,
)
hardware = HardwareSpec(
    profile_id="m4-air-32gb",
    platform="macOS",
    architecture="arm64",
    chip="Apple M4",
    memory_bytes=32 * 1024**3,
)
run = RunSpec(
    run_id="example-run",
    operation=Operation.EVALUATE,
    model=model,
    hardware=hardware,
    seed=42,
)

print(run.to_json())

inspection = inspect_model("organization/model", revision="main")
print(inspection.to_json())
```

Parameter count is descriptive metadata, not an eligibility limit. Schema `1.0`
rejects incompatible versions and non-JSON metadata rather than silently losing
provenance.

## Roadmap

| Milestone | Outcome | Status |
| --- | --- | --- |
| M0 / `v0.1.0a1` | Reliable foundation and diagnostics | 🚧 in progress |
| M1 / `v0.2.0` | Unified text quality and performance | 📋 planned |
| M2 / `v0.3.0` | Vision-language evaluation | 📋 planned |
| M3 / `v0.4.0` | ASR and TTS evaluation | 📋 planned |
| M4 / `v0.5.0` | MLX-native text-model merging | 📋 planned |
| M5 / `v1.0.0` | Quality gates and release bundles | 📋 planned |

Milestones advance only after their implementation, validation, and documentation
are complete. Unfinished functionality remains experimental and is not presented
as shipped.

## Development checks

```bash
ruff check .
pytest -q
python -m build
python -m twine check dist/*
```

Model-loading and reference benchmark runs require a real Apple Silicon terminal
with Metal access and are kept separate from backend-free unit tests.

## Contributing and security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a
pull request. Report security issues using [SECURITY.md](SECURITY.md), not a public
issue.

Released under the [MIT License](LICENSE).
