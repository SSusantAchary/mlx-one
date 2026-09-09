# mlx-one Project Structure

> Structural contract for the native mlx-one stack.
>
> This document defines ownership, dependency direction, target paths, testing
> layout, and incremental migration from the current flat package. It complements
> `mlx_one_full.md`, which remains the complete product and implementation plan.

## 1. Structural goals

The repository must keep three concerns separate:

```text
native model implementation
        ↓
task execution and training
        ↓
lifecycle evidence and user workflows
```

The structure should make these rules obvious:

- MLX is the only required model-execution foundation.
- Supported architecture code lives inside `mlx_one.models`.
- Shared operations are implemented once and reused across families.
- Hugging Face artifact handling is separate from model math.
- Forward execution is separate from token or media generation.
- Training and inference use the same model classes.
- Public APIs delegate to internal services; CLI code contains no model logic.
- Evidence, hardware planning, and release lineage surround every modality.
- Optional external runtimes remain isolated migration/parity adapters.

## 2. Repository layout

```text
mlx-one/
├── .github/                    # CI, release workflows, issue templates
├── calibration/               # Maintainer calibration matrices and evidence
├── docs/                      # Public architecture and workflow documentation
├── examples/                  # Small reproducible user workflows
├── qualification/             # Hardware and quality qualification records
├── src/
│   └── mlx_one/               # Installable Python package
├── tests/                     # Backend-free, integration, parity, hardware tests
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

Generated weights, model caches, private datasets, verbose logs, and local run
outputs do not belong in the source tree.

## 3. Target package layout

```text
src/mlx_one/
├── __init__.py                # Stable public exports only
├── _version.py
│
├── core/                      # Lowest-level contracts and runtime primitives
│   ├── config.py
│   ├── capabilities.py
│   ├── model.py
│   ├── outputs.py
│   ├── batches.py
│   ├── cache.py
│   ├── registry.py
│   ├── loading.py
│   ├── serialization.py
│   ├── quantization.py
│   ├── generation.py
│   ├── sampling.py
│   ├── errors.py
│   └── typing.py
│
├── models/                    # Native MLX architecture implementations
│   ├── shared/
│   │   ├── attention.py
│   │   ├── activations.py
│   │   ├── masks.py
│   │   ├── mlp.py
│   │   ├── moe.py
│   │   ├── norms.py
│   │   ├── position.py
│   │   └── rope.py
│   ├── language/
│   │   ├── qwen2/
│   │   ├── qwen2_moe/
│   │   ├── qwen3/
│   │   ├── openelm/
│   │   ├── llama/
│   │   ├── gemma/
│   │   ├── smollm/
│   │   └── phi/
│   ├── embeddings/
│   │   ├── minilm/
│   │   ├── bge/
│   │   └── qwen_embedding/
│   ├── vision/
│   │   ├── clip/
│   │   └── siglip/
│   ├── vision_language/
│   │   ├── smolvlm/
│   │   ├── qwen2_vl/
│   │   └── paligemma/
│   └── audio/
│       ├── whisper/
│       ├── parakeet/
│       └── kokoro/
│
├── processors/                # Text, image, audio, and multimodal preprocessing
│   ├── base.py
│   ├── text.py
│   ├── image.py
│   ├── audio.py
│   ├── multimodal.py
│   └── registry.py
│
├── generation/                # Task-level decoding, separate from model forward
│   ├── text.py
│   ├── multimodal.py
│   ├── audio.py
│   ├── sampling.py
│   ├── logits.py
│   ├── stopping.py
│   └── streaming.py
│
├── training/                  # Native training services
│   ├── trainer.py
│   ├── sft.py
│   ├── lora.py
│   ├── qlora.py
│   ├── losses.py
│   ├── datasets.py
│   ├── collators.py
│   ├── schedulers.py
│   ├── checkpoint.py
│   └── adapters.py
│
├── embeddings/                # Embedding and reranking task workflows
│   ├── pooling.py
│   ├── similarity.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── losses.py
│   └── evaluation.py
│
├── audio/                     # Shared ASR/TTS runtime services
│   ├── codecs.py
│   ├── resampling.py
│   ├── chunking.py
│   ├── streaming.py
│   └── generation.py
│
├── inspect/                   # Metadata-only artifact inspection
│   ├── huggingface.py
│   ├── local.py
│   ├── safetensors.py
│   └── service.py
│
├── hardware/                  # Detection and workload-aware planning
│   ├── detect.py
│   ├── profiles.py
│   ├── memory.py
│   ├── planner.py
│   └── calibration.py
│
├── evaluation/                # Metrics and reproducible evaluation profiles
│   ├── common/
│   ├── text/
│   ├── retrieval/
│   ├── vlm/
│   ├── asr/
│   └── tts/
│
├── benchmark/                 # Performance measurement, not model execution
│   ├── inference.py
│   ├── training.py
│   ├── memory.py
│   └── profiler.py
│
├── registry/                  # Architecture resolution and support evidence
│   ├── architectures.py
│   ├── processors.py
│   ├── capabilities.py
│   ├── evidence.py
│   └── service.py
│
├── lifecycle/                 # Cross-task reproducibility and release services
│   ├── schemas.py
│   ├── migrations.py
│   ├── provenance.py
│   ├── runs.py
│   ├── compare.py
│   ├── gates.py
│   ├── verify.py
│   ├── export.py
│   ├── release.py
│   └── lineage.py
│
├── workers/                   # Isolated subprocess entry points
│   ├── benchmark.py
│   ├── calibration.py
│   ├── conversion.py
│   ├── generation.py
│   └── training.py
│
├── compat/                    # Thin optional compatibility and fallback adapters
│   ├── unsloth.py
│   ├── trl.py
│   └── external_runtime.py
│
├── cli/                       # Argument parsing and presentation only
│   ├── main.py
│   ├── doctor.py
│   ├── inspect.py
│   ├── plan.py
│   ├── run.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── export.py
│   └── registry.py
│
└── data/                      # Small package-owned static records
    ├── hardware/
    ├── calibrations/
    └── capabilities.json
```

Every package directory contains an `__init__.py`, but internal package
`__init__` files should remain small and avoid eager MLX initialization.

## 4. Package ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `core` | Base types, caches, registry/loading contracts, outputs, generation and quantization interfaces | Concrete model families, CLI behavior, evaluation policy |
| `models` | Native architecture math and family-specific weight transforms | Dataset workflows, benchmarking, release logic |
| `processors` | Artifact-to-standard-batch preprocessing | Model forward execution or evaluation scoring |
| `generation` | Decoding, sampling, stopping, streaming | Architecture-specific Transformer blocks |
| `training` | Loss/update loops, PEFT, datasets, checkpoints | Separate training-only model implementations |
| `embeddings` | Pooling, similarity, retrieval, reranking, and embedding losses | Encoder architecture implementations |
| `audio` | Shared preprocessing, codecs, streaming, and ASR/TTS task runtime | Audio model architecture implementations |
| `inspect` | Safe metadata and tensor-header inspection | Weight materialization or remote-code execution |
| `hardware` | Detection, calibrated estimates, fit planning | Support promotion or model-quality claims |
| `evaluation` | Profiles, metrics, normalization, task results | Model loading conventions or CLI parsing |
| `benchmark` | Timing and memory measurement | Model correctness or quality qualification |
| `registry` | Architecture/processor resolution and evidence queries | Model-name-driven runtime branching |
| `lifecycle` | Schemas, provenance, runs, gates, verification, releases | Model architecture code |
| `workers` | Process isolation and protocol boundaries | Business logic duplicated from services |
| `compat` | Thin compatibility facades and temporary external adapters | Alternative authoritative implementations |
| `cli` | Commands, validation messages, rendering | Direct tensor/model implementation |

## 5. Dependency direction

Allowed dependency flow:

```text
CLI
 ↓
lifecycle / evaluation / benchmark / training / embeddings / audio
 ↓
generation / processors / registry
 ↓
models
 ↓
core
 ↓
MLX
```

Supporting services may depend on stable lifecycle schemas:

```text
inspect ────┐
hardware ───┼─> lifecycle schemas and provenance
registry ───┘
```

Forbidden directions:

- `core` must not import higher layers.
- `models` must not import CLI, evaluation, benchmark, or lifecycle workflows.
- `generation` must not select implementations by model-name substring.
- `training` must not define a second copy of a model architecture.
- `workers` must call services rather than reimplement them.
- `compat` must not become a required dependency of native execution.
- Importing `mlx_one` must not initialize Metal or download artifacts.

Cross-package cycles are architectural defects. Shared contracts should move
downward to `core` or `lifecycle.schemas`, depending on whether they describe
runtime values or reproducibility records.

## 6. Model-family layout

Each family uses the smallest applicable version of this contract:

```text
models/language/qwen2/
├── __init__.py
├── config.py                  # Typed Hugging Face config mapping
├── model.py                   # Native MLX model and forward execution
├── loader.py                  # Family loading hooks
├── weights.py                 # Explicit names and tensor transformations
├── registry.py                # Architecture/capability registration
└── architecture.py            # Only genuine family-specific layers, if needed
```

Model tests live in the repository test tree, not inside the installed package.
A family may omit files when shared implementations are sufficient. It may add
processor or generation hooks only when the generic contracts cannot express a
real architectural requirement.

No family may silently ignore unmatched weights or unsupported configuration.

## 7. Public API boundary

`mlx_one.__init__` exports only documented stable user concepts, eventually:

```python
AutoModel
AutoEmbeddingModel
AutoVLM
AutoASRModel
AutoTTSModel
AutoTokenizer
AutoProcessor
SFTTrainer
inspect_model
plan_inference
plan_training
evaluate
benchmark
verify
```

Concrete family classes remain importable from their model package for advanced
use, but are not all re-exported at the top level. Internal helpers are private
unless a public stability commitment is intentional.

The CLI and Python API call the same services and produce the same versioned
records.

## 8. Test layout

```text
tests/
├── unit/                      # No downloads, no Metal requirement
│   ├── core/
│   ├── models/
│   ├── processors/
│   ├── generation/
│   ├── training/
│   ├── lifecycle/
│   └── cli/
├── fixtures/
│   ├── configs/
│   ├── tiny_weights/
│   ├── tokenizers/
│   └── expected/
├── integration/               # Tiny local artifacts and subprocess protocols
├── parity/                    # Trusted-source numerical/task comparisons
├── hardware/                  # Explicit opt-in Apple Silicon qualification
└── release/                   # Wheel, clean-install, and bundle validation
```

Required test levels for an architecture:

1. Config and registry tests.
2. Synthetic tiny-shape forward/cache tests.
3. Tiny checkpoint load/save/generation tests.
4. Pinned real-checkpoint integration.
5. Trusted-source parity.
6. Documented hardware evidence.

Ordinary unit tests must remain backend-free where possible and must not download
models. Real-model, parity, and hardware suites are explicitly selected.

## 9. Current-to-target mapping

| Current module | Target location |
| --- | --- |
| `schemas.py`, `schema_migrations.py` | `lifecycle/schemas.py`, `lifecycle/migrations.py` |
| `inspection.py` | `inspect/service.py` plus format-specific modules |
| `hardware.py`, `planning.py` | `hardware/detect.py`, `profiles.py`, `planner.py` |
| `calibration.py` | `hardware/calibration.py` |
| `generation.py` | `generation/text.py` and task service entry point |
| `training.py`, `datasets.py` | `training/trainer.py`, `sft.py`, `datasets.py` |
| `benchmark.py` | `benchmark/inference.py` |
| `evaluation.py`, `comparison.py` | `evaluation/*`, `lifecycle/compare.py`, `gates.py` |
| `compatibility.py`, `evidence.py`, `registry.py` | `registry/capabilities.py`, `evidence.py`, `service.py` |
| `run_store.py`, `provenance.py` | `lifecycle/runs.py`, `provenance.py` |
| `*_worker.py` | `workers/*.py` |
| `workflow.py` | Task-specific lifecycle/application service |
| `cli.py` | `cli/main.py` plus command modules |
| `utils/model_loader.py` | Native `core/loading.py`; external fallback in `compat` |
| `utils/memory.py` | `hardware/memory.py` |
| `utils/checkpoint.py` | `training/checkpoint.py` or `lifecycle/lineage.py` by responsibility |

Existing public imports must continue to work during migration through explicit
re-exports or small deprecated shims.

## 10. Migration sequence

Do not reorganize the entire repository in one change.

### Stage 1 — Create non-conflicting native foundations

Create:

```text
core/
models/shared/
models/language/qwen2/
processors/
```

Keep current lifecycle modules in place. Add tests for the new contracts and
native Qwen2.5 implementation.

### Stage 2 — Introduce native runtime selection

Add `native`, `external`, and `auto` selection without changing existing default
behavior until parity gates pass. Native execution resolves through architecture
metadata and the new registry contract.

### Stage 3 — Migrate same-named modules atomically

The current files `generation.py`, `training.py`, `benchmark.py`, and
`registry.py` conflict conceptually with target packages of the same names.
Convert each file to its package in a separate atomic change:

1. Move implementation into the new package.
2. Preserve the old public symbols through the package `__init__.py`.
3. Update all internal imports.
4. Run import, CLI, unit, wheel, and clean-install tests.
5. Do not leave a module and package competing for the same import name.

### Stage 4 — Stabilize native language models

Complete Qwen2.5, Qwen3, Llama, and Gemma before broad modality expansion. Move
LoRA, QLoRA, and SFT to the same native model classes.

### Stage 5 — Add modality foundations

Proceed in order:

```text
embeddings -> VLM -> ASR -> TTS
```

Create each shared task package only when its first vertical slice is ready.

### Stage 6 — Consolidate lifecycle modules

After public import shims and schema migrations are tested, move the remaining
flat inspection, hardware, evaluation, and lifecycle modules into their target
packages. Remove shims only under the documented deprecation policy.

## 11. Architecture-first Qwen slice

The initial native architecture slice is deliberately isolated from loaders,
tokenizers, generation, training, and public auto-model APIs:

```text
src/mlx_one/core/
├── config.py
├── cache.py
├── outputs.py
├── registry.py
└── weights.py

src/mlx_one/models/
├── shared/
│   ├── transformer.py
│   ├── moe.py
│   └── vision.py
├── language/qwen2/
├── language/qwen3/
├── language/qwen2_moe/
└── vision_language/qwen2_vl/
```

Its acceptance target is native Qwen2/Qwen2.5, Qwen3, Qwen2-MoE, and full
Qwen2-VL architecture construction from tiny configurations, with registry,
synthetic shape/cache, and strict synthetic weight-contract tests. It does not
load or execute real checkpoints. Safetensors loading, tokenizer/processor
integration, generation, parity, quantization, training, and hardware
qualification follow as separate phases.

## 12. Definition of structurally complete

The structure is successful when:

- A new checkpoint in a supported architecture normally needs no model code.
- A new architecture has one predictable integration location and checklist.
- No task duplicates loading, preprocessing, model math, or evidence behavior.
- Package dependencies follow the documented direction without cycles.
- Importing lightweight inspection/schema APIs does not initialize MLX.
- Native and compatibility entry points converge on one implementation.
- Every support claim can be traced from public API to model implementation,
  exact artifact, test evidence, hardware record, and release lineage.
