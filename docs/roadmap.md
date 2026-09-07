# mlx-one Product and Engineering Roadmap

> Proposed sequence for work after the released `v0.1.0a1` foundation. A feature
> is shipped only after its exit gate passes; code on `main` may be experimental.

## Direction

mlx-one is an Apple-Silicon-first, model-size-agnostic development and evidence
layer. It owns stable workflow contracts, data validation, orchestration,
compatibility, evaluation, comparison, and artifact lineage while MLX and its
modality packages own tensor kernels and model runtimes. Native APIs are
authoritative; compatibility facades expose only documented, tested subsets.

## Milestones

| Milestone | Outcome | Exit gate |
| --- | --- | --- |
| M0 | Released diagnostics, inspection, planning, and M4 calibration | Preserved compatibility and published evidence |
| M1 | Evidence validation, capability registry, local runs, text evaluation, comparison | A pinned base/candidate evaluation reproduces without manual processing |
| M2 | Owned MLX SFT, LoRA, verified-base QLoRA, checkpoints, and export | Tiny overfit, real quality gain, resume, export load, and M4 safety tests pass |
| M3 | Optional CUDA training and conversion parity | One source/config runs on both backends and MLX evaluation accepts the artifact |
| M4 | Embedding and reranking training/evaluation | A pinned retrieval comparison reproduces |
| M5 | VLM and OCR evaluation/fine-tuning | Two families pass processor, mask, export, and quality gates |
| M6 | ASR/TTS evaluation and selected fine-tuning | Pinned audio metrics reproduce with privacy controls |
| M7 | Profiling, long-context safety, and optimization search | Measured Pareto recommendations cover two families and hardware profiles |
| M8 | Preference, continual, and distributed training | Each method passes objective, overfit, real benchmark, and restart tests |
| M9 | Release qualification and stable API | Clean end-to-end bundle validation and compatibility migrations pass |

## Current `main` boundary

M1's backend-free contracts and exact-match workflow are implemented and tested.
M2 has strict data/config contracts, an isolated `mlx-lm` LoRA adapter, adapter
continuation/export lineage, and limited compatibility facades. It is not complete
until real Qwen2.5-Coder training, post-training evaluation, adapter load/export
smoke tests, and a second architecture pass on Apple Silicon.

CUDA, embeddings, VLM, audio, preference training, distributed execution,
optimization search, serving qualification, merging, and release bundles remain
out of scope until the text vertical slice passes its hardware and quality gates.
