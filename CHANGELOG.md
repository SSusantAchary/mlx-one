# Changelog

All notable changes to mlx-one are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0;
alpha releases may refine planned interfaces.

## Unreleased

- Added extensible hardware profiles and privacy-safe local hardware detection,
  including the reference `m4-air-32gb` profile with a 24 GiB process budget.
- Added metadata-only text inference and training planning with uncertainty ranges,
  fit classifications, confidence, method recommendations, and typed Python APIs.
- Added a pinned, resumable Qwen2.5 M4 calibration matrix with isolated workers,
  atomic evidence, conversion caching, preflight skips, and memory-pressure guards.
- Published the first 0.5B–3B M4 reference matrix: 48 records, 44 completed cells,
  complete inference coverage, transparent boundary failures, and validated
  estimator error/coverage statistics.

### Added

- Metadata-only `mlx-one inspect` support for Hugging Face repositories and local
  model directories, with offline cache mode, JSON artifacts, warnings, and
  cautious backend capability hints.
- Versioned, backend-free schemas for models, hardware, run specifications,
  results, task failures, artifacts, provenance, and compatibility evidence.
- Safe `mlx-one doctor` host and backend diagnostics with JSON output.
- Outcome-based roadmap for text, vision-language, ASR, TTS, merging, and shipping.
- Open-source contribution, conduct, security, issue, and release scaffolding.

### Changed

- Made the project model-size agnostic: compatibility claims now apply to exact
  model, workload, backend, and hardware combinations instead of a parameter
  ceiling.
- Adopted the positioning: "Train anywhere. Prove it on Apple Silicon."
- Moved MLflow from core dependencies to the optional `tracking` extra.
- Avoided implicit MLX initialization during model-loader error handling.

## 0.1.0a1

- Initial package, CLI skeleton, multimodal model loaders, memory helpers, and JSON
  checkpoints.
