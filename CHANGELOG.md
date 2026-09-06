# Changelog

All notable changes to mlx-one are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0;
alpha releases may refine planned interfaces.

## Unreleased

### Added

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
