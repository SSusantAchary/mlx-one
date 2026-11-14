# Changelog

All notable changes to mlx-one are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0;
alpha releases may refine planned interfaces.

## 0.1.3

- Integrate opt-in GPT-OSS-20B MXFP4/Q4 MLXBoost dispatch and bounded prefill
  into the package, Engine and serving CLI; export `MLXBoostConfig`.
- Add `mlx-one upgrade` for Git installation using the CLI's Python interpreter,
  with ref selection, extras, dry run and optional dependency preservation.
- Preserve current main's 0.1.2 model workflows and package license notices.
- GPT-OSS acceleration defaults remain off; 128K Kilo agent workflows need
  end-to-end validation. Historical benchmark results are not new release tests.

## Unreleased

### 0.1.2 — published 2026-11-11

- Add native T5, ViT, Kokoro, XLM-R and Nomic architecture types, plus additional
  BERT retrieval profiles: ten verified checkpoint workflows and 45 types overall.
- Provide dense/sparse/ColBERT/hybrid retrieval, paired reranking, encoder-decoder
  generation, supervised image classification and bounded English speech.
- Record 2,599 passing native tests and 54 final targeted checks; retain the ViT
  hidden-state diagnostic failure alongside accepted classification results.
- Refresh README, model explorer, seven modality counts and workflow documentation.
- Publish the validated 0.1.2 wheel and source archive on PyPI. Verify on
  12 November that both uploaded SHA-256 values match the audited artifacts.
  [Release notes](docs/releases/0.1.2.md) and the
  [release decision](docs/releases/0.1.2.json) record the published files.

### 0.1.1 approved for release — 2026-11-08

- Complete real-checkpoint primary-workflow records for all 40 registered native
  model types across seven modalities, with exact checkpoint/precision/operation
  identities and applicable interface checks.
- Repair prompt/BOS handling, MoE loading, retrieval, OCR decoding/rendering,
  LFM speech state and SAM temporal tracking; retain original failed diagnostics.
- Record 2,573 passing native regression tests and 9 passing subtests, with
  95 optional tests skipped; pass wheel/sdist and isolated installed-wheel checks.
- Update the support inventory, README and documentation site from the verified
  support catalog. Archive the older 23/7/10 model audit separately.
- Maintainer approved release on 8 November. Publication remains pending; the
  published installation stays at 0.1.0a1. See [release notes](docs/releases/0.1.1.md).

### 0.1.1 integration candidate — 2026-11-04

- Integrate existing native sentiment, NER, CLIP and Silero task APIs, tests and
  scoped experimental results with the prepared README and module-boundary ADR.
- Set package, CLI, HTTP metadata and current citation version to 0.1.1.
- Preserve numerical gates, model mathematics and optional default-off MLXBoost
  behavior. No task qualification, release tag or package publication is implied.

### Proof-based alpha candidate — 2026-10-21

- Add independent cache-position/content controls, long-context shape diagnostics,
  a fixed 30-prompt evaluation per candidate, and exact upstream cache-test evidence.
- Repair the eight reproduced native test failures without weakening numerical gates.
- Redact unexpected HTTP/SSE exception details and verify request recovery.
- Pin the qualified MLX/Metal runtime, update the certificate/UI security dependencies,
  preserve bundled third-party notices, and prepare manually gated release automation.
- Prepare frozen wheel/sdist artifacts and isolated native/API smoke checks.
  Numerical-policy approval and checkpoint certification remain open; release is NO-GO.


- Added native Llama causal-model support and pinned MiniCPM5 1B/2B MLX
  integration gates, including explicit head dimensions, 4-bit loading,
  multi-EOS termination, and strict reasoning-template compatibility.
- Added coding, instruction-following, and reasoning evaluation profiles with
  per-sample records, explicit exclusions, and 95% Wilson confidence intervals.
- Added schema migrations, evidence-driven capability import/promotion, backend
  version revalidation, and exact-revision Qwen2.5-Coder and SmolLM2 candidates
  selected from the maintainer model catalog.
- Added explicit adapter-only checkpoint continuation, loss-history and peak-memory
  metrics, transactional exports, optional adapter reload smoke tests, and parent
  run lineage.
- Added a resumable inspect-to-compare workflow plus separate backend-free,
  clean-wheel, and self-hosted Apple Silicon CI lanes.

- Added strict compatibility-evidence validation and an operation-level capability
  registry seeded from checksummed Qwen2.5 M4 calibration records. Failed and
  invalid calibration cells cannot become verified evidence.
- Added atomic local run storage with parent lineage and explicit measured,
  estimated, and unavailable metric provenance.
- Added the experimental `text-exact-match-v1` evaluator, `mlx-one evaluate`, and
  protocol-safe `mlx-one compare` with JSON/Markdown output and YAML gates.
- Added versioned dataset, training, checkpoint, and capability contracts; local
  JSON/JSONL validation; response-only token masks; and privacy-safe previews.
- Added experimental native MLX LoRA/verified-base QLoRA orchestration, atomic
  adapter lineage, export verification, and limited FastLanguageModel/TRL-shaped
  compatibility facades. Held-out quality qualification remains required before
  this is released as a supported training workflow.
- Reconciled the public roadmap around vertical text-training and evaluation
  slices while retaining the `v0.1.0a1` release history.

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
