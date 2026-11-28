# Changelog

All notable changes to mlx-one are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0;
alpha releases may refine planned interfaces.

## Unreleased

### Native text-to-image

- Add the native MLX `ImageEngine`, image CLI, and OpenAI-compatible generations
  route for pinned FLUX.2 Klein 4B and Z-Image Turbo checkpoint layouts.
- Add strict safetensors loading and reusable affine denoiser export. Load
  pinned FLUX.2 Klein 4B and Z-Image Turbo community Q4 bundles directly with
  the native engine. Metal fixture checks passed. Full-checkpoint qualification
  remains pending.

- Replace the wordmark with the maintainer-supplied black-and-orange design;
  derive a compact mark for the chat UI, documentation header and favicon.
  Preserve the uploaded wordmark exactly and the historical tagged assets.
- Verify PyPI 0.1.4 publication against both recorded artifact hashes and update
  README/site/release status. Keep the existing v0.1.4 tag and distributions
  unchanged; the new bundled icon is a subsequent Git change.

## 0.1.4 — published 2026-11-15

- Refresh the three-bar logo and architecture artwork; place Architecture after
  Why MLX-One in the README and website. Prepare package metadata and local
  distributions; publication remains a maintainer action.

- Raise the context-bounded GPT-OSS UI output recommendation to 8192 tokens;
  coalesce streamed text rendering without dropping the final tail.
- Avoid redundant full-history single-query masks in the opt-in GPT-OSS
  source-aligned recipe. Keep sliding attention and unsupported layouts on
  reference execution. Record a fixed-request improvement from 19.38 to 28.91
  decode tok/s; sustained long UI responses remain unresolved at about 15.4 tok/s.

- Use low-reasoning-effort UI defaults for GPT-OSS coding requests while
  preserving explicit user output limits. Expose GPT-OSS
  reasoning effort separately from generic thinking-marker controls. Verify
  the previously blank FastAPI prompt produces visible code and stops normally.

- Fix blank/stalled-looking GPT-OSS streams by applying token-aware Harmony
  parsing to ordinary chat as well as tool requests, counting hidden reasoning,
  and emitting throttled SSE progress metadata. Add visible generation status,
  output-limit notices and context-bounded retry with more tokens. Preserve
  partial output on interruption and report incomplete streams explicitly.

- Redesign the bundled chat workspace with a dark sidebar, session tabs/history
  and search, per-message edit/regenerate/branch/delete controls, backend model
  badges, per-turn timing snapshots and a context-usage popover. Preserve audio
  and streaming workflows; distinguish decode/overall throughput and explicitly
  indicate that MCP management is unavailable. Rebuild packaged static assets.

- Preserve bidirectional actor forward requests through the runtime logging
  wrapper so batched HTTP generation and prefix isolation continue working.

- Add elapsed, terminal-colored serving lifecycle and request timing logs, with
  slot/task IDs, rate-limited decode progress, separate model/visible TTFT,
  cache/context/memory summaries and interrupted/failure status. Add
  `--log-level` and `--log-progress-interval`; honor `NO_COLOR`.

- Vectorize nonzero-temperature sampling with stable ranking and CDF selection;
  use exact BF16 integer-key ranking when applicable. Preserve Python RNG,
  normalization, top-k/top-p ordering and the greedy path. On a matched M4 test,
  median temperature-0.7 decode throughput increased from 12.90 to 32.19 tok/s,
  with identical response messages. See the sampling performance evidence.

- Skip activation-calibration identity/report construction when no activation-peak
  candidate exists, including GPT-OSS MLXBoost serving. Live registry additions
  still trigger full validation; analytical memory forecasts and admission checks
  remain active. On a fixed 88-input/256-output request (three runs, M4 32 GiB),
  median decode throughput increased from 19.39 to 38.14 tok/s, with identical
  response messages. See `docs/mlxboost/serving-overhead.md` for protocol limits.

## 0.1.3

- Integrate opt-in GPT-OSS-20B MXFP4/Q4 MLXBoost dispatch and bounded prefill
  into the package, Engine and serving CLI; export `MLXBoostConfig`.
- Add `mlx-one upgrade` for Git installation using the CLI's Python interpreter,
  with ref selection, extras, dry run and optional dependency preservation.
- Preserve current main's 0.1.2 model workflows and package license notices.
- GPT-OSS acceleration defaults remain off; 128K Kilo agent workflows need
  end-to-end validation. Historical benchmark results are not new release tests.

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
