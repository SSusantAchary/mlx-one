# README restructuring completion — 2026-11-07

## Change summary

The public [README](../README.md) now follows the requested landing-page order: introduction, value, capabilities, quick start, concise measured performance, modalities, architecture, engine/MLXBoost, APIs, documentation, a full Citation section and license information. Contribution guidance is omitted from README at the maintainer’s request; the existing contributor document is retained. Branding, badges, architecture artwork, text chart and Web UI screenshot remain. VLM, audio and OCR performance tables/charts also remain directly in README, alongside historical HTTP serving and measured additional-task latency. Embeddings and SAM3 explicitly disclose measurement gaps. Detailed reports retain companion tables and methodology.

The refactor changes documentation, presentation generators, documentation tests and documentation CI only. Existing runtime edits are preserved byte-for-byte. No evidence, qualification threshold, manifest status, package dependency, API behavior or acceleration default changed.

## Documentation mapping

| Former README material | Current home |
| --- | --- |
| Model-Specific Documentation + Supported Models | [Model index](models/index.md), with merged native components and family guides |
| Text checkpoint status | Unchanged [generated inventory](benchmarks/inventory.md) and [admission policy](model-admission.md); redundant table removed |
| Full text performance tables | [Text performance](benchmarks/text-performance.md); compact single-cohort summary remains in README |
| GPT-OSS HTTP serving | README serving table retained; [detailed provenance](benchmarks/http-performance.md) remains linked |
| VLM, audio and OCR tables/charts | Retained directly in README; companion reports: [VLM](benchmarks/vlm/performance.md), [audio](benchmarks/audio/performance.md), [OCR](benchmarks/ocr/performance.md) |
| Experimental native quality results and blocked candidates | [Native task experiments](experimental/native-tasks.md); measured task latency remains in README |
| MLXBoost experiments and comparison | [MLXBoost](experimental/mlxboost.md); existing ADR and full evidence retained |
| Engine mode/cache investigations | [Runtime experiments](experimental/inference-runtime.md) |
| Scattered limitations | [Known limitations](experimental/known-limitations.md) |
| Full install, server and task quickstarts | [Getting started](getting-started.md) |
| CLI and Python reference examples | [API guide](api/index.md) |
| Implemented-capability list and architecture principles | [Architecture](architecture/index.md) |
| Validation and historical roadmap | [Development](development.md); [existing roadmap](roadmap.md) retained |
| Citation | Full BibTeX entry restored in README, matching [CITATION.bib](../CITATION.bib) |

## README comparison

| Metric | Before | After |
| --- | ---: | ---: |
| Lines | 1,285 | 485 |
| Tables | 28 | 14 |

Reduction: **800 lines / 62.3%**. Six redundant tables were removed or consolidated: five overlapping native-component tables were merged into family-guide tables in the model index, and the ten-row checkpoint-status table was removed. The homepage has 14 fewer tables overall. Per the maintainer’s clarification, performance tables and charts for every measured modality remain in README; experimental qualification logs and optimization investigations live separately. The initial 287-line version was superseded by this correction.

## Validation

Setup leads with the published `mlx-one==0.1.0a1` PyPI package and version-pinned extras, matching the [GitHub release](https://github.com/SSusantAchary/mlx-one/releases/tag/0.1.0a1). Source cloning and editable installs are development setup. Documentation links, all four benchmark checks, requirement syntax and whitespace checks pass after this correction.

- All four benchmark generators pass their existing `--check` modes. Text: 17 measurements / 13 checkpoint revisions. VLM: 40 records, 20 measured. Audio: 14 selected records, 13 measured and one failed primary case. OCR: seven synthetic selections remain unmeasured; the separate real-image page/handwriting selections validate.
- All four generators were run in write mode twice after migration. Each round changed zero file hashes; regeneration preserves all modality performance tables while experimental qualification regions remain outside README.
- The portable documentation check passes for 50 documents: relative links, heading/HTML anchors, images, table widths, closed fences, Python syntax, marker ownership and forbidden homepage content. The independent MarkdownIt publication check also passes for 52 documents. External URLs were not fetched.
- Audio/model documentation check passes: 40 registered model types resolve to 28 family/shared guide files. Families without adjacent READMEs use explicit existing shared-guide mappings; no registry coverage was skipped.
- **56 tests passed, 9 subtests passed** across the text reporter, VLM/audio benchmark and OCR evidence/corpus tests (`pytest --no-cov`). Missing, reversed and duplicate audio marker pairs are rejected without modifying their input. A regression check verifies that all three modality performance blocks retain the same tables, charts, values and caveats in README and companion reports.
- The publication self-test rejects altered task accuracy, latency, HTTP rate and blocked-to-qualified status. Its claim checks follow the relocated reports; original evidence-map and historical-preview bytes are preserved.
- Four Python constructor/method signatures and five actual README CLI argument sets were checked against the current source. CLI help paths passed. A temporary WAV supplied the local-file argument for parsing the transcription example. No model loading, inference or downloads were needed.
- Ruff passes on affected Python files. The repository lint gate reports zero new findings and retains 65 inherited findings. `git diff --check` passes.
- All 873 protected source/configuration/evidence/artwork files in the focused preservation set are byte-identical to the starting tree. Across the complete 1,658-file baseline, 1,620 existing files are unchanged; the other 38 are the intended edits listed below. No original file was deleted.
- The incremental documentation patch passes reverse application checks against the final working tree and forward application checks against the reconstructed, hash-verified starting documentation files.

No inference regression suite or release/package build was run: runtime and dependency bytes are unchanged. The alpha packaging script’s document-output change was checked statically and linted; candidate build/install/commit operations were not invoked. The full historical publication audit was not rerun because it asserts the frozen 2026-11-04 source/protected-file/index baseline; those safeguards remain unchanged. An initial benchmark-only test invocation emitted no-runtime-coverage warnings; the final focused run used `--no-cov`.

## Unresolved historical inconsistencies

1. The older GPT-OSS HTTP result lacks an immutable local checkpoint revision and recorded MLX version. It cannot be matched safely to the current pinned MXFP4/Q4 or TQ3 inventory targets. Both the historical measurement and pending current runs remain unchanged.
2. Older audio family summaries predate the 2026-10-24 real-checkpoint campaign, especially LFM2.5 Audio. They are labeled as historical scope; single-clip ASR/TTS/short-STS success does not satisfy broader speech-quality, language, timestamp or streaming gates.
3. Earlier cache/engine roadmaps describe older stages. The current tree contains actor/batch/paged/hybrid contracts and an optional SSD cold-prefix snapshot store. Those implementations are distinct from production continuous batching or active-request SSD paging qualification.
4. MLXBoost retains cached/fresh numerical failures, unresolved cross-stack SmolLM2 EOS behavior, missing thermal control and no rollout approval. Independent neutral-host optimization evidence does not promote model/release qualification.

## File changes

Created: **20**. Modified relative to the captured starting tree: **38**. Deleted: **0**. The three modified files that were already untracked remain untracked; their pre-refactor hashes were verified when constructing the incremental patch.

| Change | File | Purpose |
| --- | --- | --- |
| Created | [benchmarks/documentation.py](../benchmarks/documentation.py) | Shared link rebasing and strict generated-region replacement. |
| Created | [docs/api/index.md](api/index.md) | Preserve extended CLI and Python examples; link HTTP reference. |
| Created | [docs/architecture/index.md](architecture/index.md) | Preserve implemented path, capability details and architecture principles. |
| Created | [docs/benchmarks/audio/performance.md](benchmarks/audio/performance.md) | Retain audio tables/plots and failed-versus-short STS distinction. |
| Created | [docs/benchmarks/http-performance.md](benchmarks/http-performance.md) | Preserve GPT-OSS HTTP table and document missing identity metadata. |
| Created | [docs/benchmarks/ocr/performance.md](benchmarks/ocr/performance.md) | Retain page/handwriting tables, failure/truncation scope and plots. |
| Created | [docs/benchmarks/text-performance.md](benchmarks/text-performance.md) | Retain all historical/source/standardized text measurements and plots. |
| Created | [docs/benchmarks/vlm/performance.md](benchmarks/vlm/performance.md) | Retain full VLM table, token counts, protocol and artwork. |
| Created | [docs/development.md](development.md) | Preserve validation commands and historical roadmap. |
| Created | [docs/documentation-generation.md](documentation-generation.md) | Document owners, target paths, commands and archival checks. |
| Created | [docs/experimental/README.md](experimental/README.md) | Experimental navigation, status definitions and qualification boundaries. |
| Created | [docs/experimental/inference-runtime.md](experimental/inference-runtime.md) | Runtime/cache/memory/scheduler experiments and stage distinctions. |
| Created | [docs/experimental/known-limitations.md](experimental/known-limitations.md) | Centralize model, parity, runtime, modality and deployment limits. |
| Created | [docs/experimental/mlxboost.md](experimental/mlxboost.md) | Retain off/on results, failures, ADR and proposal boundaries. |
| Created | [docs/experimental/native-tasks.md](experimental/native-tasks.md) | Retain quality, latency, blocked candidates and modality evidence. |
| Created | [docs/getting-started.md](getting-started.md) | Preserve install extras, server configuration and extended task examples. |
| Created | [docs/models/index.md](models/index.md) | Consolidate native architecture coverage and model-specific guides. |
| Created | [docs/readme-restructuring-audit.md](readme-restructuring-audit.md) | Record initial state, section ranges, duplication and destinations. |
| Created | [docs/readme-restructuring-results.md](readme-restructuring-results.md) | Record final deliverables, tests, file list and unresolved evidence gaps. |
| Created | [scripts/release/check_documentation.py](../scripts/release/check_documentation.py) | Portable local-link, Markdown and region-ownership checks. |
| Modified | [.github/workflows/model-performance.yml](../.github/workflows/model-performance.yml) | Run documentation checks for navigation/generator changes. |
| Modified | [README.md](../README.md) | 485-line landing page with measured performance for every modality and explicit gaps. |
| Modified | [benchmarks/audio/readme_generator.py](../benchmarks/audio/readme_generator.py) | Refresh audio performance in README and companion report; reject insertion fallback. |
| Modified | [benchmarks/audio_documentation_check.py](../benchmarks/audio_documentation_check.py) | Validate canonical model-index coverage, including shared guides. |
| Modified | [benchmarks/model_performance/report.py](../benchmarks/model_performance/report.py) | Generate concise homepage summary and full relocated report. |
| Modified | [benchmarks/ocr/report.py](../benchmarks/ocr/report.py) | Refresh OCR performance in README and companion report; reject insertion fallback. |
| Modified | [benchmarks/vlm/readme_generator.py](../benchmarks/vlm/readme_generator.py) | Refresh VLM performance in README and companion report; retain destination option. |
| Modified | [docs/architecture/mlxboost-module-boundary.md](architecture/mlxboost-module-boundary.md) | Repair the README summary link; retain ADR decisions. |
| Modified | [docs/benchmarks/README.md](benchmarks/README.md) | Add report navigation and clarify generation targets/protocol separation. |
| Modified | [docs/benchmarks/audio/README.md](benchmarks/audio/README.md) | Document README and companion performance-report generation. |
| Modified | [docs/benchmarks/figures/README.md](benchmarks/figures/README.md) | Describe charts retained in both README and benchmark reports. |
| Modified | [docs/benchmarks/ocr/README.md](benchmarks/ocr/README.md) | Document README and companion performance-report generation. |
| Modified | [docs/benchmarks/vlm/README.md](benchmarks/vlm/README.md) | Document README and companion performance-report generation. |
| Modified | [docs/model-catalog.md](model-catalog.md) | Point the candidate backlog at the canonical model index. |
| Modified | [docs/publication/2026-11-04/verify_publication.py](publication/2026-11-04/verify_publication.py) | Follow relocated claims/markers and keep historical preview output intact. |
| Modified | [docs/validation/ocr-hf-2026-10-25/report.md](validation/ocr-hf-2026-10-25/report.md) | Repair the moved OCR table link; measurements unchanged. |
| Modified | [scripts/qualification/alpha_package.py](../scripts/qualification/alpha_package.py) | Write per-run benchmark docs instead of appending README tables. |
| Modified | [src/mlx_one/models/audio/lfm2_audio/README.md](../src/mlx_one/models/audio/lfm2_audio/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/audio/whisper/README.md](../src/mlx_one/models/audio/whisper/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/embeddings/bert/README.md](../src/mlx_one/models/embeddings/bert/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/embeddings/lfm2_colbert/README.md](../src/mlx_one/models/embeddings/lfm2_colbert/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/embeddings/mpnet/README.md](../src/mlx_one/models/embeddings/mpnet/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/embeddings/qwen3_embedding/README.md](../src/mlx_one/models/embeddings/qwen3_embedding/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/embeddings/qwen3_reranker/README.md](../src/mlx_one/models/embeddings/qwen3_reranker/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/gpt2/README.md](../src/mlx_one/models/language/gpt2/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/lfm2/README.md](../src/mlx_one/models/language/lfm2/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/lfm2_moe/README.md](../src/mlx_one/models/language/lfm2_moe/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/llama/README.md](../src/mlx_one/models/language/llama/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/openelm/README.md](../src/mlx_one/models/language/openelm/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/qwen2/README.md](../src/mlx_one/models/language/qwen2/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/qwen2_moe/README.md](../src/mlx_one/models/language/qwen2_moe/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/language/qwen3/README.md](../src/mlx_one/models/language/qwen3/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/vision_language/lfm2_vl/README.md](../src/mlx_one/models/vision_language/lfm2_vl/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/vision_language/qwen2_5_vl/README.md](../src/mlx_one/models/vision_language/qwen2_5_vl/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/vision_language/qwen2_vl/README.md](../src/mlx_one/models/vision_language/qwen2_vl/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [src/mlx_one/models/vision_language/qwen3_5/README.md](../src/mlx_one/models/vision_language/qwen3_5/README.md) | Repair canonical model-index and/or evidence-definition backlinks. |
| Modified | [tests/test_audio_benchmark.py](../tests/test_audio_benchmark.py) | Guard against reinserting missing or malformed audio regions. |
| Modified | [tests/unit/test_model_performance_reporting.py](../tests/unit/test_model_performance_reporting.py) | Check compact summary, cohort separation and full report preservation. |

## Git summary

- Branch: `chore/alpha-repository-cleanup`; HEAD remains `c0bcfba3b0d19cd50a6b547ab170e89592083438`.
- Final status: 56 tracked files modified and 102 untracked status entries. This includes the pre-existing dirty tree: 22 tracked modifications and 86 untracked entries at baseline. Directory entries may contain multiple files.
- The incremental refactor contains 58 files (20 new, 38 changed). Existing runtime/test/dependency changes are excluded. No files are staged or deleted by this task. No commit, push, merge or publication occurred.
- A separate incremental patch, baseline README/diff/status, hash inventory, validation record and final Git status are retained in the task’s temporary baseline directory. The patch targets the captured working-tree baseline, not clean HEAD; pre-existing native-task and MLXBoost work is a prerequisite, not included implementation work.
- See [baseline audit](readme-restructuring-audit.md) and [generation instructions](documentation-generation.md) for review and reproduction.
