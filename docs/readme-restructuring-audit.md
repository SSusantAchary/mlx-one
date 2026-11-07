# README restructuring audit — 2026-11-07

## Baseline

- Branch: `chore/alpha-repository-cleanup`.
- Commit: `c0bcfba3b0d19cd50a6b547ab170e89592083438`.
- Starting README: 1,285 lines, including pre-existing local edits.
- The tree was dirty: 22 tracked files modified and 86 untracked status entries. Runtime, tests, dependencies, task evidence and MLXBoost work predate this refactor.
- Baseline README, Git diff/status and SHA-256 of all 1,658 tracked/unignored files were captured outside the repository before editing. The final comparison is against that working-tree baseline, not HEAD.
- No reset, checkout, clean, staging, commit or publication is part of this task. A documentation-only patch is exported separately from existing runtime edits.

## Original section ranges

| Section | Lines |
| --- | ---: |
| Why mlx-one? | 35–62 |
| Model Performance | 63–119 |
| Vision-Language Model Performance | 120–148 |
| Audio Model Performance | 149–212 |
| OCR Model Performance | 213–258 |
| Architecture | 259–282 |
| Native inference engine | 283–323 |
| Cache runtime qualification | 324–335 |
| What is implemented | 336–376 |
| Installation | 377–433 |
| Quick start | 434–596 |
| Model-Specific Documentation | 597–689 |
| Experimental native task results | 690–771 |
| MLXBoost: optional execution module | 772–806 |
| Supported Models | 807–909 |
| Capability levels | 910–925 |
| CLI | 926–1064 |
| Python API | 1065–1188 |
| Architecture principles | 1189–1201 |
| Validation | 1202–1246 |
| Citation | 1247–1262 |
| Roadmap | 1263–1277 |
| Contributing | 1278–1285 |

## Findings and destinations

| Original content | Destination |
| --- | --- |
| Full text tables and methodology references | [Text performance](benchmarks/text-performance.md), existing [methodology](benchmarks/README.md) |
| GPT-OSS HTTP serving | Retained in README with [historical HTTP provenance](benchmarks/http-performance.md) |
| VLM / audio / OCR performance | Retained in README; companion `performance.md` reports under [benchmarks](benchmarks/README.md) |
| Installation and extended quick start | [Getting started](getting-started.md) |
| Model-Specific Documentation and Supported Models | [Model index](models/index.md): family guides and native component information merged |
| Ten-row text checkpoint status table | Removed as redundant; unchanged [generated inventory](benchmarks/inventory.md) remains authoritative |
| Native task results, latency and blocked candidates | [Native tasks](experimental/native-tasks.md) |
| MLXBoost measurements and proposals | [MLXBoost experiments](experimental/mlxboost.md); ADR retained |
| Engine modes and cache qualification | [Runtime experiments](experimental/inference-runtime.md) |
| CLI / Python reference examples | [API guide](api/index.md) |
| Architecture principles | [Architecture](architecture/index.md) |
| Validation commands and historical roadmap | [Development](development.md); existing roadmap retained |

## Sources of truth and inconsistencies

- `docs/model-catalog.md` is a candidate backlog, not the implemented model catalog. `docs/models/index.md` supplies the latter without copying checkpoint JSON.
- “Architecture implemented”, “loadable”, “functional smoke”, “reference parity”, “measured” and “qualified” describe different evidence. The functional admission policy supersedes earlier numerical admission proposals; frozen numerical failures remain failures.
- Full historical GPT-OSS serving ran, while current pinned direct-model GPT-OSS targets remain pending. The older local checkpoint revision and MLX version are unrecorded: identity cannot be reconciled safely. See the HTTP report.
- Older audio guide summaries predate the 2026-10-24 real-checkpoint benchmark campaign, especially LFM2.5 Audio. Historical single-clip measurements do not complete broader speech-quality gates. Preserve both dated scopes.
- Cache Runtime V1 and the inference-engine roadmap describe different implementation stages. Actor/batch/paged source exists in the current tree; advanced-path qualification remains separate. Continuous production batching, active-request SSD paging and new MLXBoost kernels are not certified by these documents.

## Generated ownership

See [generation and validation](documentation-generation.md) for exact marker owners, commands and target files. The native-task and MLXBoost markers were editorial publication boundaries; no active producer was found for them. They move intact to experimental documents.

## Validation and final inventory

The completion results and file list are recorded in [restructuring results](readme-restructuring-results.md).
