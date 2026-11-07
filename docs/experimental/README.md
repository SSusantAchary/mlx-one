# Experimental development

This is the entry point for implemented but unqualified features, bounded task
results, runtime investigations and remaining blockers. An experimental workflow
may execute successfully without meeting every checkpoint, task, quality or
release gate. No status or acceptance threshold changes with this reorganization.

| Area | Evidence and investigations |
| --- | --- |
| Native tasks | [Text, VLM, audio, OCR, retrieval, segmentation and added task profiles](native-tasks.md) |
| MLXBoost | [Optional host optimization, measurements, regressions and proposals](mlxboost.md) |
| Inference runtime | [Caches, prefix reuse, attention, memory, scheduling and context experiments](inference-runtime.md) |
| Blockers | [Known limitations by model, runtime, modality and deployment](known-limitations.md) |
| Checkpoints | [Authoritative inventory](../benchmarks/inventory.md), [model index](../models/index.md) |
| Performance | [Benchmark navigation and methodology](../benchmarks/README.md) |

## Status definitions

The [admission policy](../model-admission.md) owns functional model admission.
[Model evidence definitions](../models/index.md#reading-the-evidence) distinguish
architecture, loading, smoke execution, reference parity, measured performance
and qualification. `EXPERIMENTAL`, `BLOCKED`, `WIP`, `NO-GO` and `NOT IMPLEMENTED`
retain their original report scope; they are not renamed to imply approval.
Runtime correctness/performance qualification is also scoped to exact execution
facts, rather than inherited from a family name.

## Current investigations and blockers

- Four bounded sentiment/NER/CLIP/VAD profiles execute through shared APIs;
  broader quality and release review remain open. Other new task candidates
  retain intermediate-parity or processor blockers.
- MLXBoost's opt-in host lookup shortcut has HTTP ablation evidence. Cached/fresh
  model numerical failures remain failed, and rollout is unapproved.
- Cache/paged/batched execution and long-context capacity need matching native
  evidence. Dense single-sequence execution remains the reference default.
- Audio language, timestamp and streaming coverage, OCR layout pipelines,
  retrieval quality and natural multi-object tracking retain separate gaps.

See [native-task blockers](../native_tasks/unresolved-blockers.md),
[MLXBoost qualification](../mlxboost/qualification-matrix.md),
[P1 failures](../p1_native_failure_log.md) and
[known limitations](known-limitations.md) for preserved decisions and failures.

## Qualification requirements

Use exact checkpoint/reference revisions, asset/source hashes, supported
precision and cache topology, named workloads, software versions and hardware.
Run applicable functional, task/API/lifecycle, clean-wheel and quality gates under
their existing policies. Measured speed is independent evidence. Earlier
exhaustive numerical admission proposals are optional research under the current
functional policy; their historical failures are not erased. New task and
optimization-specific gates retain their own unchanged requirements.

The [candidate catalog](../model-catalog.md), [development roadmap](../roadmap.md)
and [MLXBoost ADR](../architecture/mlxboost-module-boundary.md) distinguish future
work from current implementations. [Generation ownership](../documentation-generation.md)
explains how detailed generated reports stay outside the project landing page.
