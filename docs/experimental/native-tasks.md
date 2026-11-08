# Experimental native tasks

**Current support:** the selected sentiment, NER, CLIP and Silero workflows are
included in the [40-model support inventory](../../status_models_testing.md).
[0.1.1 is approved for release](../releases/0.1.1.md); publication is pending.
The November 3 measurements and optional blocked profiles below retain their
original historical labels, accuracy results and workload bounds.

[Experimental index](README.md) · [Known limitations](known-limitations.md)

<!-- mlx-one:verified-task-results:start -->
## Experimental native task results

Four bounded FP32 profiles run through the shared MLX-One registry, Python API,
CLI and HTTP server. These are **EXPERIMENTAL**, not release-qualified. Results
below were measured on **2026-11-03**, Apple M4 / 32 GiB, macOS 15.8.1,
MLX 0.32.2. These historical measurements describe the locally verified
0.1.0a1 wheel. The [0.1.1 integration candidate](../integration/2026-11-04/merge-readiness.md)
revalidates the bounded task profiles; package publication and wider checkpoint
qualification remain separate.

| Task / exact checkpoint | Measured quality | Admitted scope and remaining limits |
|---|---|---|
| Sentiment — `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | **91.06%** SST-2 validation accuracy (794/872); HF top-1 agreement 872/872 | Python/CLI/HTTP; 1–2 texts, up to 512 tokens each; FP32 only; evaluation data reuse terms unspecified |
| NER — `dslim/bert-base-NER` | **60.36%** strict entity F1 on 100 reconstructed CoNLL validation sentences; matches HF | Python/CLI/HTTP; NONE/SIMPLE aggregation, 1–2 texts, 512 tokens each; bounded subset, unspecified dataset reuse terms |
| Zero-shot image classification — `openai/clip-vit-base-patch32` | **6/6** correct with two labels; **5/6** with ten labels, six real images; source/native top-1 agrees on all 18 comparisons | Python/CLI/HTTP; one image, 1–10 labels, fixed prompt template; 100-label profile rejected; separate weight-license review pending |
| VAD — `snakers4/silero-vad-16k` | Full 60-second source/native probability and recurrent-state parity; clip presence: 10 speech clips + 3 authored negatives correct | Offline Python/CLI/HTTP; stateful Python streaming; mono 16 kHz, 512-sample chunks; human segment-boundary quality and broader noise/music evaluation pending |

Checkpoint revisions, processor/conversion hashes, source commits and individual
qualification gates are recorded in the [publication evidence map](../publication/2026-11-04/evidence-map.json)
and [task qualification matrix](../native_tasks/qualification-matrix.md).
Transformers reference: v4.57.1 commit
`8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`; Silero reference:
`60b7ffa243625ebdc1070275a29f18c87843786a`. Silero's source graph was rebound to
the selected safetensors weights; this does not claim equality with the original
JIT's different learned weights. Evaluation text, images and model weights are
not redistributed in the wheel.

### Task latency (historical, not a service SLA)

| Python task | Median (ms) | p95 (ms) | Workload |
|---|---:|---:|---|
| DistilBERT SST-2 | 14.95 | 17.58 | One text |
| BERT NER | 18.90 | 26.14 | One text, SIMPLE aggregation |
| CLIP ViT-B/32 | 80.19 | 89.68 | One image, two labels |
| Silero VAD | 16.35 | 22.97 | One second of mono 16 kHz audio |

Each row is 20 timed Python task calls in one process after one excluded warmup;
loading is excluded. Median and nearest-rank p95 include task preparation and
synchronized model execution. These are separate workloads, not HTTP latency,
concurrent-load measurements or a cross-model speed comparison. See the
[latency protocol and raw-run locators](../native_tasks/task-performance.md).

### Candidates with blocked task admission

| Candidate | Decision | Remaining gate / public task support |
|---|---|---|
| ViT-base-patch16-224 | EXPERIMENTAL helper | Intermediate FP32 discrepancy; no shared task/CLI/HTTP admission; shared OCR encoder unchanged |
| SigLIP2-base-patch16-224 (fixed-resolution SiglipModel) | BLOCKED | Vision-layer intermediate FP32 failures; no task APIs |
| DETR ResNet50 | BLOCKED | Five of 12 real COCO traces fail decoder/logit gates; no task APIs |
| RT-DETR | NOT IMPLEMENTED | DETR prerequisite gate remains unresolved |
| Qwen2.5-VL document QA | BLOCKED | Internal candidate answers 5/6 authored questions on two pages; dense vision parity fails; no dedicated task APIs |
| Qwen2.5-VL video QA | BLOCKED | Real-video pixel parity fails; full video-answer inference not implemented |
| Depth Anything V2 Small | BLOCKED | All six whole-trace FP32 comparisons fail; five final maps pass; no task APIs or independently labeled depth-quality result |

Existing VLM, OCR and segmentation support does not establish these task
qualifications. The [0.1.1 integration verification](../integration/2026-11-04/verification.json)
records **2,238 passed, 79 skipped, 9 subtests** in the default suite and
**160 passed, 17 skipped** in the native selection; these selections overlap and
must not be added. Clean-wheel checks pass for Python, CLI and HTTP across all
four profiles, including four live servers with bearer authentication.
Source-oracle replay passes 17 real-checkpoint cases covering 1,566 named tensor
comparisons. The [readiness report](../integration/2026-11-04/merge-readiness.md) records
the remaining skips and qualification limits. These checks did not rerun the
historical latency benchmarks.

### Sentiment example

Use a local checkpoint matching the locked revision in the evidence map:

```python
from mlx_one import classify
result = classify("/local/pinned-distilbert-sst2", "I enjoyed this film.")
print(result.to_dict())
```

```sh
mlx-one classify /local/pinned-distilbert-sst2 "I enjoyed this film." --json-output
mlx-one serve /local/pinned-distilbert-sst2 --offline --alias sentiment --no-warmup
```

The server accepts `POST /v1/tasks/text-classification` with JSON
`{"model":"sentiment","input":"I enjoyed this film."}` and configured bearer
authentication. HTTP requires the existing `server` extra; CLIP image processing
requires the `vision` extra. The [API contracts](../native_tasks/api-contracts.md) cover all
four profiles, input limits and score semantics. Clean-wheel verification runs
outside the checkout with reference frameworks absent and network access disabled.
<!-- mlx-one:verified-task-results:end -->

## Text generation

The [checkpoint inventory](../benchmarks/inventory.md) retains the exact
revisions, primary precision and bounded status of the ten original text rows
and three later large targets. Dense B1 with FP16 KV is the existing bounded
text checkpoint scope. The [admission policy](../model-admission.md) separates
functional support from optional exhaustive numerical research diagnostics.
Gemma3 sliding-boundary numerical and answer-quality limitations and Qwen3.5
converted-primary limitations remain recorded; a stored-source smoke is not a
primary-Q4 qualification. See the [historical support matrix](../qualification/alpha_release/model-support-matrix.md)
and [benchmark coverage](../benchmarks/coverage.md).

## Vision-language and OCR

Pinned static-image experiments are recorded in [VLM performance](../benchmarks/vlm/performance.md),
[VLM validation](../validation/vlm-hf-2026-10-24/report.md) and the
[VLM runbook](../VLM_TESTING.md). They do not qualify the blocked document/video
QA task profiles above. Native OCR covers TrOCR line crops and the implemented
page-model paths. See [OCR performance](../benchmarks/ocr/performance.md),
[100-page corpus](../benchmarks/ocr/omni100-20261025/README.md),
[10-line handwriting corpus](../benchmarks/ocr/hand10-20261025/README.md) and
[OCR runbook](../OCR_TESTING.md). Failed and truncated page requests remain in
the corpus, without invented layout, confidence or extraction scores.

## Audio experiments

The 2026-10-24 [audio campaign](../validation/audio-hf-2026-10-24/report.md)
records real-checkpoint ASR, alignment, TTS and a separate short speech-to-speech
case. These postdate some older guide summaries. The primary 10-second STS case
still fails; the short-question success does not replace it. A single-clip WER,
returned word timestamps or generated waveform does not establish corpus quality,
independent alignment accuracy or human listening quality. See
[audio performance](../benchmarks/audio/performance.md) and the earlier
[audio qualification report](../audio-qualification-m4-32gb.md) for preserved
language, timestamp, exact-anchor and streaming failures.

### Historical audio implementation progress — WIP

This 2026-10-04 implementation snapshot predates the later single-clip benchmark
campaign linked above. Standardized measurements were pending at that stage.
The [native audio runbook](../AUDIO_TESTING.md) documents checkpoint procedures
and remaining limits; this historical table retains its original gate decisions.

As of 2026-10-04, unresolved qualification gates remain **WIP** and no longer
block implementation of later families. This changes development sequencing,
not acceptance thresholds: WIP does not mean qualified, and unavailable tasks
are not advertised as working loaders.

| Stage | Checkpoints | Implementation / qualification status |
| --- | --- | --- |
| 3 | Qwen3-ASR 0.6B / 1.7B; ForcedAligner 0.6B | Implemented; **WIP** exact anchors, alignment timestamps, streaming parity and missing language/corpus gates |
| 4 | Parakeet TDT v2 / v3; Nemotron 3.5 ASR | Parakeet offline implemented; **WIP** full qualification. Nemotron contract frozen; native cached streaming not implemented |
| 5 | Voxtral Mini 3B; Voxtral Realtime 4B | Checkpoint/tensor/processor contracts frozen; **WIP / not implemented** native offline/streaming transcription and qualification |
| 6 | Original VibeVoice 1.5B multi-speaker TTS | Checkpoint contract, original source and supplemental tokenizer pinned; **WIP / not implemented** native synthesis, reference parity and fixture permission evidence |

See the [audio report](../audio-qualification-m4-32gb.md) for the numerical
investigations, checkpoint limitations and remaining validation work.

## Embeddings and retrieval

BERT, MPNet, Qwen3 embeddings/reranking and LFM2 ColBERT have native architecture
paths. The [family guides](../models/index.md#retrieval-guides) retain candidate
and synthetic-validation scope. LFM2 ColBERT emits token vectors for masked
MaxSim; it is not compatible with the generic single-vector `embed` contract.
Synthetic validation does not establish a measured checkpoint or retrieval quality.

## Segmentation and tracking

SAM3 concept/interactive image fixtures and separate offline/streaming 32-frame
single-object comparisons are recorded in the [SAM3 qualification report](../sam3-qualification-m4-32gb.md).
Minimum mask IoU was 0.99998, maximum score difference 0.00007511, with exact boxes,
stable object ID `0` and approximately 8.41 GB peak MLX allocation. These controlled
single-truck clips do not establish general tracking quality, multi-object
occlusion handling or isolated performance. Automatic-mask/crop-layer parity,
refinement, box exemplars and broader task qualification remain open. The
[model guide](../../src/mlx_one/models/segmentation/sam3/README.md) documents dedicated
Python/CLI workflows; the standalone temporal tracker is architecture-only.
