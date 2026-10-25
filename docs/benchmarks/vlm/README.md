# Native static-image VLM benchmarks

The initial source implementation used tiny synthetic fixtures only. The follow-up
real-checkpoint campaign uses the separate **VLM-HF** image protocol below; current
checkpoint status and actual measurements are recorded in the plan and result JSON.
The original twenty synthetic-workload records remain `NOT_RUN` with null metrics. [Coverage JSON](coverage.json) and the [checkpoint plan](benchmark_plan.json)
separate source implementation from actual measurements.

The [manual runbook](../../VLM_TESTING.md) covers pinned downloads, Python, CLI,
HTTP/SSE, cancellation, multi-turn images, repeated-request memory and benchmarks.
These are functional model-addition checks. Historical numerical investigations
are retained and are not admission gates.

## Protocol

Each invocation runs **one already-downloaded model in one process**, on native
MLX/Metal. Batch size 1, greedy generation, one warmup, five measured repetitions,
per-column median, output budget 128. EOS is respected: output speed uses actual
emitted token IDs, never the requested budget. Terminal EOS IDs are recorded
separately and excluded from emitted text-token counts. Decode TPS uses
`(actual_output_tokens - 1) / (last_token_time - first_token_time)`; fewer than
two emitted tokens means null TPS. A request producing no emitted token has null
TTFT. No fixed-length EOS override or numerical qualification suite is used.

| Workload | Input | Prompt |
|---|---|---|
| VLM-S1 | One 512 × 512 PNG | Describe the visible shapes, colors, and their spatial arrangement. |
| VLM-S2 | One 1024 × 1024 PNG | Same as S1 |
| VLM-S3 | Four 512 × 512 PNGs | Same as S1; ordered images |
| VLM-S4 | One 512 × 512 synthetic document | Extract all text exactly, preserving line breaks. |

Versioned PNGs live in `benchmarks/vlm/images/`. The runner records image hashes,
source sizes and **actual** model-specific processed sizes, tiles, patch counts,
visual tokens, text/framing tokens, fused input tokens and emitted output IDs.
S4 includes expected text for optional external OCR scoring; completion alone
is not a quality score. Unsupported workloads must remain pending, never marked
as successfully measured. The processor may enlarge, reduce or tile a source;
512 pixels is a workload input size, not a universal model resize.

## Timing and memory definitions

Chat-template rendering is performed before the timed direct-model request;
image decoding, processor tokenization and tensor preparation are inside it.
MLX is synchronized before request timing. Processor tensors and logits are
materialized at the respective boundaries. Native cached decode reuses the
existing generation loop, with a fresh private cache per repetition, FP16
full-attention KV and model-specific hybrid recurrent state. No image prefix
sharing or cache restoration across unrelated images occurs.

- `preprocessing_ms`: decode, EXIF, RGB/alpha conversion, resize/crop,
  normalization, patch construction and MLX materialization.
- `visual_prefill_ms`: synchronized **combined** vision encoder, projector,
  multimodal fusion, language prefill and last-token logits. Separate
  `vision_encoder_ms`, `vision_projection_ms`, `language_prefill_ms` are null;
  the current adapter cannot isolate those phases reliably.
- `prefill_tps`: actual fused input tokens / combined visual-prefill seconds.
  It is not pure text prefill speed and is not comparable to text-only tables.
- `ttft_ms`: request start through selection of the first emitted model token;
  includes preprocessing and visual/language prefill, excludes load and HTTP.
- `total_ms`: complete request including preprocessing, decoding and cleanup.
- `peak_mlx_memory_gib`: peak allocator memory after reset immediately before
  each measured request; includes resident model allocations and request state.
- `peak_process_memory_bytes`: macOS process lifetime RSS high-water mark;
  it cannot be reset between repetitions. It is not request-only MLX memory.
- `model_load_ms`: model load/materialization once, before warmup; excludes
  checkpoint hash verification. Weight files are hashed before model allocation.

Successful cache cleanup is observed after generator finalization, not inferred
from a requested token count. Python Engine, CLI and HTTP/SSE functional smokes
are separate interfaces; their latency and throughput are not mixed into these
**direct-model** records. Do not run another inference server or GPU workload
while collecting performance evidence on the M4.

## Provenance and publication

The plan pins all five revisions, original config/processor/template hashes,
tokenizer Git blob identity and full publisher LFS weight SHA-256 hashes.
The runner rejects mismatched or incomplete local files, never downloads weights,
and records actual tokenizer SHA-256, native implementation/runner fingerprints,
MLX/Python/Pillow versions, device name, physical memory and OS. Renaming a tiny
fixture or assigning a checkpoint ID cannot satisfy its content checks.

```bash
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
  --model-id mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --workload VLM-S1 \
  --output /tmp/V01-VLM-S1-measured.json
```

Inspect raw output and metadata, then replace **only** the corresponding pending
record with that real run. Keep other workloads separate. Raw files are immutable
runner outputs; the runner refuses to overwrite an existing filename.

```bash
cp /tmp/V01-VLM-S1-measured.json docs/benchmarks/vlm/results/V01-VLM-S1.json
python -m benchmarks.vlm.readme_generator
python -m benchmarks.vlm.readme_generator --check
```

The generator validates every result before updating only the VLM README markers.
It rejects mock/synthetic/incomplete records, absent hardware, unpinned revisions,
invalid counts, mismatched hashes, wrong protocols/statistics and duplicate
checkpoint/workload rows. S1 hardware/runtime/implementation cohorts get separate
tables; S2/S3/S4 are never pooled with S1. Full sample metrics remain here, outside
the landing page. Pending records cannot contain numerical scores.

## Real-checkpoint HF documentation image protocol (2026-10-24)

`VLM-HF-S1` through `VLM-HF-S4` use the user-selected
[Hugging Face documentation-images dataset](https://huggingface.co/datasets/huggingface/documentation-images)
at commit `c8c3cf72b3b724ce734259e4a086d30618a9198b`.
The pinned paths are `coco_sample.png` (two cats), `transformers/tasks/car.jpg`,
`transformers/tasks/cat.jpg`, and `transformers/model_doc/ocr_receipt.jpeg`.
The dataset card declares **CC BY-NC-SA 4.0**. Original image bytes and transformed
images are temporary test inputs; this repository retains their provenance/hashes
rather than redistributing those images under the project's code license.

The [manifest](../../../benchmarks/vlm/hf-images.json) records source hashes,
revision, attribution, Pillow version, exact transformation, and derived hashes.
Preparation applies EXIF orientation, RGB conversion and centered LANCZOS padding
onto a white square; S1/S3/S4 use 512 × 512 and S2 uses 1024 × 1024. Individual
checkpoint processors retain their own subsequent resize/tiling rules.

| Workload | Images | Prompt and purpose |
|---|---:|---|
| VLM-HF-S1 | One padded cat scene, 512-square | Describe objects and spatial relationships |
| VLM-HF-S2 | Same scene, 1024-square | Same description prompt; separate high-resolution latency |
| VLM-HF-S3 | Four padded images, 512-square each | Cats, car, cat portrait, receipt; ordered multi-image inference |
| VLM-HF-S4 | One padded receipt, 512-square | Read document number, date and total; inspect raw answer |

The HF protocol is separate from the original synthetic `VLM-S1`–`VLM-S4`
protocol. Earlier pending records remain pending; the README selects measured
HF-S1 evidence. Batch 1, greedy output up to 128 actual emitted tokens, one warmup,
five measured repetitions and medians remain unchanged. No results are inferred
from requested output budgets. Full per-repetition answers, counts, timing,
memory and processor metadata remain in the result JSON. OCR completion is not
an accuracy guarantee; image-dependent answers and OCR fields are reviewed
separately from speed.

Explicitly prepare only the small image inputs, then benchmark an already local
checkpoint. The benchmark runner never downloads models:

```bash
python -m benchmarks.vlm.prepare_images --output-dir /tmp/mlx-one-vlm-hf-images
python -m benchmarks.vlm.run \
  --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
  --model-id mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --workload VLM-HF-S1 --image-dir /tmp/mlx-one-vlm-hf-images \
  --output /tmp/V01-VLM-HF-S1-measured.json
```

To exercise exact checkpoint Python/CLI/live HTTP/SSE before publishing scores:

```bash
python -m benchmarks.vlm.smoke --id V01 \
  --checkpoint ~/mlx-one-models/vlm/qwen2-vl-2b \
  --image-dir /tmp/mlx-one-vlm-hf-images \
  --output /tmp/V01-checkpoint-smoke.json
```

The validator reads the committed manifest without opening downloaded images, so
all raw model/image cache files can be removed after the campaign while result
validation and README regeneration continue to work. Only task-created downloads
are removed; unrelated user checkpoints and caches are preserved.
