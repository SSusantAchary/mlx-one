# Native OCR implementation report — 2026-10-25

Implemented six native architecture families and a reusable OCR task subsystem, with seven
primary checkpoint inventories. The focused acceptance run passed **168 tests**, including
native FP32/FP16/BF16/Q4 execution, Python/CLI/authenticated HTTP/SSE, PDF/TIFF ingestion,
coordinates, cache continuation, EOS, repeat requests, cleanup, metrics/publication and
existing text/VLM/audio/server/registry regressions. The wheel builds and its extracted
OCR command imports successfully. Seven immutable checkpoint tensor headers match the
native parameter names, shapes and packed formats.

**No full checkpoint was downloaded or executed. No OCR model performance/accuracy score
is published.** Source integration is `NATIVE_IMPLEMENTED` for the documented task scope;
real checkpoint execution remains `NOT_RUN`. Full Paddle layout parsing and the alternative
HF GOT layout remain outside implemented support. This is not a claim of complete end-to-end
Paddle document understanding or production qualification.

Branch: `chore/alpha-repository-cleanup`. Base source commit:
`86b5a0c64babcf72b4816223220538fb3eed6b36`. Changes are uncommitted; no commit, push or release
was performed. [Machine-readable summary and source hashes](ocr-native-2026-10-25/summary.json),
[test output](ocr-native-2026-10-25/test-results.txt), and
[header results](ocr-native-2026-10-25/header-checks.json) accompany this report.

## Reuse and architecture integration

Inspected existing model/core/runtime registries, safetensors and quantization loaders,
VLM image processors/message preparation and generation, `InferenceService`/task adapters,
self/cross-attention caches, model pooling/device ownership, server generation scheduling,
`ModelManager`, authentication/body limits/SSE, public `load_model`, CLI and benchmark tools.
There was no dedicated OCR task or bounded PDF subsystem. Existing VLM OCR benchmark prompts
were not a dedicated recognizer implementation.

`ocr/generation.py` wraps model-specific visual embeddings or encoder state for the existing
`text.generation._stream_bundle`; it does not implement a second sampler/generation engine.
The existing actor schedules OCR work; state, image buffers, visual embeddings and cross-KV
are request-owned. Models share Qwen2/Qwen3 and sparse expert code where the mathematics
match. Dedicated TrOCR/BART encoder–decoder paths preserve their actual cross-attention.
GOT, DaViT, Paddle ERNIE, Pixtral and SAM/CLIP/MoE frontends are native local model code.
[Reference notices and local changes](../../src/mlx_one/models/ocr/UPSTREAM.md) are packaged.

## Changed source and artifacts

- `src/mlx_one/models/ocr/`: six architectures, native layer/cache adapters and provenance.
- `src/mlx_one/ocr/`: typed inputs/results, capability registry, strict loader/tokenizers,
  image/PDF ingestion, coordinates, model-specific processors, output parsing, generation,
  public engine and CLI command.
- `src/mlx_one/engine/{contracts.py,runtime_specs.py,task_runners/ocr.py}`: first-class tasks
  and existing-actor adapter bindings.
- `src/mlx_one/{models/registry.py,utils/model_loader.py,cli.py}`: lazy registration and
  `load_model(..., task="ocr.*")`/`mlx-one ocr` entry points.
- `src/mlx_one/server/{ocr.py,app.py,model_manager.py}`: native OCR bundle lifecycle,
  authenticated uploads, page-event SSE, body limits, aliases and chat capability rejection.
- `pyproject.toml`: optional Pillow/PDFium/tiktoken utilities and packaged attribution assets.
- `benchmarks/ocr/`: reproducible fixtures, native runner, text/region/table/formula metrics,
  header auditing and fail-closed README publication.
- `docs/benchmarks/ocr/`: pinned inventory, header/config evidence, pending per-model records,
  selected publication paths, methodology and coverage. Eight synthetic fixtures are stored;
  real handwriting is deliberately missing.
- `README.md` and `docs/OCR_TESTING.md`: OCR inventory/performance scope and runnable commands.
  Existing text/VLM/audio measurements and chart positions are preserved.
- `tests/ocr/`: tiny checkpoints and functional/native/interface/benchmark tests. Two existing
  registry tests now expect 36 architectures instead of 30; other local test edits were retained.

Unrelated working-tree work, including prior charts, audio/VLM artifacts and private tests,
was preserved. The retired exhaustive numerical qualification suite was not run or restored.

## Model and task support

| Family | Native source scope | Checkpoint execution | Limits |
|---|---|---|---|
| TrOCR printed / handwritten | ViT + cached TrOCR; standalone RoBERTa BPE | NOT_RUN | Cropped lines; handwriting capability needs the handwriting checkpoint; no detector, layout or Markdown |
| GOT-OCR 2.0 Q4 | Native vision encoder/projector + shared Qwen2, original plain/format prompts, two EOS ids | NOT_RUN | Pinned original MLX format; crop-first regions; no predicted geometry or GOT multi-crop mode |
| Florence-2 Base FT | DaViT + BART, OCR / OCR_WITH_REGION, parsed pixel quadrilaterals | NOT_RUN | OCR tasks only; region geometry is never inferred in plain mode |
| PaddleOCR-VL 1.5 Q4 | Dynamic vision/connector, native ERNIE with 3D sectioned RoPE, page/crop OCR, table/formula prompts | NOT_RUN | Separate layout detector/full Paddle page pipeline absent; HTML structure parsed when present, other table markup raw |
| LightOnOCR 2 1B | Pixtral vision/projector + shared Qwen3; OCR-specific image tokens, page text/Markdown | NOT_RUN | No bbox-variant support or independent reading-order/LaTeX quality claim |
| DeepSeek-OCR Q4 | SAM/CLIP compression + native non-MLA MoE, global/local crops and original plain OCR prompts | NOT_RUN | Original pinned variant only; nine local tiles; no measured compression benefit or geometry parser |

Direct safetensors exist for all seven primary checkpoints; no PyTorch/Transformers converter
is needed. Quantized modules execute packed affine weights without loading an entirely
dequantized model. Original TrOCR vocab/merges and GOT `qwen.tiktoken` are supported.
Native parameter formats were compared to actual headers, not assumed from model cards.
The alternative `stepfun-ai/GOT-OCR-2.0-hf` has a distinct tensor/vision/prompt layout and is
explicitly rejected rather than silently treated as the original Q4 checkpoint.

Task contracts: `ocr.recognize`, `ocr.detect`, `ocr.document`, `ocr.batch`, and selectively
`ocr.region`, `ocr.handwriting`, `ocr.table`, `ocr.formula`. The loaded checkpoint's capabilities
gate requests and model-list advertisement. Generic chat is not advertised for dedicated OCR.

## Documents, outputs and streaming

PNG/JPEG/WebP/grayscale/RGBA, sequential TIFF, PDF paths/bytes and owned Pillow copies are
supported. EXIF orientation, alpha-on-white, model-specific sizing and bounded crops are
explicit. PDFium rendering is sequential and serialized across documents; source page count,
selected ranges, DPI, rotation/cropbox and point dimensions are retained. `visual`,
`embedded_text` and deterministic page-level `auto` never concatenate duplicate OCR/embedded
text. Embedded extraction is labeled and never scored as OCR.

`OCRResult`/`OCRPage`/`LayoutElement` expose text, optional model-generated Markdown, ordered
pages, genuine predicted geometry/structure, model/task/revision, timings and token metadata.
External geometry uses top-left pixel xyxy/polygons and zero-based pages. PDF coordinates
are rendered pixels at the requested DPI; original point sizes remain metadata. Crop/resize
inverse transforms are tested. No fake confidence, page elements or geometry are fabricated.
HTML table rows/cells/spans are preserved only when emitted; formulas retain generated text.

File/page/pixel/DPI/tile/rendered-byte/transient-memory/output/context/time limits are enforced.
Cancellation is cooperative between bounded native steps; pending generator cleanup drains
in-flight work. Page state/cross-KV is released on success, error or cancellation. Closing an
engine releases its model lease; externally supplied bundles retain their caller's ownership.

`POST /v1/ocr` accepts authenticated multipart uploads and real model aliases, rejects URLs
and server-side paths, and returns JSON with the versioned `mlx-one-ocr-v1` schema. Optional
SSE emits `page_started`, `page_completed`, `document_completed`/`error`. These are page events;
internal decoder token generation is incremental but token-delta SSE is not exposed.

## Validation commands and results

```bash
MLX_ONE_RUN_MLX_TESTS=1 .venv/bin/python -m pytest \
  tests/ocr tests/test_cli.py tests/test_model_loader.py tests/test_server.py \
  tests/test_task_service.py tests/test_registry_plugins.py \
  tests/test_adapter_registry.py tests/test_runtime_registry.py \
  tests/test_vlm_pipeline_native.py tests/test_audio_pipeline_native.py --no-cov -q
# 168 passed; 2 existing Starlette/httpx deprecation warnings; 9.24 seconds.

.venv/bin/python -m benchmarks.ocr.audit_headers
# Seven native parameter/header contracts match; no numerical qualification or full model run.

.venv/bin/python -m benchmarks.ocr.report --check
# Seven records; zero measured; seven primary checkpoints awaiting benchmarks.

.venv/bin/python -m benchmarks.model_performance.report --check
.venv/bin/python -m benchmarks.vlm.readme_generator --check
.venv/bin/python -m benchmarks.audio.readme_generator --check
# Existing text/VLM/audio evidence and generated sections validate unchanged.

.venv/bin/ruff check src/mlx_one/ocr src/mlx_one/models/ocr \
  src/mlx_one/engine/task_runners/ocr.py src/mlx_one/server/ocr.py benchmarks/ocr tests/ocr
# OCR files pass. This is not a claim that unrelated existing files are lint-clean.

.venv/bin/python -m build --wheel --no-isolation --outdir /tmp/mlx-one-ocr-wheel
# Wheel built; extracted wheel includes the six families, APIs and license/provenance assets;
# python -m mlx_one.cli ocr --help succeeds against the extracted wheel.
```

## Manual hardware handoff and remaining work

[The runbook](../OCR_TESTING.md) contains exact pinned downloads, installation, Python,
line/full-page/PDF/region/table/formula CLI examples, authenticated HTTP/SSE, outputs and
repeat-request/cleanup guidance. [Benchmark methodology](../benchmarks/ocr/README.md) defines
batch 1, fresh load, one warmup, five repetitions and medians, with direct-model and Python
Engine protocols separate. Raw metrics include actual tokens/pages, stage timings, peak
MLX/process memory, provenance and accuracy; no requested-token-derived speed is published.
Optional region/reading-order/table/LaTeX metrics require annotated ground truth.

Run each downloaded checkpoint on the M4 / 32 GiB machine one at a time, inspect actual text,
then test the advertised modes and record failures by checkpoint/task. Review truncation,
real-page accuracy, formatted outputs, region geometry, repeatability and memory after cleanup.
Only promote `CHECKPOINT_SMOKE_PASSED` after a genuine image run, and `EXPERIMENTAL_SUPPORTED`
after functional/quality review. Select actual reliable JSON in `publication.json` before
regenerating README. Hardware throughput and quality remain pending for every primary model.

Remaining scope: full Paddle layout detection/structured-page parsing, non-HTML table markup
parsers, alternative HF GOT, bbox variants, a page detector for TrOCR, real handwriting fixtures,
annotated structural datasets, token-delta SSE and automated CLI/HTTP benchmark drivers.
No test in the implemented, narrowed scope currently fails; real-checkpoint behavior remains
unverified. No upstream published benchmark result was relabeled as an MLX-One measurement.
