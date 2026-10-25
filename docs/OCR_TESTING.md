# Native OCR and document testing

OCR is experimental source support. All six architecture families have native source paths,
strict safetensors loading, processors, Python/CLI/HTTP integration and tiny-model execution
tests. The seven pinned primary checkpoints have matching native parameter names/shapes
against their published safetensors headers. The [initial implementation report](validation/ocr-native-2026-10-25.md)
used metadata and tiny models only. Full-checkpoint execution and measured limitations are
now tracked separately in the [100-page corpus evidence](benchmarks/ocr/omni100-20261025/README.md)
and [10-line handwriting evidence](benchmarks/ocr/hand10-20261025/README.md).
These experimental corpus runs do not establish every advertised task's accuracy.

## Architecture inventory

Reuse: `core/registry.py`, `models/registry.py`, `text/loading.py` safetensors/quantization,
`text/generation.py` sampler and decoding, existing Qwen2/Qwen3 models, selected expert MLX
operations, `core/cache.py` self/cross state, `InferenceService` and the existing device actor,
`ModelManager`, FastAPI authentication/upload limits and the CLI command group.

New: `models/ocr/` model-specific native vision and text math; `ocr/` schemas, document
rendering, coordinate transforms, tokenizer assets, processor selection and postprocessing;
`engine/task_runners/ocr.py`; `server/ocr.py`; and `benchmarks/ocr/`.
No second generic VLM engine, generation scheduler or paged-KV implementation was added.

| Family | Implemented source scope | Not advertised / limitations |
|---|---|---|
| TrOCR | ViT, cross-attention TrOCR decoder, cached line recognition | No page detector; printed checkpoint is not a handwriting model; no boxes/Markdown |
| GOT-OCR 2.0 | SAM-style frontend, Qwen2, Q4, original task prompts, plain/formatted page OCR | Original MLX checkpoint only; crop-first regions; no native region geometry or GOT multi-crop mode |
| Florence-2 | DaViT, BART encoder/decoder, OCR and OCR-with-region parsing | OCR tasks only; quadrilaterals only when emitted, no arbitrary vision tasks |
| PaddleOCR-VL 1.5 | Dynamic-resolution vision/projector, ERNIE decoder with sectioned 3D RoPE, Q4, page/crop OCR and table/formula generation | No separate layout detector, text-spotting parser or complete Paddle structured-page pipeline; HTML cells parsed when actually emitted, other table markup stays raw |
| LightOnOCR 2 | Pixtral frontend/projector plus exact Qwen3 decoder; page-to-text/Markdown | Base variant has no bbox task; reading order/LaTeX remain model output, not independently established quality |
| DeepSeek-OCR | SAM + CLIP compression, native non-MLA MoE decoder, affine Q4, 1024 global / 640 local crops, page OCR/Markdown | Pinned original OCR variant only; at most nine local crops; no measured compression speedup or geometry parser |

`ARCHITECTURE_ONLY` means no complete task; `NATIVE_IMPLEMENTED` means synthetic native task
integration passes; `CHECKPOINT_SMOKE_PASSED` requires the exact real checkpoint; and
`EXPERIMENTAL_SUPPORTED` additionally requires the advertised real-image functional cases.
Use `WIP` for a known defect or incomplete task, and `UNSUPPORTED` for a missing implementation.
Checkpoint execution is independent of source support; consult the per-corpus coverage and
per-image records for actual runs, failures and truncations. The synthetic OCR-S* fixtures
retain their own independent NOT_RUN status until measured with that protocol.
Full Paddle layout parsing and the alternative `stepfun-ai/GOT-OCR-2.0-hf` layout are not supported.

## Install and manually download

```bash
cd /Users/susant-achary/dev_work/git_codes/c_base/mlx-one
source .venv/bin/activate
python -m pip install -e '.[ocr,server]'
```

Only Pillow, PDFium and the original GOT tokenizer are added as optional lightweight
utilities. Safetensors and standalone tokenizers are already native dependencies.
Never enable remote model Python code. No offline converter is required for the pinned
primary repositories: all seven publish safetensors. Original TrOCR's vocab/merges are loaded
with standalone tokenizers; GOT's `qwen.tiktoken` uses optional tiktoken.

Pinned manual downloads (these commands are **not** run automatically):

```bash
hf download microsoft/trocr-base-printed \
  --revision 93450be3f1ed40a930690d951ef3932687cc1892 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/trocr-base-printed

hf download microsoft/trocr-base-handwritten \
  --revision eaacaf452b06415df8f10bb6fad3a4c11e609406 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/trocr-base-handwritten

hf download mlx-community/GOT-OCR2_0-4bit \
  --revision 0e28c87720de5d8af67389a0444cf8397524cb61 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/got-ocr2

hf download microsoft/Florence-2-base-ft \
  --revision f6c1a25888ffc1d945ee8a1a77ac833c7303d46e \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/florence2-base

hf download mlx-community/PaddleOCR-VL-1.5-4bit \
  --revision 3404726b40ac30b6e4e3d37c504e3f96c9c6c984 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/paddleocr-vl

hf download lightonai/LightOnOCR-2-1B \
  --revision be7c41313c30e940fe5571de59e5fabc779e00c8 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/lightonocr2

hf download mlx-community/DeepSeek-OCR-4bit \
  --revision 4958b6b0d46d7dc3cf27556989dab659996ea426 \
  --exclude '*.bin' '*.pt' '*.pth' \
  --local-dir ~/mlx-one-models/ocr/deepseek-ocr

```

The [inventory](benchmarks/ocr/inventory.json) records processor/tokenizer assets and licenses. Review the linked pinned model cards before downloading; printed TrOCR does not declare a license in the retrieved card metadata, so its terms must be checked explicitly.

## Python

```python
from mlx_one import load_model

with load_model("~/mlx-one-models/ocr/got-ocr2", task="ocr.recognize", offline=True) as ocr:
    result = ocr.recognize(image="receipt.png", max_tokens=1024)
    print(result.text)
    result.save("receipt.json")

with load_model("~/mlx-one-models/ocr/lightonocr2", task="ocr.document", offline=True) as ocr:
    result = ocr.parse_document("document.pdf", pages=(0, 1, 2),
                                output_format="markdown", dpi=144, max_tokens=2048)
    result.save("document.md")
```

Use an in-memory Pillow image, encoded image bytes or PDF bytes in the same calls.
`OCRInput` and `OCRLimits` expose typed request settings and safety bounds.
`ocr.batch(images)` is a lazy sequence of document results with a maximum image count;
close the iterator if stopping early. `ocr.stream(source)` emits page events and must also
be closed when abandoned. Each page gets fresh visual features and self/cross-attention
state; adjacent pages never reuse a visual prefix. Close the engine to release its model lease.

## CLI

```bash
mlx-one ocr --model ~/mlx-one-models/ocr/trocr-base-printed \
  --image line.png --format text --offline

mlx-one ocr --model ~/mlx-one-models/ocr/trocr-base-handwritten \
  --image handwriting.png --task ocr.handwriting --format json --offline

mlx-one ocr --model ~/mlx-one-models/ocr/got-ocr2 \
  --image receipt.png --region 100 150 600 320 --format markdown --offline

mlx-one ocr --model ~/mlx-one-models/ocr/lightonocr2 \
  --document document.pdf --pages 1-5 --dpi 144 --max-pages 10 \
  --max-tokens 2048 --format markdown --output document.md --offline

mlx-one ocr --model ~/mlx-one-models/ocr/florence2-base \
  --image receipt.png --task ocr.detect --format json --offline

mlx-one ocr --model ~/mlx-one-models/ocr/paddleocr-vl \
  --image table-crop.png --task ocr.table --format json --offline

mlx-one ocr --model ~/mlx-one-models/ocr/paddleocr-vl \
  --image formula-crop.png --task ocr.formula --format json --offline

mlx-one ocr --model ~/mlx-one-models/ocr/got-ocr2 \
  --document document.pdf --format json --stream --offline
```

Python page indices are zero-based. CLI/HTTP ranges are one-based inclusive (`1-5,8`).
Pages are returned in document order. Nonexistent/duplicate/out-of-limit selections fail;
no page is silently skipped. `--output` writes exactly the requested path and selected format.

## HTTP

```bash
mlx-one serve --model ~/mlx-one-models/ocr/got-ocr2 \
  --offline --host 127.0.0.1 --port 8085 --alias got-ocr2 --api-key local-ocr --no-warmup

curl --fail-with-body http://127.0.0.1:8085/v1/ocr \
  -H 'Authorization: Bearer local-ocr' \
  -F 'model=got-ocr2' -F 'file=@receipt.png' \
  -F 'task=ocr.recognize' -F 'output_format=json' -F 'max_tokens=1024'
```

Pass the exact `id` returned by `GET /v1/models` in `model`; to set a predictable id,
add `--alias got-ocr2` to the serve command. The `got-ocr2` request above assumes that alias.
Uploads are multipart files, never server-side paths or URLs. PDFs and supported images are
decoded by file contents. Authentication, size, type, page range, timeout, disconnect and
model/task errors are handled explicitly. Responses use JSON with the result schema even
when a text/Markdown representation was requested. Add `-F 'stream=true'` and `curl -N` for
SSE `page_started`, `page_completed`, `document_completed` (or `error`) events. These are
**page-level**, not token-level, events. Token generation itself remains incremental internally.
Dedicated OCR models are not advertised as chat-completions models; existing VLM chat
image inputs continue using their existing endpoint and runtime.

## Input policies, coordinates and limits

Default bounds: 32 MiB encoded input; 100 source/selected pages; 20 million decoded pixels
per page; DPI 36–300; nine local DeepSeek tiles; 256 MiB aggregate rendered bytes; 8 GiB
estimated request transients; 300 seconds per document (server timeout can lower this).
Resident model weights are separate. GPU operations are bounded and cancellation is checked
between generation steps; an in-flight Metal operation cannot be preempted.
PDFium use is serialized across documents because its library is not thread-safe.

PNG/JPEG/WebP and sequential TIFF frames support RGB, grayscale and alpha-on-white.
EXIF orientation is applied before defining the public page coordinate system. `region`
is an explicit crop-first workflow in integer oriented source pixels. External boxes are
`[xmin,ymin,xmax,ymax]`, top-left origin; polygons map through the documented crop/resize
transform. PDF coordinates use rendered pixels at the requested DPI; original point sizes,
rotation and cropbox are separate metadata. Florence's 1000-bin coordinate centers are
mapped back into that source system; no boxes or confidence are created for plain text.

PDF policy `visual` always renders and runs OCR. `embedded_text` returns only existing text,
including an explicitly empty page when absent. `auto` chooses embedded text when that page
has any non-whitespace embedded characters, otherwise visual OCR. It never appends both.
Embedded extraction is not OCR and does not support inferred region/table/formula structure.

JSON includes `text`, optional `markdown`, ordered `pages`, `model`, `task`, `language`,
`timings`, `metadata`, and schema version `mlx-one-ocr-v1`. Pages carry original/rendered
sizes, text, optional predicted structure and geometry, timings and processor/token metadata.
Model-emitted text/Markdown/HTML is untrusted output; it is never executed as instructions,
tool arguments or filesystem operations. Markdown is only exposed for the models/modes
that generate it; no detailed document structure is invented from plain transcription.

## Tests and benchmark handoff

```bash
MLX_ONE_RUN_MLX_TESTS=1 python -m pytest tests/ocr --no-cov -q
python -m benchmarks.ocr.prepare
python -m benchmarks.ocr.run --help
python -m benchmarks.ocr.report --check
```

See [the benchmark protocol and command](benchmarks/ocr/README.md) for actual M4 measurements.
Start with GOT or TrOCR, inspect every output and then exercise full-page, PDF, region,
table/formula and repeated requests where applicable. Monitor peak memory and compare two
identical requests. Check truncation metadata on dense pages, review accuracy against ground truth,
and record real failures as WIP at the checkpoint/task level. Keep raw JSON and
workload hashes, then explicitly select reliable measurements for README publication.
