# Native OCR: 100 random document images

This is a direct-model MLX-One experiment on the public
[OmniAI OCR dataset](https://huggingface.co/datasets/getomni-ai/ocr-benchmark/tree/4ed0d95271ca00107726230f7a0944ed9e90d897),
not a reproduction of OmniAI's official leaderboard. No extraction LLM, external inference
backend, layout detector or reference OCR engine is used.

## Data and selection

The test split has 1,000 page images. `sample.json` pins the dataset revision, source metadata
hash, all 100 original image paths and SHA-256 hashes, dimensions, source references, decoded
references and dataset metadata. Selection is without replacement:

```python
random.Random(20261025).sample(rows_sorted_by_numeric_id, 100)
```

Every page-capable checkpoint receives these same images in the same order. Images are not
selected or replaced based on model output. Source `true_markdown_output` values are decoded
as JSON strings, then literal `\n`/`\t` separators become newlines/tabs. Unicode and other
backslashes remain unchanged. Original reference values are retained for audit.

TrOCR checkpoints recognize cropped lines. This dataset supplies page images with Markdown
references and no annotated line boxes/transcripts, so both TrOCR entries are explicitly
incompatible with this page protocol. They are measured separately on real handwriting lines
in [hand10](../hand10-20261025/README.md).

## Execution and statistics

- One model per process on Apple M4, 32 GiB unified memory; exact MLX, Python and OS versions
  are recorded in every measurement JSON.
- Native MLX/Metal, batch size 1, direct-model execution through the native OCR page adapter.
  These are not Python Engine, CLI or HTTP serving measurements.
- One warmup on the first sampled image with a 16-token limit, excluded from reported timing.
  Then one measured request for each of the 100 images, greedy decoding, 1,023 output tokens
  maximum, 180-second cooperative per-request timeout. There are no five-repeat claims for
  this corpus protocol. It is separate from the synthetic OCR-S* five-repeat protocol.
- Input dimensions and resulting per-model visual/prompt token counts are retained per image.
  Each checkpoint uses its own pinned native processor and OCR prompt. Florence uses
  `ocr.recognize`; the other page checkpoints use `ocr.document`, all in text output mode.
  Default document input bounds apply before model-specific resizing: 20,000,000 decoded
  pixels per image and 32 MiB encoded size. Oversized originals are recorded as failures,
  not silently resized or substituted with easier images.
- Wall time includes image opening/decoding, preprocessing, visual encoding, prefill and
  autoregressive generation. Model load, warmup, transcript scoring, evidence writes and
  cache cleanup are outside the timed interval. GPU synchronization brackets measurements.
  Image hashes are checked before timing, so file bytes can be in the filesystem cache;
  these are not cold-storage I/O measurements.
- Median latency covers completed requests. Throughput is **completed image count divided
  by total measured request time**, including failed-request time. It is not the reciprocal
  of median latency. Actual emitted token counts, finish reasons and truncations are recorded;
  requested output budgets are never used as generated token counts.
- The README shows the maximum per-request peak MLX allocation in GiB, including resident
  weights. The allocator peak is reset after warmup/before each request. Separate process RSS
  high-water values include loading and warmup; they are not called MLX memory.

## Accuracy and limitations

CER/WER are corpus micro averages: sum of per-image Levenshtein edit counts divided by sum
of reference characters/words. Both strings receive Unicode NFKC and whitespace collapsing;
case, punctuation and Markdown syntax remain. Failed requests are empty predictions and
therefore contribute deletion errors. CER can exceed 100% with long/repeated predictions.
RapidFuzz computes exact edit distances outside inference timing. Raw predictions and edit
counts allow all published scores to be recomputed.

The reference is Markdown, while some models produce plain text. These scores therefore
include formatting and reading-order disagreement and are not a pure optical character
recognition ranking. The common output budget can truncate long documents; the README
reports truncation counts. No retry output replaces a poor prediction. No bounding-box,
table structure, JSON extraction, semantic accuracy or layout pipeline score is inferred
from text similarity. PaddleOCR-VL is direct page transcription, not the full Paddle layout
pipeline. All native OCR integrations remain experimental.

## Evidence, failures and cleanup

`O01.json`–`O07.json` identify each checkpoint and immutable revision. Measured records contain
per-image output, timing, actual tokens, allocator peak, errors and aggregates. `O*-assets.json`
retain hashes and sizes of downloaded weights/configuration/tokenizer files. Implementation
and runner digests identify the local code, alongside the Git base commit; the working tree
can contain uncommitted implementation changes. `coverage.json` identifies missing or failed
coverage. Logs and separately named failed/interrupted attempts are retained, excluded from
publication. The initial 1,024-token Florence attempt exceeded the decoder's available context;
the corrected shared protocol uses 1,023. It is not a measured performance row.
The interrupted Paddle attempt also exposed a byte-fallback tokenizer issue: an incomplete
UTF-8 sequence could temporarily revise an earlier newline. Paddle now uses the existing
token-events generation path and decodes once at page completion, consistent with public
OCR page streaming. The failed attempt, exact triggering token sequence and corrected-image
smoke are retained separately; only the complete rerun is selected for publication.
LightOn's first warmup exposed a separate prompt-construction failure: text-only chat
validation rejected its structured image message. Its pinned template is now rendered by
the existing sandbox compiler with an explicitly constructed OCR message. The template,
failure record and regression description are retained; only the corrected full run is selected.

Downloads use a dedicated task-owned temporary directory. Each checkpoint is deleted after
its page/line runs, and remaining dataset/cache files are removed at completion. `cleanup.json`
records the removed byte counts and verifies the owned directory is gone. Existing user
models and shared Hugging Face caches are never cleanup targets. Images and weights are not
stored in this evidence directory.

## Reproduce manually

From the repository root, in an Apple Silicon Python environment:

```bash
python -m pip install -e '.[ocr,ocr-bench,viz]'
WORK=$(mktemp -d /tmp/mlx-one-ocr-omni100.XXXXXX)
printf 'omni100-20261025\n' > "$WORK/.mlx-one-ocr-benchmark-owned"
python -m benchmarks.ocr.corpus prepare --work "$WORK"
python -m benchmarks.ocr.handwriting --work "$WORK"
python -m benchmarks.ocr.campaign --work "$WORK"
```

This explicitly downloads and deletes this run's temporary assets. Evidence paths are fixed
to this cohort, so archive an existing run before reproducing it. To run a single page
checkpoint, pass e.g. `--ids O03` to the campaign. Do not run two GPU benchmarks concurrently.

After all selected records are complete, validate and generate the plots/table:

```bash
python -m benchmarks.ocr.plot
python -m benchmarks.ocr.report --write
python -m benchmarks.ocr.report --check
```

The historical synthetic-fixture protocol and its still-unmeasured fixtures remain documented
in [the parent methodology](../README.md); its scores are never pooled with these images.
