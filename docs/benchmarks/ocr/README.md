# OCR benchmark methodology and coverage

Real-image corpus runs have separate protocols and evidence:

- [100 random OmniAI document images](omni100-20261025/README.md), with per-checkpoint
  predictions, latency, throughput, memory, Markdown-reference CER/WER and failures.
- [10 random handwriting test images](hand10-20261025/README.md), including TrOCR
  Handwritten and a separate printed-checkpoint baseline.

The seven **synthetic OCR-S*** records in `results/` remain explicitly `NOT_RUN`;
their missing performance values are null. Native tiny-model tests and header-shape checks
do not establish OCR quality or full-checkpoint speed. Do not mix the real-image cohorts
with the synthetic five-repetition protocol described below.
The [inventory](inventory.json) pins checkpoint revisions, precision, tokenizer assets and
primary workloads; [coverage.json](coverage.json) lists missing primary measurements.

Use Python 3.12 on the intended Apple M4 / 32 GiB machine with the project's pinned MLX
and Metal versions. Install `pip install -e '.[ocr,server]'`. Download checkpoints manually
using [the OCR runbook](../../OCR_TESTING.md). Runtime OCR defaults to local/cache access;
never benchmark a different checkpoint under an inventory name.

## Workloads

`python -m benchmarks.ocr.prepare` creates small, stored synthetic fixtures and a manifest
with exact file hashes. The generated manifest and files should be retained together.
Regeneration can produce a new PDF identity; retain its manifest as a distinct workload
revision. No real-world corpus accuracy follows from these fixtures.

| ID | Fixture | Meaning |
|---|---|---|
| OCR-S1 | Printed English line | TrOCR printed baseline |
| OCR-S2 | Receipt | Numeric/date text fidelity, not receipt field inference |
| OCR-S3 | Page | Heading/paragraph text transcription |
| OCR-S4 | Two columns and footnote | Known textual reading order |
| OCR-S5 | Simple table | Text and structure must be assessed separately |
| OCR-S6 | Formula text | Textual LaTeX match is not mathematical equivalence |
| OCR-S7 | Ten-page scanned PDF | Fixed DPI, sequential visual processing |
| OCR-S8 | **Missing** real handwriting | Supply a licensed scan and transcript; printed fonts are not handwriting |
| OCR-S9 | Rotated/blurred text | Synthetic degradation; not a TrOCR line workload |

Real scans must use a separately identified manifest with dataset name, revision, license,
fixture hash and transcript. Do not merge real and synthetic workloads.
Table markup, annotated polygons and formula structure references should be recorded
separately when available. The metric implementation publishes text CER/WER and exact match. With separately supplied
`reference_regions`, it computes page-scoped greedy descending-IoU matching at IoU ≥ 0.5,
precision/recall/F1, matched-box mean IoU/coordinate error, and reading-order pair accuracy
on matched regions (with evaluated pair count). This is box IoU, not polygon IoU or COCO AP.
No detection scores are computed for text-only tasks. `reference_tables` enables row/cell-count
exact match and position-aligned cell-text/span fidelity, not TEDS. `reference_latex` enables
whitespace-normalized exact match, not mathematical equivalence. These optional metrics
remain null without annotations; no structure or annotations are inferred from text.
Table/formula output is retained for independent review.

## Protocol

One checkpoint per fresh benchmark process; batch 1; model load timed separately; one
warmup; five measured repetitions; per-column median. MLX operations synchronize around
wall-clock measurements; encoder timings evaluate their arrays before stopping. The
end-to-end measurement includes document decoding/rendering, preprocessing, visual encoding,
language prefill and cached generation. Per-page stage timings and actual processed sizes,
visual tokens and emitted output counts are retained outside the README. Page throughput
uses actual completed pages, never requested pages. Early EOS and truncation are recorded.

Peak MLX memory resets after warmup before each request, includes resident weights, and is
reported in GiB. Process memory is `getrusage().ru_maxrss` on macOS: the process high-water
mark including loading/warmup, **not** a resettable request-only RSS peak. It is retained
in raw samples, separate from MLX allocator memory. Model revision, checkpoint asset hash,
workload/fixture hashes, implementation/runner hashes, hardware, MLX, Python and OS are required.
Downloaded config/weight metadata must agree with `--revision`; snapshot directory names
are also accepted. Checkpoints copied without provenance must be re-downloaded manually
or their original metadata restored before publication.

Raw text scores preserve case, punctuation and whitespace. Normalized scores use Unicode
NFKC and collapse whitespace; they still preserve case and punctuation. CER is character
Levenshtein distance divided by reference characters, WER uses whitespace-separated words,
and exact match compares the complete strings. Errors can exceed 100%; no score is clipped.
Embedded PDF text is never an OCR accuracy measurement; benchmark PDFs always use `visual`.

## Commands

From the repository root, after manual download:

```bash
python -m benchmarks.ocr.run \
  --model ~/mlx-one-models/ocr/got-ocr2 \
  --checkpoint mlx-community/GOT-OCR2_0-4bit \
  --revision 0e28c87720de5d8af67389a0444cf8397524cb61 \
  --workload OCR-S3 --task ocr.document \
  --max-tokens 1024 \
  --output docs/benchmarks/ocr/results/O03-OCR-S3-direct.json
```

Use `--measurement-type python-engine` and a distinct output filename for the Python
Engine path. CLI and HTTP runs are different protocols and must not be relabeled as direct
inference. The runner currently automates direct-model and Python Engine measurements;
CLI/HTTP timing automation remains a separate integration task.

Inspect successful output and accuracy, then explicitly replace the matching pending
record path in `publication.json` with the measured JSON path. Each checkpoint/task/workload/
measurement type can be selected only once. Do not select tiny-model/mock records.

```bash
python -m benchmarks.ocr.report --write
python -m benchmarks.ocr.report --check
```

The generator reads only selected records, ignores mock entries, rejects incomplete
provenance and duplicate selections, and preserves all other README sections. Each row
links to evidence and identifies its workload/hardware cohort; different OCR tasks and
workloads are not pooled. No chart is generated while all checkpoints remain unmeasured.
