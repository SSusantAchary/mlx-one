# Native OCR dataset benchmark validation — 2026-10-25

Completed on Apple M4, 32 GiB unified memory, macOS 15.8.1, Python 3.12.11 and MLX 0.32.2.
These are direct-model native MLX/Metal measurements, batch 1. Python Engine, CLI and HTTP
serving performance are not inferred from these records.

## Coverage

- **100 pinned OmniAI test images × five page checkpoints:** 500 attempts, 485 completed.
  Every model rejected the same three originals (`62`, `32`, `371`) above the default
  20-million-pixel limit. Those failures count as empty predictions in corpus error rates.
- **10 pinned handwriting test lines × seven checkpoints:** 70 completed requests.
  TrOCR Printed is an out-of-domain baseline. TrOCR Handwritten recorded 4.4% micro CER,
  10.6% micro WER and 4.670 images/s on this small sample.
- TrOCR page entries are explicitly incompatible: these checkpoints require line crops, and
  OmniAI supplies no annotated line crops. No full-page TrOCR speed is invented.

One 16-token warmup precedes each measured pass. Each image is measured once; page output
is capped at 1,023 tokens and handwriting output at 128. Input/visual token counts vary by
image and processor. Median request latency, actual completed-count throughput, maximum
peak MLX allocation, transcript-derived micro CER/WER and truncations are retained separately.
The page metric compares against Markdown, including formatting/reading-order disagreement;
it is not an official OmniAI leaderboard score or pure character recognition ranking.

The [performance tables](../../benchmarks/ocr/performance.md#ocr-model-performance) and both charts are generated
from validated per-image JSON. Complete records, immutable dataset/checkpoint revisions,
sample selection, references, weight/tokenizer hashes and reproduction commands are in:

- [100-page methodology and evidence](../../benchmarks/ocr/omni100-20261025/README.md)
- [Handwriting methodology and evidence](../../benchmarks/ocr/hand10-20261025/README.md)
- [Page coverage](../../benchmarks/ocr/omni100-20261025/coverage.json)
- [Handwriting coverage](../../benchmarks/ocr/hand10-20261025/coverage.json)

## Bugs found and corrected

1. **Paddle byte-fallback decoding:** an incomplete UTF-8 byte run could temporarily replace
   an already decoded newline. The token streaming helper correctly rejected the revision.
   Paddle now reuses the existing token-events path and decodes once at page completion,
   consistent with public OCR page streaming. A native regression covers newline + split
   Unicode; the affected real image (`947`) completed after the fix. The entire Paddle
   sample was rerun; its partial failed attempt is excluded from publication.
2. **LightOn image prompt:** its real checkpoint template consumes structured image content,
   which the text-only chat validator rejected. The OCR adapter now renders its explicitly
   constructed image message through the existing sandbox compiler, preserving strict text
   validation. Native template coverage and the complete corrected checkpoint run passed.
   The failed pre-inference warmup has no published performance numbers.

The initial Florence request budget also needed correction from 1,024 to 1,023 output tokens
to reserve its decoder start position. Its initial all-rejected context-budget attempt is
retained separately and excluded from publication. No failed or interrupted attempt is pooled
with the complete selected runs.

## Checks and cleanup

**179 targeted tests passed** in 8.75 seconds, with two existing dependency deprecation
warnings. Coverage includes all native OCR families, Unicode/template regressions, evidence
validation, registry/loader, Python service, CLI, HTTP/SSE and existing VLM/audio integrations.
The wheel builds successfully and contains both corrected OCR modules. Exact command,
wheel hash and current code identities are in [summary.json](summary.json).

The retired exhaustive numerical qualification suite was not run. These benchmarks establish
execution and measured behavior on the selected tasks, not blanket task support, PDF/region
quality, full Paddle layout support or a broad handwriting qualification.

Both task-owned model/data/cache directories are verified removed. Individual removal events,
including repeated downloads needed for fixes, remain in the cleanup manifests. Existing user
models and pre-existing Hugging Face chunk caches were preserved; the shared-cache audit
found no task-created files remaining.

- [Page campaign cleanup](../../benchmarks/ocr/omni100-20261025/cleanup.json)
- [TrOCR handwriting cleanup](../../benchmarks/ocr/hand10-20261025/cleanup.json)
- [Shared cache audit](../../benchmarks/ocr/omni100-20261025/shared-cache-audit.json)

README content outside the OCR performance block is preserved, apart from replacing one stale
OCR “full-checkpoint validation remains pending” sentence with the current experimental status.
Both new plots were visually inspected and are positioned above their respective tables.
