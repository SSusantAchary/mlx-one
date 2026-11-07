# README publication audit — 2026-11-04

This is a documentation audit of the existing dirty development tree at base
`c0bcfba3b0d19cd50a6b547ab170e89592083438`. It introduces no model implementation,
MLXBoost optimization, dependency change or qualification promotion. Historical
performance is not a fresh measurement. The source tree and Git index remain
unchanged.

The [machine-readable evidence map](evidence-map.json) contains exact selectors,
hashes, checkpoint revisions, measurements and decisions. Run
`python docs/publication/2026-11-04/verify_publication.py` with the development
environment to verify the local evidence again. It requires the existing local
sealed runs and sibling pinned Transformers repository; a public checkout
without those artifacts can inspect the report but cannot reproduce their hash
verification. `--markdown-only` checks publication links, anchors and fences
without requiring the local model/evaluation artifacts. The
[HTML preview](README.preview.html) preserves the logo, architecture image and
tables; GitHub renders the Markdown source and its alert/diagram extensions.

## Claim decisions

| README claim group | Decision | Evidence and scope |
|---|---|---|
| Existing text performance table | HISTORICAL-VERIFIED | Existing generator validates all 17 measurements / 13 checkpoint revisions; 16 historical 128-input/16-output diagnostics and one standard GPT-2 128/128 measurement; table bytes unchanged |
| VLM performance | HISTORICAL-VERIFIED | Existing validator: 40 records, 20 measured; README selects five VLM-HF-S1 real-image records, not every run; processor/forward/decode timing columns retain their original definitions |
| Audio performance | HISTORICAL-VERIFIED | 14 selected records, 13 measured, one failed speech-to-speech workload; one excluded warmup and five repeats per measured workload; quality is fixture-specific |
| OCR performance | HISTORICAL-VERIFIED | Existing corpus validator selects 100-page and ten-handwriting-image runs; failed/truncated samples remain in the evidence; synthetic fixture records publish no speed figures |
| GPT-OSS HTTP / Kilo | HISTORICAL-VERIFIED report only | Five 175-input/91-output requests, separate native TTFT and first actionable HTTP delta; historical report has no newly verified immutable checkpoint lock; no 128K reliability claim |
| SAM3 video | HISTORICAL-VERIFIED report only | Pinned checkpoint/reference report, controlled single-object 32-frame fixtures; raw archived numerical traces are not certified by this publication audit; no natural-video qualification or isolated timing claim |
| Existing audio qualification summaries | HISTORICAL-VERIFIED report only | The [audio qualification report](../../audio-qualification-m4-32gb.md) retains exact revisions, 100-clip/16-anchor scope, language limits and failed alignment/ASR gates; no new closure decision |
| Existing text checkpoint admission | VERIFIED current policy / historical smoke | [Functional evidence summary](../../qualification/functional-evidence-summary.json), [admission policy](../../model-admission.md) and pinned [inventory](../../benchmarks/inventory.json); architecture/smoke status is not numerical release qualification |
| Four native task profiles | HISTORICAL-VERIFIED | DistilBERT, BERT NER, CLIP 1–10 labels and Silero16k; Python/CLI/HTTP evidence; all EXPERIMENTAL |
| Native-task latency | HISTORICAL-VERIFIED | Four 20-request series, one excluded warmup/process, median and nearest-rank p95 recomputed from every raw value; VAD input is one second of decoded PCM |
| Native-task regression counts | HISTORICAL-VERIFIED | M5 log summaries: 2,238 passed / 79 skipped / 9 subtests; native selection 157 passed; overlapping selections are not summed |
| MLXBoost defaults, module ownership and fallback | VERIFIED current code | Configuration, mode resolution, construction and pressure/calibration hook plus focused regressions; optional internal module, flags default false |
| MLXBoost HTTP rates | HISTORICAL-VERIFIED | Raw per-request CSV medians recomputed for each model/flag, preserving all timed requests and original cohorts |
| Standalone `mlx-one-boost` | PROPOSED ONLY | [Boundary ADR](../../architecture/mlxboost-module-boundary.md); no distribution or migration is shipped |

The qualifiers in model-guide tables retain the scope of their linked reports.
In particular, Moonshine short-form English passes, partial Qwen ASR/aligner
failures and Parakeet language-specific passes do not establish full-family
qualification. Architecture checkmarks describe implementation/synthetic gates;
the fresh default suite validates registry and protocol regressions, not every
real checkpoint. Training/MTP/CPT/RL in the unchanged artwork remain planned
where the README says they are planned.

## Native task provenance

All 1,379 files in the five sealed M1–M5 manifests were rehashed: 114, 335, 365,
280 and 285 respectively. The four indexed manifest hashes match the completion
record; M5 is separately hashed. All available assets in all eleven checkpoint
lock entries match their recorded hashes; RT-DETR explicitly has no measured
asset inventory. All 44 locked Transformers source files match Git object bytes
at `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc` (v4.57.1). Silero source/weight
assets are independently locked to `60b7ffa243625ebdc1070275a29f18c87843786a`.

The initial NOT MEASURED paragraphs in the historical task-quality/performance
notes predate their later M1–M3 results; the sealed measured runs and current
qualification matrix take precedence.

The public evidence map includes checkpoint/processor/weight hashes but no
evaluation text, raw images, audio, credentials or absolute user paths. It links
run identifiers as evidence locators; ignored `local/` artifacts are not uploaded.
Independent readers need those exact artifacts to repeat the full raw-data
audit. The source lock, current matrix, API contracts, task-quality and latency
reports accompany the patch.

Exact admitted checkpoint revisions:

| Checkpoint | Revision |
|---|---|
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| `dslim/bert-base-NER` | `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` |
| `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| `snakers4/silero-vad-16k` | `60b7ffa243625ebdc1070275a29f18c87843786a` |

CLIP weights were converted offline from the locked original; all stored
tensors round-trip exactly. The Silero oracle rebinds the source JIT graph to
the selected safetensors parameters because the original JIT has 14 different
learned tensors. No comparison to those different original weights is claimed.
SST-2/CoNLL reuse terms and CLIP weight licensing remain open. VAD clip-presence
checks and source timestamp agreement are not human-labeled boundary quality.

## MLXBoost evidence

Runs are under `local/mlxboost/runs/confirmation-20261101/`. Select
`backend=mlx-one`, the named model, and every timed row of `per_request.csv`.
The rate is each row's `completion_tokens / latency_seconds`, then the median.
This includes the complete HTTP SSE response; it is not decode-only rate.

| Model | Off run / on run | Timed rows per flag | Median off / on (tok/s) |
|---|---|---:|---:|
| Qwen2.5-0.5B Q8 | `expanded-qwen2-off` / `expanded-qwen2-on` | 50 | 22.94 / 109.17 |
| Qwen3-0.6B Q8 | `expanded-qwen3-off` / `expanded-qwen3-on` | 50 | 21.42 / 96.42 |
| SmolLM2-1.7B BF16 | `original-protocol-off` / `original-protocol-fast` | 6 | 15.39 / 24.56 |

Qwen: five processes per flag, ten timed requests/process, one excluded warmup.
SmolLM2: three processes per flag, two timed requests/process, one excluded
warmup. Loading is excluded. Qwen2 ran on then off; Qwen3 reversed that order.
The Qwen comparison report records exact output text/count/finish-reason matches
for all 100 paired cases. Thermals and external host load were not controlled;
these cohorts do not establish a service SLA or general model-forward gain.

The source study pins MLX baseline `1f8e74e3f12f31365464a6867c6579f0e9b29d85`,
MLX comparison `36969c094354086e15d2a327a9c50feb20b9cf11`, MLX-LM
`93e6d4a6447b3080861985947f0820e76fcb957d` (MIT) and oMLX
`9de79bc39a01c1f3fec1e849903aaf0dd5478d7a` (Apache-2.0, with separately
identified MLX MIT kernel portions). No oMLX code or custom kernels are adopted.
The README off/on comparison uses native MLX 0.32.2 on both sides. Separate
historical competitor environments used 0.32.4; their rates are not mixed into
this table. Existing strict cached-vs-fresh failures remain failures.

## Omitted or unavailable claims

- No new task is QUALIFIED. ViT helper access is not shared task admission;
  SigLIP2, DETR, document/video QA and depth remain BLOCKED; RT-DETR remains
  NOT IMPLEMENTED.
- No full-dataset NER accuracy, general CLIP accuracy, VAD boundary F1, labeled
  depth accuracy, document benchmark score or end-to-end video-answer result.
- No GPU-kernel gain inferred from HTTP rates, no production MLXBoost rollout,
  and no standalone-package installation instructions.
- No fresh performance claim from regression tests or clean-wheel smoke runs.
- GPT-OSS immutable historical checkpoint locking and raw SAM3 archives are
  UNVERIFIED by this audit. Their unchanged README numbers are attributed to
  the historical reports with this provenance limitation.

## Fresh verification

Fresh test, wheel, Markdown/link, preservation and patch results are recorded in
[verification.json](verification.json). Logs stay in the ignored publication run
directory, with public SHA-256 values in that summary. Warnings, skips and
external-link failures are disclosed there. No historical manifest is rewritten.

## PR scope

The documentation-only patch targets the audited development tree, whose model
implementation was already dirty and uncommitted. It does not ship those model
changes. It must be applied on top of that implementation before merging; a
documentation commit on the old base alone cannot make new task APIs available.
GitHub CLI and authenticated connector/token credentials were unavailable, so
no remote branch or PR is claimed. See [PR preparation](pr.md) for the exact
patch scope and application instructions.
