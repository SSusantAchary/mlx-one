# Native task integration and 0.1.1 publication readiness

The implementation and documentation are integrated on
`integration/native-tasks-publication-20261104`, ready for PR review against
`main`. No remote branch, PR, release tag, package publication or automatic merge
has been performed. GitHub CLI, connector authentication and token credentials
were unavailable to this workflow. [Push and PR commands](pr.md) are prepared.

Version **0.1.1** is consistent across package metadata, Python, CLI, OpenAPI and
current citations. The original dirty worktree remains unchanged. The version
update exists on the isolated integration branch. Historical 0.1.0a1 measurements
retain their original identities and do not become new performance measurements.
All task decisions remain unchanged; no task is QUALIFIED.

## Integrated scope and base

The branch includes the existing native models/processors, shared registry and
inference-service integration, Python facades, CLI commands, authenticated HTTP
routes, model-specific tests, examples and reference/conversion harnesses.
Implementation files were copied byte-for-byte from the frozen development tree;
only the subsequently requested package-version strings changed in
`pyproject.toml` and `src/mlx_one/__init__.py`. No model equations, tolerances,
MLXBoost behavior or configuration defaults changed.

Documentation commit `81ea77a4db45741aefe8c18a2df6ddec2a5b03d9` was applied after
implementation commit `e9dee7c`. Its historical publication records remain
unchanged. This integration report supersedes their former uncommitted-source
merge prerequisite: the required implementations are now committed together
with the README and [MLXBoost boundary ADR](../../architecture/mlxboost-module-boundary.md).
The README's design, artwork and exact model-performance block are preserved.

The remotely verified `main` head was
`eb8bd3887ace3217ba23dc14525cef9c8a77e50d`. It already contains the frozen source
base `c0bcfba3b0d19cd50a6b547ab170e89592083438` and has the **same tree**.
A stale local tracking ref initially suggested divergence; an explicit fetch
resolved that discrepancy. Aligning the integration branch with verified main
changed no files. The PR can target `main` directly. Update and revalidate if
upstream changes before merge.

The [changed-file inventory](changed-files.json) lists every integration path,
status and ownership group. Unrelated alpha/release investigation scripts,
validation scratch files, prompts and ZIP attachments remain outside the PR.
The original 268 dirty paths were archived locally before integration; all 1,658
nonignored original files and the original HEAD/index remain unchanged.

## Fresh clean-checkout verification

A detached checkout containing only committed code was used. Locked native
checkpoints and sealed oracle traces were supplied as ignored external fixtures.
The source captures used by MLXBoost's provenance checker were copied inside the
checkout, satisfying its unchanged containment guard. No weights or evaluation
data are committed or bundled in the wheel.

| Check | Actual result |
|---|---|
| Default suite, 0.1.1 | 2,238 passed, 79 skipped, nine subtests; two existing dependency deprecation warnings |
| Explicit native/model selection, 0.1.1 | 160 passed, 17 skipped; two dependency warnings |
| Source-oracle tensor replay | 17 real-checkpoint cases, 1,566 named tensor comparisons passed at unchanged FP32 gates |
| DistilBERT SST-2 | 794/872 correct, 91.06%; native/source top-1 agreement 872/872 |
| BERT NER | 100 reconstructed sentences; TP153, FP156, FN45; strict entity F1 60.36%; source entity decisions agree |
| CLIP | Six images: 6/6 correct at two labels, 5/6 at ten; 12 complete traces pass; 100-label API input rejected |
| Silero16k | Full 60-second/1,875-chunk state/probability trace passes; ten speech clips and three authored negatives correct; source segments agree on those 13 clips |
| Installed wheel | Fresh native-only environment; 496 package files match source and installed bytes; Python/CLI/HTTP checks pass for all four profiles |
| Real HTTP serving | Four actual CLI servers over loopback; Python/HTTP equality, bearer-auth 401/200 behavior, repeatability and clean shutdown pass |
| Version | Distribution, Python, CLI and OpenAPI all report 0.1.1 |
| MLXBoost | Default off; reference_dense default; explicit eligible opt-in, unsupported fallback and strict failure verified |
| Frozen evidence | 1,379 sealed M1–M5 artifacts, all four admitted checkpoint asset sets, original CLIP conversion input and 44 pinned Transformers files rehashed successfully |
| Protected files | 46/46 unchanged in both original and integrated trees |

Default/native selections overlap; their counts must not be summed. The 17
native skips comprise two separately gated Whisper/MiniCPM5 checkpoint sessions
and 15 tests requiring an unavailable local reference snapshot. These skips do
not imply a pass. The default suite's skips retain their stated opt-in/device
requirements. The replay uses sealed HF/Silero oracle artifacts from the pinned
sources; it does not execute Transformers, Torch or ONNX Runtime in production.
The clean inference environment also excludes MLX-LM and oMLX.

The [verification summary](verification.json) records environment, wheel/source
hashes, checkpoint revisions, raw evidence locators, log hashes and per-case
parity results. Raw files stay under the ignored
`local/publication-integration/runs/20261104/` directory. The
[replay command](reproduce.md) uses a new output directory and never overwrites
historical evidence.

Initial setup failures are retained: a provenance-source symlink was correctly
rejected by the containment guard, and the new replay runner initially resolved
its venv interpreter to the system interpreter. Only the fixture setup and new
verification runner were corrected. Subsequent clean runs pass; production
models, MLXBoost and numerical gates were unchanged.

## Qualification and publication limits

Four shared task profiles remain EXPERIMENTAL: DistilBERT SST-2, BERT NER,
CLIP one-image/1–10 labels and mono16k Silero. ViT is an EXPERIMENTAL helper with
blocked task admission. SigLIP2, DETR, document/video QA and depth remain BLOCKED;
RT-DETR remains NOT IMPLEMENTED. See the unchanged
[qualification matrix](../../native_tasks/qualification-matrix.md).

No fully reproducible GPT-OSS checkpoint-provenance claim is made: its historical
checkpoint revision/weight lock was not independently established. SAM3's
report-backed controlled-fixture summaries are retained with experimental scope;
raw archive availability/numerics were not independently verified here. Neither
result establishes broad task qualification. The independently validated four
profiles are unaffected by those limitations.

Task-quality dataset reuse terms, CLIP weight licensing, VAD boundary/noise/music
quality and all recorded blocked-model numerical gates remain open. This is PR
readiness for the explicitly experimental scope, not release certification or
broader checkpoint/device qualification. MLXBoost remains an internal optional
module; standalone `mlx-one-boost` remains a proposal.

## Logical ownership, without moving files

| Logical boundary | Existing implementation locations |
|---|---|
| Models | `src/mlx_one/models/` |
| Task inference | `classification/`, `audio/vad.py`, `engine/task_runners/` |
| Processors | `vision/`, audio processing and task-specific preparation |
| Reference engine | `engine/`, `core/` |
| Optional accelerator | `mlxboost/` configuration/provenance and existing core hooks |
| Public interfaces | Package exports, `cli.py`, `server/` |
| Qualification | Model-specific tests, `benchmarks/native_tasks/`, `docs/native_tasks/` |

These are ownership boundaries, not a proposed filesystem migration. Core owns
lifecycle and fallback; models own equations and weights; MLX owns tensor/Metal
execution. No optimization was resumed and no source-neutral execution result
was used to override failed model qualification.
