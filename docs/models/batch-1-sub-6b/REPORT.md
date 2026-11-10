# Native model validation report

Completed for review on **10 November 2026**, on branch `feat/batch-1-sub-6b`,
based on `d922903`. The implementation and evidence are uncommitted local work.
No package version, release approval, GitHub remote or PyPI publication was changed.

**Ten checkpoint profiles perform their scoped primary tasks natively**, including
real Python, CLI, Engine and HTTP execution from an isolated installed wheel.
The registry grows from 40 to **45 architecture types**. BERT serves four of
these checkpoint profiles and XLM-R serves two; checkpoint count does not inflate
the architecture count. The original 40-type 0.1.1 approval remains a dated cohort.

## Results and acceptance

| ID | Checkpoint | Native and real task | Reference result | Python / CLI / Engine / HTTP | Scope |
| --- | --- | --- | --- | --- | --- |
| B01 | MiniLM MS-MARCO cross-encoder | PASS | Pair logits and trained scalar head PASS | PASS | FP32 paired reranking |
| B02 | BGE-small-en-v1.5 | PASS | CLS vector and normalization PASS | PASS | FP32 dense retrieval |
| B03 | BGE-M3 | PASS | Dense, learned sparse and ColBERT heads PASS | PASS for dense; extra heads Python/CLI | FP32 backbone, source FP16 auxiliary storage; hybrid scoring |
| B04 | multilingual MiniLM L12 | PASS | Raw masked mean, no Normalize stage, PASS | PASS | 128-token sentence profile |
| B05 | T5-small | PASS | Teacher-forced logits and exact bounded greedy tokens PASS | PASS | Encoder-decoder text-to-text |
| B06 | ViT-base patch16-224 | PASS | Processor, logits and exact top-five PASS; fixed hidden diagnostic FAIL | PASS | Supervised image classification |
| B07 | BGE reranker v2 M3 | PASS | Paired XLM-R scalar logits PASS | PASS | FP32 paired reranking |
| B08 | Kokoro-82M | PASS | Acoustic stages/durations PASS; controlled vocoder correlation ≈0.99999997 | PASS | 24 kHz English, af_heart only |
| B09 | Nomic Embed v1.5 | PASS | Default vector and all five explicit Matryoshka dimensions PASS | PASS | 768-D default or explicit 64/128/256/512/768 |
| B10 | multilingual E5-small | PASS | BERT math, role prefixes, masked mean and normalization PASS | PASS | 384-D asymmetric dense retrieval |

The [machine-readable result manifest](../../qualification/batch-1-sub-6b/results.json)
has exactly ten entries with pins, asset hashes, source identities, parameter
counts, per-mode checks, interfaces, cleanup and limits. Its `SUPPORTED` status
applies to the named primary workflow; `VALIDATED_SCOPED` records the checkpoint
admission tier. It does not approve a new package release, untested checkpoint,
precision, context, language, voice or optional task.
G0–G4 checks pass for these primary workflows. G5 evidence and documentation are
prepared for maintainer acceptance; the manifest leaves that review explicit.

The packaged development catalog exposes these exact checkpoint identities to
task discovery. Later exact-checkpoint metadata supersedes ViT's old planned
entry without editing the historical native-task manifest. Unknown revisions
or a mismatched architecture receive no support promotion.

## What was implemented and reused

- Extended the existing BERT encoder, strict loader, tokenizers, retrieval service
  and task adapter with true paired segments, scalar trained heads, CLS pooling,
  publisher normalization and explicit role prefixes. Existing MPNet and Qwen
  retrieval contracts retain their own behavior.
- Added native XLM-R non-padding positions, classification head and BGE-M3's
  three learned representations; native Nomic fused-QKV/RoPE/SwiGLU and correct
  Matryoshka preprocessing; native T5 encoder-decoder attention and relative
  buckets; native ViT patch encoder/classifier and exact publisher image processor.
- Added native Kokoro ALBERT, recurrent/acoustic stack, adaptive convolutions,
  harmonic source, vocoder and inverse FFT. Reused the existing native audio
  ISTFT helper. Only the English rule-based G2P frontend is an optional external
  tool; every neural layer runs in MLX-One.
- Added safe, pinned offline conversion for BGE-M3 and Kokoro. The inference
  environment contains none of Torch, Transformers, SentenceTransformers,
  FlagEmbedding, Kokoro, MLX-LM or MLX-VLM.
- Reused the task actor, model leases, cancellation and server lifecycle.
  Added explicit text-to-text, supervised image classification and speech routes.
  HTTP image inputs are bounded data URIs rather than server filesystem paths.
- Added a [native retrieval/reranking demo](../../../examples/batch1_retrieval.py),
  [usage guide](README.md), [provenance map](references.md), current checkpoint
  catalog, benchmark JSON/CSV and website navigation/coverage labels.

All native forward files and task interfaces are identified in the result
manifest. [Implementation file hashes](../../qualification/batch-1-sub-6b/source-files.json)
identify the actual working source; the base Git commit alone does not identify
uncommitted implementations.

## Validation

- Full applicable native suite: **2,599 passed, 95 skipped, 9 subtests passed**,
  with two inherited web-test deprecation warnings. No weight downloads occur in
  ordinary tests. The original baseline was 2,254 passed without native opt-ins.
- Full backend-free Python **3.10, 3.11, 3.12 and 3.13** matrix: all passed,
  2,255 passed / 122 skipped per interpreter, before the additive catalog lookup.
  Focused follow-up checks cover that metadata change on all four interpreters.
  Native checkpoint execution and the optional English frontend were checked on
  Python 3.12; this is not all-hardware or all-extra native qualification.
- After final Kokoro admission accounting changed, **54 targeted native,
  interface, actor and model-pool checks passed**. The final admission/catalog
  contracts also pass on Python 3.10–3.13.
- Tiny native fixtures independently generated by pinned Transformers cover
  T5, ViT and XLM-R outputs; Nomic masking/Matryoshka, BGE heads, paired token
  truncation and the Kokoro FFT regression have focused contracts.
- Every checkpoint has an installed CLI/actor/HTTP result, API-key rejection,
  malformed request rejection, wrong-model rejection, pre-canceled task and
  closed-service/shutdown checks. Existing context-manager/lease tests and the
  full regressions also pass. Cancellation occurs at bounded task boundaries.
- Lint adds **zero** findings to the inherited 65-item debt list. Candidate
  installation used only declared core and `server,vision,kokoro` dependencies;
  dependency checking, final wheel payload identity and packaged notices are
  recorded separately in [verification](../../qualification/batch-1-sub-6b/verification.json)
  and [distribution checks](../../qualification/batch-1-sub-6b/distribution.json).

The native wheel runs precede the final packaged evidence catalog. The
[source lineage receipt](../../qualification/batch-1-sub-6b/source-lineage.json)
identifies the final differences: metadata lookup additions, plus tighter Kokoro
configuration validation after the earlier non-Kokoro runs. Kokoro itself was
rerun with those constraints. Final native equations match the per-task wheel
runs; the final wheel's files and metadata are checked against actual installed
bytes. No claim is made that all differently dated candidate wheels have one hash.

## Measurements and retained failures

ViT's strict hidden-state comparison at `atol=rtol=1e-4` remains **FAIL**.
Class logits pass (maximum absolute error about `1.72e-5`), and top-five indices
match exactly. Against an independent publisher FP64 output, publisher FP32's
maximum hidden error is `0.00038655`, versus native FP32's `0.00021761`.
The [functional decision](../../qualification/batch-1-sub-6b/B06/functional-decision.json)
accepts supervised classification and makes no universal hidden-state parity claim.
All original failed attempts and thresholds remain present.

Kokoro's initial controlled vocoder discrepancy came from real FFT endpoint
phase roundoff: DC/Nyquist are exactly real, and signed imaginary roundoff could
flip the phase feature between −π and +π. Enforcing the real-FFT endpoint
contract increased correlation from about `0.998906` to `0.99999997`. The before
and after measurements and regression fixture are retained. Acoustic stage
checks pass, predicted durations match exactly, and output is finite at 24 kHz.

The maintainer judged the [saved English sample](../../qualification/batch-1-sub-6b/B08/native-english.wav)
clear and intelligible. Native Whisper transcribed the exact ten-word text
(0 word errors on this single sample). Listening and ASR refer to that exact WAV
hash; the later installed synthesis has its own hash. No exact stochastic
waveform equality, corpus WER or general listening-study claim is made.

Earlier harness failures are also retained: evidence path serialization, BGE-M3
stored FP16 head versus FP32 oracle execution, the Nomic capacity field, Engine
batch admission settings, and tuple-versus-JSON-list comparison. The ViT actor
cost now counts 197 patch/CLS tokens instead of 50,176 pixels for a 224 image.
Final Kokoro review also corrected its transient reservation from 256 MiB to
3 GiB, with a 4 GiB server task budget. The pinned conversion was byte-identical
and real speech/actor/HTTP checks passed again with this conservative allowance.
Passing replays link to their actual results and do not overwrite failed attempts.

## Apple M4 performance

Hardware: Apple M4, **32 GiB**, macOS 15.8.2, Python 3.12, MLX/Metal 0.32.2.
FP32 native execution, **MLXBoost off**, no foreign backend, paging or speculative
execution. Loaded Python pipelines use one warmup and five synchronized timed
repetitions. Inputs and timing boundaries are retained in
[JSON](../../qualification/batch-1-sub-6b/benchmarks.json) and
[CSV](../../qualification/batch-1-sub-6b/benchmarks.csv).

| Checkpoint | Selected workload | Median pipeline latency | Throughput |
| --- | --- | ---: | ---: |
| B01 | One 128-token pair | 5.86 ms | 170.68 pairs/s |
| B02 | One 128-token text | 7.53 ms | 132.78 texts/s |
| B03 | One 128-token text, dense pipeline | 45.09 ms | 22.18 texts/s |
| B04 | One 128-token text | 7.18 ms | 139.23 texts/s |
| B05 | Two fixed translation prompts, output bounded at 24 tokens | 45.06 ms | End-to-end generation; not isolated decoder throughput |
| B06 | One image / batch of eight images | 25.54 / 156.43 ms | About 39.15 / 51.14 images/s |
| B07 | One 128-token pair | 43.14 ms | 23.18 pairs/s |
| B08 | 100 / 500 English characters, final reservation check | 3262.52 / 14666.44 ms | RTF 0.4962 / 0.4997 |
| B09 | One 128-token text | 17.55 ms | 56.97 texts/s |
| B10 | One 128-token text | 7.92 ms | 126.33 texts/s |

Retrieval includes 32/128/512-token workloads where the checkpoint allows them,
embedding batches 1/8, and reranker candidate counts 1/10/100 with neural chunks
bounded at eight. B04's requested 512-token case is recorded NOT_RUN because its
sentence profile allows 128. BGE-M3 sparse/token heads are correctness-checked
but not separately timed; its table is dense-only. T5 records encoder/first-token
timing breakdown as a single diagnostic. ViT timings include preprocessing;
Kokoro includes G2P and generated audio duration. MLX allocation peaks are saved
per timed workload and include retained worker arrays, rather than only weights.

Kokoro's earlier run measured 428.17 / 1910.84 ms (RTF about 0.0651) on the same
100/500-character workload. Both runs are retained. Host load, power state and
thermals were not controlled; the reservation is admission accounting and makes
no neural speed claim. The final run remains faster than real time.

External runtime speed comparisons, standalone process-RSS peaks, long-context
2K/8K reference qualification, full per-stage latency breakdowns, and MLXBoost
promotion are **NOT_RUN**. No corresponding speed, memory, context or acceleration
claim is made. These optional measurement extensions do not change the supported
bounded primary workflows. Tiny retrieval examples and two cat images do not
establish publisher benchmark accuracy.

## Storage and distribution

The serial campaign verified task-owned download markers and pinned hashes,
recorded results before cleanup, then removed model and conversion bytes. Shared
Hugging Face caches and the user's retained models were not used as deletion
targets. Kokoro and the additional Whisper diagnostic have separate cleanup
receipts. Large oracle arrays remain under ignored `local/evidence/batch1` with
portable hashes/shapes/previews in public receipts; small independent unit
fixtures, task outputs and the reviewed audio remain available in the repository.

Model weights, voice tensors, reference frameworks, private paths and large raw
oracle tensors are excluded from the distribution. New implementation attribution,
the relevant MIT license and Kokoro's shape contract are packaged. Optional
frontend dependencies retain their own terms; checkpoint licenses are recorded
separately. The local package remains version **0.1.1**; publishing or a version
change requires a separate maintainer action.
