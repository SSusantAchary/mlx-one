# Known limitations

[Experimental index](README.md) · [Model index](../models/index.md)

For the current release, see [supported models and current quality/scope limits](../../status_models_testing.md#known-quality-and-scope-limitations)
and [0.1.1 release notes](../releases/0.1.1.md). The older investigations below
retain their exact workloads; current alignment, Q8 speech and standalone SAM
tracking records supersede earlier primary-workflow gaps.

These are evidence boundaries, not changes to implementation or qualification.
Dated reports remain authoritative for their exact workloads and failed gates.

## Model compatibility

Architecture registration does not establish checkpoint compatibility. Exact
revisions, tensor contracts, processor assets and primary precision live in the
[text inventory](../benchmarks/inventory.md), family guides and modality runbooks.
Qwen3.5 source-text smoke does not qualify its converted primary Q4 path. SAM3
requires authorized gated access, float32 and its own checkpoint license;
quantized SAM3 and SAM3.1 are unsupported.

## Numerical parity

Gemma3 sliding-boundary and answer-quality limitations remain disclosed under the
[functional admission policy](../model-admission.md). MLXBoost’s inherited
cached/fresh failures remain failures; the neutral host optimization does not
certify model math. See [numerical evidence](../mlxboost/numerical-parity.md),
[P1 failures](../p1_native_failure_log.md) and
[task parity blockers](../native_tasks/unresolved-blockers.md). SigLIP2, DETR,
depth, and document/video QA task APIs remain blocked by their respective gates;
ViT remains a helper without shared task admission.

## Checkpoint qualification

Bounded dense B1 smoke, synthetic architecture tests, reference comparisons,
benchmark timing and release approval are different claims. Large LFM2.5/GPT-OSS
targets retain pending current-inventory runs. The
[historical GPT-OSS HTTP case](../benchmarks/http-performance.md) lacks an immutable
checkpoint lock and cannot certify current pins. Publication approval remains
separate from functional admission. No existing threshold or status is changed.

## Runtime limitations

[Cache qualification](../cache-benchmark-m4-32gb.md) did not demonstrate divergent
suffix APC reuse or qualify block APC/sharing/parity. Hybrid block/APC is rejected.
The recorded Qwen2.5 dense run reached 16K input and failed around 32K under that
budget. Configured context limits are not feasible capacity guarantees.
[Runtime experiments](inference-runtime.md) explain source components and gated
optimized modes; production continuous batching and active-request SSD paging are not established.
The optional cold-prefix SSD snapshot store is implemented separately; persistence
is not qualification of live-state paging.
The [MLXBoost matrix](../mlxboost/qualification-matrix.md) marks proposed custom
kernels and compilation unavailable. Default execution remains dense B1 with
MLXBoost disabled.

## Modality coverage

- Audio: [earlier qualification](../audio-qualification-m4-32gb.md) retains Qwen
  1.7B exact-anchor, alignment (800 ms versus 80 ms), streaming and Parakeet
  original-NeMo/timestamp gaps. [Later single-clip measurements](../validation/audio-hf-2026-10-24/report.md)
  do not resolve those gates. Primary long STS fails; the short-question case is
  separate. Nemotron/Voxtral/VibeVoice inference candidates remain unimplemented.
- OCR: [page and handwriting results](../benchmarks/ocr/performance.md) retain
  failures/truncations and high errors. TrOCR requires line crops; page OCR does
  not qualify a complete layout/structured extraction pipeline.
- Retrieval: candidate/synthetic evidence is not retrieval quality; ColBERT token
  vectors/MaxSim are distinct from generic single-vector embeddings.
- Segmentation: [SAM3 fixtures](../sam3-qualification-m4-32gb.md) cover controlled
  single-object clips, not natural multi-object quality, all prompts, refinement
  or crop-layer automatic masking. No SAM3 HTTP or Web UI integration is provided.
- Added tasks: [native-task evidence](native-tasks.md) retains small quality sets,
  FP32 input bounds, CLIP’s rejected 100-label profile, and VAD boundary/noise gaps.

## Performance validation

Historical text 128/16, standardized text 128/128, VLM images, audio clips, OCR
corpora, task Python calls and HTTP SSE rates use separate protocols. See
[benchmark methodology](../benchmarks/README.md). Missing metrics/metadata stay
unavailable. Single-clip WER is not corpus accuracy; generated speech has no human
MOS rating. MLXBoost comparisons retain dependency mismatches, uncontrolled
thermals/load, different cohorts and unresolved SmolLM2 EOS behavior. Historical
SAM3 raw archive availability is not certified by the publication audit.

## Deployment limitations

The package remains alpha. Hardware execution requires Apple Silicon and the
pinned MLX/Metal environment. Optional HTTP, vision and alignment dependencies
and FFmpeg requirements are documented in [getting started](../getting-started.md).
[Server authentication](../ui.md) is optional; `/health` and static UI are public.
No production rollout approval or separately distributed MLXBoost package is
claimed. [Regression records](../mlxboost/api-regression.md) preserve a pre-existing
monolithic-suite hang and distinguish isolated passing selections.
