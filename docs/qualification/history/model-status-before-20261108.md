# Historical qualification audit before the 8 November campaign

This archived assessment preserves the pre-campaign 23/7/10 categories and their original evidence. See the [current supported models](../../../status_models_testing.md) and [0.1.1 release notes](../../releases/0.1.1.md) for current status.

**Historical record only.** The following S/P/A and experimental labels describe evidence available before execution. They are superseded by the current 40-model support inventory linked above.

**23 of the 40 registered model types (57.5%) have documented bounded real-checkpoint functional or fixture evidence for at least one workflow. All 23 remain scoped/experimental; this is not full-model or production certification.**

| Audit category | Model types | Share of 40 | What this means |
| --- | ---: | ---: | --- |
| **S — Scoped experimental functionality** | **23** | **57.5%** | At least one pinned checkpoint/workflow has recorded functional smoke, experimental task admission, or a passing real-checkpoint fixture. Only that recorded scope is supported by the evidence. |
| **P — Partial / measured / WIP** | **7** | **17.5%** | Real checkpoint execution or benchmark evidence exists, but accuracy/task gates remain unresolved; it is not counted as a supported workflow in this conservative audit. |
| **A — Architecture / checkpoint pending** | **10** | **25.0%** | Native architecture/synthetic/kernel tests exist, but a qualifying full-checkpoint workflow is not established by the reviewed sources. |
| **Total** | **40** | **100%** | Each registered model type is counted once. |

**Broad, fully qualified support across the advertised family/task surface: 0 established by these sources.** This does not mean every workflow fails. Functional admission and administrative release approval are separate: pending release approval alone is not used to erase passing functional evidence.

S/P/A are **audit rollups**, not new upstream registry statuses or promotions. The tables below preserve the actual scope and failures. A type can have one passing path and another blocked path; S never means that every size, precision, task or interface works. OCR's measured status is retained as P rather than upgraded from benchmark completion to general task support.

## Counts by modality

| Modality | Registered types | S: scoped functionality | P: partial/measured/WIP | A: checkpoint pending |
| --- | ---: | ---: | ---: | ---: |
| Text generation | 11 | 7 | 0 | 4 |
| Vision-language | 4 | 4 | 0 | 0 |
| Audio & speech | 7 | 6 | 1 | 0 |
| OCR & documents | 6 | 0 | 6 | 0 |
| Embeddings & reranking | 5 | 0 | 0 | 5 |
| Segmentation | 4 | 3 | 0 | 1 |
| Classification | 3 | 3 | 0 | 0 |
| **Total** | **40** | **23** | **7** | **10** |

## All 40 model types

The six homepage cards contain 37 registered types; classification contributes the other three. The counts come from the actual [native registry](../../../src/mlx_one/models/registry.py), including [OCR registrations](../../../src/mlx_one/ocr/registry.py).

These are model **types**, not 40 independent checkpoints or 40 unrelated architecture families. Sizes, quantizations and checkpoint revisions do not create additional entries; the four SAM3 types are separate registered image/tracking components of the same family. Qwen3.5 is counted once under its primary vision-language modality despite also supporting text paths. The separate 49-row dashboard counts checkpoint/task/precision profiles and must not be substituted for this denominator.

### Text generation

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `gemma3_text` | **S** — Scoped experimental functionality | Gemma3 1B text has bounded real-checkpoint functional smoke. | Historical cached numerical differences and answer-quality limitations remain disclosed. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `gpt2` | **S** — Scoped experimental functionality | GPT-2 124M FP32; `EXPERIMENTAL_SUPPORTED`, real-checkpoint Python/CLI/HTTP/SSE smoke. | Larger GPT-2 sizes and other workloads are not covered by the 124M test. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `gpt_oss` | **A** — Architecture / checkpoint pending | MXFP4 target is `ARCHITECTURE_IMPLEMENTED`; TQ3 target is `QUANTIZATION_BACKEND_TESTED`. | Both pinned 20B targets lack qualifying full-checkpoint smoke. Older local HTTP runs do not identify these pinned targets. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `lfm2` | **S** — Scoped experimental functionality | Pinned LFM2-350M has `EXPERIMENTAL_SUPPORTED` functional smoke. | No blanket LFM2/LFM2.5 family claim; retain block/precision-specific rejection limits. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `lfm2_moe` | **A** — Architecture / checkpoint pending | LFM2.5-24B-A2B target is `ARCHITECTURE_IMPLEMENTED`. | Full-checkpoint download/manual smoke remains pending in the support manifest. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `llama` | **S** — Scoped experimental functionality | Pinned SmolLM2-1.7B and Llama-3.2-1B entries have `EXPERIMENTAL_SUPPORTED` functional smoke; MiniCPM5 has separate pinned gates. | Does not qualify every Llama-compatible model. Training quality failures and broader qualification remain separate. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `openelm` | **A** — Architecture / checkpoint pending | Native architecture and synthetic tests are documented. | Exact OpenELM checkpoint integration qualification is pending. | [Source](../../../src/mlx_one/models/language/openelm/README.md) |
| `phi3` | **S** — Scoped experimental functionality | Phi-3 Mini-4K has bounded real-checkpoint functional smoke. | Wider context, other precision and other Phi checkpoints remain unqualified. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `qwen2` | **S** — Scoped experimental functionality | Pinned Qwen2.5 0.5B/1.5B Instruct entries have `EXPERIMENTAL_SUPPORTED` serving/runtime smoke. | Size, revision and precision matter; Coder/training evidence and quality gates are separate. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |
| `qwen2_moe` | **A** — Architecture / checkpoint pending | Native architecture and synthetic tests are documented. | No qualified real-checkpoint workflow recorded in the reviewed evidence. | [Source](../../../src/mlx_one/models/language/qwen2_moe/README.md) |
| `qwen3` | **S** — Scoped experimental functionality | Qwen3-0.6B pinned primary profile has `EXPERIMENTAL_SUPPORTED` functional smoke. | Other sizes, precisions, devices and configured long contexts are not automatically admitted. | [Source](../../../src/mlx_one/data/compatibility/alpha_release.json) |

### Vision-language

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `lfm2_vl` | **S** — Scoped experimental functionality | Pinned Q8 450M and Q4 3B checkpoints record `CHECKPOINT_SMOKE_PASSED`. | Experimental static-image scope; other revisions, video and wider workloads need evidence. | [Source](../../benchmarks/vlm/benchmark_plan.json) |
| `qwen2_5_vl` | **S** — Scoped experimental functionality | Pinned Q4 Qwen2.5-VL-3B static-image checkpoint records `CHECKPOINT_SMOKE_PASSED`. | Dedicated document QA and video QA profiles are BLOCKED in native_tasks.json; their failures do not disappear because image chat passed. | [Source](../../benchmarks/vlm/benchmark_plan.json) |
| `qwen2_vl` | **S** — Scoped experimental functionality | Pinned Q4 Qwen2-VL-2B records `CHECKPOINT_SMOKE_PASSED`. | Experimental static-image workflow; a video-capable architecture does not establish video task support. | [Source](../../benchmarks/vlm/benchmark_plan.json) |
| `qwen3_5` | **S** — Scoped experimental functionality | Targeted 0.8B MLX Q4 image checkpoint records `CHECKPOINT_SMOKE_PASSED`. | Different text/source/converted-Q4 profiles have separate qualification. The text converted-primary result remains unqualified. | [Source](../../benchmarks/vlm/benchmark_plan.json) |

### Audio & speech

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `lfm2_audio` | **S** — Scoped experimental functionality | A10/A12: pinned Q4 checkpoint records `PASS_SCOPED` ASR and TTS waveform generation. | A13 10-second speech-to-speech workload is FAILED/WIP; separate short-question success does not replace that failure. No listening/MOS qualification. | [Source](../../../benchmarks/audio/checkpoints.json) |
| `moonshine` | **S** — Scoped experimental functionality | A04–A05: Tiny and Base record `PASS_SCOPED` native offline English ASR. | Streaming, timestamps, translation and broader workloads are not qualified. | [Source](../../../benchmarks/audio/checkpoints.json) |
| `parakeet_tdt` | **S** — Scoped experimental functionality | A09: pinned v3 MLX export has `PASS_SCOPED` native English transcription. | A08 original v2 NeMo path remains WIP for parity; timestamps and broader qualification remain open. | [Source](../../../benchmarks/audio/checkpoints.json) |
| `qwen3_asr` | **S** — Scoped experimental functionality | A06–A07: pinned Q8 exports record `PASS_SCOPED` English transcription; the 0.6B has separate bounded language evidence. | Publisher-BF16 exact-anchor discrepancies, remaining languages and streaming parity retain WIP status. Q8 functional execution is a different scope. | [Source](../../../benchmarks/audio/checkpoints.json) |
| `qwen3_forced_aligner` | **P** — Partial / measured / WIP | A11 ran the known-text alignment workload and records `PASS_SCOPED` execution, but its source status remains WIP. | Completed execution is not alignment-accuracy qualification; current clip boundary accuracy is unmeasured and historical differences reached 800 ms against an 80-ms limit. | [Source](../../../benchmarks/audio/checkpoints.json) |
| `silero_vad` | **S** — Scoped experimental functionality | FP32 mono 16-kHz streaming VAD has functional/source-parity tests, shared APIs and clean-wheel evidence. | EXPERIMENTAL; human segment-boundary quality and broad music/noise robustness remain open. | [Source](../../../src/mlx_one/data/native_tasks.json) |
| `whisper` | **S** — Scoped experimental functionality | A01–A03: Tiny, Large-v3-Turbo and Distil-large-v3 record `PASS_SCOPED` native transcription. | Recorded real English fixture/corpus scopes only; language, translation and long-form scope vary. | [Source](../../../benchmarks/audio/checkpoints.json) |

### OCR & documents

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `deepseek_ocr` | **P** — Partial / measured / WIP | Pinned native Q4 DeepSeek-OCR has page/line benchmark evidence. | Page test: 97/100 completed, 41 truncated, CER 51.2%; broader document quality remains unqualified. | [Source](../../benchmarks/ocr/performance.md) |
| `florence2` | **P** — Partial / measured / WIP | Pinned native FP16 Florence-2 has page/line benchmark execution evidence. | Page test: 97/100 completed, 61 truncated, CER 78.3%; handwriting CER 97.9% in the small line sample. | [Source](../../benchmarks/ocr/performance.md) |
| `got_ocr2` | **P** — Partial / measured / WIP | Pinned native Q4 OCR has page/line benchmark execution evidence. | Page test: 97/100 completed, 56 truncated, Markdown-reference CER 74.5%; this is experimental measurement, not full OCR task qualification. | [Source](../../benchmarks/ocr/performance.md) |
| `lightonocr` | **P** — Partial / measured / WIP | Pinned native BF16 LightOnOCR-2-1B has page/line benchmark evidence. | Page test: 97/100 completed, 64 truncated, CER 67.7%; broader document quality remains unqualified. | [Source](../../benchmarks/ocr/performance.md) |
| `paddleocr_vl` | **P** — Partial / measured / WIP | Pinned native Q4 page transcription has page/line benchmark evidence. | Page test: 97/100 completed, 53 truncated, CER 59.8%; separate layout pipeline is excluded. | [Source](../../benchmarks/ocr/performance.md) |
| `trocr` | **P** — Partial / measured / WIP | Printed and handwritten checkpoints were measured on 10 cropped line images. | No full-page support without line crops. Handwritten CER 4.4% / WER 10.6% is a small-sample result; printed checkpoint is an out-of-domain handwriting baseline. | [Source](../../benchmarks/ocr/performance.md) |

### Embeddings & reranking

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `bert` | **A** — Architecture / checkpoint pending | all-MiniLM-style embedding implementation and synthetic validation; exact-checkpoint parity, quality and hardware qualification pending. | Architecture coverage is not a qualified real-checkpoint retrieval workflow. | [Source](../../../src/mlx_one/models/embeddings/bert/README.md) |
| `lfm2_colbert` | **A** — Architecture / checkpoint pending | Token embeddings/MaxSim architecture exists; publisher checkpoint workflow unqualified and generic single-vector embed contract incompatible. | Architecture coverage is not a qualified real-checkpoint retrieval workflow. | [Source](../../../src/mlx_one/models/embeddings/lfm2_colbert/README.md) |
| `mpnet` | **A** — Architecture / checkpoint pending | MPNet embedding implementation and synthetic validation; exact-checkpoint qualification pending. | Architecture coverage is not a qualified real-checkpoint retrieval workflow. | [Source](../../../src/mlx_one/models/embeddings/mpnet/README.md) |
| `qwen3_embedding` | **A** — Architecture / checkpoint pending | Instructed pooling/dimensions and synthetic tests; real-checkpoint retrieval quality/parity/hardware results unqualified. | Architecture coverage is not a qualified real-checkpoint retrieval workflow. | [Source](../../../src/mlx_one/models/embeddings/qwen3_embedding/README.md) |
| `qwen3_reranker` | **A** — Architecture / checkpoint pending | Pair scoring, probabilities and synthetic tests; real-checkpoint quality/parity/hardware results unqualified. | Architecture coverage is not a qualified real-checkpoint retrieval workflow. | [Source](../../../src/mlx_one/models/embeddings/qwen3_reranker/README.md) |

### Segmentation

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `sam3` | **S** — Scoped experimental functionality | Pinned SAM3 native concept-image fixture passed the recorded reference gate. | One retained truck mask; box exemplars, current-preprocessing recapture and wider quality remain pending. Package qualified=False. | [Source](../../sam3-qualification-m4-32gb.md) |
| `sam3_tracker` | **S** — Scoped experimental functionality | Pinned SAM3 interactive image fixture passed the corrected-preprocessing gate. | One positive center-point fixture; box prompts, refinement and automatic-mask parity remain open. Package qualified=False. | [Source](../../sam3-qualification-m4-32gb.md) |
| `sam3_tracker_video` | **A** — Architecture / checkpoint pending | Standalone temporal-tracker architecture participates inside composite video execution. | Composite sam3_video success does not independently qualify the standalone temporal tracker. | [Source](../../models/index.md) |
| `sam3_video` | **S** — Scoped experimental functionality | Pinned composite SAM3 video passed separate offline/streaming 32-frame CPU-reference fixture gates. | Controlled single-object rolled image, not natural multi-object video. MPS comparison failed and is retained; lifecycle/quality gates remain pending. Package qualified=False. | [Source](../../sam3-qualification-m4-32gb.md) |

### Classification

| Model type | Audit status | Recorded testing / functional scope | Remaining limits | Evidence |
| --- | --- | --- | --- | --- |
| `bert_token_classification` | **S** — Scoped experimental functionality | Pinned BERT NER checkpoint has FP32 B1/B2 none/simple aggregation, functional/source parity and shared API/wheel evidence. | EXPERIMENTAL; diagnostic strict-span F1 0.604 on 100 sentences; maximum 512 tokens; broader quality/release work pending. | [Source](../../../src/mlx_one/data/native_tasks.json) |
| `clip` | **S** — Scoped experimental functionality | Pinned CLIP checkpoint has audited local safetensors conversion, source parity, shared API/wheel evidence and small real-image tests. | EXPERIMENTAL; one image and 1–10 labels only. The 100-label gate fails and that scope is rejected. | [Source](../../../src/mlx_one/data/native_tasks.json) |
| `distilbert` | **S** — Scoped experimental functionality | Pinned sentiment checkpoint has FP32 B1/B2 functional/source parity, SST-2 evaluation, Python/CLI/HTTP and wheel evidence. | EXPERIMENTAL; maximum 512 tokens; broader regimes and release qualification pending. | [Source](../../../src/mlx_one/data/native_tasks.json) |

## Important distinctions

- **No blanket family claim:** the S group means that at least one documented workflow works in its recorded scope. It does not certify every architecture-compatible checkpoint, configured context length, quantization, language or device.
- **Preserved blocked paths:** Qwen2.5-VL image chat smoke does not admit its separately BLOCKED document/video QA tasks. LFM2 Audio ASR/TTS completion does not turn the FAILED 10-second speech-to-speech case into a pass. SAM3 CPU-reference fixture success does not change the failed MPS comparison.
- **Boundary accuracy matters:** ForcedAligner has successful execution but unresolved timing accuracy, so it stays in P. A11's `PASS_SCOPED` is an execution claim, not a word-boundary accuracy certificate.
- **Source and date matter:** some older family summaries say checkpoint testing is pending; later pinned benchmark manifests record bounded execution. Where they differ, this audit records the later, narrower checkpoint result and retains unresolved broader gates. It never infers broad quality from speed.
- **The 49 dashboard profiles are a different set:** ViT, SigLIP, DETR, RT-DETR and Depth Anything candidates appear there but are not part of the homepage's 40 registered types. Their statuses are not silently added to or subtracted from this table.
- **Historical tests versus local availability:** recent cleanup removed several checkpoint weight files. Passing historical evidence remains documented, but tests using deleted weights require those exact checkpoints to be restored before replay. This report is not a claim that all weights are installed locally today.
- **No new qualification issued:** official statuses, runtime defaults, manifests and benchmark measurements were not changed by this report.

## How to reproduce the inventory and review

1. Read `builtin_registrations()` in `src/mlx_one/models/registry.py` and OCR's `CAPABILITIES`; group distinct registered types by primary modality. `website.insights.modality_coverage()` performs the same metadata-only enumeration without importing MLX.
2. Match text types to `src/mlx_one/data/compatibility/alpha_release.json`; use the exact pinned checkpoint and primary precision, not just a similar model name.
3. Match VLM types to `docs/benchmarks/vlm/benchmark_plan.json`, audio to `benchmarks/audio/checkpoints.json`, OCR to `docs/benchmarks/ocr/inventory.json` and its corpus reports, and native tasks to `src/mlx_one/data/native_tasks.json`.
4. Read family guides for retrieval architecture-only status and the SAM3 qualification report for separate concept, interactive, standalone-temporal and composite-video scopes.
5. Recalculate these 40 rows whenever a registry entry or its evidence changes. This is a dated audit, not an automatically generated qualification manifest.

## Evidence index

- [Functional admission policy](../../model-admission.md)
- [Model index and evidence boundaries](../../models/index.md)
- [Text support manifest](../../../src/mlx_one/data/compatibility/alpha_release.json)
- [VLM checkpoint plan](../../benchmarks/vlm/benchmark_plan.json)
- [Audio checkpoint manifest](../../../benchmarks/audio/checkpoints.json) and [coverage](../../benchmarks/audio/coverage.md)
- [OCR inventory](../../benchmarks/ocr/inventory.json) and [measured accuracy, failures and truncations](../../benchmarks/ocr/performance.md)
- [Native task manifest](../../../src/mlx_one/data/native_tasks.json) and [task evidence](../../experimental/native-tasks.md)
- [SAM3 partial qualification](../../sam3-qualification-m4-32gb.md)
