# Model support and validation records

Updated: **12 November 2026**. The current source has **45 native architecture
types across seven modalities**, represented by **50 verified checkpoint
profiles**: the dated 40-type release cohort below and ten additional profiles.
The additional workflows meet the technical release criteria for their recorded
tasks. Version **0.1.2** was published on PyPI on 11 November; its
[release record](docs/releases/0.1.2.json) identifies the verified artifacts.

[Model workflows and examples](docs/models/batch-1-sub-6b/README.md) ·
[Ten-checkpoint validation results](docs/qualification/batch-1-sub-6b/results.json) ·
[Validation report and limitations](docs/models/batch-1-sub-6b/REPORT.md).

## Current native coverage

| Modality | Native architecture types |
| --- | ---: |
| Text generation | 12 |
| Vision-language | 4 |
| Audio & speech | 8 |
| OCR & documents | 6 |
| Embeddings & reranking | 7 |
| Segmentation | 4 |
| Classification | 4 |
| **Total** | **45** |

New source workflows include T5, supervised ViT, Kokoro, BGE-M3, BGE Reranker,
Nomic and additional BERT retrieval profiles. ViT classification passes with its
failed fixed hidden-state diagnostic retained. Kokoro is qualified for bounded
`af_heart` English speech. Exact revisions and limits are in the linked results.

## Dated 0.1.1 release cohort

Recorded: **8 November 2026**. **40/40 model types have passing primary-workflow records** under the MLX-LM-aligned functional standard. Each selected checkpoint is **SUPPORTED** for the operations below. **0.1.1 was approved on 8 November 2026 and is published on PyPI.** See the [release notes](docs/releases/0.1.1.md).

The required checks are architecture/state correctness, real checkpoint load and task output, applicable public interfaces, regressions and distribution checks. MLX-LM does not prescribe a blanket experimental label or universal broad-reliability certification. Corpus accuracy and reference comparisons remain recorded diagnostics. See the [release guide](mlx-one_model_release_guide.md) for upstream sources and policy.

[Current technical readiness](docs/qualification/model-release-20261108/RELEASE_CANDIDATE.md) · [machine-readable support catalog](src/mlx_one/data/model_support_release_20261108.json) · [repairs and diagnostic limits](docs/qualification/model-release-20261108/RESOLUTIONS.md).

<a id="current-counts-by-modality"></a>

### 0.1.1 counts by modality

| Modality | Model types | Supported |
| --- | ---: | ---: |
| Text generation | 11 | 11 |
| Vision-language | 4 | 4 |
| Audio & speech | 7 | 7 |
| OCR & documents | 6 | 6 |
| Embeddings & reranking | 5 | 5 |
| Segmentation | 4 | 4 |
| Classification | 3 | 3 |
| **Total** | **40** | **40** |

<a id="current-supported-models"></a>

## 0.1.1 supported models

These are 40 registered model types, each represented by an exact checkpoint and precision. A model type does not represent every checkpoint size or optional task extension. Full revisions and result hashes are stored in the catalog.

| Model type | Status | Tested checkpoint | Primary operations | Evidence |
| --- | --- | --- | --- | --- |
| `gpt2` | **SUPPORTED** | `openai-community/gpt2` (float32) | text-generation | [Result](docs/qualification/model-release-20261108/followups/interface-replay/gpt2.json) |
| `qwen2` | **SUPPORTED** | `Qwen/Qwen2.5-0.5B-Instruct` (affine_q8) | text-generation | [Result](docs/qualification/model-release-20261108/followups/interface-replay/qwen2.json) |
| `qwen3` | **SUPPORTED** | `Qwen/Qwen3-0.6B` (affine_q8) | text-generation | [Result](docs/qualification/model-release-20261108/followups/interface-replay/qwen3.json) |
| `lfm2` | **SUPPORTED** | `LiquidAI/LFM2-350M` (affine_q4) | text-generation | [Result](docs/qualification/model-release-20261108/followups/lfm2-bos/lfm2.json) |
| `llama` | **SUPPORTED** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` (affine_q4) | text-generation | [Result](docs/qualification/model-release-20261108/followups/interface-replay/llama.json) |
| `gemma3_text` | **SUPPORTED** | `google/gemma-3-1b-it` (bfloat16) | text-generation | [Result](docs/qualification/model-release-20261108/followups/interface-replay/gemma3_text.json) |
| `phi3` | **SUPPORTED** | `microsoft/Phi-3-mini-4k-instruct` (affine_q4) | text-generation | [Result](docs/qualification/model-release-20261108/phi3.json) |
| `lfm2_moe` | **SUPPORTED** | `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` (mixed_affine_q4_embedding_q6_router_fp32) | text-generation | [Result](docs/qualification/model-release-20261108/lfm2_moe.json) |
| `gpt_oss` | **SUPPORTED** | `mlx-community/gpt-oss-20b-MXFP4-Q4` (mixed_mxfp4_affine_q4_q8) | text-generation | [Result](docs/qualification/model-release-20261108/followups/gpt-oss-512/gpt_oss.json) |
| `openelm` | **SUPPORTED** | `apple/OpenELM-270M` (publisher storage precision) | text-generation | [Result](docs/qualification/model-release-20261108/followups/openelm-bos/openelm.json) |
| `qwen2_moe` | **SUPPORTED** | `mlx-community/Qwen1.5-MoE-A2.7B-4bit` (stored_affine_q4) | text-generation | [Result](docs/qualification/model-release-20261108/followups/qwen2-moe-q4/qwen2_moe.json) |
| `qwen2_vl` | **SUPPORTED** | `mlx-community/Qwen2-VL-2B-Instruct-4bit` (Q4) | image-text-generation | [Result](docs/qualification/model-release-20261108/qwen2_vl.json) |
| `qwen2_5_vl` | **SUPPORTED** | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (Q4) | image-text-generation | [Result](docs/qualification/model-release-20261108/qwen2_5_vl.json) |
| `qwen3_5` | **SUPPORTED** | `mlx-community/Qwen3.5-0.8B-4bit` (Q4) | image-text-generation | [Result](docs/qualification/model-release-20261108/qwen3_5.json) |
| `lfm2_vl` | **SUPPORTED** | `mlx-community/LFM2-VL-450M-8bit` (Q8) | image-text-generation | [Result](docs/qualification/model-release-20261108/lfm2_vl.json) |
| `whisper` | **SUPPORTED** | `openai/whisper-tiny` (FP32) | transcription | [Result](docs/qualification/model-release-20261108/whisper.json) |
| `moonshine` | **SUPPORTED** | `UsefulSensors/moonshine-tiny` (FP32) | transcription | [Result](docs/qualification/model-release-20261108/moonshine.json) |
| `qwen3_asr` | **SUPPORTED** | `mlx-community/Qwen3-ASR-0.6B-8bit` (Q8 / BF16) | transcription | [Result](docs/qualification/model-release-20261108/qwen3_asr.json) |
| `parakeet_tdt` | **SUPPORTED** | `mlx-community/parakeet-tdt-0.6b-v3` (FP32) | transcription | [Result](docs/qualification/model-release-20261108/parakeet_tdt.json) |
| `qwen3_forced_aligner` | **SUPPORTED** | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` (Q8 / BF16) | forced-alignment | [Result](docs/qualification/model-release-20261108/followups/resolution-alignment/qwen3_forced_aligner.json) |
| `lfm2_audio` | **SUPPORTED** | `mlx-community/LFM2.5-Audio-1.5B-8bit` (Q8 / BF16) | transcription, text-to-speech, speech-to-speech | [Result](docs/qualification/model-release-20261108/followups/resolution-lfm-q8-final/lfm2_audio.json) |
| `silero_vad` | **SUPPORTED** | `snakers4/silero-vad` (publisher storage precision) | voice-activity-detection | [Result](docs/qualification/model-release-20261108/silero_vad.json) |
| `trocr` | **SUPPORTED** | `microsoft/trocr-base-printed` (FP32) | line-ocr | [Result](docs/qualification/model-release-20261108/trocr.json) |
| `got_ocr2` | **SUPPORTED** | `mlx-community/GOT-OCR2_0-4bit` (Affine Q4) | ocr.recognize | [Result](docs/qualification/model-release-20261108/followups/resolution-ocr2/got_ocr2.json) |
| `florence2` | **SUPPORTED** | `microsoft/Florence-2-base-ft` (FP16) | ocr.recognize | [Result](docs/qualification/model-release-20261108/followups/functional-florence/florence2.json) |
| `paddleocr_vl` | **SUPPORTED** | `mlx-community/PaddleOCR-VL-1.5-4bit` (Affine Q4) | ocr.recognize | [Result](docs/qualification/model-release-20261108/paddleocr_vl.json) |
| `lightonocr` | **SUPPORTED** | `lightonai/LightOnOCR-2-1B` (BF16) | ocr.recognize | [Result](docs/qualification/model-release-20261108/followups/resolution-ocr2/lightonocr.json) |
| `deepseek_ocr` | **SUPPORTED** | `mlx-community/DeepSeek-OCR-4bit` (Affine Q4) | ocr.recognize | [Result](docs/qualification/model-release-20261108/followups/resolution-ocr2/deepseek_ocr.json) |
| `bert` | **SUPPORTED** | `sentence-transformers/all-MiniLM-L6-v2` (publisher storage precision) | embedding | [Result](docs/qualification/model-release-20261108/followups/retrieval-serving/bert.json) |
| `mpnet` | **SUPPORTED** | `sentence-transformers/all-mpnet-base-v2` (publisher storage precision) | embedding | [Result](docs/qualification/model-release-20261108/followups/retrieval-serving/mpnet.json) |
| `qwen3_embedding` | **SUPPORTED** | `Qwen/Qwen3-Embedding-0.6B` (publisher storage precision) | embedding | [Result](docs/qualification/model-release-20261108/followups/resolution-embedding/qwen3_embedding.json) |
| `qwen3_reranker` | **SUPPORTED** | `Qwen/Qwen3-Reranker-0.6B` (publisher storage precision) | reranking | [Result](docs/qualification/model-release-20261108/followups/retrieval-serving/qwen3_reranker.json) |
| `lfm2_colbert` | **SUPPORTED** | `LiquidAI/LFM2-ColBERT-350M` (publisher storage precision) | token-embedding, maxsim-scoring | [Result](docs/qualification/model-release-20261108/followups/colbert-expansion/lfm2_colbert.json) |
| `sam3` | **SUPPORTED** | `facebook/sam3` (publisher storage precision) | image-segmentation | [Result](docs/qualification/model-release-20261108/followups/resolution-sam3/sam3.json) |
| `sam3_tracker` | **SUPPORTED** | `facebook/sam3` (publisher storage precision) | point-prompted-image-segmentation | [Result](docs/qualification/model-release-20261108/followups/resolution-sam3/sam3_tracker.json) |
| `sam3_video` | **SUPPORTED** | `facebook/sam3` (publisher storage precision) | video-segmentation, offline, streaming | [Result](docs/qualification/model-release-20261108/followups/functional-sam-video/sam3_video.json) |
| `sam3_tracker_video` | **SUPPORTED** | `facebook/sam3` (publisher storage precision) | point-prompted-video-tracking, offline, streaming | [Result](docs/qualification/model-release-20261108/followups/resolution-sam3/sam3_tracker_video.json) |
| `distilbert` | **SUPPORTED** | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (float32) | text-classification | [Result](docs/qualification/model-release-20261108/followups/interface-replay/distilbert.json) |
| `bert_token_classification` | **SUPPORTED** | `dslim/bert-base-NER` (float32) | token-classification | [Result](docs/qualification/model-release-20261108/followups/resolution-ner-final/bert_token_classification.json) |
| `clip` | **SUPPORTED** | `openai/clip-vit-base-patch32` (float32) | zero-shot-image-classification | [Result](docs/qualification/model-release-20261108/followups/clip-cli/clip.json) |

## Known quality and scope limitations

The forced aligner now passes independently annotated TIMIT checks: **29.472 ms mean boundary error**, **68.1875 ms p95**, and full matched-word coverage. The older 800-ms discrepancy remains archived.

Florence plain OCR executes successfully but its recorded page CER exceeds the original 25% benchmark floor; optional OCR region parsing is excluded. The small CoNLL NER sample scores F1 0.6024, matching the publisher implementation. LFM2 Audio selects the tested Q8 profile; weaker Q4 speech-quality results remain recorded. SAM video has passing native offline/streaming and CLI results; the optional MPS reference streaming discrepancy remains documented. These measurements do not become passing benchmarks when support policy changes.

## Historical qualification audit

The long pre-campaign S/P/A audit has moved to the [archived report](docs/qualification/history/model-status-before-20261108.md). Its 23/7/10 counts and experimental labels remain historical evidence. The current support decision is the 40-model inventory above.
