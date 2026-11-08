# Model support and release readiness

Technical checks: **40/40**. Decision: **READY_FOR_RELEASE**.

Maintainer release approval: **APPROVED** on 2026-11-08. Publication has not been performed. The source changes are local; frozen-source/publication controls still apply when publishing.

Supported models pass implementation, real task and interface checks under the user's MLX-LM-aligned standard. Historical benchmark failures remain visible and are not rewritten as benchmark passes.

Final native suite: **2,573 passed**, **95 skipped**, **9 subtests passed**. Wheel/sdist payload, clean installed-wheel and downloaded-weight cleanup checks pass. Protected GPT-OSS model hashes are unchanged.

| Model type | Exact profile | Technical decision | Evidence |
| --- | --- | --- | --- |
| `gpt2` | `openai-community/gpt2` at `607a30d783df`; float32 | SUPPORTED | [Result](followups/interface-replay/gpt2.json) |
| `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf`; affine_q8 | SUPPORTED | [Result](followups/interface-replay/qwen2.json) |
| `qwen3` | `Qwen/Qwen3-0.6B` at `c1899de289a0`; affine_q8 | SUPPORTED | [Result](followups/interface-replay/qwen3.json) |
| `lfm2` | `LiquidAI/LFM2-350M` at `0b6ea934a94b`; affine_q4 | SUPPORTED | [Result](followups/lfm2-bos/lfm2.json) |
| `llama` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` at `31b70e2e869a`; affine_q4 | SUPPORTED | [Result](followups/interface-replay/llama.json) |
| `gemma3_text` | `google/gemma-3-1b-it` at `dcc83ea841ab`; bfloat16 | SUPPORTED | [Result](followups/interface-replay/gemma3_text.json) |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` at `f39ac1d28e92`; affine_q4 | SUPPORTED | [Result](phi3.json) |
| `lfm2_moe` | `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` at `667408d494bd`; mixed_affine_q4_embedding_q6_router_fp32 | SUPPORTED | [Result](lfm2_moe.json) |
| `gpt_oss` | `mlx-community/gpt-oss-20b-MXFP4-Q4` at `f356f2747216`; mixed_mxfp4_affine_q4_q8 | SUPPORTED | [Result](followups/gpt-oss-512/gpt_oss.json) |
| `openelm` | `apple/OpenELM-270M` at `e2b9003235d5`; stored publisher precision | SUPPORTED | [Result](followups/openelm-bos/openelm.json) |
| `qwen2_moe` | `mlx-community/Qwen1.5-MoE-A2.7B-4bit` at `11aaad5b454a`; stored_affine_q4 | SUPPORTED | [Result](followups/qwen2-moe-q4/qwen2_moe.json) |
| `qwen2_vl` | `mlx-community/Qwen2-VL-2B-Instruct-4bit` at `01af461cdb95`; Q4 | SUPPORTED | [Result](qwen2_vl.json) |
| `qwen2_5_vl` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` at `46d4cf06a06f`; Q4 | SUPPORTED | [Result](qwen2_5_vl.json) |
| `qwen3_5` | `mlx-community/Qwen3.5-0.8B-4bit` at `da28692b5f13`; Q4 | SUPPORTED | [Result](qwen3_5.json) |
| `lfm2_vl` | `mlx-community/LFM2-VL-450M-8bit` at `2422c0686f1f`; Q8 | SUPPORTED | [Result](lfm2_vl.json) |
| `whisper` | `openai/whisper-tiny` at `169d4a4341b3`; FP32 | SUPPORTED | [Result](whisper.json) |
| `moonshine` | `UsefulSensors/moonshine-tiny` at `390624ed33d5`; FP32 | SUPPORTED | [Result](moonshine.json) |
| `qwen3_asr` | `mlx-community/Qwen3-ASR-0.6B-8bit` at `89e96d92ba34`; Q8 / BF16 | SUPPORTED | [Result](qwen3_asr.json) |
| `parakeet_tdt` | `mlx-community/parakeet-tdt-0.6b-v3` at `ed2b7e8c15f9`; FP32 | SUPPORTED | [Result](parakeet_tdt.json) |
| `qwen3_forced_aligner` | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` at `0e1a68e91d81`; Q8 / BF16 | SUPPORTED | [Result](followups/resolution-alignment/qwen3_forced_aligner.json) |
| `lfm2_audio` | `mlx-community/LFM2.5-Audio-1.5B-8bit` at `a569a7805a8e`; Q8 / BF16 | SUPPORTED | [Result](followups/resolution-lfm-q8-final/lfm2_audio.json) |
| `silero_vad` | `snakers4/silero-vad` at `60b7ffa24362`; stored publisher precision | SUPPORTED | [Result](silero_vad.json) |
| `trocr` | `microsoft/trocr-base-printed` at `93450be3f1ed`; FP32 | SUPPORTED | [Result](trocr.json) |
| `got_ocr2` | `mlx-community/GOT-OCR2_0-4bit` at `0e28c87720de`; Affine Q4 | SUPPORTED | [Result](followups/resolution-ocr2/got_ocr2.json) |
| `florence2` | `microsoft/Florence-2-base-ft` at `f6c1a25888ff`; FP16 | SUPPORTED | [Result](followups/functional-florence/florence2.json) |
| `paddleocr_vl` | `mlx-community/PaddleOCR-VL-1.5-4bit` at `3404726b40ac`; Affine Q4 | SUPPORTED | [Result](paddleocr_vl.json) |
| `lightonocr` | `lightonai/LightOnOCR-2-1B` at `be7c41313c30`; BF16 | SUPPORTED | [Result](followups/resolution-ocr2/lightonocr.json) |
| `deepseek_ocr` | `mlx-community/DeepSeek-OCR-4bit` at `4958b6b0d46d`; Affine Q4 | SUPPORTED | [Result](followups/resolution-ocr2/deepseek_ocr.json) |
| `bert` | `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4`; stored publisher precision | SUPPORTED | [Result](followups/retrieval-serving/bert.json) |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` at `e8c3b32edf54`; stored publisher precision | SUPPORTED | [Result](followups/retrieval-serving/mpnet.json) |
| `qwen3_embedding` | `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d`; stored publisher precision | SUPPORTED | [Result](followups/resolution-embedding/qwen3_embedding.json) |
| `qwen3_reranker` | `Qwen/Qwen3-Reranker-0.6B` at `e61197ed4502`; stored publisher precision | SUPPORTED | [Result](followups/retrieval-serving/qwen3_reranker.json) |
| `lfm2_colbert` | `LiquidAI/LFM2-ColBERT-350M` at `9d727d7964d3`; stored publisher precision | SUPPORTED | [Result](followups/colbert-expansion/lfm2_colbert.json) |
| `sam3` | `facebook/sam3` at `3c879f39826c`; stored publisher precision | SUPPORTED | [Result](followups/resolution-sam3/sam3.json) |
| `sam3_tracker` | `facebook/sam3` at `3c879f39826c`; stored publisher precision | SUPPORTED | [Result](followups/resolution-sam3/sam3_tracker.json) |
| `sam3_video` | `facebook/sam3` at `3c879f39826c`; stored publisher precision | SUPPORTED | [Result](followups/functional-sam-video/sam3_video.json) |
| `sam3_tracker_video` | `facebook/sam3` at `3c879f39826c`; stored publisher precision | SUPPORTED | [Result](followups/resolution-sam3/sam3_tracker_video.json) |
| `distilbert` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` at `714eb0fa89d2`; float32 | SUPPORTED | [Result](followups/interface-replay/distilbert.json) |
| `bert_token_classification` | `dslim/bert-base-NER` at `d1a3e8f13f8c`; float32 | SUPPORTED | [Result](followups/resolution-ner-final/bert_token_classification.json) |
| `clip` | `openai/clip-vit-base-patch32` at `3d74acf9a28c`; float32 | SUPPORTED | [Result](followups/clip-cli/clip.json) |

See [machine-readable gates and scope](release-candidate.json), [original cohort](README.md), and [repair decisions](RESOLUTIONS.md).

Final audits: [regression](regression-resolution-audit.json), [distribution payload](package-audit-resolution-final.json), [clean wheel](clean-wheel-resolution-final.json), [source/build identity](candidate-build-resolution-final.json), [installed lineage](candidate-lineage-resolution.json), [weight cleanup](cleanup-resolution-audit.json).

The existing publication workflow targets the earlier restricted-alpha proof cohort. This report records current model support; it does not claim that the legacy publication job has been executed for this cohort.
