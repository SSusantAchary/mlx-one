# Historical extended qualification campaign

**Current support decisions:** see [Model support and release readiness](RELEASE_CANDIDATE.md). All 40 selected model types have passing primary-workflow records under the user's MLX-LM-aligned functional standard. The stricter campaign rollup below is retained for comparison; its `release_qualified` fields and experimental-review counts are historical and are not the current model-support decision. Original failed accuracy/reference measurements are preserved. See [current policy](SUPPORT_POLICY.json) and [resolutions](RESOLUTIONS.md).

Release gate: **BLOCKED**. Recorded attempts: **40/40**. Functional probes passed: **38**. Pending: **0**. Release-qualified: **0**.

A functional pass applies to the exact checkpoint revision, precision, fixture and operations recorded. It does not certify every checkpoint in a family, every modality/task variant, corpus quality, long context or other hardware. Publication approval and existing package support decisions are unchanged.
The campaign tests one representative per registered type. Passing one representative does not execute every checkpoint/task/precision profile in the guide.

Tests execute on the Apple M4 32-GiB host using the built candidate wheel outside the checkout. The native worker rejects installed/imported reference frameworks. Hugging Face fixtures are pinned and identified in [datasets.json](datasets.json). Raw sample inputs stay in the temporary work directory.

## Results by modality

| Modality | Types | Functional passes |
| --- | ---: | ---: |
| text | 11 | 11 |
| vlm | 4 | 4 |
| audio | 6 | 6 |
| vad | 1 | 1 |
| ocr | 6 | 5 |
| retrieval | 5 | 5 |
| segmentation | 4 | 3 |
| classification | 3 | 3 |

**37** records are candidates for a bounded experimental-support review: their recorded quality sanity checks and applicable interfaces pass. This is separate from release approval. Any remaining quality or interface gaps are listed in the per-type records and next steps.

## All 40 model types

| Model type | Probe | Sample quality | Interfaces | Weights removed |
| --- | --- | --- | --- | --- |
| `gpt2` | [PASS_FUNCTIONAL](followups/interface-replay/gpt2.json) ([initial](gpt2.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `qwen2` | [PASS_FUNCTIONAL](followups/interface-replay/qwen2.json) ([initial](qwen2.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `qwen3` | [PASS_FUNCTIONAL](followups/interface-replay/qwen3.json) ([initial](qwen3.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `lfm2` | [PASS_FUNCTIONAL](followups/lfm2-bos/lfm2.json) ([initial](lfm2.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `llama` | [PASS_FUNCTIONAL](followups/interface-replay/llama.json) ([initial](llama.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `gemma3_text` | [PASS_FUNCTIONAL](followups/interface-replay/gemma3_text.json) ([initial](gemma3_text.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `phi3` | [PASS_FUNCTIONAL](phi3.json) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `lfm2_moe` | [PASS_FUNCTIONAL](lfm2_moe.json) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `gpt_oss` | [PASS_FUNCTIONAL](followups/gpt-oss-512/gpt_oss.json) ([initial](gpt_oss.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `openelm` | [PASS_FUNCTIONAL](followups/openelm-bos/openelm.json) ([initial](openelm.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `qwen2_moe` | [PASS_FUNCTIONAL](followups/qwen2-moe-q4/qwen2_moe.json) ([initial](qwen2_moe.json)) | PASS_FIVE_EXCERPT_WIKITEXT_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_GREEDY_EQUALS_NONSTREAMING; http: PASS; http_authentication: PASS | Yes |
| `qwen2_vl` | [PASS_FUNCTIONAL](qwen2_vl.json) | PASS_TWO_IMAGE_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_VISION_COMPLETION; http: PASS; http_authentication: PASS | Yes |
| `qwen2_5_vl` | [PASS_FUNCTIONAL](qwen2_5_vl.json) | PASS_TWO_IMAGE_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_VISION_COMPLETION; http: PASS; http_authentication: PASS | Yes |
| `qwen3_5` | [PASS_FUNCTIONAL](qwen3_5.json) | PASS_TWO_IMAGE_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_VISION_COMPLETION; http: PASS; http_authentication: PASS | Yes |
| `lfm2_vl` | [PASS_FUNCTIONAL](lfm2_vl.json) | PASS_TWO_IMAGE_SANITY_ONLY | python: PASS; cli: PASS; sse: PASS_VISION_COMPLETION; http: PASS; http_authentication: PASS | Yes |
| `whisper` | [PASS_FUNCTIONAL](whisper.json) | PASS_ONE_ENGLISH_CLIP_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `moonshine` | [PASS_FUNCTIONAL](moonshine.json) | PASS_ONE_ENGLISH_CLIP_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `qwen3_asr` | [PASS_FUNCTIONAL](qwen3_asr.json) | PASS_ONE_ENGLISH_CLIP_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `parakeet_tdt` | [PASS_FUNCTIONAL](parakeet_tdt.json) | PASS_ONE_ENGLISH_CLIP_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `qwen3_forced_aligner` | [PASS_FUNCTIONAL](followups/resolution-alignment/qwen3_forced_aligner.json) ([initial](qwen3_forced_aligner.json)) | PASS_TEN_TIMIT_HUMAN_BOUNDARY_CLIPS | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `lfm2_audio` | [PASS_FUNCTIONAL](followups/resolution-lfm-reference2/lfm2_audio.json) ([initial](lfm2_audio.json)) | ASR_MEASURED_SPEECH_REVIEW_REQUIRED | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `silero_vad` | [PASS_FUNCTIONAL](silero_vad.json) | PASS_UTTERANCE_PRESENCE_ONLY_NO_BOUNDARY_GOLD | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `trocr` | [PASS_FUNCTIONAL](trocr.json) | PASS_ONE_PRINTED_LINE_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `got_ocr2` | [PASS_FUNCTIONAL](followups/resolution-ocr2/got_ocr2.json) ([initial](got_ocr2.json)) | PASS_ONE_LABELED_PAGE_TEXT_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `florence2` | [FAIL](followups/resolution-florence-reference/florence2.json) ([initial](florence2.json)) | MEASURED_ONE_LABELED_PAGE | Pending | Yes |
| `paddleocr_vl` | [PASS_FUNCTIONAL](paddleocr_vl.json) | PASS_ONE_LABELED_PAGE_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `lightonocr` | [PASS_FUNCTIONAL](followups/resolution-ocr2/lightonocr.json) ([initial](lightonocr.json)) | PASS_ONE_LABELED_PAGE_TEXT_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `deepseek_ocr` | [PASS_FUNCTIONAL](followups/resolution-ocr2/deepseek_ocr.json) ([initial](deepseek_ocr.json)) | PASS_ONE_LABELED_PAGE_TEXT_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `bert` | [PASS_FUNCTIONAL](followups/retrieval-serving/bert.json) ([initial](bert.json)) | PASS_32_PAIR_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `mpnet` | [PASS_FUNCTIONAL](followups/retrieval-serving/mpnet.json) ([initial](mpnet.json)) | PASS_32_PAIR_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `qwen3_embedding` | [PASS_FUNCTIONAL](followups/resolution-embedding/qwen3_embedding.json) ([initial](qwen3_embedding.json)) | PASS_32_PAIR_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `qwen3_reranker` | [PASS_FUNCTIONAL](followups/retrieval-serving/qwen3_reranker.json) ([initial](qwen3_reranker.json)) | PASS_THREE_PAIR_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `lfm2_colbert` | [PASS_FUNCTIONAL](followups/colbert-expansion/lfm2_colbert.json) ([initial](lfm2_colbert.json)) | PASS_THREE_PAIR_LATE_INTERACTION_SANITY_ONLY | python: PASS_TASK_SERVICE_MAXSIM; cli: NOT_ADVERTISED_FOR_TOKEN_FEATURES; http: NOT_ADVERTISED_FOR_TOKEN_FEATURES | Yes |
| `sam3` | [PASS_FUNCTIONAL](followups/resolution-sam3/sam3.json) ([initial](sam3.json)) | PASS_PINNED_IMAGE_MASK_REFERENCE | python: PASS; cli: PASS; http: NOT_ADVERTISED_FOR_SEGMENTATION | Pending |
| `sam3_tracker` | [PASS_FUNCTIONAL](followups/resolution-sam3/sam3_tracker.json) ([initial](sam3_tracker.json)) | PASS_PINNED_IMAGE_MASK_REFERENCE | python: PASS; cli: PASS; http: NOT_ADVERTISED_FOR_SEGMENTATION | Pending |
| `sam3_video` | [PASS_FUNCTIONAL](followups/resolution-sam2/sam3_video.json) ([initial](sam3_video.json)) | PASS_PINNED_NATURAL_VIDEO_REFERENCE | python: PASS; cli: PASS; http: NOT_ADVERTISED_FOR_SEGMENTATION | Yes |
| `sam3_tracker_video` | [FAIL](followups/resolution-sam2/sam3_tracker_video.json) ([initial](sam3_tracker_video.json)) | PENDING | Pending | Yes |
| `distilbert` | [PASS_FUNCTIONAL](followups/interface-replay/distilbert.json) ([initial](distilbert.json)) | PASS_32_EXAMPLE_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `bert_token_classification` | [PASS_FUNCTIONAL](followups/resolution-ner/bert_token_classification.json) ([initial](bert_token_classification.json)) | PASS_BOUNDED_REFERENCE_EQUIVALENCE | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |
| `clip` | [PASS_FUNCTIONAL](followups/clip-cli/clip.json) ([initial](clip.json)) | PASS_THREE_CANDIDATE_SANITY_ONLY | python: PASS; cli: PASS; http: PASS; http_authentication: PASS | Yes |

## Recorded blockers

- `florence2`: labeled HF page normalized CER exceeds declared 25% floor
- `sam3_tracker_video`: SAM tracking reference failed; see sam3_tracker_video-reference-False.log

## Evidence and cleanup

- [Initial candidate identity](candidate-build.json), [final candidate identity](candidate-build-final.json) and [candidate lineage](candidate-lineage.json). The campaign includes explicit retests across successive candidate builds, identified by each worker's installed RECORD digest.
- [Exact checkpoint inventory](checkpoint-plan.json)
- [All explicit retest histories and final selections](retests.json)
- [Remaining release work](NEXT_STEPS.md) and [cleanup audit](cleanup-audit.json)
- [Final native regression log](regression-final.log); earlier failures and corrections are retained separately.
- [Final package audit](package-audit-final2.json)
- [Cleanup safety tests](campaign-safety.log)
- Each `*-assets.json` records file sizes and SHA-256 hashes before inference.
- Each `*-cleanup.json` records that durable results preceded removal of campaign-owned weights.
- Existing GPT-OSS directories remain protected. Reused existing weights are verified and retained.
- Failed and timed-out attempts retain their evidence; a failure does not become a pass after cleanup.

See [PROTOCOL.md](PROTOCOL.md) for scope, thresholds and replay commands.
