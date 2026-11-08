# Model fixes and release decisions

The final selection has **40 passing primary workflows**. Decisions follow the user's explicit request to use MLX-LM-style support admission: implementation tests, native real-checkpoint output, working relevant interfaces and regressions. No blanket experimental label or extra broad-reliability certificate is required. [Current support decisions](RELEASE_CANDIDATE.md) bind these checks to exact checkpoints. [Policy](SUPPORT_POLICY.json) records the upstream sources and mandatory checks.

## Repairs and evidence

| Area | Change or finding | Recorded result |
| --- | --- | --- |
| OpenELM and LFM2 text | Corrected BOS/template ownership and cache/token accounting. | Real generation plus interfaces pass; original failures retained. |
| Qwen2 MoE | Added packed routed-expert quantization metadata handling and corrected top-k normalization. | Tiny quantized load and real pinned Q4 workflow pass. |
| BERT/MPNet retrieval | Drop only validated, unused pooler tensors; preserve strict loading elsewhere. | Native embedding workflows pass. |
| Qwen retrieval | Support publisher ByteLevel/Sequence tokenizer composition and bare-backbone weights. | Embedding/reranking workflows pass. |
| Retrieval serving | Add native model-manager/server integration and retain leases during scheduled work. | Applicable Python/CLI/authenticated HTTP workflows and regressions pass. |
| LFM2 ColBERT | Match published prefixes, projection, query expansion, punctuation and unmasked expansion convolution. | Independent stock publisher vector difference below 3.7e-7; token/MaxSim interfaces pass. |
| Qwen3 Embedding | Keep BF16 storage and explicitly execute the backbone in FP32. | Batch/single max difference 2.398e-7 and publisher max difference 4.321e-7; original strict consistency check passes. |
| Forced alignment | Use ten pinned TIMIT samples with independently annotated word boundaries. | Mean error 29.472 ms, p95 68.1875 ms, coverage 1.0; original 80/200-ms checks pass. |
| NER | Verify tokenizer IDs/masks/offsets, all layers, logits and entity aggregation against an isolated publisher runtime. | Native/reference tensors, shapes and keys match; logits max difference 6.58035e-5. Sample F1 0.6024 matches the publisher and remains a diagnostic. |
| OCR text rendering | Compare visible plain text with plain-text references; preserve requested raw Markdown and numeric content. | Versioned `ocr-visible-text-v3` metric. GOT CER 0.1806, LightOn 0.0590 and DeepSeek 0.1204. Original failed metric results retained. |
| GOT/Florence decoding | Restore full-token byte decoding; implement publisher no-repeat n-gram constraint and native beam decoding. | Actual OCR task and applicable CLI/HTTP/authentication checks pass. Florence benchmark limits remain below. |
| LFM2 Audio | Fix interleaved text/audio EOS state ordering, use publisher speech sampling defaults and request-local RNG. Preserve 512-frame guard. | Final pinned Q8 profile passes ASR/TTS/short and longer STS. Speech CLI, HTTP, streaming, early-close/recovery and invalid-input checks pass. |
| SAM tracker video | Implement standalone temporal checkpoint/session adapter, explicit object prompts and bounded lifecycle. | Standalone native offline/streaming and CLI pass, including the 32-frame natural-video reference run. |
| SAM preprocessing/runtime | Match torchvision v2 uint8 resize rounding/antialiasing and keepdim tensor behavior; normalize per-object score shape. | Image/point references and native composite video workflows pass. Exact masks/boxes/IDs and CLI receipts are archived. |
| Packaging and claims | Package notices/data and all 40 canonical types; bind supported claims to immutable result hashes. Remove blanket unqualified help labels. | Exact wheel/sdist payload audit and final clean-wheel registry/catalog/import/help checks pass. |

## Preserved benchmark limits

- **Florence-2-base plain OCR:** recorded page CER 0.3287 exceeds the original 0.25 benchmark floor. The independent publisher run also exceeded that floor (0.2824). The final functional record passes actual task/interface checks and retains `page_quality_diagnostic: FAIL`. Optional OCR region output produced a malformed geometry parse and is excluded from the supported primary operations.
- **NER:** F1 0.6024 is the same on the selected small CoNLL slice in the publisher runtime; implementation support does not promise a different model accuracy.
- **LFM2 Audio Q4:** ASR and execution now complete, but longer-speech round-trip WER remained weak (0.625 with Whisper Tiny; 0.5833 with Qwen ASR). Final support selects the separately pinned **8-bit** checkpoint. These Q4 measurements are not promoted into Q8 accuracy evidence. Reference forced-history/codec diagnostics remain separate.
- **SAM composite video reference:** optional MPS reference streaming disagreed with its own offline behavior; native offline/streaming outputs remained nearly identical and actual native/CLI workflows pass. The failed cross-runtime comparison remains recorded. Support does not assert all-device reference parity.
- Full leaderboards, every size/precision, all optional task variants, every device and broad reliability certification are not universal admission gates in MLX-LM. Additional accuracy or performance claims still require their own evidence.

## Evidence and artifact identity

`checkpoint-plan-release.json` and `release-retests.json` select the exact passing workflow for each type; the original plan and strict retest history are preserved. Per-model installed RECORD identities map to immutable wheels through `candidate-lineage.json` and `candidate-lineage-resolution.json`. The final wheel source manifest is `candidate-build-resolution-final.json`. Only support catalog/help text changed after inference build 8; real task records retain their actual historical build identities.

Final native regression, package payload, clean-wheel and cleanup audits are linked from the technical readiness report. Optional skipped tests are reported separately and never counted as passes. Campaign-downloaded checkpoint weights were removed after outputs were saved, while the user's protected GPT-OSS model assets were preserved and hash-verified. No artifact was published.
