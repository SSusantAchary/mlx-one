# Source-to-native task map

The existing model registry, runtime adapter registry and InferenceService remain the dispatch authority. data/native_tasks.json attaches exact-checkpoint evidence without promoting architecture registration to release support. See qualification-matrix.md for accepted profiles and failed gates.

| Task | Native implementation / reused infrastructure | Task admission |
|---|---|---|
|T01 ViT|vision/vit_classification.py, unchanged shared TrOCR encoder|Blocked by whole-trace FP32; raw helper retained|
|T02 sentiment|models/classification/distilbert.py, existing task actor/pool/device owner|Experimental Python/CLI/HTTP|
|T03 NER|models/classification/bert_ner.py reuses existing BERT embeddings/encoder; task-specific head/mask/postprocess|Experimental Python/CLI/HTTP|
|T04 CLIP|models/classification/clip.py reuses LFM2 attention/shared QuickGELU; bounded native image processor; audited conversion|Experimental one image/1–10 labels, Python/CLI/HTTP|
|T04 fixed SigLIP2|models/classification/siglip_fixed.py reuses LFM2 blocks/attention and CLIP embeddings; actual fixed SiglipModel source|Blocked intermediate FP32; raw helper|
|T05 Silero|models/audio/silero, existing Parakeet PredictorLSTM, audio input processing, shared actor and request-private host state|Experimental mono16k offline/Python streaming, CLI/HTTP offline|
|T06 DETR|models/detection/detr.py, source-specific ResNet50/frozenBN/decoder/boxes and reused native attention projections|Blocked whole traces; raw helper|
|T06 RT-DETR|No native port; retained independent source/checkpoint lock|Not implemented behind DETR gate|
|T07 document QA|vision/document_qa.py wraps existing VisionLanguageEngine/Qwen2.5 model/processor; no duplicate engine or OCR pipeline|Blocked dense vision; internal one-page candidate only|
|T08 video QA|vision/video_processing.py reuses local FFmpeg/image infrastructure; Qwen2.5-specific per-grid attention/time positions corrected against source|Blocked processor/vision; end-to-end QA not implemented|
|T09 relative depth|models/depth/depth_anything.py reuses compatible attention/GELU MLP, adds DINOv2/taps/reassembly/fusion/head/interpolation; checkpoint-selected DPT processor|Blocked FP32 and independent quality; raw helper|

No MLXBoost code, defaults, benchmarks, global tolerances or shared OCR encoder mathematics were changed. Source/refactor details and licenses are in source-audit.md and the exact lockfiles.
