# A8 public API and HTTP scope

Latest independent qualification: **Qwen3 quality FAIL, bounded coverage PASS**, 15/27 distinct cases passing across five deterministic repetitions. Native execution/lifecycle and stream/generate consistency passed. See [release-readiness.md](release-readiness.md). The results below retain their stated earlier source identity and scope. No alpha support promotion.

Current authorized native validation: pinned Qwen3 Q8 reference B1 passed real Python/CLI/live HTTP chat/SSE, auth, disconnect, cancellation/recovery and cleanup on M4. Full quality coverage and release installation remain incomplete; other tasks retain their prior candidate scope. See [native Qwen3 validation](native-qwen3-validation.md). The implementation validation below records the earlier A8 stage.

All 82 baseline root exports and 27 architecture names preserved exactly (compared with private A0 manifests). Package, text, vision, audio, retrieval, training and compatibility imports remain backend-free. No generic load_model facade added: existing task-aware loaders and registry already dispatch without duplicating runtime ownership. Engine remains text-specific.

| Python workflow | Signature / example |
|---|---|
| Engine text | Engine(model, revision=..., offline=True); generate(prompt, max_tokens=32, temperature=0); stream(prompt, ...) |
| Stateless text | generate(model, prompt, options=TextGenerationOptions(...)); stream_generate/stream_chat preserved |
| VLM | load_vlm_model(model, revision=..., offline=True); stream_vlm(bundle, prompt, images, options=...) |
| ASR | transcribe(model, audio, revision=..., offline=True, options=WhisperDecodeOptions(...)) |
| Retrieval | embed(model, texts, max_length=None, ...); rerank(model, query, documents, ...) |

| Existing endpoint | Scope / behavior |
|---|---|
| GET /health, /v1/models, /v1/runtime | Health, served identity, effective runtime/capabilities |
| POST /v1/chat/completions | Native text and existing bounded multimodal content; SSE [DONE] |
| POST /v1/embeddings | Embedding task service required |
| POST /v1/rerank | Pair-scoring service required; not ordinary embeddings |
| POST /v1/audio/transcriptions | Multipart; optional Whisper sidecar; 25 MiB file, 26 MiB request, 300s decoded duration, 120s decoder timeout |
| /v1/completions | Absent; no compatibility guarantee or duplicate route added |

Preserved authentication, alias checks, body limits (text 1 MiB, multimodal 32 MiB), queue timeout/backpressure, private principal namespaces and unload/cancel behavior. Unsupported task services return structured capability errors. Native combinations do not fall back to another inference engine. API route availability is not checkpoint qualification.

Implementation: sidecar now uses the existing bounded decoder before native transcription, with exact format/rate and guaranteed temporary-file cleanup on decoder errors. A6 owner supplied the optional closed input-format selector during this integration. AudioProcessingError becomes AudioUploadError; existing route returns structured 415. OpenAPI now reports actual 0.1.0a1, avoiding a false final-release version. No new model residency policy or MLX execution owner.

Validation: a8 log/JUnit includes decoder safety, new import-denial subprocess and server tests plus server/controls/scheduler/media/transcription/CLI/compat/Engine/model-pool regression suites. 68 passed, two dependency deprecation warnings. Exact test list is in execution-report/progress and JUnit. All root exports and model names identical to A0. HTTP client --help PASS; no real model or live network smoke run. Real advertised-task usability remains HARDWARE_PENDING.
