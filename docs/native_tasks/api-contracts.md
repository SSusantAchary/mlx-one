# Native task API contracts

M1 adds `text-classification` to the existing core architecture registry, runtime adapter registry and `InferenceService`. It uses the same actor, device owner, model pool, leases, cancellation and deadline checks as other native tasks. `ClassificationEngine` is a facade over that service. ViT is not admitted to the task service because its intermediate FP32 gate still fails.

```python
from mlx_one import ClassificationEngine, classify

result = classify("/local/pinned-sst2", "I enjoyed this film.")
with ClassificationEngine("/local/pinned-sst2") as classifier:
    result = classifier.classify(["Wonderful.", "Tedious."], max_length=128, top_k=2)
print(result.to_dict())
```

The checkpoint must contain the exact locked FP32 assets. Local/cached loading is the Python and classification CLI default. `offline=False` or CLI `--download` explicitly enables the existing Hub acquisition tool at the immutable revision. Inference never selects another checkpoint or runtime.

```sh
mlx-one classify /local/pinned-sst2 "I enjoyed this film." --json-output
mlx-one serve /local/pinned-sst2 --offline --alias sentiment --api-key SECRET --no-warmup
```

`POST /v1/tasks/text-classification`, JSON:

```json
{"model":"sentiment","input":["Wonderful.","Tedious."],"max_length":512,"top_k":2}
```

Use the server's configured bearer authentication. The result has `schema_version=mlx-one-classification-v1`, `task`, immutable `model`, `revision`, `source_commit`, `score_semantics`, and `predictions`. Each prediction contains input `index`, post-truncation `token_count`, `truncated`, and `labels` with `label`, `index` and numerical `score`. No raw input is echoed. A serving alias is accepted in the request; the response preserves the canonical checkpoint identity.

One or two nonempty, non-whitespace texts are accepted. Each text is limited to 32,768 UTF-8 bytes; aggregate text is limited to 65,536 bytes. `max_length` is an integer 2–512 inclusive and includes CLS/SEP; truncation is from the right, padding from the right with ID 0, and tokenization options are request-local. `top_k` is 1 or 2. Scores are the checkpoint's mutually exclusive softmax outputs; no probability-calibration claim is made. Sort order is descending score, then ascending class index for ties. Scores are emitted as round-trippable JSON numbers. Only FP32 is admitted.

Errors reuse the server's `error` envelope: 401 authentication, 404 unknown loaded alias, 400 task capability mismatch, 422 malformed input/options, 413 route body limit, 429 queue/capacity, 408 timeout, and cancellation handling. A disconnected HTTP client sets the same cancellation event. Metal work is bounded and is not forcibly interrupted mid-kernel; model leases remain held until work has stopped. `submit(...).cancel()` and explicit monotonic `deadline` are available on the Python engine. No classification SSE stream is advertised.

Existing completion/chat, OCR, audio, embeddings and rerank endpoints retain their contracts. A classification-only model rejects generation requests.

## M2 token classification

`TokenClassificationEngine.tag` and `token_classify` reuse the same service. The locked dslim BERT checkpoint is admitted in FP32 with one or two texts, the same byte/token bounds as sentiment, and `aggregation_strategy="none"` (default) or `"simple"`. Other aggregation modes are rejected. Entity start/end offsets are Python Unicode codepoint offsets, end exclusive, in the original input. NONE returns winning non-O token labels; SIMPLE follows source BIO grouping, averages winning-token scores and filters O groups after grouping. Offsets are truncated with input tokens; `truncated` is explicit. Scores are uncalibrated token softmax/group means.

```python
from mlx_one import token_classify
print(token_classify("/local/pinned-bert-ner", "Marie Curie lived in Paris.",
                     aggregation_strategy="simple").to_dict())
```

```sh
mlx-one ner /local/pinned-bert-ner "Marie Curie lived in Paris." --aggregation-strategy simple --json-output
```

`POST /v1/tasks/token-classification` accepts JSON `{"model":"entities","input":"Marie Curie lived in Paris.","max_length":512,"aggregation_strategy":"simple"}`. Launch with existing `mlx-one serve /local/pinned-bert-ner --offline --alias entities --no-warmup`. Results include immutable model/source identity and `predictions[].entities`. Duplicate names remain separate by offsets.

## M2 zero-shot image classification

`ZeroShotImageEngine.classify` and `zero_shot_classify` admit the audited local CLIP conversion only. Input is one RGB/grayscale image and 1–10 nonempty candidate labels, at most 512 UTF-8 bytes per label. The fixed source prompt is `This is a photo of {}.`; prompts over 77 tokens are rejected without truncation. Duplicate labels are preserved and tied scores sort by input index. CLIP scores are softmax conditional on the supplied candidate set, not calibrated independent probabilities. Image batch >1, 100-label profiles, arbitrary templates, transparent images and SigLIP2 task admission are unavailable.

```python
from mlx_one import zero_shot_classify
print(zero_shot_classify("/local/audited-clip", "/local/cat.png", ["cat", "dog"]).to_dict())
```

```sh
mlx-one zero-shot-image /local/audited-clip /local/cat.png --label cat --label dog --json-output
```

`POST /v1/tasks/zero-shot-image-classification` requires multipart form fields `model`, `candidate_labels` (a JSON string array), optional `top_k`, and one PNG/JPEG/WebP `file`. No remote URLs or server filesystem paths are accepted. The upload cap is 20 MiB; decoded image dimensions/pixels use the existing bounded image decoder. Launch with existing `serve` and configured bearer auth. Results include canonical identity, original image size, template, candidate index/label, logit and score. The outer HTTP body guard is 32 MiB, with tighter inner upload/label bounds. Errors/lifecycle follow the shared service contract above; HTTP media type errors use 415.

## M3 voice activity detection

```python
from mlx_one import VADEngine
with VADEngine("/local/pinned-silero") as vad:
    result = vad.detect("/local/mono16k-pcm16.wav")
    session = vad.session()
    # Each process_chunk accepts exactly 512 mono samples at16kHz.
    # session.process_chunk(pcm512) returns one frame speech probability.
    # session.finalize(tail) accepts an optional final1–511 samples.
print(result.to_dict())
```

The local directory must contain the exact pinned `silero_vad_16k.safetensors`. Loading is offline; no checkpoint replacement or JIT execution occurs in production. Accepted PCM is a finite mono floating array or int16, sample_rate exactly16000. Encoded input is mono PCM16 WAV only; maximum600s/25MiB. No resampling, stereo mixing, music robustness or alternate-rate claim. Empty PCM produces no frames/segments. Partial final frames are zero-padded only for inference; timestamp endpoints are bounded by original sample count.

Frame size512/context64, two128-element recurrent states per stream. Session process/finalize/reset calls serialize; sessions hold separate host state and share only model weights through the existing task actor. Reset clears probability/context/state history. After finalization reset is required. Finalize applies the same offline source timestamp policy; no live VADIterator event protocol or HTTP stateful stream is advertised.

Default options match source: threshold0.5, negative threshold max(threshold−0.15,0.01), minimum speech250ms, silence100ms, padding30ms. Public options expose threshold strictly(0,1) and these three durations in[0,60000]ms. Outputs contain immutable source/checkpoint identity, frame probabilities, original sample count and absolute sample `start`/`end` (end exclusive). Scores are uncalibrated model sigmoid outputs. Output is bounded to18,750 frames.

```sh
mlx-one vad /local/pinned-silero /local/mono16k-pcm16.wav --json-output
mlx-one serve /local/pinned-silero --offline --alias vad --no-warmup
```

Authenticated `POST /v1/tasks/voice-activity-detection`: multipart `model`, one WAV `file`, optional threshold/min_speech_duration_ms/min_silence_duration_ms/speech_pad_ms. Duplicate/unknown fields, URLs and filesystem paths are rejected. Shared request leases/cancellation/deadlines, auth/error envelope and cleanup apply. Work yields between chunks on the shared device owner. DETR and RT-DETR have no task route; raw DETR diagnostic loading does not imply support.

## M4 internal candidates, no admitted task routes

`mlx_one.vision.document_qa.DocumentQAEngine` is an internal diagnostic wrapper over existing VisionLanguageEngine/InferenceService, exact prior Q4 checkpoint, one image page and greedy1–64-token answer. It returns answer/generated IDs/image metadata and supplied page_index1; no inferred citations/confidence/retrieval claim. Source vision gates fail, so there is no registered document-QA Python facade, CLI command or HTTP route. The runnable candidate example is examples/native_tasks/document_qa_candidate.py. Video frame/patch helpers exist for development; there is no complete video-answer task or public route. Existing generic VLM image APIs keep their prior contracts.
