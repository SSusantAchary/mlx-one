# Native model workflows

Run dense and hybrid retrieval, paired reranking, encoder-decoder text generation,
supervised image classification and English speech with native MLX models.
The ten checkpoint profiles below cover BERT, XLM-R, Nomic, T5, ViT and Kokoro;
the library has **45 architecture types across seven modalities**. Multiple
checkpoints can share one architecture type.

The dated 40-type **0.1.1 release approval remains unchanged**. These changes
need to be merged and published before they appear in an installation from PyPI.
All ten profiles pass the technical release checks for their documented tasks.
The next package release decision is pending. See
[results](../../qualification/batch-1-sub-6b/results.json), the
[engineering report](REPORT.md), [source identities](references.md), and
[development guidelines](../../../mlx_one_development_guidelines_v1.0.md).

## Checkpoint profiles

| Pinned checkpoint | Native type | Primary operation and limits |
| --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `bert` | Paired cross-encoder, trained pooler and scalar head; 512 tokens including specials |
| `BAAI/bge-small-en-v1.5` | `bert` | 384-D CLS + L2 dense retrieval; 512 tokens |
| `BAAI/bge-m3` | `xlm-roberta` | 1024-D dense, learned sparse weights, 1024-D token vectors and hybrid scoring; converted safe bundle |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `bert` | 384-D masked mean, **no L2 normalization**; 128-token sentence limit |
| `google-t5/t5-small` | `t5` | Greedy encoder-decoder text-to-text, teacher forcing; 512 encoder tokens, at most 511 new tokens; batches 1–8 |
| `google/vit-base-patch16-224` | `vit` | Supervised ImageNet classifier, fixed 224×224 RGB, patch size 16; batches 1–8 |
| `BAAI/bge-reranker-v2-m3` | `xlm-roberta` | Paired XLM-R classification head, raw scalar logits and separate sigmoid probabilities |
| `hexgrad/Kokoro-82M` | `kokoro` | Native 24 kHz mono English speech, `af_heart` only, 1–510 phonemes, speed 0.5–2, at most 30 seconds |
| `nomic-ai/nomic-embed-text-v1.5` | `nomic_bert` | Normalized 768-D default; explicit Matryoshka 64/128/256/512/768, full-vector layer norm → truncate → normalize |
| `intfloat/multilingual-e5-small` | `bert` | 384-D masked mean + L2, XLM-R tokenizer with **BERT neural math**, query/passage prefixes; 512 tokens |

FP32 execution is the admitted baseline. BGE-M3's auxiliary heads retain the
publisher's FP16 storage and execute with its FP32 backbone. No quantized profile,
alternate Kokoro voice/language, causal T5 chat, ViT zero-shot classification,
streaming speech, or MLXBoost speedup is implied.

BGE-M3, BGE Reranker and Nomic configurations allow contexts up to 8192 tokens, but the
repeated reference/performance workload is bounded at 512. Longer contexts need
separate measurements. Multilingual MiniLM explicitly rejects a requested limit above 128.

## Install from the working tree

```bash
python -m pip install -e '.[server,vision,kokoro]'
```

The `kokoro` extra provides a rule-based English eSpeak frontend; its dependencies
retain their own licenses. Phoneme-input synthesis requires only the core package.
Torch/Transformers and publisher inference packages are development-only oracles
and conversion tools. They are absent from the clean inference environment.

## Dense retrieval and paired reranking

```python
from mlx_one.retrieval import embed, rerank

vectors = embed("BAAI/bge-small-en-v1.5", ["Paris is the capital of France."])
scores = rerank("cross-encoder/ms-marco-MiniLM-L6-v2",
                "What is the capital of France?",
                ["Paris is the capital of France.", "Bananas are yellow."])
print(scores.items[0].score)        # raw logit; larger is more relevant
print(scores.items[0].probability)  # separate sigmoid, not a calibrated probability
```

Use the immutable revisions in [references](references.md) for reproducibility.
The local [retrieval/reranking demo](../../../examples/batch1_retrieval.py) selects
documents with BGE-small or multilingual E5 and reranks with MiniLM or BGE
Reranker, entirely through MLX-One.

```bash
mlx-one embed BAAI/bge-small-en-v1.5 'Paris is in France.' --json-output
mlx-one rerank cross-encoder/ms-marco-MiniLM-L6-v2 \
  'capital of France' 'Paris is in France.' 'Bananas are yellow.' --json-output
```

### Pooling and prefixes

For multilingual MiniLM, use `embed()` identically and preserve the returned raw mean vectors.
Cosine-based retrieval can normalize copies explicitly; normalization is not
silently inserted into the checkpoint's sentence-transformer pipeline.

E5 uses `input_type="query"` → `query: ` and the default document role →
`passage: `. Nomic uses `search_query: ` / `search_document: `. The Hub identity
selects these documented profiles. **An arbitrary local copy needs an explicit
`mlx_one_retrieval.json` sidecar** so its intended role is not guessed from a
folder name:

```json
{"query_prefix": "query: ", "document_prefix": "passage: "}
```

Use `apply_prefix=False` (CLI `--raw-text`) for already formatted input. An
explicit `instruction=` is query-only and is concatenated as supplied. For
BGE-small, a publisher-style instruction may be supplied explicitly; this
campaign's controlled retrieval profile uses no added instruction.

```python
from mlx_one.retrieval import embed

query = embed("intfloat/multilingual-e5-small", "Quelle est la capitale de la France?",
              input_type="query")
short = embed("nomic-ai/nomic-embed-text-v1.5", "capital of France",
              input_type="query", dimensions=128)
```

For Nomic, omitting `dimensions` uses the normalized full-vector profile;
explicit `dimensions=768` requests Matryoshka preprocessing even at full width.
Those operations have separate actor batching identities.

## BGE-M3's three representations

Convert the pinned publisher `.bin` and two `.pt` heads offline as described in
[references](references.md#safe-offline-conversion). Native runtime accepts the
converted safetensors bundle and rejects missing trained auxiliary heads.

```python
from mlx_one.retrieval import encode_multi, hybrid_scores, load_retrieval_model

model = load_retrieval_model("./bge-m3-native", task="embedding", offline=True)
query = encode_multi(model, "capital of France")
documents = encode_multi(model, ["Paris is in France.", "Bananas are yellow."])
print(query.dense_embeddings, query.sparse_weights, query.token_embeddings)
print(hybrid_scores(query, documents))
```

Sparse output takes the maximum learned positive weight per token ID, excluding
the publisher's special-token set. ColBERT excludes CLS and padding, retains EOS,
and uses mean query-token MaxSim. The default hybrid weights are dense `1`,
sparse `0.3`, ColBERT `1`; individual component scores remain visible. Query and
document features must come from the same loaded execution identity.

`mlx-one embed-multi ./bge-m3-native 'capital of France' --offline` emits all
three fields. Dense-only `embed()` also works. Multi-representation output is
currently a direct Python/CLI API; it has no multi-head actor or HTTP contract.

## T5 encoder-decoder text generation

```python
from mlx_one.seq2seq import generate

result = generate("google-t5/t5-small",
                  "translate English to German: The house is wonderful.",
                  max_new_tokens=24)
print(result.texts, result.finish_reasons)
```

```bash
mlx-one seq2seq google-t5/t5-small \
  'translate English to German: The house is wonderful.' --max-new-tokens 24
```

The native decoder reuses encoder outputs and recomputes its bounded decoder
prefix. No incremental KV-cache, sampling, beam-search or causal-chat claim is made.

## ViT supervised image classification

```python
from mlx_one.classification import classify_image

result = classify_image("google/vit-base-patch16-224", "./cat.jpg", top_k=5)
print(result.predictions)
```

```bash
mlx-one classify-image google/vit-base-patch16-224 ./cat.jpg --top-k 5 --json-output
```

Input is a local path, PIL image, or uint8 RGB/HWC array, capped at 32 megapixels.
The processor matches the pinned resize/rescale/normalization settings. Logits
and top-five indices match the independent publisher on two CC0 cat images.
The stricter hidden-state `1e-4` diagnostic **remains failed**; an independent
FP64 comparison shows the native FP32 error is smaller than publisher FP32 error.
The accepted scope is supervised classification, not universal hidden-state parity.

## Kokoro native English speech

Safely convert the publisher PTH and `af_heart.pt` in the isolated reference
environment first. All acoustic and waveform neural layers then run in MLX.

```python
from mlx_one.audio.kokoro import synthesize

speech = synthesize("./kokoro-native", "Hello, world. This is native speech.")
speech.save("./speech.wav")
print(speech.sample_rate, speech.to_dict())
```

```bash
mlx-one tts ./kokoro-native 'Hello, world.' --output ./speech.wav --json-output
```

Pass `phonemes=True` (CLI `--phonemes`) for publisher-alphabet phoneme text
without the optional G2P tools. Unknown voices, languages, phonemes, invalid speed,
overlong utterances and predicted audio above 30 seconds fail explicitly.
The maintainer confirmed the [saved 4.7-second sample](../../qualification/batch-1-sub-6b/B08/native-english.wav)
is clear and intelligible. Native Whisper independently returned its exact text;
neither result establishes general perceptual quality or corpus accuracy.

## Engine and HTTP

Use the existing `InferenceService`, with explicit batch/admission limits:

```python
from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.service import InferenceService
from mlx_one.seq2seq import load_seq2seq_model

service = InferenceService(batching_config=BatchingConfig(
    max_sequences=8, max_scheduled_tokens=8192, task_transient_budget_bytes=1024**3))
try:
    service.register_task("t5", load_seq2seq_model("google-t5/t5-small"))
    result = service.run_task("t5", "text-to-text-generation", {
        "texts": "translate English to German: Hello.", "max_new_tokens": 16}).output
finally:
    service.close()
```

Use `embedding`, `reranking`, `image-classification`, or `speech-generation` with
the corresponding loaded bundle. For Kokoro, set
`task_transient_budget_bytes=4*1024**3`: its bounded 30-second synthesis reserves
3 GiB of transient capacity in addition to resident weights. The server sets
this speech budget automatically. Engine cancellation is bounded at task execution
boundaries; it does not interrupt an already executing Metal kernel. Service
closure drains leases and rejects later submissions.

Start serving a **local complete bundle** with `mlx-one serve ./bundle --alias local`.
These HTTP paths are tested with real weights:

| Operation | Endpoint | Request fields beyond `model` |
| --- | --- | --- |
| Dense embeddings | `/v1/embeddings` | `input`, `input_type`, optional `dimensions`, `instruction`, `apply_prefix` |
| Paired reranking | `/v1/rerank` | `query`, `documents`, optional `top_n` |
| T5 | `/v1/tasks/text-to-text-generation` | `input`, `max_length`, `max_new_tokens` |
| ViT | `/v1/tasks/image-classification` | `images`: 1–8 PNG/JPEG/WebP base64 data URIs, `top_k` |
| Kokoro | `/v1/tasks/speech-generation` | `input`, `voice`, `language`, `phonemes`, `speed`; mono PCM16 WAV response |

The new task routes use the existing API-key validation, actor, leases and shutdown
path. HTTP images do not accept local paths or remote URLs. The existing 1 MiB
request-body cap still applies. These tasks do not become chat-completion models.
