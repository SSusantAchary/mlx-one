# Python, CLI and HTTP APIs

**0.1.2 is published on PyPI**, with 45 model types and 50 verified
checkpoint profiles. [Release notes](../releases/0.1.2.md) identify the tested
scope. Interface availability remains task-specific; SAM3 uses its Python/CLI
segmentation paths.

[Getting started](../getting-started.md) · [Server and Web UI](../ui.md) · [HTTP compatibility](openai-compatibility.md)

The source checkout adds [native Python, CLI, Engine and HTTP workflows](../models/batch-1-sub-6b/README.md):
paired BERT/XLM-R reranking, dense/sparse/token retrieval, T5, supervised ViT and Kokoro.
These workflows are included in 0.1.2; the dated 0.1.1 cohort remains identifiable.

## CLI

### Inspect and plan

```bash
mlx-one inspect MODEL [--revision REVISION] [--offline] [--json-output]
mlx-one hardware list
mlx-one hardware detect --json-output
mlx-one plan inference --model MODEL --hardware PROFILE
mlx-one plan train --model MODEL --hardware PROFILE --method auto
```

Inspection is metadata-only: it does not execute remote model code or open
pickle checkpoints.

### Generate text

```bash
mlx-one generate MODEL PROMPT \
  [--revision REVISION] [--offline] [--cache-dir DIRECTORY] \
  [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] \
  [--seed N] [--stop TEXT] [--stream] [--json-output]
```

GPT-2 uses the native path for direct generation, evaluation, and inference
benchmarking. Native GPT-2 training and adapters are intentionally rejected
until their own implementation and validation gates are complete.

### Embed and rerank

```bash
mlx-one embed MODEL TEXT... \
  [--input-type query|document] [--instruction TEXT] [--dimensions N] \
  [--max-length N] [--batch-size N] [--offline] [--json-output]

mlx-one rerank MODEL QUERY DOCUMENT... \
  [--instruction TEXT] [--top-k N] [--max-length N] [--batch-size N] \
  [--offline] [--json-output]
```

Both commands use the native Qwen3 tokenizer and strict safetensors loader.
Document embeddings are unprefixed; query embeddings use the documented Qwen3
instruction format. Reranking returns the original document index, raw yes/no
logit difference, and two-class probability.

### Segment images and track video

```bash
mlx-one segment facebook/sam3 image.jpg --mode concept --text "person" \
  --output results-concept
mlx-one segment facebook/sam3 image.jpg --mode interactive \
  --prompts-json prompts.json --output results-interactive
mlx-one segment facebook/sam3 image.jpg --mode automatic \
  --points-per-side 32 --output results-automatic

mlx-one track facebook/sam3 video.mp4 --text "person" --output results-offline
mlx-one track facebook/sam3 frames/ --streaming --text "person" --output results-stream
```

Both commands accept `--revision`, `--offline`, and `--cache-dir`. The publisher's
SAM3 checkpoint defaults to the pinned revision recorded in the qualification
report. An interactive `prompts.json` can contain:

```json
{"points": [[100, 120]], "point_labels": [1]}
```

Coordinates must lie within the original image; labels `1` and `0` mean positive
and negative points. The [SAM3 guide](../../src/mlx_one/models/segmentation/sam3/README.md)
documents box exemplars, refinement masks, and timed video prompt JSON for
corrections/removal. Exports contain mask PNGs and a JSON manifest with scores,
pixel-space XYXY boxes, identifiers, dimensions, and checkpoint provenance.
Existing output directories are never overwritten.

Streaming processes frames chronologically and rejects retroactive corrections;
offline sessions retain frames for replay. Streaming omits future-dependent
hotstart removal, so its outputs are qualified separately. Automatic generation
currently uses a single-image point grid and mask-IoU NMS, not a qualified
crop-layer pipeline. These workflows remain experimental; no SAM3 training,
HTTP endpoints, or Web UI integration is provided.

### Validate training data

```bash
mlx-one data validate \
  --dataset examples/text-sft/train.jsonl \
  --layout prompt-completion
```

Supported layouts are `instruction`, `messages`, `prompt-completion`, and
`text`. A tokenizer can be supplied for truncation and loss-mask previews.

### Train and export

```bash
mlx-one train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --dataset examples/text-sft/train.jsonl \
  --config examples/text-sft/train.yaml

mlx-one export \
  --checkpoint runs/qwen-coder-sft/adapters/mlx-one-checkpoint.json \
  --output artifacts/qwen-coder-sft
```

Training is experimental. Start with the planner and pin the model revision.

### Evaluate, compare, and benchmark

```bash
mlx-one evaluate \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --dataset examples/text-sft/eval.jsonl \
  --predictions examples/text-sft/predictions-smoke.json \
  --runs-dir runs \
  --json-output

mlx-one compare runs/base/result.json runs/candidate/result.json \
  --gate examples/text-sft/quality-gate.yaml \
  --format markdown

mlx-one benchmark inference \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision PINNED_REVISION \
  --prompts examples/text-sft/prompts.json
```

ASR evaluation uses the pinned `whisper-wer-v1` and `whisper-cer-v1` profiles.

### Evidence and registry

```bash
mlx-one registry list
mlx-one registry validate registry.json
mlx-one evidence publish --result runs/example/result.json --output evidence.json
```

## Python API

Inspect and plan without initializing Metal:

```python
from mlx_one import (
    Precision,
    WorkloadKind,
    WorkloadSpec,
    estimate_memory,
    inspect_model,
    load_hardware_profile,
)

model = inspect_model("Qwen/Qwen2.5-1.5B").model
hardware = load_hardware_profile("m4-air-32gb")
workload = WorkloadSpec(
    kind=WorkloadKind.INFERENCE,
    precision=Precision.BF16,
    context_length=2048,
)

estimate = estimate_memory(model, hardware, workload)
print(estimate.to_json())
```

Transcribe a file or mono 16-kHz waveform:

```python
from mlx_one import WhisperDecodeOptions, transcribe

result = transcribe(
    "openai/whisper-tiny",
    "recording.wav",
    language="en",
    word_timestamps=True,
    options=WhisperDecodeOptions(seed=0),
)

print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.text)
```

Embed and rerank through typed native retrieval results:

```python
from mlx_one import embed, rerank

query = embed(
    "Qwen/Qwen3-Embedding-0.6B",
    "How does MLX use unified memory?",
    input_type="query",
    dimensions=512,
)

ranking = rerank(
    "Qwen/Qwen3-Reranker-0.6B",
    "How does MLX use unified memory?",
    ["MLX arrays share CPU and GPU memory.", "A recipe for sourdough."],
    top_k=1,
)

print(query.embeddings[0])
print(ranking.items[0].index, ranking.items[0].score)
```

Segment an image with Meta SAM3 through the lazy segmentation namespace:

```python
from mlx_one.segmentation import load_segmentation_model, segment_image

model = load_segmentation_model("facebook/sam3", task="concept")
result = segment_image(model, "image.jpg", text="person")
print(result.masks.shape, result.scores, result.boxes)
```

For interactive masks, load with `task="interactive"` and call
`predict_masks(model, image, points=..., point_labels=...)`. Interactive
`predicted_iou` is distinct from concept confidence `scores`; refinement uses
`low_res_logits`. All masks are original-resolution boolean arrays.

Track lexically ordered, fixed-dimension frames with a session-owned state:

```python
from mlx_one.segmentation import load_segmentation_model

model = load_segmentation_model("facebook/sam3", task="video", offline=True)
with model.video_session("frames/") as session:
    prompt_id = session.add_prompt("person")
    for result in session.propagate():
        print(result.frame_index, result.object_ids, result.prompt_ids)
```

`offline=True` requires a complete cached checkpoint. Use zero-padded frame
filenames. Python RGB iterators and local encoded-video paths are also accepted.
For live Python frames, use `model.video_session(streaming=True)` and
`session.process_frame(frame)`. Context exit releases session resources without
unloading another session's shared model. SAM3 temporal memory is separate from
the text dense/APC/block cache runtime. See the
[model guide](../../src/mlx_one/models/segmentation/sam3/README.md) for prompt validation,
corrections, cancellation, and current qualification limits.

Train through the typed SFT contract:

```python
from mlx_one import SFTTrainer, TrainConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    revision="PINNED_REVISION",
    train_dataset="examples/text-sft/train.jsonl",
    args=TrainConfig(
        output_dir="runs/qwen-coder-sft",
        max_seq_length=512,
        method="lora",
        max_steps=10,
    ),
)

result = trainer.train()
print(result.to_json())
```


### Native retrieval serving

Local Sentence Transformer pooling metadata selects the embedding loader when
starting the server. Bundles without task metadata require an explicit task:

```bash
mlx-one serve /path/to/pinned-reranker --offline --retrieval-task reranking
```

`ModelManager.load(path, offline=True, retrieval_task="reranking")` provides the
same explicit selection in Python. The server wires `/v1/embeddings` or
`/v1/rerank` to the loaded native retrieval bundle and holds its model lease until
the scheduled operation completes. ColBERT uses the
[token-feature workflow](../../src/mlx_one/models/embeddings/lfm2_colbert/README.md)
with MaxSim. See the [campaign results](../qualification/model-release-20261108/README.md)
for exact checkpoint passes and failures, including the Qwen3 embedding batch check.
