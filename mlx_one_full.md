# mlx-one — Full Strategy, Architecture, Implementation Standard, and Roadmap

> Consolidated project strategy for `mlx-one`.
>
> This document merges the current implementation state, long-term product direction,
> outcome-based roadmap, README-level public positioning, and the renewed decision to
> build a **native, owned MLX model stack** across language, embeddings, vision-language,
> and audio.
>
> Reference development machine: Apple M4 MacBook Air, 32 GiB unified memory.
>
> Primary platform: Apple Silicon + MLX.
>
> First implementation wave: models **below 3B parameters** across supported modalities.
>
> Long-term policy: model-size agnostic. The <3B boundary is the initial engineering and
> maintainer-validation scope, not a permanent software ceiling.

---

# 1. Executive Direction

`mlx-one` should become a **single MLX-native package** for the end-to-end lifecycle of
small and efficient multimodal models.

The renewed direction is intentionally different from a pure orchestration layer.

The project should eventually own:

- language-model architectures and execution;
- embedding models and retrieval;
- vision-language models;
- ASR and TTS models;
- model loading;
- processors and tokenizers where needed;
- generation;
- quantization integration;
- LoRA / QLoRA / SFT;
- evaluation;
- benchmarking;
- artifact lineage;
- compatibility evidence;
- release packaging.

The core product principle is:

> **One package. One implementation style. One lifecycle. Native MLX.**

The first target is broad, qualified support for popular models below 3B parameters across:

```text
LM
VLM
Embeddings
ASR
TTS
```

Existing MLX projects remain extremely valuable, but their role changes.

They are:

- implementation references;
- parity references;
- upstream sources of ideas;
- compatibility targets;
- sources from which MIT-licensed implementation patterns may be adapted when
  legally and technically appropriate.

They are **not intended to remain mandatory runtime dependencies** for the final native
mlx-one stack.

The relevant reference projects include:

```text
https://github.com/Blaizzy/mlx-vlm
https://github.com/Blaizzy/mlx-audio
```

Both are MIT licensed and can be studied for architecture, conventions, processors,
weight conversion patterns, and model integration style.

For language models, existing MLX-LM implementations should also be studied closely as
reference implementations, while mlx-one gradually takes ownership of its own native
model code.

---

# 2. Product Vision

The long-term lifecycle remains:

```text
Discover
  ↓
Inspect
  ↓
Plan
  ↓
Load / Convert
  ↓
Train / Fine-tune
  ↓
Evaluate
  ↓
Verify
  ↓
Benchmark
  ↓
Gate
  ↓
Ship
```

A more implementation-oriented view is:

```text
Hugging Face / Local Artifact
          ↓
     mlx-one inspect
          ↓
 Architecture Registry
          ↓
 Native Model Loader
          ↓
 ┌────────┼────────┬────────────┐
 │        │        │            │
 LM      VLM    Embeddings     Audio
 │        │        │            │
 └────────┼────────┴────────────┘
          ↓
   Native MLX Runtime
          ↓
 Generation / Embed / ASR / TTS
          ↓
 LoRA / QLoRA / SFT / Evaluation
          ↓
 Quantize / Benchmark / Verify
          ↓
 Reproducible Release Bundle
```

The target user experience should eventually resemble:

```bash
mlx-one inspect MODEL

mlx-one run MODEL

mlx-one generate MODEL --prompt "..."

mlx-one embed MODEL --text "..."

mlx-one transcribe MODEL audio.wav

mlx-one speak MODEL --text "..."

mlx-one train MODEL --dataset train.jsonl --method lora

mlx-one evaluate MODEL --profile text-core

mlx-one benchmark MODEL

mlx-one quantize MODEL --bits 4

mlx-one export MODEL --output artifact/

mlx-one ship MODEL
```

And through Python:

```python
from mlx_one import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-1.7B")
output = model.generate("Explain KV cache.")
```

The modality-specific APIs should feel like parts of the same package rather than
separate ecosystems.

---

# 3. Core Strategic Change

## Previous interpretation

Earlier planning positioned `mlx-one` primarily as:

> an orchestration and evidence layer over MLX-LM, MLX-VLM, MLX-Audio, and
> MLX-Embeddings.

That approach was useful to establish:

- schema design;
- inspection;
- hardware planning;
- evidence;
- evaluation;
- compatibility records;
- calibration;
- initial SFT workflows.

However, it creates long-term disadvantages:

- users still install multiple packages;
- behavior differs across modalities;
- model support depends on external package release cycles;
- APIs diverge;
- training interfaces diverge;
- processors and loaders differ;
- fixes require upstream coordination;
- a model release may work in one MLX package but not another;
- mlx-one cannot fully control lifecycle guarantees.

## Renewed interpretation

`mlx-one` should become:

> **a unified native MLX model stack that owns its supported model
> implementations and lifecycle while using MLX itself as the execution
> foundation.**

MLX remains the underlying tensor/autograd/compiler/runtime technology.

mlx-one owns the model layer above MLX.

That means the long-term architecture is closer to:

```text
                  mlx-one

 ┌──────────────────────────────────────────────┐
 │                Unified APIs                  │
 │ load / run / train / eval / quant / export │
 └──────────────────────────────────────────────┘
                       ↓
 ┌──────────────────────────────────────────────┐
 │          Native Model Implementations        │
 │                                              │
 │ LM       VLM       Embeddings       Audio   │
 └──────────────────────────────────────────────┘
                       ↓
 ┌──────────────────────────────────────────────┐
 │       Shared Training / Generation Core      │
 └──────────────────────────────────────────────┘
                       ↓
 ┌──────────────────────────────────────────────┐
 │                    MLX                       │
 │ arrays / nn / optimizers / metal / compile  │
 └──────────────────────────────────────────────┘
```

External MLX packages become references and optional migration aids, not core architectural
dependencies.

---

# 4. Project Positioning

Short description:

> `mlx-one` is a unified native MLX package for loading, running, fine-tuning,
> evaluating, quantizing, and shipping language, vision-language, embedding,
> speech, and audio models on Apple Silicon.

Longer positioning:

> `mlx-one` provides one implementation style and lifecycle across local AI
> modalities. It owns supported model implementations above MLX, beginning with
> popular models below 3B parameters, while preserving reproducible hardware
> evidence, evaluation, compatibility records, and release lineage.

Primary tagline:

> **One MLX stack. Every small model.**

Existing tagline may remain useful for qualification and deployment:

> **Train anywhere. Prove it on Apple Silicon.**

The two messages can coexist:

```text
Product:
One MLX stack. Every small model.

Lifecycle:
Train anywhere. Prove it on Apple Silicon.
```

---

# 5. Initial Scope

The initial engineering target is:

> Implement popular models below 3B parameters across all major modalities.

This scope is intentionally aggressive but bounded enough to establish architecture,
coding standards, and integration velocity.

The first-wave modalities are:

1. Language models
2. Embedding models
3. Vision-language models
4. ASR
5. TTS

Later:

6. Rerankers
7. Audio-language models
8. Multimodal embeddings
9. Vision encoders
10. Audio encoders
11. Preference training
12. Model merging
13. Distributed training
14. Cross-backend conversion

The <3B target is not a hard check such as:

```python
if params > 3_000_000_000:
    reject()
```

That must **never** become the design.

Instead:

```text
software support = architecture capability
hardware verification = exact workload evidence
```

---

# 6. Implementation Philosophy

Every supported model family should look structurally familiar to contributors.

A contributor integrating a new language model, VLM, embedding model, or audio model
should not need to invent a new architecture inside the repository.

The project therefore needs a **standard implementation style**.

The implementation style should optimize for:

- readability;
- explicit configuration;
- Hugging Face compatibility;
- minimal hidden behavior;
- testability;
- isolated model-specific logic;
- shared reusable components;
- modality consistency;
- inference/training parity;
- easy addition of new architectures;
- straightforward code review;
- reproducible model loading.

---

# 7. Standard Repository Architecture

The operational directory ownership, dependency rules, current-module mapping,
and safe migration sequence are defined in `PROJECT_STRUCTURE.md`. The layout
below is the architectural baseline; `PROJECT_STRUCTURE.md` governs filesystem
migration details and public-import compatibility.

Recommended long-term package layout:

```text
mlx_one/
├── __init__.py
│
├── core/
│   ├── config.py
│   ├── registry.py
│   ├── model.py
│   ├── outputs.py
│   ├── cache.py
│   ├── generation.py
│   ├── sampling.py
│   ├── quantization.py
│   ├── loading.py
│   ├── serialization.py
│   └── utils.py
│
├── models/
│   │
│   ├── language/
│   │   ├── qwen2/
│   │   ├── qwen2_5/
│   │   ├── qwen3/
│   │   ├── llama/
│   │   ├── gemma/
│   │   ├── smollm/
│   │   ├── phi/
│   │   └── ...
│   │
│   ├── embeddings/
│   │   ├── minilm/
│   │   ├── bge/
│   │   ├── qwen_embedding/
│   │   └── ...
│   │
│   ├── vision_language/
│   │   ├── qwen_vl/
│   │   ├── smolvlm/
│   │   ├── paligemma/
│   │   └── ...
│   │
│   ├── vision/
│   │   ├── siglip/
│   │   ├── clip/
│   │   └── ...
│   │
│   ├── audio/
│   │   ├── whisper/
│   │   ├── parakeet/
│   │   ├── kokoro/
│   │   └── ...
│   │
│   └── shared/
│       ├── attention.py
│       ├── rope.py
│       ├── norms.py
│       ├── mlp.py
│       ├── moe.py
│       ├── position.py
│       └── activations.py
│
├── processors/
│   ├── text.py
│   ├── image.py
│   ├── audio.py
│   ├── multimodal.py
│   └── registry.py
│
├── generation/
│   ├── text.py
│   ├── multimodal.py
│   ├── streaming.py
│   ├── logits.py
│   └── stopping.py
│
├── training/
│   ├── trainer.py
│   ├── sft.py
│   ├── lora.py
│   ├── qlora.py
│   ├── losses.py
│   ├── datasets.py
│   ├── collators.py
│   ├── checkpoint.py
│   ├── scheduler.py
│   └── compatibility/
│       ├── fast_language_model.py
│       └── sft_trainer.py
│
├── embeddings/
│   ├── pooling.py
│   ├── similarity.py
│   ├── retrieval.py
│   ├── reranking.py
│   ├── losses.py
│   └── evaluation.py
│
├── audio/
│   ├── codecs.py
│   ├── resampling.py
│   ├── chunking.py
│   ├── streaming.py
│   └── generation.py
│
├── evaluation/
│   ├── text/
│   ├── retrieval/
│   ├── vlm/
│   ├── asr/
│   ├── tts/
│   └── common/
│
├── benchmark/
│   ├── inference.py
│   ├── training.py
│   ├── memory.py
│   └── profiler.py
│
├── hardware/
│   ├── detect.py
│   ├── profiles.py
│   └── planner.py
│
├── inspect/
│   ├── huggingface.py
│   ├── safetensors.py
│   └── metadata.py
│
├── registry/
│   ├── models.py
│   ├── capabilities.py
│   └── evidence.py
│
├── lifecycle/
│   ├── verify.py
│   ├── compare.py
│   ├── gates.py
│   ├── export.py
│   ├── release.py
│   └── lineage.py
│
└── cli/
    ├── main.py
    ├── inspect.py
    ├── run.py
    ├── train.py
    ├── evaluate.py
    ├── benchmark.py
    └── export.py
```

This structure separates:

```text
model implementation
from
task workflow
from
lifecycle infrastructure
```

That separation is critical.

---

# 8. Standard Model Family Layout

Every model family should follow a predictable structure.

Example:

```text
models/language/qwen3/
├── __init__.py
├── config.py
├── model.py
├── attention.py        # only if architecture-specific
├── convert.py
├── loader.py
├── generation.py       # only if special handling required
├── lora.py             # only if special mapping required
├── registry.py
└── tests/
    ├── test_config.py
    ├── test_shapes.py
    ├── test_loading.py
    ├── test_forward.py
    └── test_parity.py
```

For most models, architecture-specific files should be minimal.

Shared Transformer logic belongs in:

```text
models/shared/
```

Do not duplicate standard attention, RMSNorm, RoPE, or MLP implementations merely
because Hugging Face puts them inside individual modeling files.

---

# 9. Required Model Integration Contract

Every new model integration should implement the same conceptual contract.

## 9.1 Config

Each architecture must expose a typed configuration object.

Example:

```python
@dataclass
class Qwen3Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
```

The config parser should map directly from Hugging Face `config.json`.

Unknown fields may be retained for provenance, but required model fields must be explicit.

---

## 9.2 Model class

Every model class should expose a familiar MLX module surface:

```python
class Model(nn.Module):
    def __init__(self, config):
        ...

    def __call__(self, input_ids, cache=None):
        ...
```

Where useful:

```python
class ModelOutput:
    logits
    hidden_states
    cache
```

Avoid model-specific return conventions.

---

## 9.3 Weight mapping

Every architecture must explicitly declare how Hugging Face tensor names map to mlx-one
tensor names.

Example:

```python
WEIGHT_MAP = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    ...
}
```

For systematic mappings, use deterministic functions rather than huge static dictionaries.

The mapping layer should handle:

- tensor renames;
- qkv splitting;
- qkv fusion;
- transpose requirements;
- convolution layout changes;
- tied weights;
- MoE experts;
- quantized tensors;
- vision projectors;
- audio convolutions.

No silent unmatched tensor should be accepted during verified loading.

---

## 9.4 Sanitization hook

Each architecture may implement:

```python
def sanitize(weights, config):
    ...
```

Use it only for deterministic conversion behavior such as:

- splitting fused tensors;
- combining tensors;
- transposes;
- removing unused buffers;
- fixing known naming mismatches.

The sanitizer must be testable independently.

---

## 9.5 Processor contract

Models may declare:

```python
processor_class
tokenizer_class
image_processor_class
audio_processor_class
```

The runtime should resolve processors through registries.

---

## 9.6 Task capabilities

Each architecture should explicitly register capabilities.

Example:

```python
ModelCapabilities(
    text_generation=True,
    embeddings=False,
    vision=False,
    asr=False,
    tts=False,
    training=True,
    lora=True,
    quantization=True,
)
```

Never infer high-level functionality only from model names.

---

# 10. Registry-Driven Architecture

A central architecture registry should map Hugging Face metadata to mlx-one implementations.

Conceptually:

```python
register_model(
    model_type="qwen3",
    config=Qwen3Config,
    model=Qwen3Model,
    loader=Qwen3Loader,
    capabilities=...
)
```

Loading becomes:

```python
config = read_config(repo)

impl = registry.resolve(
    model_type=config["model_type"],
    architectures=config.get("architectures")
)

model = impl.from_pretrained(...)
```

This avoids giant conditional blocks such as:

```python
if "qwen" in model_name:
elif "llama" in model_name:
elif "gemma" in model_name:
```

Architecture detection should primarily use model metadata.

Model-name heuristics are fallback diagnostics only.

---

# 11. Native Language Model Foundation

Language models are the first and most important native foundation.

mlx-one should own:

- architecture implementations;
- model loading;
- safetensor handling;
- generation;
- KV cache;
- prompt processing;
- sampling;
- quantized loading;
- LoRA;
- QLoRA;
- SFT;
- checkpointing;
- adapter export;
- evaluation;
- benchmarking.

Initial families:

```text
Qwen2
Qwen2.5
Qwen3
Llama
Gemma
SmolLM
Phi
```

Priority foundation:

```text
Qwen2.5
Qwen3
Llama
Gemma
```

These families should establish the implementation patterns from which later models inherit.

---

# 12. Language Architecture Layers

Reusable components should include:

```text
Embedding
RMSNorm
LayerNorm
RoPE
Scaled Dot Product Attention
Grouped Query Attention
Multi Query Attention
Sliding Window Attention
MLP
SwiGLU
GeGLU
KV Cache
Causal Mask
Logits Projection
```

Conceptual implementation:

```python
class TransformerBlock(nn.Module):

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attention(self.norm1(x), mask, cache)
        return h + self.mlp(self.norm2(h))
```

Architecture-specific differences should be configuration or small subclasses whenever possible.

---

# 13. Generation Core

Generation must not be reimplemented independently by every model.

Shared API:

```python
generate(
    model,
    tokenizer,
    prompt,
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

Streaming:

```python
for token in stream_generate(...):
    print(token.text, end="")
```

The generation system should support:

- greedy decoding;
- temperature;
- top-k;
- top-p;
- min-p;
- repetition penalty;
- stop tokens;
- EOS handling;
- logits processors;
- streaming;
- batched generation;
- speculative decoding later;
- prompt caching later.

Models only need to expose compatible logits and cache behavior.

---

# 14. KV Cache Standard

KV cache is one of the most important shared runtime components.

Define a common interface:

```python
class KVCache:
    def update(self, keys, values):
        ...

    def state(self):
        ...

    def reset(self):
        ...
```

Support:

```text
standard cache
rotating/sliding cache
quantized cache
paged cache later
```

KV behavior should not be hidden inside individual model classes.

This enables:

- benchmarking;
- memory estimation;
- long-context optimization;
- cache quantization;
- reuse across models.

---

# 15. Quantization Strategy

Quantization should be a first-class shared service.

Initial goals:

```text
8-bit
6-bit
4-bit
```

Later:

```text
3-bit
2-bit experiments
mixed-bit
layer-wise search
activation-aware quantization
KV-cache quantization
```

The API should resemble:

```python
model = quantize(
    model,
    bits=4,
    group_size=64,
)
```

CLI:

```bash
mlx-one quantize MODEL \
  --bits 4 \
  --group-size 64 \
  --output model-4bit
```

Quantization results must preserve:

- original revision;
- quantization recipe;
- tensor checksums;
- calibration settings if used;
- compatibility evidence.

---

# 16. Native Training Compatibility

mlx-one should provide a native training implementation while exposing convenient APIs
inspired by familiar libraries.

Two levels should exist:

## Native authoritative API

```python
from mlx_one.training import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=config,
)

trainer.train()
```

## Compatibility-style API

```python
from mlx_one import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-1.7B",
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
)
```

The compatibility API maps onto native mlx-one components.

It must not become a separate implementation.

---

# 17. SFT Training Stack

Training should include:

```text
dataset parsing
chat templates
tokenization
packing
response-only masking
batching
gradient accumulation
optimizer
scheduler
LoRA
QLoRA
checkpointing
resume
evaluation
adapter export
merge
```

Standard data formats:

```json
{"text": "..."}
```

```json
{
  "prompt": "...",
  "completion": "..."
}
```

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Trainer config:

```yaml
model: Qwen/Qwen3-1.7B
method: lora

max_seq_length: 2048
batch_size: 1
gradient_accumulation_steps: 8

learning_rate: 2e-4
max_steps: 1000

lora:
  rank: 16
  alpha: 32
  dropout: 0.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

---

# 18. Training Roadmap

Initial:

```text
LoRA
QLoRA
SFT
checkpoint resume
adapter export
adapter reload
merge
```

Later:

```text
full fine-tuning
continued pretraining
DPO
ORPO
SimPO
KTO
GRPO
distillation
multi-adapter training
adapter composition
```

Training implementation should remain modality extensible.

Eventually:

```python
Trainer(task="text_sft")
Trainer(task="embedding_contrastive")
Trainer(task="vlm_sft")
Trainer(task="asr")
Trainer(task="tts")
```

---

# 19. Embeddings

Embeddings are the second major vertical after LM.

Initial model families:

```text
MiniLM
BGE
Qwen Embedding
```

Potential first targets:

```text
all-MiniLM-L6-v2
BGE-small
BGE-base where suitable
Qwen embedding family below 3B
```

mlx-one should own:

```text
encoder loading
tokenization
pooling
normalization
similarity
batch encoding
contrastive training
retrieval evaluation
reranking
```

User API:

```python
from mlx_one import EmbeddingModel

model = EmbeddingModel.from_pretrained("BAAI/bge-small-en-v1.5")

vectors = model.encode([
    "Apple Silicon inference",
    "MLX model optimization",
])
```

CLI:

```bash
mlx-one embed MODEL \
  --input texts.jsonl \
  --output embeddings.npy
```

---

# 20. Embedding Pooling

Shared pooling implementations:

```text
CLS
mean
weighted mean
last token
EOS token
```

Example:

```python
embedding = pool(
    hidden_states,
    attention_mask,
    strategy="mean",
)
```

Normalization:

```python
embedding = embedding / ||embedding||
```

Pooling behavior must be model metadata rather than hidden logic.

---

# 21. Embedding Training

mlx-one should eventually support:

```text
contrastive learning
MultipleNegativesRankingLoss
triplet loss
InfoNCE
cosine similarity loss
margin loss
knowledge distillation
```

Training data:

```json
{
  "query": "...",
  "positive": "...",
  "negative": "..."
}
```

Evaluation:

```text
Recall@K
Precision@K
MRR
NDCG
MAP
STS correlation
```

---

# 22. Rerankers

Rerankers belong beside embeddings.

Initial support:

```text
cross-encoder rerankers
BGE reranker family
small Qwen reranker models where applicable
```

API:

```python
reranker.score(
    query,
    documents,
)
```

Workflow:

```text
embedding retrieval
        ↓
top 100
        ↓
reranker
        ↓
top 10
```

---

# 23. Vision-Language Foundation

The VLM implementation should learn from mlx-vlm while adopting mlx-one conventions.

Core components:

```text
vision encoder
image processor
projector
token merger
language decoder
multimodal prompt processor
generation
```

Initial target families:

```text
Qwen-VL family below 3B
SmolVLM
PaliGemma variants below 3B
```

Potential priority:

```text
SmolVLM
Qwen2-VL 2B
small Qwen multimodal families
```

The exact supported list should be determined from current Hugging Face architectures and
models at implementation time.

---

# 24. VLM Architecture

Canonical design:

```text
Image
  ↓
Image Processor
  ↓
Vision Encoder
  ↓
Projection / Resampler
  ↓
Vision Tokens
  ↓
Multimodal Token Merge
  ↓
Language Model
  ↓
Generation
```

Common API:

```python
model.generate(
    prompt="Describe this image.",
    images=[image],
)
```

Multi-image:

```python
model.generate(
    prompt=prompt,
    images=[img1, img2],
)
```

The processor should return a standard internal batch:

```python
MultimodalBatch(
    input_ids=...,
    pixel_values=...,
    image_positions=...,
    attention_mask=...,
)
```

---

# 25. VLM Training

After inference support:

```text
projector tuning
LoRA on language layers
LoRA on vision projector
selective vision tuning
full VLM SFT where practical
```

Dataset layout:

```json
{
  "messages": [...],
  "images": ["image.jpg"]
}
```

Evaluation:

```text
VQA
OCR
document understanding
chart understanding
captioning
multimodal reasoning
```

---

# 26. OCR Evaluation

OCR should become a standard VLM qualification task.

Metrics may include:

```text
CER
WER
exact match
ANLS
field extraction accuracy
```

Use cases:

```text
documents
screenshots
receipts
forms
UI understanding
tables
```

---

# 27. Audio Foundation

Audio should follow the same model registration and runtime philosophy.

Initial tasks:

```text
ASR
TTS
```

Later:

```text
audio classification
speaker embeddings
audio-language models
speech translation
voice conversion
```

---

# 28. ASR

Initial model families:

```text
Whisper
Parakeet
```

Priority should favor compact models that fit the <3B initial scope.

API:

```python
from mlx_one import ASRModel

model = ASRModel.from_pretrained(...)

result = model.transcribe("audio.wav")
```

Return:

```python
TranscriptionResult(
    text=...,
    segments=...,
    timestamps=...,
    language=...,
)
```

CLI:

```bash
mlx-one transcribe MODEL audio.wav
```

---

# 29. Audio Preprocessing

Shared components:

```text
decode audio
resample
normalize
chunk
mel spectrogram
feature extraction
streaming windows
```

Do not duplicate audio loading and resampling across ASR families.

---

# 30. ASR Evaluation

Standard metrics:

```text
WER
CER
RTF
latency
peak memory
audio duration
language
sample rate
```

Streaming metrics later:

```text
first partial latency
finalization latency
chunk RTF
```

---

# 31. TTS

Initial compact TTS target:

```text
Kokoro
```

The implementation should learn from existing mlx-audio patterns.

Common API:

```python
from mlx_one import TTSModel

model = TTSModel.from_pretrained(...)

audio = model.generate(
    text="Hello from mlx-one.",
    voice="...",
)
```

CLI:

```bash
mlx-one speak MODEL \
  --text "Hello from mlx-one." \
  --output output.wav
```

---

# 32. TTS Runtime Components

Shared audio generation infrastructure:

```text
text normalization
phonemization
speaker conditioning
codec / vocoder
chunk generation
streaming
waveform assembly
resampling
```

TTS evaluation should distinguish objective proxies from subjective listening quality.

Measure:

```text
real-time factor
generated duration
round-trip ASR intelligibility
speaker similarity when appropriate
peak memory
```

Do not present automated metrics as MOS.

---

# 33. Audio Codec Layer

As multimodal audio models grow, mlx-one should eventually provide a common codec layer.

Potential abstraction:

```python
class AudioCodec:

    def encode(self, waveform):
        ...

    def decode(self, tokens):
        ...
```

This supports:

```text
neural audio codecs
speech tokenizers
audio-language models
future duplex speech models
```

---

# 34. Shared Processor System

The processor stack should be registry driven.

```python
processor = AutoProcessor.from_pretrained(model_id)
```

Processor responsibilities:

```text
text tokenization
chat templates
image preprocessing
audio preprocessing
multimodal packing
special token insertion
batch collation
```

Internal output should use standardized batch classes.

Examples:

```python
TextBatch
EmbeddingBatch
MultimodalBatch
AudioBatch
```

---

# 35. Hugging Face Integration

Hugging Face should remain the primary model artifact ecosystem.

mlx-one needs to support:

```text
config.json
*.safetensors
tokenizer.json
tokenizer.model
vocab files
special_tokens_map.json
tokenizer_config.json
processor_config.json
preprocessor_config.json
generation_config.json
```

Potential additional model-specific files must be preserved.

When a new model is released, mlx-one should inspect:

1. `config.json`
2. `architectures`
3. `model_type`
4. tensor names
5. tensor shapes
6. tokenizer
7. processor
8. special tokens
9. generation config
10. custom code requirement
11. license
12. upstream modeling implementation

This should be enough to determine whether:

```text
existing architecture already supports it
OR
a new model implementation is needed
```

---

# 36. New Model Integration Workflow

When a model is released on Hugging Face:

```text
1. Inspect metadata
2. Identify model_type
3. Check architecture registry
4. Inspect tensor names/shapes
5. Compare with known family
6. Inspect upstream modeling code if required
7. Implement missing operations
8. Add weight sanitizer/mapping
9. Add processor support
10. Load tiny/real checkpoint
11. Forward-pass parity
12. Generation parity
13. Quantized-load test
14. Training compatibility test
15. Benchmark
16. Register support evidence
```

The goal is eventually:

> Most new checkpoints within an already-supported architecture require zero
> model code changes.

Only architectural changes should require implementation work.

---

# 37. Parity Validation

Native ownership makes parity testing essential.

For every architecture, compare mlx-one against a trusted source implementation.

Tests should include:

```text
tensor structure
weight mapping
embedding outputs
hidden states
logits
greedy token output
processor outputs
quantized behavior
```

Where numerically appropriate:

```python
max_abs_error
mean_abs_error
cosine_similarity
top-k logit agreement
```

For quantized models, use task-aware tolerances.

Parity does not require bit-exact equivalence where kernels or quantization differ.

---

# 38. Testing Standard

Each architecture should have several test layers.

## Level 1 — Config tests

No model weights.

```text
parse config
validate dimensions
registry resolution
capability detection
```

## Level 2 — Synthetic shape tests

Tiny randomly initialized configurations.

```text
forward pass
cache
batch dimensions
sequence dimensions
loss
```

## Level 3 — Tiny checkpoint tests

Small stored fixture or miniature generated weights.

```text
load
save
reload
generation
```

## Level 4 — Real model integration

Actual Hugging Face model.

```text
download
load
generate
benchmark
```

## Level 5 — Parity qualification

Compare against source runtime.

## Level 6 — Hardware evidence

Run on documented Apple Silicon hardware.

---

# 39. Coding Standard for Model Implementations

Every model implementation should follow these rules.

### Rule 1 — Config drives architecture

Avoid magic numbers.

### Rule 2 — Reuse shared layers

Do not copy generic Transformer blocks unnecessarily.

### Rule 3 — Weight transformation is explicit

No silent reshape or rename.

### Rule 4 — No model-name hacks

Resolve via metadata.

### Rule 5 — Forward and generation are separate

The model computes logits.

Generation chooses tokens.

### Rule 6 — Training uses the same model

Do not create separate training-only architecture classes.

### Rule 7 — Quantization is compositional

Quantized modules should preserve the same logical API.

### Rule 8 — Every model has parity evidence

Support claims require evidence.

### Rule 9 — Unknown behavior fails visibly

Never silently ignore unsupported settings.

### Rule 10 — Model-specific code stays local

Shared abstractions remain shared.

---

# 40. Unified Public Python API

Long-term top-level API:

```python
from mlx_one import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoEmbeddingModel,
    AutoVLM,
    AutoASRModel,
    AutoTTSModel,
)
```

Examples:

## Language

```python
model = AutoModel.from_pretrained("Qwen/Qwen3-1.7B")
model.generate("Explain attention.")
```

## Embeddings

```python
model = AutoEmbeddingModel.from_pretrained(...)
model.encode(["hello", "world"])
```

## Vision-language

```python
model = AutoVLM.from_pretrained(...)
model.generate("What is shown?", images=[image])
```

## ASR

```python
model = AutoASRModel.from_pretrained(...)
model.transcribe(audio)
```

## TTS

```python
model = AutoTTSModel.from_pretrained(...)
model.generate("Hello")
```

---

# 41. Unified CLI

Target CLI:

```text
mlx-one doctor
mlx-one inspect
mlx-one plan

mlx-one run
mlx-one generate
mlx-one embed
mlx-one transcribe
mlx-one speak

mlx-one train
mlx-one evaluate
mlx-one benchmark
mlx-one compare

mlx-one quantize
mlx-one convert
mlx-one export
mlx-one verify
mlx-one ship

mlx-one registry
```

Model type may be inferred where safe.

Example:

```bash
mlx-one run Qwen/Qwen3-1.7B
```

For task-specific ambiguity:

```bash
mlx-one run MODEL --task embeddings
```

---

# 42. Lifecycle Infrastructure Already Built

The renewed strategy should **retain**, not discard, the substantial work already present.

According to the uploaded development summary and README, the repository already includes
or has implemented experimental/current-main versions of the following.

## Foundation and schemas

Completed:

- backend-free schema `1.0`;
- typed model identity records;
- hardware records;
- workload records;
- inspection records;
- memory estimates;
- plans;
- runs;
- results;
- compatibility evidence.

## Metadata-only inspection

Implemented:

```bash
mlx-one inspect MODEL
mlx-one inspect MODEL --revision REVISION
mlx-one inspect MODEL --offline
mlx-one inspect MODEL --json-output
```

Inspection avoids:

- MLX initialization;
- Transformers import;
- remote-code execution;
- loading tensor values.

This should remain a core mlx-one capability.

---

# 43. Hardware Planning Already Built

Reference profile:

```text
Apple M4 MacBook Air
32 GiB unified memory
24 GiB MLX process budget
8 GiB system/application reserve
```

Existing commands:

```bash
mlx-one hardware list
mlx-one hardware show m4-air-32gb
mlx-one hardware detect
```

Existing planner covers:

```text
text inference
full SFT
BF16 LoRA
4-bit QLoRA
scratch training
```

The estimator already accounts for:

```text
weights
runtime overhead
KV cache
activations
temporary buffers
gradients
optimizer state
adapters
```

Fit states:

```text
comfortable
possible
risky
does-not-fit
unknown
```

This planning system should expand to VLM, embeddings, and audio rather than be replaced.

---

# 44. Existing Calibration Evidence

The uploaded development snapshot records the following reference calibration matrix.

Models:

```text
Qwen2.5 0.5B
Qwen2.5 1.5B
Qwen2.5 3B
```

Coverage:

```text
BF16 inference
4-bit inference
BF16 LoRA
4-bit QLoRA

context:
512
1024
2048
4096
```

Total:

```text
48 cells
44 completed
```

All 2048-token workloads completed.

Measured 2048-token device performance reported in the README:

| Model | BF16 inference | 4-bit inference | BF16 LoRA | 4-bit QLoRA |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 0.5B | 88.7 tok/s · 1.49 GiB | 233.2 tok/s · 1.02 GiB | 894.3 tok/s · 7.54 GiB | 798.5 tok/s · 6.89 GiB |
| Qwen2.5 1.5B | 30.5 tok/s · 3.38 GiB | 93.0 tok/s · 1.51 GiB | 393.4 tok/s · 11.17 GiB | 333.9 tok/s · 9.11 GiB |
| Qwen2.5 3B | 15.3 tok/s · 6.20 GiB | 50.3 tok/s · 2.26 GiB | 176.5 tok/s · 15.62 GiB | 157.8 tok/s · 11.49 GiB |

These measurements are device-performance evidence, not model-quality scores.

---

# 45. Existing Training Work

The uploaded README reports native/owned text training work already on current `main`,
including:

```text
dataset validation
training schemas
LoRA orchestration
QLoRA preflight
checkpoint lineage
adapter continuation
adapter export
FastLanguageModel facade
SFTTrainer-style facade
isolated inference benchmarking
```

Hardware-qualified LoRA runs reported:

```text
Qwen2.5-Coder 1.5B Instruct
SmolLM2 1.7B Instruct
```

Both passed adapter update and reload checks.

The tiny held-out evaluation did not establish quality improvement, therefore they remain
hardware-verified rather than quality-verified.

This distinction should remain part of the renewed project.

---

# 46. Evidence-Based Compatibility

Even after mlx-one owns model implementations, support should remain evidence-scoped.

Status levels can continue to include:

```text
candidate
upstream-documented
integration-tested
hardware-verified
quality-verified
unsupported
deprecated
```

A support claim applies to:

```text
model revision
+
mlx-one version
+
operation
+
precision
+
workload
+
settings
+
hardware
```

Native ownership does not remove the need for evidence.

It makes the evidence more meaningful because mlx-one controls the implementation.

---

# 47. Reference Implementation Policy

MIT-licensed repositories can be used as engineering references.

When adapting implementation ideas:

1. preserve license requirements;
2. retain required copyright/license notices;
3. document substantial adapted components where appropriate;
4. avoid blind copy-paste;
5. refactor into mlx-one conventions;
6. write independent parity tests;
7. keep API and layout consistent across mlx-one.

Reference projects should help answer:

```text
How does the architecture work?
How are weights mapped?
How is preprocessing implemented?
What tensor transforms are required?
How is generation wired?
What implementation details are MLX-specific?
```

mlx-one should then implement those concepts in its own common architecture.

---

# 48. First-Wave Native LM Implementation Plan

Order:

## LM-1 — Shared Transformer primitives

Implement and test:

```text
RMSNorm
RoPE
attention
GQA
causal masking
MLP
SwiGLU
KV cache
quantized linear abstraction
```

## LM-2 — Qwen2.5

Port/implement natively.

Acceptance:

```text
load HF BF16
forward parity
greedy generation parity
4-bit load
LoRA
SFT
checkpoint
benchmark
```

## LM-3 — Qwen3

Reuse Qwen foundation where possible.

Implement deltas only.

## LM-4 — Llama

Add Llama-family config and architectural differences.

## LM-5 — Gemma

Add Gemma-specific norms, embeddings, attention differences as required.

## LM-6 — SmolLM / Phi

Use to validate architecture extensibility.

---

# 49. First-Wave Embedding Plan

## EMB-1

Implement common encoder-only Transformer support.

## EMB-2

MiniLM.

## EMB-3

BGE.

## EMB-4

Qwen embedding model.

## EMB-5

Pooling registry.

## EMB-6

Batch encoding.

## EMB-7

Retrieval evaluator.

## EMB-8

Contrastive trainer.

## EMB-9

Reranker support.

Exit condition:

```text
encode
batch encode
similarity
retrieval benchmark
fine-tune
export
```

all available through mlx-one.

---

# 50. First-Wave VLM Plan

Study mlx-vlm implementation patterns.

Implement:

```text
AutoImageProcessor
vision encoder abstraction
vision-language projector
multimodal processor
image-token merge
VLM generation
```

Recommended implementation order:

```text
SmolVLM
Qwen-VL / Qwen2-VL small family
PaliGemma small family
```

Exact checkpoint choices should remain based on current availability and <3B total-model
scope during implementation.

Acceptance:

```text
single image
multi image
processor parity
generation parity
OCR benchmark
LoRA SFT
```

---

# 51. First-Wave Audio Plan

Study mlx-audio architecture patterns.

## ASR

Start with:

```text
Whisper
Parakeet
```

Implement:

```text
audio loader
resampler
feature extraction
encoder/decoder
beam/greedy decode where needed
timestamps
chunking
streaming
```

## TTS

Start with:

```text
Kokoro
```

Implement:

```text
text normalization
phoneme/token processing
speaker conditioning
acoustic generation
vocoder/codec
waveform export
streaming
```

---

# 52. Phase Roadmap

The earlier roadmap should be reinterpreted around native model ownership.

## Phase 0 — Foundation

Already substantially complete.

```text
schemas
inspect
hardware detect
planner
calibration
registry foundations
run records
evaluation foundations
```

## Phase 1 — Native LM Core

```text
shared Transformer layers
Qwen2.5
Qwen3
Llama
Gemma
native loading
generation
cache
quantization
```

## Phase 2 — Training Compatibility

```text
FastLanguageModel
SFTTrainer
LoRA
QLoRA
dataset processing
checkpointing
resume
adapter export
merge
```

## Phase 3 — Embeddings

```text
MiniLM
BGE
Qwen embedding
pooling
similarity
contrastive training
rerankers
retrieval evaluation
```

## Phase 4 — Vision Language

```text
vision encoders
projectors
processors
SmolVLM
small Qwen-VL
VLM generation
VLM LoRA/SFT
OCR evaluation
```

## Phase 5 — Audio

```text
Whisper / Parakeet
Kokoro
audio preprocessing
chunking
streaming
ASR evaluation
TTS evaluation
audio training
```

## Phase 6 — Advanced Lifecycle

```text
quality benchmark suite
profiling
quantization search
preference training
merging
release bundles
distributed workflows
cross-backend conversion
```

---

# 53. Milestone Definition

A model family is **not supported** because it loads once.

Minimum acceptance:

```text
[ ] config parsed
[ ] registry entry
[ ] BF16 weights load
[ ] forward pass
[ ] source parity check
[ ] deterministic generation or task output
[ ] save/reload
[ ] benchmark
[ ] memory measurement
[ ] integration tests
[ ] documented known limitations
```

For trainable models:

```text
[ ] LoRA injection
[ ] gradient update verified
[ ] checkpoint save
[ ] checkpoint resume
[ ] adapter reload
[ ] evaluation after reload
```

For quantized support:

```text
[ ] quantized weights load
[ ] generation/task works
[ ] quality tolerance documented
[ ] memory benchmark
```

---

# 54. Model Support Matrix

Documentation should eventually generate a matrix such as:

| Family | Params | Inference | 4-bit | LoRA | QLoRA | SFT | Eval | Hardware Verified |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 | <3B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen3 | <3B | ... | ... | ... | ... | ... | ... | ... |
| Llama | <3B | ... | ... | ... | ... | ... | ... | ... |
| Gemma | <3B | ... | ... | ... | ... | ... | ... | ... |

Equivalent matrices should exist for:

```text
Embeddings
VLM
ASR
TTS
```

---

# 55. Architecture Support vs Checkpoint Support

Important distinction:

```text
Architecture support
≠
checkpoint verification
```

For example:

```text
Qwen3 architecture supported
```

means mlx-one can theoretically load compatible Qwen3 checkpoints.

But:

```text
Qwen/Qwen3-1.7B revision XYZ
hardware-verified
```

requires an actual run.

Documentation should distinguish:

```text
architecture-supported
checkpoint-tested
hardware-verified
quality-verified
```

---

# 56. Model Discovery

mlx-one can eventually build a Hugging Face inventory.

Suggested bands:

```text
0–100M
100M–500M
500M–1B
1B–3B
3B–6B
6B–10B
10B+
unknown
```

The first model-support campaign should prioritize `<3B`.

The discovery system can classify:

```text
architecture
modality
parameter count
license
processor
remote-code requirement
existing mlx-one support
verification state
```

This becomes both:

- a development backlog;
- a public compatibility catalog.

---

# 57. Hardware Policy

The reference machine remains:

```text
M4 MacBook Air
32 GiB unified memory
```

But support should remain model-size agnostic.

Never state:

> mlx-one only supports models below 3B.

Instead state:

> The first implementation and maintainer qualification wave targets models
> below 3B. Larger models are valid when the architecture is supported and the
> workload fits available hardware.

---

# 58. Memory Planning Expansion

Existing text memory estimation should extend to:

## Embeddings

```text
weights
sequence length
batch size
hidden states
pooling
```

## VLM

```text
language weights
vision weights
image resolution
patch count
vision tokens
projector
KV cache
```

## ASR

```text
audio duration
feature frames
encoder activations
decoder cache
```

## TTS

```text
text length
acoustic tokens
codec/vocoder
waveform buffers
```

This makes planning one of mlx-one's strongest differentiators.

---

# 59. Benchmarking Standard

Every modality should expose common measurements:

```text
load time
wall time
peak memory
model size
cold run
warm run
```

Task-specific metrics:

## LM

```text
prompt tok/s
decode tok/s
TTFT
```

## Embeddings

```text
texts/s
tokens/s
batch latency
```

## VLM

```text
vision encoder latency
prefill latency
decode tok/s
```

## ASR

```text
RTF
audio seconds processed/s
```

## TTS

```text
RTF
time-to-first-audio
```

---

# 60. Evaluation Strategy

Evaluation should be native at the orchestration level even when metrics use pinned
external datasets.

## LM

```text
instruction following
reasoning
knowledge
coding
exact-match fixtures
```

## Embeddings

```text
retrieval
STS
clustering
classification where useful
```

## VLM

```text
VQA
OCR
document understanding
visual reasoning
```

## ASR

```text
WER
CER
```

## TTS

```text
RTF
ASR round trip
speaker similarity
manual listening
```

---

# 61. Release Artifacts

Every released model artifact should eventually include:

```text
config
weights
tokenizer / processor
mlx-one model metadata
quantization metadata
training manifest
evaluation results
benchmark results
hardware profile
lineage
checksums
license metadata
model card
```

Release bundle:

```text
release/
├── model/
├── processor/
├── manifest.json
├── lineage.json
├── benchmark.json
├── evaluation.json
├── checksums.json
└── MODEL_CARD.md
```

---

# 62. Cross-Backend Training

The project can still preserve:

> Train anywhere. Prove it on Apple Silicon.

Native MLX training should be first-class, but CUDA workflows remain valuable.

Long-term flow:

```text
CUDA / TPU / other training
        ↓
canonical Hugging Face artifact
        ↓
mlx-one inspect
        ↓
mlx-one import / convert
        ↓
native MLX verification
        ↓
Apple Silicon benchmark
```

The interchange boundary remains a complete Hugging Face artifact.

---

# 63. Distributed Training

Deferred until the native single-device foundations are stable.

Potential later directions:

```text
MLX distributed
multi-GPU source training
FSDP source workflows
DeepSpeed source workflows
TPU source workflows
```

mlx-one should not build cluster provisioning.

It should own:

```text
training specs
artifact compatibility
lineage
conversion
verification
```

---

# 64. Advanced Optimization

After modality foundations:

```text
mixed-bit quantization
quantization search
KV cache quantization
speculative decoding
continuous batching
prompt caching
prefix caching
compile optimization
kernel profiling
memory scheduling
streaming generation
```

Optimization should be benchmark-driven.

---

# 65. Advanced Training

Later:

```text
DPO
ORPO
SimPO
KTO
GRPO
distillation
preference datasets
reward modeling
continued pretraining
```

These should build on the same trainer primitives.

---

# 66. Model Merging

Retain the previously planned lifecycle feature.

Initial:

```text
linear
SLERP
task arithmetic
```

Validation must include:

```text
architecture match
tensor shape match
tokenizer match
config compatibility
reload
evaluation
```

Quantized weights should not be naively averaged.

---

# 67. Compatibility Registry

The registry remains strategically important.

Each evidence record should include:

```text
model id
model revision
architecture
mlx-one version
hardware
precision
task
context
batch
operation
status
metrics
artifact checksums
```

Community submissions should use the same schema.

---

# 68. Community Contribution Model

Native ownership changes contribution opportunities.

Contributors can add:

```text
new architecture
new model family
new processor
new weight mapping
new parity fixture
new hardware evidence
new benchmark
new training task
new quantization method
new evaluation profile
```

Ideal contributor issue:

> Add architecture `X`.

Checklist:

```text
[ ] config
[ ] model
[ ] registry
[ ] weights
[ ] parity
[ ] tests
[ ] docs
```

This is much more scalable than unrelated per-package contributions.

---

# 69. Architectural Governance

New code should pass several questions.

1. Is this model-specific or reusable?
2. Can another modality reuse the abstraction?
3. Does it match the mlx-one standard API?
4. Is loading deterministic?
5. Is unsupported behavior explicit?
6. Can it be tested without downloading a real model?
7. Is there a parity test?
8. Does the implementation preserve lifecycle evidence?

Avoid introducing a new abstraction merely because an upstream repository uses one.

mlx-one should have its own architectural consistency.

---

# 70. Dependency Philosophy

Core dependencies should stay small.

Likely core:

```text
mlx
huggingface_hub
safetensors
numpy
configuration / CLI dependencies
```

Tokenizer and media dependencies may be modular where necessary.

Avoid making these permanent core dependencies:

```text
mlx-lm
mlx-vlm
mlx-audio
mlx-embeddings
transformers
```

They may remain:

```text
development references
parity-test dependencies
optional compatibility dependencies
```

during migration.

The final supported runtime should increasingly depend on:

```text
mlx + mlx-one
```

rather than an ecosystem of modality packages.

---

# 71. Migration Strategy from Current Repository

Do not rewrite everything at once.

## Stage A

Keep current orchestration functionality working.

Add:

```text
mlx_one/models/
mlx_one/core/
```

## Stage B

Implement native Qwen2.5 behind a feature flag.

Example:

```bash
mlx-one generate MODEL --runtime native
```

Compare with existing backend.

## Stage C

Make native runtime preferred for supported architectures.

Fallback to external runtime temporarily.

## Stage D

Add Qwen3, Llama, Gemma.

## Stage E

Remove external LM runtime requirement for supported families.

## Stage F

Repeat pattern for embeddings.

## Stage G

Repeat for VLM.

## Stage H

Repeat for audio.

This protects existing users while moving toward full ownership.

---

# 72. Runtime Selection During Migration

Temporary runtime selector:

```text
native
external
auto
```

Example:

```bash
mlx-one generate MODEL --runtime native
```

`auto`:

```text
native implementation available?
    yes → native
    no  → external fallback if installed
```

Eventually:

```text
native is standard
external adapters become optional
```

---

# 73. Definition of Native

A model is considered `native` only when:

```text
model architecture lives in mlx_one
weight loading lives in mlx_one
forward execution uses MLX directly
task runtime uses mlx_one core
no external model runtime package is required
```

Using:

```text
mlx
huggingface_hub
safetensors
tokenizer libraries
```

does not make it non-native.

---

# 74. Version Roadmap — Renewed

A possible renewed roadmap:

## v0.2 — Native LM Foundation

```text
Qwen2.5 native
shared Transformer core
generation
KV cache
quantization integration
parity suite
```

## v0.3 — Native LM Expansion + Training

```text
Qwen3
Llama
Gemma
FastLanguageModel
SFTTrainer
LoRA
QLoRA
checkpointing
```

## v0.4 — Embeddings

```text
MiniLM
BGE
Qwen embeddings
pooling
retrieval
contrastive training
rerankers
```

## v0.5 — Vision Language

```text
vision encoders
SmolVLM
Qwen-VL small
multimodal processor
OCR evaluation
VLM SFT
```

## v0.6 — Audio

```text
Whisper / Parakeet
Kokoro
streaming
chunking
ASR/TTS evaluation
```

## v0.7 — Optimization

```text
quantization search
cache optimization
profiling
advanced benchmarking
```

## v0.8 — Advanced Training

```text
preference training
distillation
continued pretraining
merge
```

## v0.9 — Registry and Community Qualification

```text
community hardware evidence
architecture catalog
support matrix
generated docs
```

## v1.0 — Unified Native MLX Stack

Exit condition:

A fresh user can:

```text
install mlx-one
load supported LM/VLM/embedding/audio model
run it
fine-tune where supported
evaluate
benchmark
quantize
verify
export
ship
```

without installing separate MLX model-runtime packages for the supported native
architectures.

---

# 75. v1.0 Definition

`mlx-one v1.0` should mean:

> A coherent native MLX model ecosystem exists, not merely a CLI that delegates
> to other packages.

Minimum v1.0 scope should include:

### Language

```text
Qwen2.5
Qwen3
Llama
Gemma
```

### Embeddings

```text
MiniLM
BGE
Qwen embedding
```

### Vision-language

```text
at least two representative families
```

### Audio

```text
at least one ASR family
at least one TTS family
```

### Training

```text
LM LoRA/QLoRA/SFT
embedding contrastive training
at least one VLM tuning path
basic audio training/tuning where practical
```

### Lifecycle

```text
inspect
plan
evaluate
benchmark
verify
quantize
export
registry
ship
```

---

# 76. Immediate Development Order

The recommended order is now:

```text
1. Freeze standard implementation conventions
2. Build native Transformer shared core
3. Implement native Qwen2.5
4. Add parity tests against trusted implementation
5. Move current SFT/LoRA stack onto native Qwen
6. Implement Qwen3
7. Implement Llama
8. Implement Gemma
9. Implement native embedding base
10. MiniLM
11. BGE
12. Qwen embeddings
13. Build native VLM primitives
14. SmolVLM / small Qwen-VL
15. Build audio primitives
16. Whisper / Parakeet
17. Kokoro
18. Expand advanced lifecycle
```

Do not start by implementing dozens of models.

First establish the architecture standard with four strong LM families.

Then model additions should become progressively cheaper.

---

# 77. First Coding Milestone

The best first native coding milestone is:

> **Native Qwen2.5 inference from Hugging Face safetensors without mlx-lm.**

Acceptance test:

```bash
mlx-one generate \
  Qwen/Qwen2.5-0.5B-Instruct \
  --runtime native \
  --prompt "What is MLX?"
```

Requirements:

```text
[ ] read config
[ ] load tokenizer
[ ] resolve qwen2 model class
[ ] load safetensors
[ ] sanitize weights
[ ] initialize MLX modules
[ ] create KV cache
[ ] prefill
[ ] decode
[ ] sample
[ ] stream output
```

Then test:

```text
0.5B
1.5B
3B
```

Then:

```text
BF16
4-bit
```

Then:

```text
LoRA
SFT
```

This milestone proves that the renewed architecture is real.

---

# 78. Second Coding Milestone

> **Native Qwen3 with architecture reuse.**

The purpose is not only Qwen3 support.

It validates whether the common abstractions created for Qwen2.5 were actually reusable.

If Qwen3 requires rewriting the whole stack, the design is too model-specific.

---

# 79. Third Coding Milestone

> **Native Llama and Gemma.**

This tests whether the architecture can support genuinely different families.

After these four:

```text
Qwen2.5
Qwen3
Llama
Gemma
```

the LM integration standard should be considered stable enough for rapid expansion.

---

# 80. Engineering Success Metrics

Track:

```text
time to add a checkpoint within existing architecture
time to add a new architecture
lines of architecture-specific code
shared-code reuse
parity error
test coverage
peak memory
throughput
number of runtime dependencies
```

A key success metric should be:

> **How quickly can a newly released <3B Hugging Face model be supported?**

Ideal long-term targets:

```text
existing architecture:
hours or no code change

minor architecture variant:
1 day

new Transformer family:
1–3 days
```

after the platform matures.

---

# 81. What mlx-one Should Not Become

Do not turn mlx-one into:

```text
a wrapper around Transformers
a collection of copied model files
a benchmark-only project
a giant CLI with no coherent Python API
an MLX-LM fork with extra folders
a UI-first application
a model downloader
a Hugging Face clone
```

The differentiator is:

```text
unified native implementation
+
consistent APIs
+
training
+
multimodality
+
hardware awareness
+
reproducible lifecycle
```

---

# 82. Technical Principles

1. **MLX is the execution foundation.**
2. **mlx-one owns supported model implementations.**
3. **Hugging Face is the primary artifact ecosystem.**
4. **Model architecture, not model name, drives loading.**
5. **One implementation style spans every modality.**
6. **Inference and training use the same model classes.**
7. **Generation is separate from forward execution.**
8. **Processors are first-class.**
9. **Quantization is shared infrastructure.**
10. **Parity is required before support claims.**
11. **Compatibility is evidence-scoped.**
12. **<3B is the first implementation campaign, not a permanent limit.**
13. **External MLX projects are references, not long-term mandatory runtimes.**
14. **Unknown means unknown.**
15. **Every artifact keeps lineage.**

---

# 83. Final Product Architecture

```text
                        mlx-one
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   Developer APIs                         CLI / Workflows
        │                                     │
        └──────────────────┬──────────────────┘
                           │
              Unified Model / Task Layer
                           │
     ┌──────────┬──────────┼──────────┬───────────┐
     │          │          │          │           │
    LM      Embeddings    VLM        ASR         TTS
     │          │          │          │           │
     └──────────┴──────────┼──────────┴───────────┘
                           │
                 Native Model Registry
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
 Shared Layers       Shared Processors    Shared Runtime
     │                     │                     │
 attention             tokenizer              generation
 RoPE                  image                  KV cache
 norms                  audio                  sampling
 MLP                    multimodal             quantization
     │                     │                     │
     └─────────────────────┼─────────────────────┘
                           │
                          MLX
                           │
                    Apple Silicon
```

Lifecycle surrounds the entire stack:

```text
Inspect
Plan
Train
Evaluate
Benchmark
Verify
Compare
Gate
Export
Ship
Registry
```

---

# 84. Final Strategic Statement

The renewed plan for `mlx-one` is:

> Build **one native MLX package** that supports language models, embeddings,
> vision-language models, ASR, and TTS through a shared implementation style.
>
> Begin by implementing and qualifying popular models below 3B parameters.
>
> Own architecture implementations, loading, generation, quantization, training,
> evaluation, and lifecycle behavior rather than making separate MLX model
> packages mandatory runtime dependencies.
>
> Use projects such as MLX-LM, mlx-vlm, and mlx-audio as technical references and
> parity sources. Where licenses permit, adapt useful implementation patterns
> into a consistent mlx-one architecture.
>
> Preserve the project's existing strengths in model inspection, hardware-aware
> planning, reproducible calibration, evidence-scoped compatibility, evaluation,
> checkpoint lineage, and release qualification.
>
> Establish Qwen2.5, Qwen3, Llama, and Gemma as the native language foundation;
> MiniLM, BGE, and Qwen models as the embedding foundation; small Qwen-VL and
> SmolVLM-style architectures as the vision-language foundation; Whisper or
> Parakeet for ASR; and Kokoro as the first compact TTS family.
>
> Once those foundations are stable, expand toward preference training,
> quantization search, rerankers, model merging, distributed workflows,
> multimodal models, and increasingly broad architecture coverage.

The defining product idea is no longer merely:

```text
one CLI over many packages
```

It is:

```text
one MLX model stack
one implementation standard
one lifecycle
one package
```

That is the architecture `mlx-one` should grow into.

---

# 85. Immediate Checklist

## Architecture

- [ ] Freeze `mlx_one/models` standard.
- [ ] Freeze configuration contract.
- [ ] Freeze registry contract.
- [ ] Freeze weight mapping/sanitization contract.
- [ ] Freeze processor contract.
- [ ] Freeze model output types.

## Native LM

- [ ] Shared RMSNorm.
- [ ] Shared RoPE.
- [ ] Shared attention.
- [ ] Shared GQA.
- [ ] Shared MLP/SwiGLU.
- [ ] Shared KV cache.
- [ ] Qwen2.5 native model.
- [ ] Native safetensors loader.
- [ ] Native generation.
- [ ] Qwen2.5 parity tests.
- [ ] Native 4-bit loading.
- [ ] Connect LoRA.
- [ ] Connect SFTTrainer.
- [ ] Qwen3.
- [ ] Llama.
- [ ] Gemma.

## Embeddings

- [ ] Encoder-only base.
- [ ] Pooling registry.
- [ ] MiniLM.
- [ ] BGE.
- [ ] Qwen embedding.
- [ ] Batch encode.
- [ ] Retrieval evaluation.
- [ ] Contrastive training.
- [ ] Reranker API.

## VLM

- [ ] Vision encoder interface.
- [ ] Image processor interface.
- [ ] Projector interface.
- [ ] Multimodal batch.
- [ ] SmolVLM.
- [ ] Small Qwen-VL.
- [ ] VLM generation.
- [ ] OCR evaluation.
- [ ] VLM LoRA/SFT.

## Audio

- [ ] Audio decode/resample.
- [ ] Chunking.
- [ ] Streaming abstractions.
- [ ] Whisper or Parakeet.
- [ ] WER/CER evaluation.
- [ ] Kokoro.
- [ ] TTS waveform export.
- [ ] TTS RTF evaluation.
- [ ] Audio training path.

## Lifecycle

- [x] Inspect.
- [x] Hardware profiles.
- [x] Text memory planning.
- [x] Calibration harness.
- [x] Compatibility evidence foundations.
- [x] Initial LoRA/SFT workflow.
- [ ] Native runtime evidence.
- [ ] Embedding planner.
- [ ] VLM planner.
- [ ] Audio planner.
- [ ] Full quality suites.
- [ ] Cross-modal benchmark schema.
- [ ] Quantization search.
- [ ] Preference training.
- [ ] Merge.
- [ ] Release bundle.
- [ ] Community registry.

---

# 86. North-Star Test

The project is succeeding when a user can do this:

```bash
pip install mlx-one
```

then:

```python
from mlx_one import AutoModel

model = AutoModel.from_pretrained("some-new-small-model")
print(model.generate("Hello"))
```

and, if the architecture is already supported, mlx-one understands the checkpoint without
requiring the user to discover and install a separate LM, VLM, embedding, or audio
package.

The same package should then let the user:

```text
fine-tune it
quantize it
evaluate it
benchmark it
verify it
ship it
```

on Apple Silicon through the same lifecycle conventions.

That is the long-term standard for `mlx-one`.
