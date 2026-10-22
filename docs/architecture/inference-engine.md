# mlx-one Native Inference Engine

## Status

mlx-one already has a **native MLX inference stack**. It does not delegate serving or generation to `mlx-lm`, `llama.cpp`, Ollama, vLLM, or SGLang.

The goal of this roadmap is therefore **not to create an inference engine from scratch**. The goal is to evolve the existing mlx-one runtime into a **vLLM-class, Apple-Silicon-native inference and serving engine** designed around MLX and unified memory.

> **Long-term definition**
>
> **mlx-one Engine = a native, unified-memory-aware inference and serving runtime for MLX, analogous in responsibility to vLLM, but designed from the ground up for Apple Silicon and multimodal models.**

---

# 1. Current State

mlx-one already has the following native inference components.

| Component | Current responsibility | Status |
|---|---|---|
| `GenerationEngine` | Adapts chat/API requests, reasoning, prompt caching, metrics, speculative options | ✅ Implemented |
| Native generation core | Prefill, cached decoding, sampling, streaming, cancellation, stop handling, context enforcement | ✅ Implemented |
| `GenerationScheduler` | Bounded FIFO queue and cooperative inference slots on one MLX thread | ✅ Implemented |
| `ModelManager` | Single loaded model lifecycle and metadata | ✅ Implemented |
| FastAPI application | OpenAI-compatible HTTP serving | ✅ Implemented |
| Native MLX model execution | Executes directly through mlx-one model implementations + MLX | ✅ Implemented |
| External inference backend | mlx-lm / llama.cpp / Ollama / vLLM / SGLang | ❌ Not required |

Current source locations:

```text
src/mlx_one/
├── server/
│   ├── generation_engine.py
│   ├── scheduler.py
│   ├── model_manager.py
│   └── app.py
│
└── text/
    └── generation.py
```

Key implementation references:

```text
GenerationEngine
src/mlx_one/server/generation_engine.py

Native generation core
src/mlx_one/text/generation.py

GenerationScheduler
src/mlx_one/server/scheduler.py

ModelManager
src/mlx_one/server/model_manager.py

FastAPI application
src/mlx_one/server/app.py
```

---

# 2. Current Runtime Flow

Today, inference is conceptually:

```text
                         Client
                           │
                  OpenAI-compatible API
                           │
                           ▼
                  FastAPI Application
                           │
                           ▼
                    GenerationEngine
                           │
              ┌────────────┼────────────┐
              │            │            │
        Prompt logic    Metrics    Speculative options
              │
              ▼
                GenerationScheduler
                           │
                    bounded FIFO
                           │
                  cooperative slot
                           │
                           ▼
                 Native Generation Core
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
       Prefill          Decode          Streaming
          │                │                 │
          ├── KV cache     ├── sampling      │
          ├── prompt cache ├── stopping      │
          └── context      ├── cancellation  │
                           └── context guard  │
                           │
                           ▼
                    mlx-one Model
                           │
                           ▼
                          MLX
                           │
                           ▼
                         Metal
                           │
                           ▼
                     Apple Silicon
```

This is already a real inference engine.

The remaining work is primarily about **throughput, memory efficiency, scheduling sophistication, long-context execution, multimodal execution, and Apple-specific optimization**.

---

# 3. Architectural Principle

mlx-one should keep a strict separation between four layers.

```text
Application / API
      │
      ▼
mlx-one Runtime Engine
      │
      ▼
mlx-one Model Implementations
      │
      ▼
MLX
      │
      ▼
Metal / Apple Silicon
```

## Responsibility boundary

### mlx-one Runtime Engine

Owns:

- request lifecycle
- scheduling
- prompt preparation
- prefill orchestration
- decoding
- KV-cache policy
- prompt/prefix caching
- sampling
- logits processing
- stopping
- streaming
- cancellation
- batching
- memory budgeting
- speculative decoding orchestration
- metrics
- serving semantics

### mlx-one Models

Own:

- model architecture
- attention
- MLP
- normalization
- RoPE
- MoE routing
- multimodal projectors
- encoder/decoder structure
- weight mapping
- model-specific cache behavior

### MLX

Owns:

- tensors
- lazy graph execution
- matrix operations
- fast attention primitives
- quantized operations
- memory allocation
- device execution
- Metal dispatch

### Metal / Apple Silicon

Owns:

- GPU execution
- memory bandwidth
- hardware kernels
- unified-memory hardware behavior

---

# 4. Target Engine Architecture

The long-term mlx-one inference architecture should become:

```text
                         Applications
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
         CLI              Python API        OpenAI API
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                     ┌─────────────────┐
                     │ mlx-one Engine  │
                     └────────┬────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
      Request Layer      Scheduler         Memory Planner
           │                  │                  │
           └──────────────┬───┴──────────────────┘
                          ▼
                       KV Manager
                          │
                 ┌────────┴────────┐
                 │                 │
             Block KV         Prefix Cache
                 │                 │
                 └────────┬────────┘
                          ▼
                    Model Executor
                          │
              ┌───────────┼───────────┐
              │           │           │
           Prefill      Decode    Speculative
              │           │           │
              └───────────┼───────────┘
                          ▼
                  Native mlx-one Models
                          │
                          ▼
                         MLX
                          │
                          ▼
                   Metal / Unified
                       Memory
```

---

# 5. Core Engine Components

## 5.1 Request Manager

Normalize all entry points into one request type.

```python
GenerationRequest(
    prompt="Explain attention",
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    seed=42,
    stop=["</answer>"],
)
```

Sources may include:

```text
CLI
Python API
FastAPI
OpenAI-compatible API
Internal multimodal pipelines
```

Target responsibilities:

- request validation
- request ID
- priority
- arrival timestamp
- context size
- sampling config
- cancellation token
- modality metadata
- speculative configuration
- cache hints

---

# 5.2 ModelRunner

The scheduler should never depend directly on Qwen, Llama, Gemma, Phi, etc.

It should call a common execution interface.

```python
runner.prefill(batch)
runner.decode(batch)
```

Conceptually:

```text
Scheduler
   │
   ▼
ModelRunner
   │
   ├── prepare inputs
   ├── execute model.forward()
   ├── control MLX evaluation
   ├── capture logits
   ├── update cache
   └── return execution result
           │
           ▼
        Model
           │
           ▼
          MLX
```

Future runner types:

```text
CausalLMRunner
VLMRunner
ASRRunner
EmbeddingRunner
RerankerRunner
AudioGenerationRunner
```

---

# 5.3 Prefill Engine

Current mlx-one already performs prefill.

The next evolution should optimize it for:

- very long prompts
- batched prompts
- prefix reuse
- prompt cache reuse
- chunked execution
- multimodal embeddings
- memory pressure

Long-context example:

```text
128K prompt
    │
    ▼
Chunk 1: 4096
    │
Decode waiting requests
    │
Chunk 2: 4096
    │
Decode waiting requests
    │
Chunk 3: 4096
    │
...
```

This prevents a single large prompt from blocking all active decode traffic.

Primary metric:

> **TTFT — Time To First Token**

---

# 5.4 Decode Engine

The existing decode path should evolve toward batched and continuously scheduled decoding.

```text
new token
   +
KV state
   │
   ▼
model forward
   │
   ▼
logits
   │
   ▼
logits processors
   │
   ▼
sampler
   │
   ▼
next token
```

Primary metrics:

- decode tokens/sec
- inter-token latency
- batch efficiency
- memory bandwidth utilization

---

# 5.5 KV Cache Manager

The existing cached decode mechanism should evolve into a dedicated cache subsystem.

## Stage 1

```text
Request
  │
  ▼
Contiguous KV cache
```

## Target

```text
                   KV Block Pool

┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │
└─────────┴─────────┴─────────┴─────────┴─────────┘
     ▲         ▲                    ▲
     │         │                    │
 Request A ────┘                    │
                                    │
 Request B ─────────────────────────┘
```

Suggested subsystem:

```text
engine/cache/
├── kv_cache.py
├── block_manager.py
├── prefix_cache.py
├── eviction.py
├── allocator.py
└── stats.py
```

Desired capabilities:

- dynamic growth
- block allocation
- block reuse
- prefix sharing
- sliding-window cache
- quantized KV
- eviction
- cache accounting
- memory pressure handling

A suitable long-term name is:

> `UnifiedMemoryKVManager`

---

# 5.6 Prefix Cache

mlx-one already has prompt caching behavior.

This should evolve into generalized prefix KV reuse.

Example:

```text
System prompt: 10K tokens

             Cached Prefix
                  │
         ┌────────┼────────┐
         │        │        │
      User A   User B   User C
```

Without prefix reuse:

```text
10K prefill
10K prefill
10K prefill
```

With prefix reuse:

```text
10K prefill once
      │
      ├── reuse
      ├── reuse
      └── reuse
```

Target capabilities:

- deterministic prefix hashing
- block-level cache matching
- multimodal cache-key support
- cache invalidation
- eviction
- cache-hit metrics
- tenant/session isolation where needed

---

# 5.7 Logits Processing

Keep logits processing separate from sampling.

```text
raw logits
   │
   ├── repetition penalty
   ├── frequency penalty
   ├── presence penalty
   ├── token masks
   ├── min-token constraints
   ├── grammar constraints
   └── custom processors
   │
   ▼
processed logits
```

This makes it easier to extend generation without modifying core decode logic.

---

# 5.8 Sampling

Baseline support:

- greedy
- temperature
- top-k
- top-p
- deterministic seeded sampling

Later:

- typical-p
- grammar-constrained sampling
- structured output
- DRY-style repetition control
- speculative acceptance
- model-specific sampling behavior

---

# 5.9 Stop Controller

Centralize termination behavior.

Finish conditions:

- EOS
- maximum generated tokens
- maximum context
- stop token IDs
- stop strings
- user cancellation
- client disconnect
- scheduler cancellation
- runtime error

Standard finish reasons:

```text
stop
length
cancelled
context_limit
error
```

---

# 5.10 Streaming

Streaming should remain independent of model execution.

```text
Token IDs
   │
   ▼
Incremental Decoder
   │
   ▼
Text Chunk
   │
   ├── CLI
   ├── Python iterator
   └── SSE / OpenAI API
```

Future support:

- streaming token usage
- partial reasoning
- multimodal stream events
- cancellation propagation
- structured events

---

# 6. Scheduler Evolution

The current scheduler already provides:

- bounded FIFO
- cooperative inference slots
- one MLX inference thread

This corresponds to a strong early-stage serving runtime.

The next step is a token-aware scheduler.

## Current

```text
Request A ──► ACTIVE

Request B ──► FIFO
Request C ──► FIFO
Request D ──► FIFO
```

## Target

```text
Waiting:

A   prefill  8000 tokens
B   decode      1 token
C   decode      1 token
D   prefill  1000 tokens
E   decode      1 token

Scheduler considers:

token budget
+
memory budget
+
request state
+
priority
+
latency objective
```

Conceptual interface:

```python
batch = scheduler.schedule(
    token_budget=2048,
    memory_budget=available_memory,
)

outputs = executor.run(batch)

scheduler.update(outputs)
```

---

# 7. Continuous Batching

This is one of the major steps required to reach vLLM-class throughput behavior.

Sequential execution:

```text
time ─────────────────────────────────>

A  P D D D D D D D
B                    P D D D D
C                              P D D
```

Continuous batching:

```text
time ─────────────────────────────────>

A  P D D D D D D
B      P P D D D D
C          P D D D
D              P D
```

Where:

```text
P = prefill
D = decode
```

Requests dynamically enter and leave execution batches.

Benefits:

- higher throughput
- less idle GPU time
- better latency under concurrency
- better serving utilization

---

# 8. Chunked Prefill

Chunked prefill is important for 32K, 64K, 128K and larger contexts.

Instead of:

```text
Request A

████████████████████████████ 128K PREFILL ████████████████████████████

All decodes wait
```

Use:

```text
A: P4096
B: D
C: D

A: P4096
B: D
C: D

A: P4096
B: D
C: D
```

The scheduler should tune:

```text
prefill chunk size
×
decode latency target
×
memory bandwidth
×
unified-memory pressure
×
number of active sequences
```

---

# 9. Apple Unified-Memory Scheduler

This should become a major mlx-one differentiator.

Traditional discrete GPU serving commonly reasons about:

```text
CPU RAM
   │
PCIe / NVLink
   │
GPU VRAM
```

Apple Silicon instead provides:

```text
                   Unified Memory

CPU ──────────────────┐
GPU ──────────────────┼── shared memory pool
Neural Engine ────────┘
```

Therefore mlx-one should eventually schedule against the total machine memory picture.

Conceptual budget:

```text
Total unified memory
        │
        ├── OS reserve
        ├── model weights
        ├── KV cache
        ├── activations
        ├── Metal working set
        ├── image embeddings
        ├── audio embeddings
        └── runtime buffers
```

Possible API:

```python
MemoryBudget(
    total_unified_memory=32_GB,
    reserve_for_os=6_GB,
    model=15_GB,
    kv_cache=8_GB,
    runtime=3_GB,
)
```

Potential scheduling signals:

- total physical memory
- currently available memory
- wired/compressed memory
- model footprint
- KV-cache footprint
- active sequence count
- context lengths
- prefill batch size
- multimodal embedding size
- Metal allocation behavior
- memory pressure

---

# 10. Speculative Decoding

The current `GenerationEngine` already exposes speculative options.

Long term, speculative execution should become an explicit engine subsystem.

## Draft-model mode

```text
Small Draft Model
       │
       ▼
candidate tokens
       │
       ▼
Target Model
       │
       ▼
verify / accept
```

## MTP mode

```text
Main model
   │
   ├── t+1
   ├── t+2
   ├── t+3
   └── ...
        │
        ▼
 verification
```

Suggested structure:

```text
engine/speculative/
├── base.py
├── draft.py
├── verifier.py
├── mtp.py
└── acceptance.py
```

Metrics:

- draft acceptance ratio
- target calls avoided
- effective tok/s
- verification overhead
- memory overhead

---

# 11. Multimodal Engine Strategy

mlx-one should not create four unrelated inference engines.

Use one runtime with modality-specific runners.

```text
                         mlx-one Engine
                               │
                   Shared Runtime Services
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
 CausalLMRunner           VLMRunner              ASRRunner
        │                      │                      │
 Generation Core        Vision Encoder          Audio Encoder
                               │                      │
                           Projector             ASR Decoder
                               │
                        Generation Core
```

Shared:

- scheduler
- request lifecycle
- memory planner
- cancellation
- metrics
- model lifecycle
- queueing
- streaming abstractions
- executor infrastructure

Specialized:

- preprocessing
- encoder execution
- cache semantics
- decoding algorithm
- output format

---

# 12. LM Runtime

```text
Text
  │
Tokenizer
  │
Token IDs
  │
Prefill
  │
KV Cache
  │
Decode
  │
Sampling
  │
Text
```

Uses the full autoregressive generation core.

---

# 13. VLM Runtime

```text
Image + Text
     │
     ├── Image Processor
     │
     ▼
Vision Encoder
     │
Projector
     │
     └────────────┐
                  ▼
           Text Tokenizer
                  │
                  ▼
        Multimodal Embeddings
                  │
                  ▼
               Prefill
                  │
                  ▼
               KV Cache
                  │
                  ▼
          Shared Decode Core
                  │
                  ▼
                Text
```

VLM should reuse the LM decode and generation machinery wherever architecture permits.

---

# 14. ASR Runtime

ASR should share the runtime infrastructure but not be forced through an LM-only execution loop.

Whisper-like architecture:

```text
Audio
  │
Feature Extraction
  │
Mel Spectrogram
  │
Audio Encoder
  │
Encoder States
  │
Text Decoder
  │
Transcript
```

Possible shared services:

- request lifecycle
- scheduler
- streaming
- metrics
- cancellation
- model manager
- memory planner

ASR-specific:

- audio preprocessing
- chunking
- timestamps
- encoder execution
- decoder behavior
- CTC or seq2seq decoding
- streaming audio state

---

# 15. Embedding Runtime

Embedding models do not need the autoregressive generation engine.

```text
Text
  │
Tokenizer
  │
Encoder
  │
Pooling
  │
Embedding Vector
```

They should still use:

- ModelManager
- common scheduler infrastructure
- memory planner
- metrics
- batching
- MLX executor abstractions

---

# 16. Unified Execution Primitives

A clean multimodal design can center on three primitives.

## `EncoderExecutor`

```text
input
  │
  ▼
representation
```

Used by:

- embeddings
- VLM vision encoders
- ASR audio encoders
- rerankers

## `GenerationExecutor`

```text
tokens / representation
          │
          ▼
autoregressive output
```

Used by:

- LM
- VLM
- generative audio
- seq2seq ASR decoders

## `SequencePipeline`

```text
preprocessor
    │
 encoder
    │
 projector
    │
 decoder
    │
postprocessor
```

Used to compose multimodal execution without duplicating the engine.

---

# 17. Proposed Package Evolution

Do not immediately move working code only for architectural aesthetics.

First establish interfaces, tests, and compatibility.

Long-term structure:

```text
src/mlx_one/
│
├── engine/
│   ├── engine.py
│   ├── request.py
│   ├── output.py
│   │
│   ├── scheduler/
│   │   ├── scheduler.py
│   │   ├── token_budget.py
│   │   └── policies.py
│   │
│   ├── cache/
│   │   ├── kv_cache.py
│   │   ├── block_manager.py
│   │   ├── prefix_cache.py
│   │   ├── allocator.py
│   │   └── eviction.py
│   │
│   ├── executor/
│   │   ├── model_runner.py
│   │   ├── encoder.py
│   │   ├── generation.py
│   │   ├── prefill.py
│   │   ├── decode.py
│   │   └── batch.py
│   │
│   ├── sampling/
│   │   ├── sampler.py
│   │   └── logits.py
│   │
│   ├── speculative/
│   │   ├── draft.py
│   │   ├── verifier.py
│   │   └── mtp.py
│   │
│   ├── streaming/
│   │   └── streamer.py
│   │
│   ├── memory/
│   │   ├── planner.py
│   │   └── pressure.py
│   │
│   └── metrics/
│       └── stats.py
│
├── runners/
│   ├── causal_lm.py
│   ├── vlm.py
│   ├── asr.py
│   ├── embeddings.py
│   └── reranker.py
│
├── models/
│   ├── qwen/
│   ├── llama/
│   ├── gemma/
│   └── ...
│
├── server/
│   └── ...
│
└── text/
    └── ...
```

Migration should be incremental.

---

# 18. Roadmap

## E0 — Native single-request inference

### Goal

Correct native mlx-one inference.

### Status

✅ Already implemented.

Includes:

- native model execution
- prefill
- decode
- sampling
- stop handling
- context enforcement

---

## E1 — Production generation primitives

### Goal

Reliable interactive generation.

### Status

✅ Already implemented substantially.

Includes:

- KV-cached decode
- streaming
- cancellation
- metrics
- prompt caching
- reasoning handling
- speculative configuration surface

---

## E2 — Server runtime

### Goal

Serve the native engine safely.

### Status

✅ Already implemented.

Includes:

- `GenerationEngine`
- bounded FIFO scheduler
- cooperative MLX inference slots
- single loaded model management
- FastAPI
- OpenAI-compatible serving

---

## E3 — Formal KV Memory Manager

### Goal

Separate KV ownership and allocation from generation logic.

### Work

- define KV-cache interface
- cache statistics
- block abstraction
- dynamic allocation
- unified-memory accounting
- eviction policy
- sliding-window support
- model-specific cache capabilities

### Status

🟡 Next major engine foundation.

---

## E4 — Generalized Prefix Cache

### Goal

Reuse previously computed prefixes.

### Current baseline

🟡 Prompt caching exists.

### Work

- cache-key standardization
- block-level prefix matching
- request-safe ownership
- eviction
- cache hit/miss metrics
- multimodal key support

---

## E5 — Token-Budget Scheduler

### Goal

Schedule work based on tokens and memory rather than request count alone.

### Work

- request states
- computed-token accounting
- token budget
- memory budget
- latency policies
- decode prioritization
- fairness
- priority support

---

## E6 — Chunked Prefill

### Goal

Prevent long prompts from monopolizing the device.

### Work

- configurable chunk sizes
- resumable prefill state
- interleave decode traffic
- benchmark TTFT versus throughput
- 32K / 64K / 128K test suite

---

## E7 — Continuous Batching

### Goal

Dynamically add/remove sequences from active execution batches.

### Work

- batch builder
- dynamic tensor preparation
- mixed request states
- decode batching
- prefill/decode interleaving
- batch-aware KV manager
- scheduler integration

---

## E8 — Unified-Memory Scheduler

### Goal

Optimize specifically for Apple Silicon.

### Work

- memory profiler
- model footprint estimation
- KV-cache forecasting
- OS memory reserve
- memory-pressure signals
- batch-size adaptation
- context admission control
- modality-aware memory cost

This is a potential mlx-one differentiator.

---

## E9 — Speculative Decode Runtime

### Goal

Increase output tokens/sec.

### Work

- draft model abstraction
- token verification
- acceptance policy
- MTP integration
- scheduler interaction
- memory accounting
- metrics

---

## E10 — Multimodal Scheduling

### Goal

Use the same runtime for text, vision and audio execution.

### Work

- generic runner protocol
- encoder scheduling
- multimodal request schema
- image/audio preprocessing workers
- multimodal memory accounting
- VLM cache semantics
- heterogeneous execution stages

---

## E11 — MLX / Metal Kernel Optimization

### Goal

Maximum Apple-Silicon performance.

Candidate work:

- fused decode operations
- fused RMSNorm/projection paths
- optimized RoPE
- optimized quantized matmul paths
- attention optimizations
- cache update kernels
- MoE routing optimization
- custom Metal kernels only where profiling proves value

Rule:

> Do not write custom Metal kernels merely to own more code. Use MLX primitives until profiling proves a meaningful bottleneck.

---

# 19. Current Roadmap Position

Based on the existing implementation:

```text
E0  Native single-request inference       ✅
E1  Generation primitives                ✅ / strong
E2  Server runtime                       ✅
E3  Formal KV memory manager             🟡 lifecycle/accounting foundation
E4  Prefix cache                         🟡 byte-bounded contiguous snapshots
E5  Token-budget scheduler               🟡 internal policy foundation
E6  Chunked prefill                      🟡 runner contract; production integration pending
E7  Continuous batching                  ⬜
E8  Unified-memory scheduler             🟡 admission/accounting foundation
E9  Speculative runtime                  🟡 interface/options exist
E10 Multimodal scheduling                🟡 additive API/media safety foundation
E11 Kernel optimization                  ⬜ profiling-driven
```

Therefore the immediate project should **not** rebuild generation.

The next systems work should be:

```text
1. Formalize runtime interfaces
2. Extract/strengthen KV-cache ownership
3. Add block-aware KV accounting
4. Generalize prompt cache → prefix KV cache
5. Introduce token-budget scheduling
6. Implement chunked prefill
7. Implement continuous batching
8. Add unified-memory-aware admission/scheduling
```

---

# 20. Recommended Near-Term Development Sequence

## Phase A — Stabilize current engine contracts

- preserve existing APIs
- add engine-level protocol/interfaces
- establish `GenerationRequest`
- establish `GenerationResult`
- establish `ModelRunner`
- establish KV-cache abstraction
- standardize metrics
- strengthen cancellation tests

Do not reorganize large amounts of code before contracts exist.

---

## Phase B — Benchmark baseline

Create repeatable benchmarks for:

```text
TTFT
prompt tok/s
decode tok/s
total latency
peak unified memory
KV-cache memory
cache hit rate
concurrent throughput
p50 latency
p95 latency
p99 latency
```

Contexts:

```text
1K
4K
8K
16K
32K
64K
128K
```

Concurrency:

```text
1
2
4
8
16
```

Representative hardware:

```text
M1
M2
M3
M4
16 GB
24 GB
32 GB
64 GB+
```

Representative model classes:

```text
<1B
1–3B
4–8B
10B+
dense
MoE
MTP-capable
```

---

## Phase C — KV subsystem

Implement:

```text
BaseKVCache
ContiguousKVCache
BlockKVCache
PrefixCache
KVAllocator
CacheStats
```

Keep contiguous KV as a safe fallback.

---

## Phase D — New scheduler

Add request state machine:

```text
WAITING
PREFILL
DECODING
FINISHED
CANCELLED
ERROR
```

Then introduce:

```text
token budget
memory budget
priority
fairness
decode latency target
```

---

## Phase E — Long-context execution

Build chunked prefill.

Primary validation:

- no correctness regression
- bounded decode starvation
- lower p95 decode latency under long-prefill traffic
- competitive TTFT
- no memory leak

---

## Phase F — Continuous batching

Only after KV and scheduler abstractions are stable.

Primary validation:

- throughput increases with concurrency
- single-user latency remains acceptable
- cancellation works inside batches
- stop handling remains per-request
- cache ownership remains correct

---

## Phase G — Apple-specific memory intelligence

Build memory-aware admission control.

Example:

```text
Can this request safely enter?

model
+
existing KV
+
new context KV estimate
+
prefill activation estimate
+
runtime reserve
<
allowed unified-memory budget
```

If not:

- queue
- reduce batch
- reduce chunk size
- reject with useful capacity error

Avoid relying on system OOM behavior.

---

# 21. Metrics Contract

Each request should expose or internally collect:

```text
request_id

prompt_tokens
generated_tokens
cached_prompt_tokens

queue_ms
tokenization_ms
prefill_ms
ttft_ms

prompt_tokens_per_sec
decode_tokens_per_sec

total_latency_ms

peak_memory_bytes
kv_cache_bytes

prefix_cache_hit
prefix_cache_tokens

speculative_draft_tokens
speculative_accepted_tokens

finish_reason
```

Server-level metrics:

```text
active_requests
waiting_requests
requests/sec

tokens/sec
prefill_tokens/sec
decode_tokens/sec

KV utilization
prefix cache hit rate

unified memory usage
engine memory reserve

p50 latency
p95 latency
p99 latency

p50 TTFT
p95 TTFT
p99 TTFT
```

---

# 22. Correctness Requirements

Optimization cannot weaken generation correctness.

Required tests:

- deterministic greedy output
- seeded sampling reproducibility
- cached vs uncached parity
- streaming vs non-streaming parity
- cancellation
- stop string across token boundaries
- EOS behavior
- max-token behavior
- max-context behavior
- prompt-cache correctness
- prefix-cache correctness
- batch isolation
- KV ownership isolation
- long-context correctness
- quantized/unquantized parity tolerance
- speculative/non-speculative output validity

---

# 23. Engine API Direction

Potential public API:

```python
from mlx_one import Engine

engine = Engine(
    model="Qwen/Qwen3.5-4B",
)

result = engine.generate(
    "Explain KV cache",
    max_tokens=512,
)
```

Streaming:

```python
for chunk in engine.stream(
    "Explain KV cache",
    max_tokens=512,
):
    print(chunk.text, end="")
```

Advanced server:

```python
engine = Engine(
    model="Qwen/Qwen3.5-4B",
    max_num_seqs=8,
    max_context=131072,
    prefix_cache=True,
    chunked_prefill=True,
)
```

The public API should remain simpler than the internal runtime.

---

# 24. What mlx-one Should Not Become

Do not make mlx-one:

### A wrapper around another inference engine

Avoid:

```text
mlx-one
   │
   └── vLLM / SGLang / llama.cpp / mlx-lm
```

Target:

```text
mlx-one Engine
   │
   └── MLX
```

### A giant monolithic `generate()` function

Scheduler, cache, execution, sampling, streaming and metrics should have clear ownership.

### A CUDA architecture copied onto Apple hardware

Apple Silicon has different constraints and opportunities.

### A custom-Metal-first project

MLX remains the compute foundation.

Custom kernels should be profiling-driven.

---

# 25. mlx-one vs vLLM Mental Model

```text
vLLM
│
├── GPU inference runtime
├── scheduler
├── KV-cache management
├── prefix caching
├── continuous batching
├── speculative decoding
└── serving
     │
     ▼
CUDA / accelerator backends
```

mlx-one target:

```text
mlx-one Engine
│
├── MLX-native inference runtime
├── scheduler
├── unified-memory-aware KV management
├── prefix caching
├── continuous batching
├── chunked prefill
├── speculative / MTP decoding
├── multimodal execution
└── serving
     │
     ▼
MLX
     │
     ▼
Metal
     │
     ▼
Apple Silicon
```

---

# 26. Primary Technical Objectives

mlx-one inference should optimize four dimensions.

## Correctness

Native model parity and deterministic runtime behavior.

## Latency

Especially:

- queue time
- TTFT
- inter-token latency

## Throughput

Especially under concurrent requests.

## Memory Efficiency

Especially:

- KV cache
- long context
- multimodal inputs
- unified-memory pressure

---

# 27. Strategic Differentiators

The mlx-one engine can differentiate through:

### 1. Unified-memory-aware inference

Treat the whole Apple machine as one memory system rather than emulating CUDA VRAM assumptions.

### 2. Native multimodal runtime

LM + VLM + ASR + embeddings under one execution architecture.

### 3. Long-context local inference

Optimize 32K–128K+ prefill and KV management on consumer Apple Silicon.

### 4. MTP-aware execution

Support emerging multi-token-prediction model families natively.

### 5. One package

```text
models
+
inference
+
training
+
quantization
+
serving
+
evaluation
```

without requiring multiple external inference frameworks.

---

# 28. Final Architecture Goal

```text
                              mlx-one
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
              Train           Infer            Serve
                                 │
                                 ▼
                        ┌───────────────────┐
                        │ mlx-one Engine    │
                        │                   │
                        │ Request Manager   │
                        │ Scheduler         │
                        │ Memory Planner    │
                        │ KV Manager        │
                        │ Prefix Cache      │
                        │ Batch Manager     │
                        │ Metrics           │
                        └─────────┬─────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
             LM Runner        VLM Runner      Audio Runner
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                          Execution Runtime
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                 Prefill        Decode      Speculative
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                         Native mlx-one Models
                                  │
                                  ▼
                                 MLX
                                  │
                                  ▼
                                Metal
                                  │
                                  ▼
                         Apple Unified Memory
```

---

# 29. Definition of Success

mlx-one should eventually be able to say:

> **mlx-one provides its own native MLX inference engine. It performs model execution, prefill, KV-cached decoding, scheduling, batching, prompt/prefix caching, sampling, streaming, speculative decoding, multimodal execution, metrics, and OpenAI-compatible serving directly on Apple Silicon. It does not require mlx-lm, llama.cpp, Ollama, vLLM, or SGLang as an inference backend.**

And at the systems level:

> **vLLM-class runtime responsibilities, MLX-native implementation, Apple-Silicon-native scheduling.**

---

# 30. Immediate Next Milestone

The next milestone should **not** be “build an inference engine.”

That already exists.

The milestone should be:

## `Engine Runtime V2 — KV + Scheduler Foundation`

Deliverables:

- [ ] Formal `GenerationRequest`
- [ ] Formal `GenerationResult`
- [ ] `ModelRunner` interface
- [ ] `KVCache` interface
- [ ] current contiguous KV implementation behind interface
- [ ] KV memory accounting
- [ ] prefix-cache contract
- [ ] scheduler request-state machine
- [ ] token-budget scheduler design
- [ ] unified engine metrics contract
- [ ] benchmark harness
- [ ] 1K–128K context benchmarks
- [ ] concurrency 1/2/4/8 benchmarks
- [ ] regression tests against current generation behavior

Then proceed to:

```text
Block KV
   ↓
Generalized Prefix Cache
   ↓
Token-Budget Scheduler
   ↓
Chunked Prefill
   ↓
Continuous Batching
   ↓
Unified-Memory Scheduler
   ↓
Speculative / MTP Runtime
   ↓
Multimodal Scheduling
   ↓
Profiling-driven Metal optimization
```

This evolves the existing mlx-one engine instead of replacing it.
