# Native inference contracts and baseline harness

PR-01–PR-12 provide runtime contracts, cache transactions, safe prefix sharing, explicit dense,
paged and hybrid batching, live memory-pressure control, and runnable qualification profiles.
These implementations remain experimental. Metal kernels have not been compiled or measured in
this sandbox; CPU readiness does not establish GPU parity or performance qualification.
Measured GPU qualification remains pending. Existing dense execution remains the default.

## API migration

These internal contract changes intentionally break old imports and capability-property access:

| Previous interface | Replacement |
| --- | --- |
| `SequenceState` enum | `Phase`; `DECODING` becomes `DECODE` |
| `SequenceContext` | `SequenceState` dataclass |
| `state`, `prompt_tokens`, `computed_tokens`, `generated_tokens` | `phase`, `input_ids`, `computed_len`, `generated_ids` |
| `ExecutionBatch` | immutable `BatchPlan` containing `ScheduledSequence` descriptors |
| runner output tuple | `RunnerOutput` with aligned request IDs, logits, and committed counts |
| `runner.capabilities` | `runner.capabilities()` |
| `engine.capabilities` | `engine.capabilities()` |

`Phase.TIMED_OUT` is retained alongside the roadmap phases. State also retains the normalized
`EngineRequest`, cancellation event, error, deadline, and model epoch. `computed_len` advances
only after evaluated model execution. Generated IDs may contain a final token that is not yet
represented in the cache. Repeating a terminal transition or finalization is harmless; leaving a
terminal phase is invalid. Event sequence numbers and model epochs let adapters reject stale events.

`LayerStateSpec` adapts the existing cache topology without replacing `CachePlan` or `CacheHandle`.
A block-compatible layout does not imply direct paged attention. Unknown state capabilities are
false. Layout precision declarations are separate from observed tensor dtypes.

```python
from mlx_one import Engine

with Engine("Qwen/Qwen2.5-0.5B", offline=True,
            inference_mode="reference_dense") as engine:
    print(engine.capabilities())
    print(engine.generate("Hello", max_tokens=32).text)
    print(engine.runtime_stats())
```

Generation, streaming, submission, and context-manager signatures otherwise remain compatible.
CLI/API generation defaults and SSE framing remain unchanged.

## Execution modes and feature inventory

`--inference-mode reference_dense` selects the existing single-request dense path.
`--inference-mode experimental_block` selects existing block storage and rejects unsupported
models. The latter requires unquantized Qwen2/Qwen2.5 attention topology. Existing `--cache-backend dense|block` selections remain supported; an explicit conflict with inference mode fails. Omitted
mode/backend still selects dense.

| Feature | Inventory status |
| --- | --- |
| Native dense generation | Experimental until exact-scope evidence is registered |
| Chunked prefill in `CausalLMRunner` | Experimental runner capability |
| Equal-offset attention decode with B>1 in `CausalLMRunner` | Experimental, restricted runner capability |
| Cooperative server slots | Separate B=1 forward calls |
| Existing block storage/prefix index | Experimental for eligible topology |
| Direct page-table attention and page-native prefill | Experimental in `paged`; Metal qualification pending |
| Continuous Python/HTTP batching | Experimental in `experimental_dense_batch` |
| Hybrid prefix restoration | Experimental in `experimental_hybrid_batch`; Metal qualification pending |
| Quantized paged KV | Harness only until exact quality evidence permits public admission |
| 128K-class admission | Harness only until exact context, quality, memory and service evidence passes |
| Automatic selection | Opt-in `auto`, with exact performance evidence or explained reference fallback |

Python diagnostics and `/v1/runtime`'s additive `inference` field share the native feature resolver.
`runner_qualification` describes runner-only capabilities separately from the selected server path.
The feature status enum is independent of the existing operation-level capability registry:
`not_implemented`, `experimental`, `correctness_passed`, `performance_qualified`.

Promotion requires an immutable checkpoint revision, execution path, topology digest, weight/KV
precision, MLX version, hardware identity, source revision, and matching native qualification result.
Artifacts use the existing `ArtifactRef` type and SHA-256 checksums. Changed evidence is downgraded;
a different scope does not inherit qualification. Existing `mlx-lm` calibration evidence cannot
qualify native execution. Baseline runs never promote features automatically.

## Running profiles

Run from the repository root after installing the existing development/server dependencies.
Listing profiles does not initialize MLX or download models:

```bash
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode reference_dense --list-profiles
```

The default matrix pins Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` and LFM2-350M-4bit revision
`18dc72abf3b2337f9123cfd6eeeb58dfa7947066`. Weights, configuration, tokenizer assets, and
source are hashed. Model loading is offline; missing cached assets cause an explicit failure.
Both checkpoints retain native floating-point weight precision, and LFM retains its 4-bit weights.
The reference profiles request BF16, unquantized attention KV. Actual dtypes are verified rather
than inferred from configuration. Convolution carries have their own byte/dtype records.

On a GPU-capable Metal host:

```bash
# Twelve fresh-process configurations: two models × 2K/8K prompts × slots 1/2/4.
.venv/bin/python benchmarks/inference_baseline.py --inference-mode reference_dense

# Shared-prefix diagnostics using the repaired restoration boundaries.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode reference_dense --workload prefix

# Bounded, single-slot 16K/32K attempts. Context/memory failures are recorded.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode reference_dense --workload long

# Forced-token dense-runner parity and independent hybrid-state/RNG checks.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode reference_dense --workload parity
```

Use `--models qwen` or `--models lfm` to restrict a run. Each HTTP configuration runs one cold
request group, one warm-up request, and three warm groups, with up to 128 output tokens per request.
Prompt lengths are measured after rendering and stored alongside requested sizes and hashes.
The main matrix disables prefix reuse and speculation. Prefix diagnostics enable the existing
cache behavior and vary suffixes. The worker uses one authenticated test principal so these
requests can share prefixes; anonymous HTTP requests intentionally cannot.

Every worker has a 600-second overall deadline and a parent RSS watchdog at 75% of detected RAM.
The watchdog polls every 100 ms and cannot guarantee prevention of a sudden allocation failure.
Unavailable memory sampling fails the run. Timeout, process failure, rejection, and memory
termination are durable outcomes, never successful qualification. Long contexts also remain
subject to checkpoint limits and existing admission behavior.

Experimental block profiles use explicit FP16 floating weights and FP16 KV. Block updates cast
incoming K/V to FP16; a declared precision must match actual observed tensors. Block parity
compares against the single-request path with the same weights and FP16 KV.

## Measurements and artifacts

Results go under `qualification/results/<host>/<checkpoint>/<run>/`:

- `manifest.json`, `checkpoint.json`, `workload.json`: exact run identities, hashes, and profiles.
- `requests.jsonl`, `requests.csv`, `summary.json`: request outcomes and timing distributions.
- `runtime.json`, `load.json`, `cleanup.json`: runtime inventory, loading, and cache cleanup state.
- `parity.json`: forced-token logit errors, greedy agreement, carry-state matching, cancellation,
  and seeded sampling replay for parity profiles.
- `supervision.json`, `worker.log`, and failure artifacts: RSS samples, exit status, and diagnostics.

The benchmark observer explicitly evaluates logits before recording forward time. Its instrumentation
is named `synchronized-forward-v1` and is enabled only for harness workers. Compare runs using the
same observer policy. SSE delivery gaps measure visible chunks; model-token gaps measure accepted
sampled tokens, which can be buffered by detokenization. They are reported separately.

Server generation TTFT excludes queueing and request setup; client TTFT includes them. Queue time
and observed tokenization/prefill time are separate metrics. Per-request decode rate excludes prefill; aggregate output rate includes request wall time and is labeled separately. Unknown
measurements remain null. No maximum-active-user or speedup claims are generated.

Parity profiles use 2K rendered inputs and 32 forced steps against the existing single-request
path, plus 512-token chunked runner prefill and interleaved independent carries. Initial absolute/relative tolerances are FP32
`1e-4/1e-4`, FP16 `1e-2/1e-2`, and BF16 `5e-2/1e-2`. `--atol` and `--rtol` must be overridden
together and produce a different profile ID. Greedy agreement is recorded separately from numeric
acceptance. Sampling isolation replays identical logits with independent RNG streams.

## Validation

```bash
PYTHONPATH=src:. .venv/bin/pytest -q --no-cov -p no:cacheprovider
.venv/bin/ruff check .

# Optional pinned checkpoint tests; requires readable offline assets and Metal access.
MLX_ONE_RUN_INFERENCE_CHECKPOINTS=1 PYTHONPATH=src:. \
  .venv/bin/pytest -q --no-cov tests/test_inference_checkpoint_parity.py
```

The inspected sandbox cannot initialize Metal. CPU tests and profile listing establish harness
readiness only; publish measured results from a GPU-capable environment before qualification.


## Safe sharing and continuous batching (PR-03–PR-06)

`experimental_dense_batch` supports unquantized Qwen2/Qwen2.5 text models with observed FP16
floating weights and FP16 KV. `execution_dtype="float16"` explicitly converts floating model
weights; omission preserves native checkpoint precision and rejects incompatible batched execution.
Intermediate arithmetic may use higher precision. Runtime diagnostics record actual weight, cache,
and logits dtypes separately. Existing BF16 baseline profiles retain native checkpoint precision.

```python
from mlx_one import BatchingConfig, Engine

with Engine(
    "Qwen/Qwen2.5-0.5B",
    revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
    offline=True,
    inference_mode="experimental_dense_batch",
    execution_dtype="float16",
    batching_config=BatchingConfig(max_sequences=4),
    cache_namespace="my-trusted-session",
) as engine:
    jobs = [engine.submit(prompt, max_tokens=128) for prompt in ("Hello", "Explain RoPE")]
    results = [job.result() for job in jobs]
    print(engine.runtime_stats())
```

Concurrent `generate`, `stream`, `submit`, and `async_generate` calls use one actor. The default
maximum active sequence count remains one. `reference_dense` retains cooperative B=1 execution;
`experimental_block` retains B=1 block storage. Explicit mode/backend conflicts and unsupported
architectures or precision fail without substitution. `experimental_dense_batch` does not enable
speculation, quantized weights/KV, mixed-phase forwards, hybrid restoration, or paged attention.

```bash
mlx-one serve Qwen/Qwen2.5-0.5B --offline \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --inference-mode experimental_dense_batch --execution-dtype float16 \
  --parallel 4 --prefill-chunk-size 512 --max-batch-tokens 2048 \
  --max-prefill-tokens 1024
```

Prefix sharing is process-local and isolated by authenticated API key principal. Anonymous HTTP
requests receive a private namespace for each request. Python requests are private unless the
caller supplies `cache_namespace`. Callers sharing a Python namespace must be within one trust
boundary. `cache_idle_slots` and slot reassignment never override principal isolation. Request body
fields cannot select an authenticated namespace. Credentials and namespace secrets are omitted from
runtime reports and result manifests.

Dense full-attention restoration finds the common token prefix and trims an isolated snapshot.
Block restoration adopts sealed complete pages. Both leave at least one prompt token to execute,
including identical prompts. Cache publication requires evaluated prompt state, consistent layer
offsets, and a live uncancelled request. Weights/adapters and tokenizer assets are immutable within
a loaded model epoch; change them through lifecycle unload/reload barriers. Context-shifted prompts
are matched using their actual rendered tokens, effective context limit, RoPE, template options,
special tokens, and checkpoint identity. Unsupported multimodal restoration fails closed.

Page IDs contain pool epoch, slot, and allocation generation; recycled slots cannot satisfy stale
references. Layout version 1 includes layer, page width, heads, dimensions, and actual K/V dtype.
The parent-linked prefix index bounds unique pinned tensor bytes and separately bounds metadata
using a conservative accounting estimate. Eviction removes leaves without invalidating a retained
path or a live request. Complete sealed pages share references; mutable tails are copied.

Cache counters have these meanings:

| Counter | Meaning |
| --- | --- |
| `logical_bytes` | Live request token state; shared tokens count once per request |
| `unique_live_physical_bytes` | Unique live/prefix-resident block storage; dense request allocations |
| `retained_free_bytes` | Reusable block storage with no live references or pins |
| `physical_bytes` | Block live plus retained free storage; dense allocations plus prefix snapshots |
| `prefix_bytes` / `pinned_bytes` | Prefix-retained storage; overlaps physical storage |
| `prefix_metadata_bytes` | Conservative index metadata accounting, separate from tensor bytes |
| `evictable_bytes` | Prefix storage releasable without live request references or other pins |
| `shared_page_savings_bytes` | Physical bytes avoided by additional live page references |
| `reserved_bytes` | Unallocated promised growth plus temporary execution reservations |

The actor reserves prompt-plus-output cache growth before forwarding and temporary packing/scatter
storage before each batch. As buffers become resident, reconciliation reduces outstanding growth
reservations. Cache budgets cover cache/packing storage, not a unified-memory activation guarantee;
the process watchdog remains necessary for GPU qualification. No calibrated pressure controller is
introduced in this milestone.

Decode and prefill run as separate microbatches under one scheduled-token budget. Pending prefill
reserves up to one chunk and at most half that budget; selection rotates both groups. Admission
uses earliest deadline with FIFO ties and oldest-first consideration after two seconds. Temporarily
unfit admissions release their slot for other feasible requests. Output channels hold at most 32
events by default; a continuously full channel expires after 30 seconds. Completion/error status
uses a separate channel, and slow clients do not block the actor. Deadlines include queued and
paused time. An in-flight Metal operation is cooperatively cancelled after evaluation.

### Metal qualification commands

```bash
# List 32 fresh-process configurations: contexts 2K/8K × B=1/2/4/8 × four chunk sizes.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode experimental_dense_batch --models qwen --workload batching --list-profiles

# Matched FP16 cooperative baseline for one selected configuration.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode reference_dense --models qwen --workload batching \
  --context-tokens 2048 --slots 4 --chunk-tokens 512

# Run its batched counterpart; supply service limits chosen for the target workload.
# Replace the directory and limits with the actual baseline run and declared budgets.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode experimental_dense_batch --models qwen --workload batching \
  --context-tokens 2048 --slots 4 --chunk-tokens 512 \
  --ttft-budget-ms "$TTFT_BUDGET_MS" --decode-gap-budget-ms "$DECODE_GAP_BUDGET_MS" \
  --reference-run "$REFERENCE_RUN_DIRECTORY"

# Cache/parity and traffic diagnostics. Each workload is independently identified.
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode experimental_block --models qwen --workload parity
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode experimental_dense_batch --models qwen --workload parity --slots 2 --chunk-tokens 512
.venv/bin/python benchmarks/inference_baseline.py \
  --inference-mode experimental_dense_batch --models qwen --workload turnover
# Also available: mixed, cancellation, slow_consumer, prefix, long.
```

All main profiles use greedy decoding, 128 output tokens, three repetitions, one cold group and
one warmup. Main profiles disable prefix reuse and speculation. Chunk profiles explicitly increase
compatible token budgets for 2K/4K chunks. Turnover, mixed prompts, cancellation, and throttled SSE
consumption have separate profile IDs. The HTTP throttling profile is a delivery diagnostic;
bounded-channel stall expiry is also verified directly with CPU actor tests.

Additional artifacts include `forward-traces.jsonl`, matched FP16 `parity.json`, `gate-c.json`,
and `evidence-<feature>.json`. Evidence uses the existing native qualification schema and
`ArtifactRef` SHA-256 checksums. Explicit registry promotion validates backing artifacts and
rechecks them on reporting; edited evidence becomes stale.
Traces record request membership, history lengths, real/padded tokens, padding efficiency, queue
age, temporary reservations, and synchronized execution duration. Reported forward amortization is
a count of actual forwards, not a measurement of GPU weight traffic. SSE delivery timing remains
separate from model-token timing. The full trace is saved as it occurs; runtime diagnostics keep a
bounded recent window.

Gate C is diagnostic without explicit TTFT and model-token decode-gap limits, and pending without
matched reference evidence. A passing gate requires numeric parity, actual B>1 decode, changing
membership, completed service, all declared latency limits, and higher aggregate throughput in all
three matched warm repetitions. Reference identity checks include checkpoint, precision, workload,
hardware, MLX, and source provenance. Failed declared gates fail the worker; no run automatically
promotes registry status. Block Gate B evidence includes shared-prefix reuse and avoided prefill
alongside logits and accounting. GPU Gates B/C remain pending in this Metal-inaccessible sandbox.

### Review boundaries

- PR-03: staged block mutation and multi-layer adoption, isolated dense snapshots, committed prefix
  publication, FP16 KV enforcement, idempotent request cleanup and reservations.
- PR-04: generation-aware IDs/layouts, parent-linked bounded APC, principal namespaces, accounting.
- PR-05: dense packing/scatter runner, inference actor, explicit mode/precision, Python/HTTP routing.
- PR-06: combined-budget chunked prefill, fairness/deadlines/backpressure, traces, profile matrix,
  parity and explicit service-budget gates.

The changes build on the existing uncommitted PR-01/02 state. No UI files, generated UI assets,
remote branches, or published PRs are changed by this implementation.

### Implementation validation (2026-10-10)

The complete backend-free suite passes: **606 passed, 23 skipped**, with two dependency
deprecation warnings. This includes cache rollback/ownership, unequal-history B=2 forwards,
dynamic membership, Python/HTTP routing, namespace isolation, cancellation, bounded consumers,
scheduling budgets, lifecycle barriers, and deterministic harness/evidence fixtures.

```bash
PYTHONPATH=src:. .venv/bin/pytest -q --no-cov -p no:cacheprovider
.venv/bin/ruff check .
git diff --check
```

Ruff reports **65 pre-existing findings and zero new findings**. Existing findings are in the CLI,
inspection/model registry, and SAM3 modules; no automatic rewrites were used. Diff checks pass.
The profile listing exposes all 32 Qwen batching configurations without initializing Metal.

The local Metal probe fails with `ImportError: [metal::load_device] No Metal device available`.
Consequently, checkpoint parity, measured shared-prefix Gate B, and throughput/latency Gate C are
**pending**. CPU fixtures and runnable profiles establish implementation readiness and do not
qualify GPU execution or change its experimental status.

## Paged execution, hybrid state and qualification (PR-07–PR-12)

The added modes are explicit. Unsupported architectures, precision, layouts and backend conflicts
raise capability errors; there is no substitution. Omitted modes retain their existing paths.
Maximum active sequences remains one unless explicitly configured. Existing generation methods,
HTTP response shapes and SSE framing are retained; diagnostics add fields.

```python
from mlx_one import BatchingConfig, CacheConfig, Engine, MemoryPressureConfig

with Engine("Qwen/Qwen2.5-0.5B",
            revision="060db6499f32faf8b98477b0a26969ef7d8b9987", offline=True,
            inference_mode="paged", execution_dtype="float16",
            cache_config=CacheConfig(backend="paged", key_type="f16", value_type="f16"),
            batching_config=BatchingConfig(max_sequences=2, admission_policy="cache_aware"),
            cache_namespace="application-tenant-42",
            memory_pressure_config=MemoryPressureConfig()) as engine:
    first = engine.submit("A document and its first question.", max_tokens=128)
    second = engine.submit("Another document and question.", max_tokens=128)
    print(first.result().text, second.result().text)
    print(engine.runtime_stats())
```

`experimental_hybrid_batch` selects native LFM2/LFM2.5 attention-plus-convolution execution.
Use `CacheConfig(backend="hybrid", key_type="bf16", value_type="bf16")` for the pinned LFM
profile and omit execution dtype to preserve its checkpoint weight quantization. Explicit
`execution_dtype="bfloat16"` is supported for eligible floating models; dtype selection casts
floating weights and records the conversion. Omission preserves native weights.

HTTP admission still derives opaque namespaces from authenticated principals. Anonymous requests
receive private namespaces. Client-supplied namespace text never grants sharing authority. Python
requests are private unless `cache_namespace` is provided. Slot reuse does not alter this policy.
Multimodal restoration fails closed.

```bash
mlx-one serve Qwen/Qwen2.5-0.5B --offline \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --inference-mode paged --execution-dtype float16 --parallel 2 \
  --admission-policy cache_aware --kv-block-size 32 --memory-pressure

mlx-one serve mlx-community/LFM2-350M-4bit --offline \
  --revision 18dc72abf3b2337f9123cfd6eeeb58dfa7947066 \
  --inference-mode experimental_hybrid_batch --parallel 2 \
  --cache-type-k bf16 --cache-type-v bf16
```

### Storage, execution and lifecycle boundaries

`PagedRunner` executes both prefill and decode directly over segmented pages. Each physical page
is a kernel-addressable segment; dispatches bind at most eight floating or four quantized segments.
GPU inputs contain validated segment bindings, logical starts, real sequence lengths, query widths
and absolute positions. The Metal kernel performs indexed GQA attention and FP32 online softmax;
partition results merge using maxima, normalizers and weighted sums. Appends update only affected
segments. Floating KV supports FP16/BF16, page widths 16/32/64/128, and head dimensions up to 256.
Qwen RoPE also requires an even head dimension. Other layouts fail explicitly.

The final PR-08 path uses page-native chunked prefill, so the earlier dense staging/conversion
step is unnecessary. `history_materializations` and `conversion_bytes` are zero for this path.
No dense-attention fallback is present. Kernel timing is synchronized host dispatch time, which
includes host dispatch/partition-merge overhead. GPU timestamp latency remains null.

Quantized layouts are separately versioned: unsigned 4/8-bit payloads, affine groups of 32
scalars per token/head, dimension padding and FP16 scales/biases. Four-bit values pack the lower
indexed scalar into the low nibble. Mutable tails remain floating; complete pages are quantized
and sealed transactionally. Kernels dequantize individual scalars during attention. Diagnostic
CPU dequantization is never a production attention fallback. A separate harness oracle reconstructs
quantized pages for kernel correctness against dequantized dense attention; its materializations
are recorded separately from production calls. Forecasts include padding, metadata,
floating tails and sealing overlap. Public quantized admission requires exact quality evidence.

`HybridRunner` keeps full/sliding KV and convolution carries request-local, packs unequal histories
with explicit absolute positions/masks, and commits only real tokens. It retains native checkpoint
weight quantization and floating KV. Carries are copied on fork/restore; padding never advances
them. Snapshots publish only at complete page boundaries, restore an exact matching boundary and
always execute a suffix containing at least one prompt token. Sliding components retain logical
positions and window starts. Hybrid attention remains component-aware dense attention.

Snapshots use `hybrid-state-v1` JSON metadata plus checksummed NPZ tensors with `allow_pickle=False`.
Topology, identity, committed positions, shapes and dtypes validate before adoption. BF16 payloads
use lossless FP32 interchange encoding. Generic recurrent components support isolated clone/reset
lifecycle tests; native SSM/DeltaNet transition math remains unavailable.

Both runners stage all layers, evaluate state/logits and then commit. Cancellation or a deadline
expiring during a forward prevents adoption/publication for that request. Failed allocations,
failed forwards, stale identities and closed handles cannot adopt staged state. Unload/reload uses
actor barriers; repeated cleanup releases reservations and ownership idempotently.

Cache-aware admission is opt-in. It orders expired/near deadlines (250 ms), then requests aged
at least two seconds in FIFO order, then longest reusable prefix with deadline/FIFO ties. Preview
does not pin pages or change namespace authority. Temporarily unfit candidates can be skipped.
Default admission, separate decode/prefill microbatches and combined token budgets are retained.

### Pressure and evidence selection

New modes and `auto` enable live pressure control by default. Existing modes require explicit
`MemoryPressureConfig` or `--memory-pressure`. `--no-memory-pressure` disables it;
`--process-ceiling-mib` lowers the process ceiling. A model-owned background sampler collects
RSS, installed RAM, MLX active/cached/peak bytes, recommended working set, OS pressure and swap
once per second. Unavailable observations are null. Scheduling boundaries apply the latest sample.

The ceiling is the minimum of 75% of installed RAM, an explicit process ceiling and available
recommended working-set guidance; cache budgets independently constrain owner allocations.
Warning starts at 90% of the ceiling. Critical pressure includes the ceiling, critical OS pressure
or two successive swap-growth samples. Recovery requires two normal samples. Warning sheds
unreferenced prefixes/free pages and allocator cache, reduces prefill service and queues admission.
Critical reservations that cannot fit fail with typed capacity errors after cleanup.

Owner counters explain physical sharing; observed RSS and MLX measurements overlap and are never
summed. Persistent growth and transient overlap reservations are separate. Conservative analytical
forecasts are labeled. Fresh `activation_peak` evidence can supply a measured increment only for
identical execution identity and exact batch/chunk/history dimensions; stale calibration falls back
to the analytical forecast.

`auto` resolves once at model load. It requires fresh checksummed performance evidence matching
checkpoint, weights/adapters, tokenizer/template/special tokens, topology, precision, page/kernel
version, source, MLX, hardware and workload envelope. Otherwise it resolves to `reference_dense`
and reports the reason. Active requests never switch backends. The envelope includes:

```json
{
  "context_tokens": 8192, "output_tokens": 128, "slots": 2,
  "chunk_tokens": 512, "page_tokens": 32, "cache_type": "f16",
  "admission_policy": "deadline", "ttft_budget_ms": 5000,
  "decode_gap_budget_ms": 100
}
```

These budget numbers illustrate configuration; they are not qualification measurements. Python
accepts `qualification_registry` and `qualification_profile`; the server accepts corresponding
configuration and `--qualification-evidence PATH` (repeatable) plus `--qualification-profile JSON`.
Configured execution must match the envelope, and admission revalidates prompt/output limits.
128K-class public requests additionally require context, quality, memory and service evidence;
the limit always includes requested output. `auto` is rejected by benchmark profiles.

Runtime diagnostics expose requested/resolved mode, selection reason/evidence scope, observed
weight/cache dtypes, component capabilities, pressure telemetry/state, forecasts, scheduler
membership and direct-attention/materialization counters. Per-forward harness traces also record
MLX active/cached/process-peak memory. Configured precision is never presented as an observation.

### Qualification commands and pending gates

The pinned matrix adds Qwen2.5-1.5B revision `8faed761d45a263340a0528343f099c05c9a4323` and
Qwen2.5-3B revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` to the earlier Qwen0.5/LFM pins.
Local config preflight limits are 32768/131072/32768/128000 respectively. Unsupported prompt-plus-
output profiles record `rejected_context` before loading weights. No checkpoint is downloaded.

```bash
export PYTHONPATH=src:.

# One page/chunk configuration; remove filters to list the full matrix.
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --slots 2 --context-tokens 8192 --chunk-tokens 512 --page-tokens 32 --list-profiles

# Matched FP16 cooperative reference, then an explicit direct-page candidate.
.venv/bin/python benchmarks/inference_baseline.py --inference-mode reference_dense \
  --qualification-profiles --models qwen --slots 2 --context-tokens 8192 --chunk-tokens 512
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --slots 2 --context-tokens 8192 --chunk-tokens 512 --page-tokens 32 \
  --reference-run /absolute/path/to/reference-run \
  --ttft-budget-ms 5000 --decode-gap-budget-ms 100 --min-decode-tok-s 10

# Attention timing, hybrid state parity, prefix-policy traffic and pressure recovery.
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --workload microbench --slots 2
.venv/bin/python benchmarks/inference_baseline.py --inference-mode experimental_hybrid_batch \
  --models lfm --workload parity --slots 2
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --workload multi_turn --slots 2 --shared-prefix-fraction .5 --admission-policy cache_aware
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --workload pressure --slots 2

# Quality limits JSON must declare all three limits listed below; no implicit tolerances.
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --workload quality --context-tokens 8192 --slots 2 --kv-bits 4 \
  --quality-limits /absolute/path/to/quality-limits.json

# A 30-minute soak (use 60 for the other duration).
.venv/bin/python benchmarks/inference_baseline.py --inference-mode paged --models qwen \
  --workload soak --context-tokens 8192 --slots 2 --soak-minutes 30 \
  --ttft-budget-ms 5000 --decode-gap-budget-ms 100 --leak-budget-bytes 0

.venv/bin/python benchmarks/inference_qualification_matrix.py \
  --results qualification/results --output qualification/matrix

MLX_ONE_RUN_INFERENCE_CHECKPOINTS=1 .venv/bin/pytest -q --no-cov \
  tests/test_inference_checkpoint_parity.py
```

Quality limits are `max_logit_error`, `max_mean_nll_delta` and `max_retrieval_accuracy_drop`.
Results record forced-token errors, greedy agreement, held-out continuation NLL changes and
three-position long-context retrieval. Missing limits remain diagnostic. For 128K-class evidence,
run matching `quality` and `soak` profiles at 130944 prompt tokens on Qwen1.5 (127872 on LFM),
then pass the checksummed quality directory to the soak with `--quality-run`. Missing context,
quality, memory or service results cannot unlock public admission.

Pressure profiles change a declared process ceiling to exercise queueing, typed reservation
rejection and recovery without allocating artificial host-memory stress. Soaks compare stable
initial/final greedy probes and post-cleanup active bytes, record live pressure recovery, and
require a declared leak budget. Traffic scenarios include deterministic arrivals/turnover, mixed
lengths, cancellation and slow SSE consumers. Delivery throttling remains a transport diagnostic;
the bounded actor channel and stalled-consumer expiry are verified separately by CPU tests.

Artifacts remain under `qualification/results/<host>/<checkpoint>/<run>/`: manifests, exact
checkpoint hashes, request JSONL/CSV, forward traces, raw memory/time series, parity/state/quality
results, failures, cleanup, gates and checksummed native evidence. The matrix exports JSON, CSV
and a readable table. Stale/incomplete artifacts yield pending gates. SSE timing, model-token
gaps, failures and regressions are separate fields. Service-qualified decoder counts require
declared TTFT/gap/decode-floor limits and count overlapping model decode intervals, excluding
queue and SSE time.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| D | Direct execution, parity/quality, actual batching where configured, measured advantage and declared service budgets against matched dense execution | Pending Metal measurements |
| E | Native LFM logits and all component-state parity, isolation and actual prefix restoration | Pending Metal measurements |
| F | Named-device 30/60-minute soak, reconciled cleanup/memory, pressure recovery and service stability | Pending Metal measurements |

### Six review boundaries

| Review | Main implementation | Verification |
| --- | --- | --- |
| PR-07 | `engine/paged.py`, `engine/paged_attention.py`, Qwen attention dispatch | Ragged GQA, partial pages, FP32 partition merge, stale IDs, rollback; opt-in native parity |
| PR-08 | Page-native prefill and versioned quantized layouts/kernels; quality profiles | Chunk masking/positions, 4/8-bit packing/padding, public quality gates |
| PR-09 | Shared parent-linked prefix index integration and actor cache-aware admission | Principal isolation, actual restoration, deadline/age/locality order, skipped candidates |
| PR-10 | `engine/components.py`, `engine/hybrid.py`, native LFM dispatch | Carries/KV parity, fork/cancel/deadline isolation, JSON/NPZ adoption, recurrent lifecycle |
| PR-11 | `engine/pressure.py`, `engine/selection.py`, Python/server/CLI integration | Sampling/lifecycle, pressure hysteresis/recovery, forecasts, stale evidence and envelope checks |
| PR-12 | `inference_qualification*.py`, isolated-worker extensions, matrix exporter and documentation | Profile/preflight validation, quality/service gates, integrity, failures and aggregation |

Shared actor/cache/API integration spans these boundaries. This delivery extends the current
uncommitted PR-01–PR-06 working state; it does not publish remote PRs or include unrelated UI work.

### PR-07–PR-12 validation (2026-10-10)

The complete backend-free suite passes: **651 passed, 23 skipped**, with two dependency
deprecation warnings. Added fixtures cover direct partition merging, GQA/ragged/partial pages,
quantized packing and admission gates, allocation rollback at every staged page layer, hybrid
carry/state isolation, cancellation and deadline expiry during forwards, HTTP/Python batching,
principal isolation, snapshot validation, cache-aware ordering, background telemetry lifecycle,
pressure hysteresis, exact/stale evidence, calibrated forecasts and matrix aggregation.

Ruff reports **65 pre-existing findings, zero new findings**. The comparison checks each finding
against the same source line at HEAD; no automatic rewrites were used. Diff checks pass. Filtered
paged, hybrid, microbenchmark and 30-minute/128K-class soak profiles list successfully without
initializing Metal. The earlier native-precision baseline profiles retain their identifiers.

The isolated MLX import fails with `[metal::load_device] No Metal device available` (exit 1).
`xcrun -f metal` also fails because the Metal compiler is unavailable (exit 72). Consequently no
Metal compilation, checkpoint parity, quality, throughput, stress or soak measurements are claimed.
Gates D–F remain **pending**, and implementation readiness does not promote GPU qualification.
