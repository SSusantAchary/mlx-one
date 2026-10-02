# Cache Runtime V1

mlx-one keeps model cache semantics in `mlx_one.core.cache` and runtime ownership in
`mlx_one.engine`. Models still receive ordinary cache objects; scheduling and allocation policy do
not enter model implementations.

```text
Generation / ModelRunner
          |
          v
     CacheManager -------- CachePlan
          |
    +-----+--------+
    |              |
CacheHandle    Prefix index
    |              |
CacheBundle     BlockPool
    |
   MLX
```

## Current-state audit

| Existing component | Responsibility | V1 treatment |
| --- | --- | --- |
| `core.cache.KVCache` | Dense attention K/V and 256-token capacity growth | Retained as the default oracle backend |
| `core.cache.QuantizedKVCache` | Mixed native/q4/q8 K/V storage | Retained behind managed dense handles |
| `EncoderDecoderKVCache`, `ConvCache`, Qwen3.5 linear cache | Cross-attention and hybrid state | Represented in plans; block reuse disabled |
| `engine.cache.CacheBundle` | Heterogeneous cache lifecycle | Extended with logical/allocated accounting and close state |
| `engine.prefix_cache.PrefixCache` | Whole-snapshot LRU reuse | Retained for dense compatibility |
| `engine.block_cache.BlockAllocator` | Experimental references and page tables | Compatibility wrapper over production `BlockPool` |
| `server.GenerationEngine` | Prompt reuse and generation adaptation | Delegates construction, reuse, reservations, and cleanup to `CacheManager` |
| `server.MemoryAdmission` | Machine/unified-memory safety | Retained; scheduler rechecks admission before execution |
| `engine.TokenBudgetScheduler` | Future token scheduling policy | Unchanged; it is not the production server executor |
| `server.GenerationScheduler` | Active cooperative execution scheduler | Adds admission callbacks and deterministic iterator cleanup |

## Ownership and lifecycle

Each request receives one `CacheHandle`. `CacheManager` is the sole owner of handle registration,
prefix indexes, reservations, and block pools. Generation uses the bundle and releases the handle
from a `finally` path. Explicit double release is an ownership error.

Reservations move through `REQUESTED`, `GRANTED`, `COMMITTED`, and `RELEASED`. Statistics separate
logical state from allocated capacity and unique physical pool storage. Prefix blocks are pinned;
live requests hold references. A physical block is reclaimable only when both counts are zero.

## Prefix reuse

Dense mode uses the compatible snapshot LRU. Block mode hashes canonical runtime identity and token
blocks, shares only sealed complete blocks, and always leaves at least one prompt token for prefill.
Generated suffixes and partial tails are never published. The fingerprint covers model/revision,
local asset identity, adapter, tokenizer, template, topology, precision, context/RoPE settings, and
the configured namespace.

## Support matrix

| Backend | Supported | Notes |
| --- | --- | --- |
| Dense KV | Yes, default | All currently supported text families |
| Quantized dense KV | Yes | Existing f16/bf16/q4/q8 behavior |
| Snapshot prefix | Yes | Dense, complete restorable bundles only |
| Block KV | Opt-in | Qwen2/Qwen2.5, text-only, B=1, f16 |
| Block APC | Opt-in | Same qualification boundary as block KV |
| Hybrid prefix restore | No | Attention-only restore is unsafe |
| Rotating/quantized block KV | No | Follow-up work |
| Chunked prefill / continuous batching | No | Contracts are prepared; scheduling is not implemented |

Server configuration keeps dense defaults. Use `--cache-backend block --kv-block-size 32` to opt in;
`--kv-cache-budget-mib` bounds cache storage explicitly. `/v1/runtime` reports the resolved server
configuration, cache capabilities/statistics, scheduler state, and unified-memory budget.
