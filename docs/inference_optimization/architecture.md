# Native inference optimization architecture

Immutable comparison baseline: `b802fd2a0e9cd500a96e4fe21e2c9b3bb43df902`.
The checkout was dirty at startup. `baseline/starting-status.txt` and
`baseline/starting-user-changes.patch` preserve that starting state. Existing
P1/P0.1 baseline identifiers remain historical and have not been replaced.

The native model registry, InferenceActor, BatchScheduler, CacheManager and
transaction owners remain authoritative. Execution mode selects dense, hybrid
or direct paged state. Request-scoped NativeMathPolicy independently selects
arithmetic; its default is reference attention. Cache-owned `attend()` takes
precedence over ordinary dense attention.

The existing dense batch adapter supports qwen2, qwen3 and llama. Hybrid state
uses the existing hybrid adapter/runner; general causal/task bindings are
enumerated in the preserved baseline registry inventory. Explicit FP16 INT8
affine execution retains rowwise gather-QMM projections and FP32 per-query
attention reductions; packed weights and FP16 KV storage remain unchanged.

Dense batching currently packs full histories into padded layer arrays, executes
and evaluates the model, constructs independent replacement caches, evaluates
them, then calls `commit_evaluated`. That transaction validates model epochs,
component shapes, cancellation and reservations before adopting state. These
ownership rules are unchanged. Persistent storage is a later phase.

Qualification now resolves correctness prerequisites from explicit execution
backends. Reference dense and dense/hybrid batching require dense parity;
paged requires paged parity. With no selection, both remain required for
compatibility. Attention results cannot borrow evidence from another backend.
The six original pinned P1 checkpoints remain in `p1_profiles.CHECKPOINTS`.
The four-model P0.1 track remains additional and separate.

Admission checks source and test digests, Git revision, software/package identity,
hardware, checkpoint file hashes/revision, exact pytest commands and JUnit case
identities, numerical comparisons, isolation, cleanup and artifact checksums.
Passing correctness permits measurements; production promotion still requires
the existing performance gates. A completed benchmark worker is not a
performance qualification pass.

References actually inspected: installed mlx-lm **0.31.3** (`models/base.py`,
`KVCache`, `BatchKVCache`); pinned mlx-lm
`93e6d4a6447b3080861985947f0820e76fcb957d` and MLX
`1f8e74e3f12f31365464a6867c6579f0e9b29d85` source snapshots already in
`qualification/results/p01_math/reference-inspection`; Rapid-MLX
`b638c2ab6cebe01446ee34b53230b9ef44881453` (`QuantizedBatchKVCache`,
`_install_batched_mask_safe_quantized_sdpa`, packed-prefix reads,
`FreeKVCacheBlockQueue`); oMLX
`03bc3dd7118a5f1bbf9cb9ac119f551c77477b09`
(`omlx/cache/paged_cache.py`: free queue, `PagedCacheManager.free_block`,
`_maybe_evict_cached_block`). Fetched sources and checksums are under
`baseline/references/`. `omlx/cache/block_aware_cache.py` returned HTTP 404;
the actual lifecycle symbols were found in `omlx/cache/paged_cache.py`.
No external code was copied into the runtime and no dependency was added.

Floating SDPA and packed quantized matmul are separate upstream paths. Rapid's
rank-5 grouped-score mask repair inserts a dimension only for its precise
per-batch mask shape. oMLX's inspected paged manager provides logical block
lifecycle, not evidence for direct page-table GPU attention.
