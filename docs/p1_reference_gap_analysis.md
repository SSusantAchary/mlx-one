# P1 reference gap analysis

Base: `6d2c9a1bbc36f1972b468ca841977e3b342fbb36`. Native qualification pending.
All references are Apache-2.0; Rapid-MLX also carries a NOTICE describing its
vllm-mlx ancestry. Implement designs independently; do not add execution dependencies.

| Pin / source | Concept and inspected lines | mlx-one equivalent / delta | Tests / decision |
|---|---|---|---|
| [oMLX 64bf5d1](https://github.com/jundot/omlx/tree/64bf5d1ba6336b45955c00f7fe0e46acd5c85095) `omlx/cache/paged_cache.py` | CacheBlock 133–176, allocation 649, release 762, prefix chain 1040–1100 | Versioned BlockPool, segmented arena, BlockPrefixIndex already exist; preserve one owner and prove INT8 identity separation | `tests/test_paged_cache.py` 470–1103: sharing/acquisition/free queue; adapt |
| Same pin, `omlx/scheduler.py` | BatchGenerator delegation 6–34; chunk state 603–631 | Existing native actor and budget scheduler; extend adapters rather than delegate | `tests/test_scheduler_chunked_prefill.py` 657 onward: interleaved prefill/decode; adapt |
| [vllm-mlx 9a939f8](https://github.com/waybarrios/vllm-mlx/tree/9a939f8b59bddaf98ccab865f4fa7e2fe48c6d65) `vllm_mlx/paged_cache.py` | Metadata/refcounts 84–115; COW 1029–1044 | Ownership is not proof of direct GPU attention; reuse mlx-one direct segmented kernels | `tests/test_prefix_cache_trim_failclosed.py` 95 onward; adapt lifecycle, reject backend replacement |
| Same pin, `vllm_mlx/scheduler.py`, `bench_serve.py` | External BatchGenerator 22, chunk patch 203; capacity classes 94–106 and classification 212 | Native scheduling, source-scoped serving evidence and honest capacity outcomes | `tests/test_batch_reshuffle_metadata.py` 166–661, bench workload tests; adapt |
| [Rapid-MLX b638c2a](https://github.com/raullenchai/Rapid-MLX/tree/b638c2ab6cebe01446ee34b53230b9ef44881453) `rapid_mlx/hybrid_state_checkpoints.py` | Exact-position constraint 1–21, immutable checkpoints 72, positive type identification 198–230 | Native typed carries and snapshots already exist; extend to attention pages without treating carries as KV | Hybrid lifecycle/state-isolation tests; adapt |
| Same pin, `rapid_mlx/bench/tier_runner.py` | Clean server session 576, profile execution 777, speed probe 453 | Existing isolated workers; add matched INT8 reference and declared P1 gates | `tests/test_bench_confirms_a_regression_against_its_own_base.py` 158–324; adapt |
| All pins | SSD tiers, MTP, independent server/kernel packs | Outside P1 | Defer / reject as runtime dependencies |

Source inspection establishes design relevance, not performance qualification.
P0 CPU baseline: 704 passed, 23 skipped; hardware acceptance remains unmeasured.
Confirmed blockers: 4-bit-only native loader/converter; Qwen2-only dense/paged
eligibility; no shared execution adapter; dense-only hybrid attention components.

## Resulting decisions

Adopt ownership and exact-boundary lifecycle principles through the existing
mlx-one owner/index, not another cache implementation. Adapt scheduling patterns
through BatchScheduler and InferenceActor; the references' external BatchGenerator
delegation is not mlx-one's execution path. Keep direct attention in the existing
segmented arena/kernel implementation. Adapt native adapter gating and full hybrid
state validation. Defer SSD tiers, speculation and MTP. Reject external engines as
runtime backends. Shared ancestry is shared design history, not independent
validation of the targets.

Source/module inspection is separate from checkpoint execution. All six P1 pins
are missing from the local offline cache at delivery; native execution, packed
checkpoint validation and performance remain pending. Synthetic runtime fixtures
exercise ownership and scheduling contracts without establishing checkpoint math.
